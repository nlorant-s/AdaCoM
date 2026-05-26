# -*- coding: utf-8 -*-
"""The explorer module"""
from __future__ import annotations

import asyncio
import math
import os
import random
import re
import time
import traceback
from collections import deque
from dataclasses import replace
from typing import List, Optional

import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from trinity.buffer.buffer import get_buffer_reader
from trinity.buffer.pipelines.experience_pipeline import ExperiencePipeline
from trinity.buffer.task_scheduler import get_taskset_scheduler
from trinity.common.config import Config
from trinity.common.constants import (
    ROLLOUT_WEIGHT_SYNC_GROUP_NAME,
    RunningStatus,
    SyncMethod,
    SyncStyle,
)
from trinity.common.models import create_inference_models
from trinity.common.models.utils import get_checkpoint_dir_with_step_num
from trinity.explorer.scheduler import Scheduler
from trinity.manager.state_manager import StateManager
from trinity.manager.synchronizer import Synchronizer
from trinity.utils.annotations import Experimental
from trinity.utils.log import get_logger
from trinity.utils.monitor import MONITOR, gather_metrics
from trinity.utils.plugin_loader import load_plugins
from trinity.utils.timer import Timer


class Explorer:
    """Responsible for exploring the taskset."""

    def __init__(self, config: Config):
        self.logger = get_logger(config.explorer.name, in_ray_actor=True)
        load_plugins()
        self.state = StateManager(
            path=config.checkpoint_job_dir, explorer_name=config.explorer.name, config=config
        )
        explorer_state = self.state.load_explorer()
        self.explore_step_num = explorer_state.get("latest_iteration", 0)
        self.last_sync_step = self.explore_step_num if self.explore_step_num > 0 else -1
        self.last_monitored_step = self.explore_step_num if self.explore_step_num > 0 else -1
        self.synchronizer = Synchronizer.get_actor(config)
        self.config = config
        self.models, self.auxiliary_models = create_inference_models(config)
        self.experience_pipeline = self._init_experience_pipeline()
        self.taskset = (
            get_taskset_scheduler(explorer_state=explorer_state, config=config)
            if self.config.mode not in {"bench", "serve"}
            else None
        )
        self.scheduler = None
        self.monitor = MONITOR.get(self.config.monitor.monitor_type)(
            project=self.config.project,
            group=self.config.group,
            name=self.config.name,
            role=self.config.explorer.name,
            config=config,
        )
        if config.explorer.over_rollout.ratio > 0.0:
            # Scale by num_agents when agent_expand == "all" (scheduler splits per agent)
            effective_batch_size = config.buffer.batch_size
            train_taskset = config.buffer.explorer_input.taskset
            if train_taskset:
                wa = train_taskset.workflow_args or {}
                if wa.get("agent_expand") == "all" and wa.get("agent_models"):
                    effective_batch_size *= len(wa["agent_models"])
            self.min_wait_num = math.ceil(
                effective_batch_size * (1 - config.explorer.over_rollout.ratio)
            )
            self.logger.info(
                f"Over rollout is enabled. Explorer will only wait for {self.min_wait_num} task@agents in each step."
            )
        else:
            self.min_wait_num = None
        self.use_nccl_sync = self.config.synchronizer.sync_method == SyncMethod.NCCL
        self.pending_eval_tasks = deque()

        # For checkpoint weights update
        # Use explorer to periodically load the latest model weights and
        # boradcast to all rollout models
        self.enable_lora = self.config.explorer.rollout_model.enable_lora
        self.model_version = -1
        self.last_sync_successful = True
        self.eval_start_time = None
        self.explore_start_time = None

        # Track whether metric has reached threshold for dynamic eval interval
        self.metric_above_threshold = False
        self.logger.info("Finished initializing Explorer.")

    async def setup_weight_sync_group(
        self, master_address: str, master_port: int, state_dict_meta: List = None
    ):
        # In checkpoint mode, we use explorer to store the model weights which has no rank
        base_offset = 1 if self.use_nccl_sync else 0
        world_size = (
            len(self.models) * self.config.explorer.rollout_model.tensor_parallel_size + base_offset
        )
        self.logger.info(
            f"Initialize process group for weight synchronization, "
            f"master_address={master_address}, master_port={master_port}, "
            f"world_size={world_size}, rank_offset={base_offset}"
        )
        # TODO: save state_dict in models
        refs = [
            model.init_process_group.remote(
                master_address=master_address,
                master_port=master_port,
                rank_offset=i * self.config.explorer.rollout_model.tensor_parallel_size
                + base_offset,
                world_size=world_size,
                group_name=ROLLOUT_WEIGHT_SYNC_GROUP_NAME,
                explorer_name=self.config.explorer.name,
                timeout=self.config.synchronizer.sync_timeout,
                state_dict_meta=state_dict_meta,
            )
            for i, model in enumerate(self.models)
        ]
        await asyncio.gather(*refs)

    async def _checkpoint_weights_update(self, step_num: Optional[int] = None) -> int:
        self.logger.info(f"Start to update model weights from checkpoint at step {step_num}.")
        step_num = await self.synchronizer.set_model_state_dict_with_step_num.remote(step_num)
        await asyncio.gather(*[model.sync_model.remote(step_num) for model in self.models])
        self.logger.info(f"Model weights updated to checkpoint at step {step_num}.")
        return step_num  # type: ignore

    async def _pull_latest_weights(self):
        self.logger.info("Start to pull latest model weights.")
        new_version = await self.synchronizer.wait_new_model_state_dict.remote(self.model_version)
        if new_version > self.model_version:
            if self.model_version != -1:
                self.logger.info(f"New model weights version: {new_version}")
                await asyncio.gather(
                    *[model.sync_model.remote(new_version) for model in self.models]
                )
            self.model_version = new_version
            self.last_sync_step = self.explore_step_num
            self.last_sync_successful = True
        else:
            self.logger.warning(
                f"No new model weights found, current version: {self.model_version}"
            )
            self.last_sync_successful = False

    async def _nccl_weights_update(self):
        new_version = await self.synchronizer.ready_to_nccl_sync.remote(
            "explorer", self.model_version
        )
        if new_version is None:
            self.logger.info("Trainer is not ready to sync weight. Skipping sync weight.")
            self.last_sync_successful = False
            return
        self.model_version = new_version
        await asyncio.gather(
            *[model.sync_model.remote(self.model_version) for model in self.models]
        )
        self.last_sync_step = self.explore_step_num
        self.last_sync_successful = True

    async def prepare(self) -> None:
        """Preparation before running."""
        try:
            # prepare experience pipeline
            if self.experience_pipeline:
                await self.experience_pipeline.prepare.remote()
            self.logger.info("Experience pipeline is ready.")
            # make sure all rollout models are ready
            run_api_ref = [model.prepare.remote() for model in self.models]
            run_api_ref.extend(
                model.prepare.remote() for models in self.auxiliary_models for model in models
            )
            await asyncio.gather(*run_api_ref)
            self.logger.info("All models are ready.")

            if not self.use_nccl_sync:
                master_address, master_port = await self.models[0].get_available_address.remote()
                await self.setup_weight_sync_group(master_address, master_port)
            if self.config.mode != "serve":
                self.scheduler = Scheduler(self.config, self.models, self.auxiliary_models)
                await self.scheduler.start()
            if self.config.explorer.eval_on_startup and self.explore_step_num == 0:
                await self.eval()

            await self.synchronizer.set_explorer_status.remote(RunningStatus.REQUIRE_SYNC)
        except Exception as e:
            self.logger.error(f"Error during explorer preparation: {traceback.format_exc()}")
            await self.shutdown()
            raise e

    async def get_weight(self, name: str) -> torch.Tensor:
        """Get the weight of the loaded model (For checkpoint weights update)."""
        return self.state_dict[name]

    async def explore(self) -> str:
        """
        The timeline of the exploration process:
                 | <--------------------------------- one period -------------------------------------> |
        explorer | <---------------- step_1 --------------> |                                           |
                 |   | <---------------- step_2 --------------> |                                       |
                 |      ...                                                                             |
                 |          | <---------------- step_n ---------------> |                               |
                 |                  | <---------------------- eval --------------------> | <-- sync --> |
                 |--------------------------------------------------------------------------------------|
        trainer  | <-- idle --> | <-- step_1 --> | <-- step_2 --> | ... | <-- step_n --> | <-- sync --> |
        """
        while True:
            try:
                # Check if trainer has stopped, if so, shutdown explorer
                trainer_status = await self.synchronizer.get_trainer_status.remote()
                if trainer_status == RunningStatus.STOPPED:
                    self.logger.info("Trainer has stopped, shutting down explorer.")
                    await self.save_checkpoint(sync_weight=False)
                    await self.synchronizer.set_explorer_status.remote(
                        RunningStatus.STOPPED,
                        old_status=RunningStatus.RUNNING
                        if self.last_sync_successful
                        else RunningStatus.REQUIRE_SYNC,
                    )
                    await self.shutdown()
                    break

                self.logger.info(f"Explore step {self.explore_step_num + 1} started.")
                explore_contionue = await self.explore_step()
                if not explore_contionue:
                    # TODO: support eval on last checkpoint
                    break
                if self.need_eval():
                    # Save checkpoint before eval when metric is below threshold
                    await self._save_before_eval_if_needed()
                    await self.eval()
                if await self.need_sync():
                    await self.sync_weight()
            except Exception:
                self.logger.error(f"Error in Explorer: {traceback.format_exc()}")
                break
        self.logger.info(
            f"--------------------\n> Explorer ({self.config.explorer.name}) finished.\n--------------------"
        )
        return self.config.explorer.name

    async def explore_step(self) -> bool:
        if self.explore_start_time is None:
            self.explore_start_time = time.time()
        try:
            tasks = await self.taskset.read_async()
        except StopAsyncIteration:
            self.logger.warning("No more tasks to explore. Stop exploring.")
            await self.save_checkpoint(sync_weight=False)
            await self.synchronizer.set_explorer_status.remote(
                RunningStatus.STOPPED,
                old_status=RunningStatus.RUNNING
                if self.last_sync_successful
                else RunningStatus.REQUIRE_SYNC,
            )
            await self.shutdown()
            return False
        self.scheduler.schedule(tasks, batch_id=self.explore_step_num + 1)
        self.explore_step_num += 1
        return True

    async def need_sync(self) -> bool:
        if self.config.synchronizer.sync_style == SyncStyle.FIXED:
            if self.explore_step_num <= self.config.synchronizer.sync_offset:
                return False
            require_sync = (
                self.explore_step_num - self.config.synchronizer.sync_offset
            ) % self.config.synchronizer.sync_interval == 0
        else:
            require_sync = False
            if self.config.synchronizer.sync_style == SyncStyle.DYNAMIC_BY_EXPLORER:
                delta = self.explore_step_num - self.last_sync_step
                if delta >= self.config.synchronizer.sync_interval:
                    require_sync = True
            else:
                require_sync = await (
                    self.synchronizer.get_trainer_status.remote() == RunningStatus.REQUIRE_SYNC
                )
        if require_sync and self.last_sync_successful:
            await self.synchronizer.set_explorer_status.remote(
                RunningStatus.REQUIRE_SYNC, old_status=RunningStatus.RUNNING
            )
        return require_sync

    def _get_current_eval_interval(self) -> int:
        """Get current eval interval based on metric threshold status."""
        # If dynamic eval interval is not configured, use fixed eval_interval
        if self.config.explorer.eval_interval_low_score is None:
            return self.config.explorer.eval_interval

        if self.metric_above_threshold:
            return self.config.explorer.eval_interval_high_score or self.config.explorer.eval_interval
        return self.config.explorer.eval_interval_low_score

    def _get_current_eval_repeat_times(self) -> int:
        """Get current eval repeat_times based on metric threshold status."""
        # If dynamic eval interval is not configured, return None to use taskset default
        if self.config.explorer.eval_interval_low_score is None:
            return None

        if self.metric_above_threshold:
            return self.config.explorer.eval_repeat_times_high_score
        return self.config.explorer.eval_repeat_times_low_score

    def need_eval(self) -> bool:
        eval_interval = self._get_current_eval_interval()
        return self.explore_step_num % eval_interval == 0

    async def _save_before_eval_if_needed(self) -> None:
        """Save checkpoint before eval when save_on_metric is configured.

        When save_on_metric is configured:
        - Always save before eval (regardless of metric threshold)
        - When metric > threshold, trainer also saves by save_interval

        When save_only_on_passing_eval is True:
        - Skip saving before eval (save only after eval if metric > threshold)
        """
        if not self.config.trainer.save_on_metric:
            return

        # Skip saving before eval when save_only_on_passing_eval is enabled
        if self.config.trainer.save_only_on_passing_eval:
            return

        self.logger.info(
            f"Saving checkpoint before eval at step {self.explore_step_num}."
        )
        try:
            trainer = ray.get_actor(self.config.trainer.name)
            ray.get(trainer.save_checkpoint_on_metric.remote())
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint before eval: {e}")

    async def eval(self):
        """Evaluation on all evaluation data samples."""
        self.eval_start_time = time.time()
        if len(self.config.buffer.explorer_input.eval_tasksets) == 0:
            self.logger.warning("No evaluation data samples. Skip evaluation.")
            return
        self.logger.info(f"Evaluation at step {self.explore_step_num} started.")

        if self.config.buffer.explorer_input.default_eval_workflow_type:
            self.logger.info(
                f"Use '{self.config.buffer.explorer_input.default_eval_workflow_type}' for evaluation."
            )

        # Get dynamic repeat_times if configured
        dynamic_repeat_times = self._get_current_eval_repeat_times()
        if dynamic_repeat_times is not None:
            self.logger.info(
                f"Using dynamic eval repeat_times={dynamic_repeat_times} "
                f"(metric_above_threshold={self.metric_above_threshold})"
            )

        for eval_taskset_config in self.config.buffer.explorer_input.eval_tasksets:
            # Per-taskset eval_interval: skip if this taskset has its own interval
            # and the current step doesn't match it
            taskset_interval = getattr(eval_taskset_config, "eval_interval", 0)
            if taskset_interval > 0 and self.explore_step_num % taskset_interval != 0:
                self.logger.info(
                    f"Skipping eval taskset '{eval_taskset_config.name}' at step {self.explore_step_num} "
                    f"(taskset eval_interval={taskset_interval})"
                )
                continue
            self.logger.info(
                f"Evaluation on {eval_taskset_config.name} at step {self.explore_step_num} started."
            )
            eval_taskset = get_buffer_reader(eval_taskset_config)
            eval_batch_id = f"{self.explore_step_num}/{eval_taskset_config.name}"
            self.pending_eval_tasks.append((self.explore_step_num, eval_taskset_config.name))

            # Collect all eval tasks first
            all_eval_data = []
            while True:
                try:
                    data = await eval_taskset.read_async()
                    # Override repeat_times if dynamic eval is configured
                    if dynamic_repeat_times is not None:
                        data = [replace(task, repeat_times=dynamic_repeat_times) for task in data]
                    all_eval_data.extend(data)
                except StopAsyncIteration:
                    break

            # Subsample if max_eval_tasks is set
            max_eval_tasks = getattr(eval_taskset_config, "max_eval_tasks", 0)
            if max_eval_tasks > 0 and len(all_eval_data) > max_eval_tasks:
                total_available = len(all_eval_data)
                all_eval_data = random.sample(all_eval_data, max_eval_tasks)
                self.logger.info(
                    f"Subsampled {max_eval_tasks} tasks from {eval_taskset_config.name} "
                    f"(total available: {total_available})"
                )

            if all_eval_data:
                self.scheduler.schedule(all_eval_data, batch_id=eval_batch_id)

    async def benchmark(self) -> bool:
        """Benchmark the model checkpoints."""
        # benchmark on the latest checkpoint
        if self.config.explorer.bench_on_latest_checkpoint:
            self.explore_step_num = await self._checkpoint_weights_update()
            await self.eval()
            await self._finish_eval_step(prefix="bench")
            return True

        # benchmark on base model
        if self.config.explorer.eval_on_startup:
            await self._finish_eval_step(prefix="bench")

        # benchmark on all checkpoints
        all_ckp_steps = sorted(
            [
                int(ckp.split("global_step_")[-1])
                for ckp in os.listdir(self.config.checkpoint_job_dir)
                if os.path.isdir(os.path.join(self.config.checkpoint_job_dir, ckp))
                and ckp.startswith("global_step_")
            ]
        )
        for step_num in all_ckp_steps:
            if step_num <= self.explore_step_num:
                continue
            self.explore_step_num = await self._checkpoint_weights_update(step_num=step_num)
            await self.eval()
            await self._finish_eval_step(prefix="bench")
        return True

    async def save_checkpoint(self, sync_weight: bool = False) -> None:
        if self.scheduler:
            await self._finish_steps(
                self.last_monitored_step + 1, self.explore_step_num, self.model_version
            )
            self.last_monitored_step = self.explore_step_num

        if sync_weight:
            # sync weights
            self.logger.info(f"Explorer sync_weights at step {self.explore_step_num} started.")
            if self.use_nccl_sync:
                await self._nccl_weights_update()
            else:  # pull weights from Synchronizer
                await self._pull_latest_weights()
            self.logger.info(
                f"Explorer sync_weights at step {self.explore_step_num} finished, model version = {self.model_version}."
            )

        # save explore checkpoint
        self.state.save_explorer(
            current_step=self.explore_step_num,
            taskset_states=self.taskset.state_dict() if self.taskset else [],
        )

    async def sync_weight(self) -> None:
        """Synchronize model weights."""
        # call this method before training start to load the latest model weights
        await self.save_checkpoint(sync_weight=True)

    async def _finish_steps(self, start_step: int, end_step: int, model_version: int) -> None:
        for step in range(start_step, end_step + 1):
            self.logger.info(f"Waiting for step {step}")
            await self._finish_explore_step(step=step, model_version=model_version)
            await self._finish_eval_step(step=step)

        # Record the time: read_task + explore_step (>=1) + eval (if any)
        if self.explore_start_time is not None:
            metric = {"time/explorer_sync_interval": time.time() - self.explore_start_time}
            self.explore_start_time = None
            self.monitor.log(metric, step=end_step)

    async def _finish_explore_step(self, step: int, model_version: int) -> None:
        # Capture scheduled count before get_results (over_rollout needs it)
        scheduled_num = self.scheduler.task_num_map.get(step, 0)

        metric = {"rollout/model_version": model_version}
        with Timer(metric, "time/wait_explore_step"):
            statuses, exps = await self.scheduler.get_results(
                batch_id=step, min_num=self.min_wait_num
            )
        pipeline_metrics = await self.experience_pipeline.process.remote(exps)
        self.taskset.update(pipeline_metrics)
        metric.update(pipeline_metrics)
        if statuses:
            all_task_metrics = [status.metrics[0] for status in statuses]
            metric["rollout/finished_task_count"] = len(statuses)

            # Per-agent rollout metrics (search_calls, iterations, etc.)
            agent_groups = {}
            for m in all_task_metrics:
                agent = m.get("_agent_model", "_default")
                agent_groups.setdefault(agent, []).append(m)
            for agent, agent_metrics in agent_groups.items():
                safe_name = str(agent).rsplit("/", 1)[-1]
                agent_gathered = gather_metrics(agent_metrics, f"rollout/agent/{safe_name}")
                # Remove score/mean — duplicates experience_pipeline/rollout/agent/{agent}/final_score/mean
                agent_gathered.pop(f"rollout/agent/{safe_name}/score/mean", None)
                metric.update(agent_gathered)
                metric[f"rollout/agent/{safe_name}/task_count"] = len(agent_metrics)

            # Over-rollout logging: log avg iterations for finished vs discarded rollouts
            if self.min_wait_num is not None:
                finished_count = len(statuses)
                discarded_count = scheduled_num - finished_count

                finished_iterations = [
                    s.metrics[0].get("iterations", 0)
                    for s in statuses
                    if s.metrics and len(s.metrics) > 0 and "iterations" in s.metrics[0]
                ]
                avg_finished_iters = (
                    sum(finished_iterations) / len(finished_iterations)
                    if finished_iterations else 0
                )

                self.logger.info(
                    f"[Over-rollout] Step {step}: "
                    f"scheduled={scheduled_num}, finished={finished_count}, discarded={discarded_count}, "
                    f"avg_rounds(finished)={avg_finished_iters:.1f}"
                )
                metric["rollout/scheduled_task_count"] = scheduled_num
                metric["rollout/discarded_task_count"] = discarded_count
                metric["rollout/avg_rounds_finished"] = avg_finished_iters

            # Collect and aggregate timing data from BCP workflow if available
            try:
                from trinity.common.workflows.envs.browse_comp_plus.bcp_simple_react_workflow import BCPSimpleToolReActWorkflow
                timing_data = BCPSimpleToolReActWorkflow.batch_timing_data
                # self.logger.info(f"Found {len(timing_data)} timing records in batch_timing_data")

                if timing_data:
                    # Calculate statistics for each timing phase
                    timing_metrics = gather_metrics(timing_data, "workflow/timing")
                    metric.update(timing_metrics)
                    self.logger.info(f"Adding timing metrics to wandb: {timing_metrics.keys()}")

                    # Also add individual phase averages for clarity
                    for phase in ['model_initialization', 'logger_setup', 'worker_creation',
                                 'task_execution', 'judging', 'experience_extraction', 'total_time']:
                        phase_times = [d.get(phase, 0) for d in timing_data if phase in d]
                        if phase_times:
                            metric[f"workflow/timing_{phase}_avg"] = sum(phase_times) / len(phase_times)

                    # Clear batch timing data for next batch
                    BCPSimpleToolReActWorkflow.batch_timing_data = []

                    self.logger.info(f"Workflow timing metrics added: {[k for k in metric.keys() if 'workflow/timing' in k]}")
                else:
                    self.logger.debug("No timing data found in BCPSimpleToolReActWorkflow.batch_timing_data")

            except ImportError as e:
                self.logger.debug(f"BCP workflow not imported: {e}")
            except Exception as e:
                self.logger.error(f"Failed to collect workflow timing data: {e}", exc_info=True)

            self.monitor.log(metric, step=step)

    async def _finish_eval_step(self, step: Optional[int] = None, prefix: str = "eval") -> None:
        if not self.pending_eval_tasks:
            return
        step = step or self.explore_step_num
        metric = {}
        while self.pending_eval_tasks:
            eval_step, eval_task_name = self.pending_eval_tasks[0]
            if eval_step != step:
                return
            self.pending_eval_tasks.popleft()
            eval_batch_id = f"{step}/{eval_task_name}"
            # Apply over_rollout to eval (eval_ratio=-1 inherits from ratio, 0 waits for all)
            eval_over_ratio = self.config.explorer.over_rollout.eval_ratio
            if eval_over_ratio < 0:
                eval_over_ratio = self.config.explorer.over_rollout.ratio
            if eval_over_ratio > 0.0:
                scheduled = self.scheduler.task_num_map.get(eval_batch_id, 0)
                eval_min_num = max(1, math.ceil(scheduled * (1 - eval_over_ratio)))
            else:
                eval_min_num = None  # wait for all
            scheduled_num = self.scheduler.task_num_map.get(eval_batch_id, 0)
            statuses, _ = await self.scheduler.get_results(
                batch_id=eval_batch_id, min_num=eval_min_num
            )
            all_run_metrics = [status.metrics[0] for status in statuses]
            finished_num = len(statuses)
            unfinished_num = max(0, scheduled_num - finished_num)
            metric[f"{prefix}/{eval_task_name}/finished_task_count"] = finished_num
            metric[f"{prefix}/{eval_task_name}/unfinished_task_count"] = unfinished_num
            metric[f"{prefix}/{eval_task_name}/scheduled_task_count"] = scheduled_num
            # Overall aggregation (all agents/searchers combined)
            metric.update(
                gather_metrics(
                    all_run_metrics,
                    f"{prefix}/{eval_task_name}",
                    output_stats=["mean", "std"],
                )
            )
            # --- Per-agent / per-searcher / cross eval breakdown ---
            # Task-level metrics (all_run_metrics) contain mean@k / pass@k from
            # calculate_task_level_metrics. With agent_expand=all, task is encoded as
            # task@agent, so _agent_model and _searcher_type are preserved per task.
            eval_prefix = f"{prefix}/{eval_task_name}"
            stats = ["mean", "std"]

            agent_task_groups = {}
            searcher_task_groups = {}
            for m in all_run_metrics:
                agent = m.get("_agent_model")
                searcher = m.get("_searcher_type")
                if agent is not None:
                    agent_task_groups.setdefault(agent, []).append(m)
                if searcher is not None:
                    searcher_task_groups.setdefault(searcher, []).append(m)

            # Per-agent (with mean@k / pass@k)
            if len(agent_task_groups) > 1:
                for agent_val, agent_metrics in agent_task_groups.items():
                    safe_name = str(agent_val).rsplit("/", 1)[-1]
                    group_prefix = f"{eval_prefix}/agent_model/{safe_name}"
                    metric[f"{group_prefix}/task_count"] = len(agent_metrics)
                    metric.update(gather_metrics(agent_metrics, group_prefix, output_stats=stats))

            # Per-searcher (with mean@k / pass@k)
            if len(searcher_task_groups) > 1:
                for st_val, st_metrics in searcher_task_groups.items():
                    group_prefix = f"{eval_prefix}/searcher_type/{st_val}"
                    metric[f"{group_prefix}/task_count"] = len(st_metrics)
                    metric.update(gather_metrics(st_metrics, group_prefix, output_stats=stats))

            # Cross: agent × searcher (task-level, with mean@k / pass@k)
            # Skip when only 1 agent — cross degenerates to per-searcher (already emitted above)
            if len(agent_task_groups) > 1:
                for agent_val, agent_metrics in agent_task_groups.items():
                    safe_agent = str(agent_val).rsplit("/", 1)[-1]
                    sub = {}
                    for m in agent_metrics:
                        st = m.get("_searcher_type")
                        if st is not None:
                            sub.setdefault(st, []).append(m)
                    if len(sub) > 1:
                        for st, st_metrics in sub.items():
                            cross_prefix = f"{eval_prefix}/agent_model/{safe_agent}/{st}"
                            metric[f"{cross_prefix}/task_count"] = len(st_metrics)
                            metric.update(gather_metrics(st_metrics, cross_prefix, output_stats=stats))
        # Compute adjusted score metrics: treat unfinished tasks as score=0
        # For any "score" metric with /mean stat, adjusted = original * finished / scheduled
        adjusted_metrics = {}
        for key, val in metric.items():
            if "/score/" in key and key.endswith("/mean") and isinstance(val, (int, float)):
                # Extract the eval_task_name from the key to find the right counts
                # Key format: {prefix}/{eval_task_name}/[subgroup/]score/.../mean
                for task_name_candidate in [
                    k.split("/")[1]
                    for k in metric
                    if k.endswith("/scheduled_task_count")
                ]:
                    task_prefix = f"{prefix}/{task_name_candidate}/"
                    if key.startswith(task_prefix):
                        s = metric.get(f"{prefix}/{task_name_candidate}/scheduled_task_count", 0)
                        f = metric.get(f"{prefix}/{task_name_candidate}/finished_task_count", 0)
                        if s > 0:
                            adjusted_metrics[key.removesuffix("/mean") + "/adjusted_mean"] = val * f / s
                        break
        metric.update(adjusted_metrics)

        if self.eval_start_time is not None:
            metric.update({"time/eval": time.time() - self.eval_start_time})
            self.eval_start_time = None
        self.monitor.log(metric, step)

        # Metric-based dynamic eval interval update
        if self.config.trainer.save_on_metric:
            metric_pattern = self.config.trainer.save_on_metric
            threshold = self.config.trainer.save_on_metric_threshold

            # Exact match: use the configured metric key directly
            metric_value = metric.get(metric_pattern)

            if metric_value is not None:
                above_threshold = metric_value > threshold

                self.logger.info(
                    f"Metric '{metric_pattern}' = {metric_value}, "
                    f"threshold = {threshold}, above = {above_threshold}"
                )

                # Update metric_above_threshold for dynamic eval interval
                old_state = self.metric_above_threshold
                self.metric_above_threshold = above_threshold
                if old_state != self.metric_above_threshold:
                    self.logger.info(
                        f"Dynamic eval state changed: metric_above_threshold={self.metric_above_threshold} "
                        f"(threshold={threshold})"
                    )

                # Sync metric_above_threshold state to trainer
                # Note: checkpoint is already saved before eval in _save_before_eval_if_needed()
                # When metric > threshold, trainer will also save by save_interval in its train loop
                try:
                    trainer = ray.get_actor(self.config.trainer.name)
                    ray.get(trainer.set_metric_above_threshold.remote(self.metric_above_threshold))

                    # Save checkpoint after eval when save_only_on_passing_eval is enabled
                    # and metric exceeds threshold
                    if (
                        self.config.trainer.save_only_on_passing_eval
                        and any_above_threshold
                    ):
                        self.logger.info(
                            f"Metric {max_metric_key}={max_metric_value} > threshold={threshold}, "
                            f"saving checkpoint at step {step}."
                        )
                        ray.get(trainer.save_checkpoint_on_metric.remote())
                except Exception as e:
                    self.logger.warning(f"Failed to sync metric state to trainer: {e}")

        elif self.config.trainer.save_only_on_passing_eval:
            # No save_on_metric configured but save_only_on_passing_eval is set: save on every eval
            try:
                trainer = ray.get_actor(self.config.trainer.name)
                self.logger.info(f"save_only_on_passing_eval without save_on_metric: saving checkpoint at step {step}.")
                ray.get(trainer.save_checkpoint_on_metric.remote())
            except Exception as e:
                self.logger.warning(f"Failed to save checkpoint after eval: {e}")

    async def shutdown(self) -> None:
        if self.scheduler:
            await self.scheduler.stop()
            self.scheduler = None
        if self.experience_pipeline:
            await self.experience_pipeline.close.remote()
            self.experience_pipeline = None
        if self.monitor:
            self.monitor.close()
            self.monitor = None
        self.logger.info(
            f"Explorer ({self.config.explorer.name}) shutdown successfully at step {self.explore_step_num}."
        )

    async def is_alive(self) -> bool:
        """Check if the explorer is alive."""
        return True

    def _init_experience_pipeline(self) -> ray.actor.ActorHandle:
        """Init experience pipeline for the explorer."""
        if self.config.mode == "bench":
            return None
        node_id = ray.get_runtime_context().get_node_id()
        return (
            ray.remote(ExperiencePipeline)
            .options(
                name=f"{self.config.explorer.name}_pipeline",
                namespace=ray.get_runtime_context().namespace,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
            )
            .remote(self.config)
        )

    @Experimental
    async def serve(self) -> None:
        """Run the explorer in serving mode.

        In serving mode, the explorer starts an OpenAI compatible server to handle requests.
        Agent applications can be deployed separately and interact with the explorer via the API.


        .. code-block:: python

            import openai


            client = openai.OpenAI(
                base_url=f"{explorer_server_url}/v1",
                api_key="EMPTY",
            )
            response = client.chat.completions.create(
                model=config.model.model_path,
                messages=[{"role": "user", "content": "Hello!"}]
            )
        """
        from trinity.explorer.api.service import ExplorerService

        self.service = ExplorerService(
            self,
            listen_address=self.config.explorer.listen_address,
            port=self.config.explorer.api_port,
        )
        await self.service.serve()
        self.server_url = f"http://{ray.util.get_node_ip_address()}:{self.service.port}"
        self.logger.info(
            f"Explorer API Server is started on {self.server_url} and listening to {self.service.listen_address}."
        )
        self.state.save_explorer_server_url(self.server_url)
        while True:
            self.explore_step_num += 1
            await asyncio.sleep(self.config.explorer.service_status_check_interval)
            # process experiences generated in the last interval
            exps = await self.service.get_all_experiences()
            metrics = await self.experience_pipeline.process.remote(exps)
            metrics.update(self.service.collect_metrics())
            self.monitor.log(metrics, self.explore_step_num)
            # get the latest checkpoint
            _, step_num = get_checkpoint_dir_with_step_num(
                self.config.checkpoint_job_dir, raise_error=False
            )
            self.service.set_latest_model_version(step_num)

    @classmethod
    def get_actor(cls, config: Config):
        """Get a Ray actor for the explorer."""
        return (
            ray.remote(cls)
            .options(
                name=config.explorer.name,
                namespace=ray.get_runtime_context().namespace,
            )
            .remote(config)
        )
