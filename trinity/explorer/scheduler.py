"""Scheduler for rollout tasks."""

import asyncio
import math
import re
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import ray

from trinity.common.config import Config
from trinity.common.experience import Experience
from trinity.common.models import InferenceModel
from trinity.common.workflows import Task
from trinity.explorer.workflow_runner import Status, WorkflowRunner
from trinity.utils.log import get_logger


@dataclass
class TaskWrapper:
    """A wrapper for a task.
    Each task can run multiple times (repeat_times) on same or different runners.
    """

    task: Task
    batch_id: Union[int, str]
    sub_task_num: int = 1  # number of sub tasks splitted from this task
    # if max_repeat_times_per_runner is set, one task may be splitted into multiple sub tasks
    results: List[Tuple[Status, List[Experience]]] = field(default_factory=list)


# Adapted from verl/trainer/ppo/metric_utils.py
def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Performs bootstrap resampling to estimate statistics of metrics.

    This function uses bootstrap resampling to estimate the mean and standard deviation
    of metrics computed by the provided reduction functions on random subsets of the data.

    Args:
        data: List of data points to bootstrap from.
        subset_size: Size of each bootstrap sample.
        reduce_fns: List of functions that compute a metric from a subset of data.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A list of tuples, where each tuple contains (mean, std) for a metric
        corresponding to each reduction function in reduce_fns.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> reduce_fns = [np.mean, np.max]
        >>> bootstrap_metric(data, 3, reduce_fns)
        [(3.0, 0.5), (4.5, 0.3)]  # Example values
    """
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Codex unbiased estimator for pass@k.

    Calculates the probability that at least one of k randomly chosen samples
    from n total samples is correct, given c correct samples.

    Formula: pass@k = 1 - C(n-c, k) / C(n, k)

    Args:
        n: Total number of samples.
        c: Number of correct samples.
        k: Number of samples to choose.

    Returns:
        The estimated pass@k probability.

    Reference:
        Chen et al., "Evaluating Large Language Models Trained on Code", 2021
        https://arxiv.org/abs/2107.03374
    """
    if n - c < k:
        # Not enough incorrect samples to fill k selections,
        # so at least one must be correct
        return 1.0
    if c == 0:
        # No correct samples
        return 0.0
    # Use math.comb for accurate integer computation
    # pass@k = 1 - C(n-c, k) / C(n, k)
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def calculate_task_level_metrics(metrics: List[Dict], is_eval: bool) -> Dict[str, float]:
    """Calculate task level metrics (mean) from multiple runs of the same task.

    Args:
        metrics (`List[Dict]`): A list of metric dictionaries from multiple runs of the same task.
        is_eval (`bool`): Whether this is an evaluation task.

    Returns:
        `Dict[str, float]`: A dictionary of aggregated metrics, where each metric is averaged over all runs.
            For eval mode, includes:
            - mean@k, std@k: mean and std of first k runs
            - pass@k: Codex unbiased estimator for binary metrics (score in {0, 1})
            - For non-binary metrics, pass@k is not computed
    """
    if not metrics:
        return {}
    aggregated_metrics: Dict[str, list] = defaultdict(list)
    for m in metrics:
        for key, value in m.items():
            aggregated_metrics[key].append(value)
    if is_eval:
        result = {}
        for key, values in aggregated_metrics.items():
            # Preserve consistent string values (e.g. _agent_model, _searcher_type)
            if values and isinstance(values[0], str):
                if len(set(values)) == 1:
                    result[key] = values[0]
                continue

            if "time/task_execution" in key or "time/run_execution" in key:
                result[key] = sum(values) / len(values)
                continue

            n_values = len(values)
            # Output mean@k for all k from 1 to n_values
            for k in range(1, n_values + 1):
                result[f"{key}/mean@{k}"] = np.mean(values[:k])
                if k > 1:
                    result[f"{key}/std@{k}"] = np.std(values[:k])

            # Check if this is a binary metric (all values are 0 or 1)
            is_binary = all(v in (0, 0.0, 1, 1.0) for v in values)

            if is_binary and n_values > 1:
                # For binary metrics, use Codex unbiased estimator for pass@k
                # Compute pass@k for k = 1, 2, 4, ..., n_values
                n = n_values  # total samples
                c = sum(1 for v in values if v > 0)  # correct samples

                # Compute pass@k for k = 1 to n_values
                ns = []
                k = 1
                while k < n_values:
                    ns.append(k)
                    k *= 2
                ns.append(n_values)

                for k in ns:
                    result[f"{key}/pass@{k}"] = pass_at_k(n, c, k)
        return result
    else:
        result = {}
        for key, values in aggregated_metrics.items():
            if not values:
                continue
            if isinstance(values[0], str) and len(set(values)) == 1:
                result[key] = values[0]
            elif isinstance(values[0], (int, float)):
                result[key] = sum(values) / len(values)
        return result


class RunnerWrapper:
    """A wrapper for a WorkflowRunner"""

    def __init__(
        self,
        runner_id: int,
        rollout_model: InferenceModel,
        auxiliary_models: List[InferenceModel],
        config: Config,
    ):
        self.logger = get_logger(__name__)
        self.runner_id = runner_id
        self.rollout_model = rollout_model
        self.auxiliary_models = auxiliary_models
        self.config = config
        self.retry_times = config.explorer.max_retry_times
        self.timeout = config.explorer.max_timeout
        self.namespace = config.ray_namespace
        self.runner = self._create_runner()
        self.state = {}

    def _create_runner(self):
        return (
            ray.remote(WorkflowRunner)
            .options(
                num_cpus=0,
                namespace=self.namespace,
                scheduling_strategy="SPREAD",
                runtime_env={
                    "env_vars": self.config.explorer.env_vars,
                },
            )
            .remote(self.config, self.rollout_model, self.auxiliary_models, self.runner_id)
        )

    async def prepare(self):
        await self.runner.prepare.remote()

    async def update_state(self) -> None:
        """Get the runner state."""
        self.state = await self.runner.get_runner_state.remote()
        self.state["running_time"] = time.time() - self.state.get("begin_time", time.time())

    async def run_with_retry(
        self, task: TaskWrapper, repeat_times: int, run_id_base: int, timeout: float
    ) -> Tuple[Status, List, int, float]:
        """
        Args:
            task (`TaskWrapper`): The task to run.
            repeat_times (`int`): The number of times to repeat the task.
            run_id_base (`int`): The base run id for this task runs.
            timeout (`float`): The timeout for each task run.

        Returns:
            `Status`: The return status of the task.
            `List`: The experiences generated by the task.
            `int`: The runner_id of current runner.
            `float`: The time taken to run the task.
        """
        last_exception_msg = None
        await self.runner.__ray_ready__.remote()
        start_time = time.time()
        status = Status(ok=False, metrics=list())
        exps = []
        task2run = replace(
            task.task,
            rollout_args=replace(
                task.task.rollout_args,
                n=repeat_times,
            ),
        )
        try:
            for attempt in range(self.retry_times + 1):
                try:
                    status, exps = await asyncio.wait_for(
                        self.runner.run_task.remote(
                            task=task2run,
                            repeat_times=repeat_times,
                            run_id_base=run_id_base,
                        ),
                        timeout=timeout,
                    )
                    if status.ok:
                        break
                    else:
                        self.logger.error(status.message)
                except asyncio.TimeoutError:
                    last_exception_msg = f"Timeout when running task of batch {task.batch_id} at runner {self.runner_id} at attempt {attempt + 1}: {task.task}"
                    self.logger.error(last_exception_msg)
                    status = Status(ok=False, metrics=list(), message=last_exception_msg)
                except Exception:
                    last_exception_msg = traceback.format_exc()
                    self.logger.warning(
                        f"Task execution attempt {attempt + 1} failed:\n{last_exception_msg}"
                    )
                    status = Status(ok=False, metrics=list(), message=last_exception_msg)
        finally:
            end_time = time.time()
            status.metrics.append({"time/task_execution": end_time - start_time})
        return status, exps, self.runner_id, end_time - start_time

    async def restart_runner(self):
        old_runner = self.runner
        self.runner = self._create_runner()
        await self.runner.prepare.remote()
        try:
            ray.kill(old_runner)
        except Exception:
            pass


def sort_batch_id(batch_id: Union[int, str]):
    """Priority of batch_id"""
    # TODO: avoid sort the batch_id every time
    if isinstance(batch_id, int):
        return (batch_id, 0)
    else:
        match = re.match(r"^(\d+)", batch_id)
        if match:
            num = int(match.group(1))
            return (num, 1)
        else:
            return (float("inf"), 1)


class Scheduler:
    """Scheduler for rollout tasks.

    Supports scheduling tasks to multiple runners, retrying failed tasks,
    and collecting results at different levels.
    """

    def __init__(
        self,
        config: Config,
        rollout_model: List[InferenceModel],
        auxiliary_models: Optional[List[List[InferenceModel]]] = None,
    ):
        self.logger = get_logger(__name__)
        self.config = config
        self.rollout_model = rollout_model
        self.auxiliary_models = auxiliary_models or []
        self.namespace = ray.get_runtime_context().namespace
        self.default_timeout = config.explorer.max_timeout * (config.explorer.max_retry_times + 1)
        self.max_retry_times = config.explorer.max_retry_times
        self.max_repeat_times = config.explorer.max_repeat_times_per_runner
        self.default_batch_size = config.buffer.batch_size
        self.running = False

        self.runner_num = len(rollout_model) * config.explorer.runner_per_model
        self.runners: Dict[int, RunnerWrapper] = dict()
        self.idle_runners = set()  # runner_id of idle runners
        self.busy_runners = dict()  # runner_id -> task

        self.pending_tasks: Dict[Union[int, str], deque] = defaultdict(
            deque
        )  # batch_id -> (task, repeat_times, run_id_base)
        self.running_tasks: Dict[Union[int, str], set[asyncio.Future]] = defaultdict(
            set
        )  # batch_id -> futures
        self.task_num_map: Dict[Union[int, str], int] = defaultdict(
            int
        )  # batch_id -> tasks scheduled under this batch_id
        self.running_task_map: Dict[asyncio.Future, TaskWrapper] = dict()  # future -> task
        self.completed_tasks: Dict[
            Union[int, str], deque[Tuple[Status, List[Experience]]]
        ] = defaultdict(
            deque
        )  # batch_id -> results

        self.scheduler_task: Optional[asyncio.Task] = None
        self.monitor_task: Optional[asyncio.Task] = None
        self.running = False

        self.total_running_time = 0.0
        self.total_completed_tasks = 0


    async def _create_runner(
        self,
        runner_id: int,
    ):
        runner = RunnerWrapper(
            runner_id=runner_id,
            rollout_model=self.rollout_model[runner_id % len(self.rollout_model)],
            auxiliary_models=[
                self.auxiliary_models[j][runner_id % len(self.auxiliary_models[j])]
                for j in range(len(self.auxiliary_models))
            ],
            config=self.config,
        )
        await runner.prepare()
        self.runners[runner_id] = runner
        self.idle_runners.add(runner_id)

    async def _restart_runner(self, runner_id: int, reason: str = "unknown"):
        """Restart a runner."""
        await self.runners[runner_id].restart_runner()

        if runner_id in self.busy_runners:
            task = self.busy_runners.pop(runner_id)
            task_id = task.task.raw_task.get("query_id", task.task.task_id) if isinstance(task.task.raw_task, dict) else task.task.task_id
            self.logger.warning(
                f"Runner {runner_id} cancelled task_id={task_id} at batch_id {task.batch_id} (reason: {reason})"
            )

        self.idle_runners.add(runner_id)
        self.logger.info(f"Runner {runner_id} restarted.")

    async def _scheduler_loop(self) -> None:
        self.logger.info("Scheduler loop started.")
        while self.running:
            try:
                await self._schedule_pending_tasks()
                await asyncio.sleep(0.01)
            except Exception:
                self.logger.error(f"Error in scheduler loop:\n{traceback.format_exc()}")
                await asyncio.sleep(0.1)
        self.logger.info("Scheduler loop stopped.")

    async def _monitor_runner_state_loop(self) -> None:
        interval = self.config.explorer.runner_state_report_interval
        if interval <= 0:
            self.logger.info("Runner state monitoring loop disabled.")
            return

        self.logger.info("Runner state monitoring loop started.")
        while self.running:
            try:
                await asyncio.gather(*[runner.update_state() for runner in self.runners.values()])
                self.print_all_state()
            except Exception:
                self.logger.error(
                    f"Error in runner state monitoring loop:\n{traceback.format_exc()}"
                )
            await asyncio.sleep(interval)
        self.logger.info("Runner state monitoring loop stopped.")

    async def _schedule_pending_tasks(self) -> None:
        if not self.idle_runners:
            return

        # TODO: Support more advanced scheduling strategies
        for batch_id in sorted(self.pending_tasks.keys(), key=sort_batch_id):
            task_queue = self.pending_tasks[batch_id]

            while task_queue and self.idle_runners:
                task, repeat_times, run_id_base = task_queue.pop()
                runner_id = self.idle_runners.pop()
                self.busy_runners[runner_id] = task
                future = asyncio.create_task(
                    self.runners[runner_id].run_with_retry(
                        task,
                        repeat_times=repeat_times,
                        run_id_base=run_id_base,
                        timeout=self.dynamic_timeout(),
                    )
                )
                self.running_task_map[future] = task
                future.add_done_callback(self.task_done_callback)
                self.running_tasks[batch_id].add(future)

            if not task_queue:
                del self.pending_tasks[batch_id]

    def task_done_callback(self, async_task: asyncio.Task):
        task = self.running_task_map.pop(async_task)
        if async_task.cancelled():
            return
        elif async_task.exception():
            self.logger.error(f"Task {task.task.task_id} failed: {async_task.exception()}")
            return
        else:
            status, exps, runner_id, run_time = async_task.result()
            if not task.task.is_eval:  # only count running time for non-eval tasks
                self.total_running_time += run_time
                self.total_completed_tasks += 1
            task.results.append((status, exps))
            self.busy_runners.pop(runner_id)
            self.idle_runners.add(runner_id)
            # If all sub runs in a task are completed
            if len(task.results) == task.sub_task_num:
                task_experiences = []
                task_metrics = []
                all_success = True
                for s, exp in task.results:
                    task_metrics.extend(s.metrics)
                    task_experiences.extend(exp)
                    if not s.ok:
                        all_success = False
                # calculate task level metrics
                task_status = Status(
                    ok=all_success,
                    metrics=[calculate_task_level_metrics(task_metrics, task.task.is_eval)],
                )
                self.completed_tasks[task.batch_id].appendleft((task_status, task_experiences))
                self.logger.debug(f"Task completed (batch_id {task.batch_id}).")

        if task.batch_id in self.running_tasks:
            self.running_tasks[task.batch_id].remove(async_task)
            if not self.running_tasks[task.batch_id]:
                del self.running_tasks[task.batch_id]

    def _get_task_id_label(self, task: TaskWrapper) -> str:
        raw = task.task.raw_task
        if isinstance(raw, dict):
            return raw.get("query_id", raw.get("task_id", str(task.task.task_id)))
        return str(task.task.task_id)

    def _clear_timeout_tasks(self, batch_id: Union[int, str]) -> None:
        timeout_ids = []
        if batch_id in self.pending_tasks:
            for tw, _, _ in self.pending_tasks[batch_id]:
                timeout_ids.append(self._get_task_id_label(tw))
            self.logger.info(f"Clear timeout pending tasks at batch_id {batch_id}.")
            del self.pending_tasks[batch_id]
        if batch_id in self.running_tasks:
            for future in self.running_tasks[batch_id]:
                if future in self.running_task_map:
                    timeout_ids.append(self._get_task_id_label(self.running_task_map[future]))
            self.logger.info(f"Clear timeout running tasks at batch_id {batch_id}.")
            for future in self.running_tasks[batch_id]:
                future.cancel()
            del self.running_tasks[batch_id]
        if timeout_ids:
            self.logger.warning(
                f"Batch {batch_id}: {len(timeout_ids)} task(s) timed out: {timeout_ids}"
            )
        self.task_num_map.pop(batch_id, None)

    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        await asyncio.gather(*[self._create_runner(i) for i in range(self.runner_num)])
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        ready_refs = [runner.runner.__ray_ready__.remote() for runner in self.runners.values()]
        await asyncio.gather(*ready_refs)
        self.monitor_task = asyncio.create_task(self._monitor_runner_state_loop())
        self.logger.info(f"Starting Scheduler with {self.runner_num} runners")

    async def stop(self) -> None:
        if not self.running:
            return

        self.running = False
        all_running_futures = []
        for futures in self.running_tasks.values():
            all_running_futures.extend(futures)

        if all_running_futures:
            self.logger.info(f"Waiting for {len(all_running_futures)} running tasks to complete...")
            await asyncio.gather(*all_running_futures, return_exceptions=True)

        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Scheduler stopped")

    def schedule(self, tasks: List[Task], batch_id: Union[int, str]) -> None:
        """Schedule the provided tasks.

        Args:
            tasks (`List[Task]`): The tasks to schedule.
            batch_id (`Union[int, str]`): The id of provided tasks. It should be an integer or a string
                starting with an integer (e.g., 123, "123/my_task")
        """
        if not tasks:
            return
        self.task_num_map[batch_id] += len(tasks)
        self._split_and_submit_tasks(tasks, batch_id=batch_id)

    def _expand_by_agent(self, task: Task) -> List[Task]:
        """If agent expansion is enabled, split one task into per-agent tasks.

        Each returned task has fixed_agent_index in task.index and repeat_times divided by num_agents.
        This makes over-rollout work at the task@agent level.
        Uses task.index (not workflow_args) to preserve workflow_args identity for config caching.
        """
        # Check per-task workflow_args instead of global num_expand_agents,
        # so eval tasksets with different agent_expand settings are handled correctly.
        wa = task.workflow_args or {}
        if wa.get("agent_expand") != "all" or not wa.get("agent_models"):
            return [task]
        num_agents = len(wa["agent_models"])
        if task.repeat_times % num_agents != 0:
            self.logger.warning(
                f"repeat_times ({task.repeat_times}) not divisible by num_agents ({num_agents}), skipping agent expansion"
            )
            return [task]
        per_agent_repeats = task.repeat_times // num_agents
        expanded = []
        for agent_idx in range(num_agents):
            new_index = dict(task.index) if task.index else {}
            new_index["fixed_agent_index"] = agent_idx
            expanded.append(replace(task, repeat_times=per_agent_repeats, index=new_index))
        return expanded

    def _split_and_submit_tasks(self, tasks: List[Task], batch_id: Union[int, str]) -> None:
        task_idx = 0
        for task in tasks:
            assert task.repeat_times is not None, "Task repeat_times should not be None"
            expanded_tasks = self._expand_by_agent(task)
            # If expanded, update task_num_map (schedule() already added len(tasks), adjust for extra)
            if len(expanded_tasks) > 1:
                self.task_num_map[batch_id] += len(expanded_tasks) - 1
            for et in expanded_tasks:
                task_wrapper = TaskWrapper(
                    task=replace(et, batch_id=batch_id, task_id=task_idx),
                    batch_id=batch_id,
                )
                task_idx += 1
                if self.max_repeat_times is None:
                    task_wrapper.sub_task_num = 1
                    self.pending_tasks[batch_id].appendleft((task_wrapper, et.repeat_times, 0))
                    continue
                sub_tasks = []
                for run_id_base in range(0, et.repeat_times, self.max_repeat_times):
                    repeat_times = min(self.max_repeat_times, et.repeat_times - run_id_base)
                    sub_tasks.append((task_wrapper, repeat_times, run_id_base))
                task_wrapper.sub_task_num = len(sub_tasks)
                self.pending_tasks[batch_id].extendleft(sub_tasks)

    def dynamic_timeout(self, timeout: Optional[float] = None) -> float:
        """Calculate dynamic timeout based on historical data."""
        max_timeout = timeout or self.default_timeout
        if not self.config.explorer.dynamic_timeout.enable:
            return max_timeout
        if self.total_completed_tasks < self.default_batch_size:
            return max_timeout
        avg_time_per_task = self.total_running_time / self.total_completed_tasks
        return min(
            max_timeout,
            avg_time_per_task * self.config.explorer.dynamic_timeout.ratio,
        )

    async def _cleanup_batch_and_restart_runners(self, batch_id: Union[int, str]) -> None:
        """Clear timeout tasks for a batch and restart associated runners."""
        self._clear_timeout_tasks(batch_id=batch_id)
        runners_to_restart = [
            runner_id
            for runner_id, task in list(self.busy_runners.items())
            if task.batch_id == batch_id
        ]
        if runners_to_restart:
            await asyncio.gather(*[self._restart_runner(rid, reason=f"batch {batch_id} timeout") for rid in runners_to_restart])

    async def get_results(
        self,
        batch_id: Union[int, str],
        min_num: Optional[int] = None,
        timeout: Optional[float] = None,
        clear_timeout_tasks: bool = True,
    ) -> Tuple[List[Status], List[Experience]]:
        """Get the result of tasks at the specific batch_id.

        Args:
            batch_id (`Union[int, str]`): Only wait for tasks at this batch.
            min_num (`int`): The minimum number of tasks to wait for. If `None`, wait for all tasks at `batch_id`.
            timeout (`float`): The timeout for waiting for tasks to finish. If `None`, wait for default timeout.
            clear_timeout_tasks (`bool`): Whether to clear timeout tasks.
        """
        timeout = timeout or self.default_timeout
        start_time = time.time()
        scheduled_num = self.task_num_map.get(batch_id, 0)
        if min_num is None:
            min_num = scheduled_num
        elif min_num > scheduled_num:
            self.logger.warning(
                f"Requested min_num {min_num} is greater than scheduled tasks {scheduled_num} at batch_id {batch_id}. Adjusting min_num to {scheduled_num}."
            )
            min_num = scheduled_num

        self.logger.debug(f"Waiting for {min_num} tasks to complete...")
        min_threshold_reached_time = None
        while time.time() - start_time <= timeout:
            completed_count = len(self.completed_tasks.get(batch_id, []))
            if completed_count >= min_num:
                min_threshold_reached_time = min_threshold_reached_time or time.time()
                if completed_count >= scheduled_num:
                    break
                if (
                    time.time() - min_threshold_reached_time
                    >= self.config.explorer.over_rollout.wait_after_min
                ):
                    if clear_timeout_tasks:
                        await self._cleanup_batch_and_restart_runners(batch_id)
                        break
            await asyncio.sleep(0.1)

        if time.time() - start_time > timeout:
            self.logger.error(
                f"Timed out waiting for tasks at batch {batch_id} to complete after {timeout} seconds"
            )
            if clear_timeout_tasks:
                await self._cleanup_batch_and_restart_runners(batch_id)

        statuses = []
        experiences = []
        completed_queue = self.completed_tasks.get(batch_id, deque())
        while completed_queue:
            status, exps = completed_queue.pop()
            statuses.append(status)
            if isinstance(exps, list):
                experiences.extend(exps)
            else:
                experiences.append(exps)

        if batch_id in self.completed_tasks and not self.completed_tasks[batch_id]:
            del self.completed_tasks[batch_id]

        completed_count = len(statuses)
        if completed_count < min_num:
            self.logger.warning(
                f"Timeout reached, only {completed_count}/{min_num} tasks completed"
            )

        return statuses, experiences

    def has_step(self, batch_id: Union[int, str]) -> bool:
        return (
            batch_id in self.completed_tasks
            or batch_id in self.pending_tasks
            or batch_id in self.running_tasks
        )

    async def wait_all(
        self, timeout: Optional[float] = None, clear_timeout_tasks: bool = True
    ) -> None:
        """Wait for all tasks to complete without poping results. If timeout reached, raise TimeoutError.

        Args:
            timeout (`float`): timeout in seconds. Raise `TimeoutError` when no new tasks is completed within timeout.
            clear_timeout_tasks (`bool`): Whether to clear timeout tasks.
        """
        timeout = timeout or self.default_timeout
        start_time = time.time()

        self.logger.debug("Waiting for all tasks to complete...")
        last_completed_count = 0
        while time.time() - start_time < timeout:
            has_pending = bool(self.pending_tasks)
            has_running = bool(self.running_tasks)

            if not has_pending and not has_running:
                self.logger.debug("All tasks completed successfully")
                return

            completed_count = sum(len(tasks) for tasks in self.completed_tasks.values())
            if completed_count != last_completed_count:
                # flush timeout when new tasks are completed
                start_time = time.time()
                last_completed_count = completed_count

            await asyncio.sleep(0.1)

        pending_count = sum(len(tasks) for tasks in self.pending_tasks.values())
        running_count = sum(len(futures) for futures in self.running_tasks.values())
        error_msg = f"Timeout after {timeout} seconds. Still have {pending_count} pending tasks and {running_count} running tasks."
        self.logger.error(error_msg)

        if clear_timeout_tasks:
            for batch_id in self.pending_tasks.keys() | self.running_tasks.keys():
                self._clear_timeout_tasks(batch_id)
            asyncio.gather(
                *[self._restart_runner(runner_id, reason="global read timeout") for runner_id in self.busy_runners.keys()]
            )

        raise TimeoutError(error_msg)

    def get_key_state(self, key: str) -> Dict:
        """Get the scheduler state.

        Args:
            key (`str`): The key of the state to get.

        Returns:
            `Dict`: A dictionary of runner ids to their state for the given key.
        """
        result = {}
        for runner in self.runners.values():
            runner_state = runner.state
            if runner_state and key in runner_state:
                result[runner.runner_id] = runner_state[key]
        return result

    def get_runner_state(self, runner_id: int) -> Dict:
        """Get the scheduler state.

        Args:
            runner_id (`int`): The id of the runner.

        Returns:
            `Dict`: The state of the runner.
        """
        runner = self.runners.get(runner_id, None)
        if runner:
            return runner.state
        else:
            return {}

    def get_all_state(self) -> Dict:
        """Get all runners' state.

        Returns:
            `Dict`: The state of all runners.
        """
        result = {}
        for runner in self.runners.values():
            runner_state = runner.state
            if runner_state:
                result[runner.runner_id] = runner_state
        return result

    def print_all_state(self) -> None:
        """Print all runners' state in a clear, aligned table format."""
        all_keys = set()
        for runner in self.runners.values():
            runner_state = runner.state
            if runner_state:
                all_keys.update(runner_state.keys())
        all_keys = sorted(all_keys)
        # Prepare header
        header = ["runner_id"] + all_keys  # type: ignore [operator]
        # Prepare rows
        rows = []
        for runner in self.runners.values():
            runner_state = runner.state or {}
            row = [str(runner.runner_id)]
            for key in all_keys:
                value = runner_state.get(key, "-")
                row.append(str(value))
            rows.append(row)
        # Calculate column widths
        col_widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
        # Print header
        header_line = " | ".join(str(h).ljust(w) for h, w in zip(header, col_widths))
        self.logger.info(header_line)
        self.logger.info("-+-".join("-" * w for w in col_widths))
        # Print each row
        for row in rows:
            line = " | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
            self.logger.info(line)