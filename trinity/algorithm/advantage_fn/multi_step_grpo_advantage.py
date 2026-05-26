"""GRPO advantage computation for multi-step scenarios
"""
from typing import Dict, List, Optional, Tuple

import torch

from trinity.algorithm.advantage_fn.advantage_fn import ADVANTAGE_FN, AdvantageFn
from trinity.buffer.operators import ExperienceOperator
from trinity.common.experience import Experience, group_by
from trinity.utils.monitor import gather_metrics
from trinity.utils.log import get_logger

logger = get_logger(__name__)


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    tensor_values = torch.tensor(values, dtype=torch.float32)
    return torch.quantile(tensor_values, q).item()


def _is_all_zero_final_score_group(run_exps: Dict[str, List[Experience]]) -> bool:
    """Whether every rollout in the task group ends with final score 0."""
    final_scores = [step_exps[-1].reward for step_exps in run_exps.values() if step_exps]
    return bool(final_scores) and all(score == 0 for score in final_scores)


def _has_nonzero_local_reward(exp: Experience) -> bool:
    """Whether an experience has an explicit non-zero step-level reward detail."""
    reward_details = (exp.info or {}).get("reward_details", [])
    return any(float(detail.get("reward", 0.0) or 0.0) != 0.0 for detail in reward_details)


def _filter_nonzero_local_reward_steps(
    run_exps: Dict[str, List[Experience]],
) -> Dict[str, List[Experience]]:
    """For all-zero task groups, keep only steps with local reward signal."""
    filtered = {}
    for run_id, step_exps in run_exps.items():
        kept_steps = [exp for exp in step_exps if _has_nonzero_local_reward(exp)]
        if kept_steps:
            filtered[run_id] = kept_steps
    return filtered


@ADVANTAGE_FN.register_module("step_wise_grpo")
class StepWiseGRPOAdvantageFn(AdvantageFn, ExperienceOperator):
    """
    An advantage function that broadcasts advantages from the last step to previous steps.
    Inspired by rLLM (https://github.com/rllm-org/rllm).
    """

    def __init__(
        self,
        epsilon: float = 1e-6,
        enable_step_norm: bool = False,
        std_cal_level: str = "group",  # 'group' (task-level) or 'batch'
        alpha_inter_reward: float = 0.1,
        inter_reward_normalization: str = "none",
        inter_reward_std_floor: float = 0.0,
        outcome_tie_std_threshold: float = 1e-6,
        outcome_tie_inter_reward_alpha: float = 1.0,
        advantage_clip_abs: float = 3.0,
        **kwargs,
    ) -> None:
        """Initialize the Step-wise GRPO advantage function.

        Advantage formula (v5.5+):
            adv_task = (final_reward - task_mean) / (task_std + eps)   # broadcast to all steps of the run
            inter_sum = sum(reward_details where reason not in OVERRIDE_REASONS)
            if inter_reward_normalization == "task_std":
                inter_adv = alpha_inter_reward * inter_sum / max(task_std, inter_reward_std_floor, eps)
            elif inter_reward_normalization == "outcome_tie":
                inter_z = (inter_sum - inter_mean) / (inter_std + eps)
                if task_std <= outcome_tie_std_threshold:
                    final_advantage = clip(outcome_tie_inter_reward_alpha * inter_z, -clip_abs, +clip_abs)
                else:
                    final_advantage = clip(adv_task + alpha_inter_reward * inter_z, -clip_abs, +clip_abs)
            elif inter_reward_normalization == "none":
                inter_adv = alpha_inter_reward * inter_sum
                final_advantage = clip(adv_task + inter_adv, -clip_abs, +clip_abs)

        Override branch: if a step has reward_details with reason in OVERRIDE_REASONS,
        final_advantage is replaced by their sum (adv_task and other inter rewards are dropped).

        When task_std is below outcome_tie_std_threshold in outcome_tie mode,
        z-scored process rewards become the primary tie-breaking signal.

        Args:
            epsilon: A small value to avoid division by zero.
            enable_step_norm: If True, normalize advantages by trajectory length.
            std_cal_level: 'group' (per-task std, default) or 'batch' (all last-step rewards).
            alpha_inter_reward: Scale applied to summed non-override reward_details.
            inter_reward_normalization: 'none' for direct scaling, or 'task_std' to divide
                non-override inter reward by max(task_std, inter_reward_std_floor, epsilon),
                or 'outcome_tie' to use z-scored process rewards as the main signal
                only when the outcome reward has no group-relative variance.
            inter_reward_std_floor: Minimum denominator for task_std-normalized inter reward.
            outcome_tie_std_threshold: Outcome std at or below which process rewards
                become the primary tie-breaking advantage.
            outcome_tie_inter_reward_alpha: Scale for z-scored process rewards in
                outcome-tie groups.
            advantage_clip_abs: Absolute bound applied to final_advantage (non-override branch).
        """
        self.epsilon = epsilon
        self.enable_step_norm = enable_step_norm
        self.std_cal_level = std_cal_level
        self.alpha_inter_reward = alpha_inter_reward
        self.inter_reward_normalization = inter_reward_normalization
        self.inter_reward_std_floor = inter_reward_std_floor
        self.outcome_tie_std_threshold = outcome_tie_std_threshold
        self.outcome_tie_inter_reward_alpha = outcome_tie_inter_reward_alpha
        self.advantage_clip_abs = advantage_clip_abs

        if self.std_cal_level not in ["group", "batch"]:
            raise ValueError("std_cal_level must be either 'group' or 'batch'")
        if self.inter_reward_normalization not in ["none", "task_std", "outcome_tie"]:
            raise ValueError(
                "inter_reward_normalization must be one of 'none', 'task_std', or 'outcome_tie'"
            )

    def calculate_last_step_statistics(
        self,
        exps: Dict[str, Experience],
        precomputed_std: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float, Dict[str, float]]:
        """Calculate mean and std from last step rewards.

        Args:
            exps (Dict[str, Experience]): One experience per run (last step), keyed by run ID.
            precomputed_std (Optional[torch.Tensor]): Precomputed standard deviation for batch-level calculation.

        Returns:
            float: Mean of last step rewards.
            float: Standard deviation (group or batch level).
            Dict[str, float]: Metrics for logging.
        """
        with torch.no_grad():
            if len(exps) == 1:
                group_reward_mean = torch.tensor(0.0)
                group_reward_std = torch.tensor(1.0)
            else:
                rewards = torch.tensor([exp.reward for exp in exps.values()], dtype=torch.float32)
                group_reward_mean = torch.mean(rewards)
                group_reward_std = torch.std(rewards)  # Default: unbiased=True

            # Select which std to use
            if self.std_cal_level == "batch" and precomputed_std is not None:
                final_std = precomputed_std
            else:
                final_std = group_reward_std

            metrics = {
                "reward_mean": group_reward_mean.item(),
                "reward_std": group_reward_std.item(),
            }

        return group_reward_mean.item(), final_std.item(), metrics

    def calculate_stepwise_advantages(
        self,
        run_exps: Dict[str, List[Experience]],
        task_mean: float,
        task_std: float,
    ) -> Tuple[Dict[str, List[Experience]], Dict[str, float]]:
        """Calculate per-step advantages (v5.5+).

        Default branch:
            adv_task = (final_reward - task_mean) / (task_std + eps)           # broadcast
            inter_sum = sum(reward_details where reason not in OVERRIDE_REASONS)
            final_advantage = clip(adv_task + alpha * inter_sum, -clip_abs, +clip_abs)

        Override branch: if a step has any reward_detail with reason in OVERRIDE_REASONS,
        final_advantage = sum of those override reward values (adv_task and other
        inter rewards at that step are discarded).
        """
        OVERRIDE_REASONS = {"parsing_error", "modification_error", "degenerate_generation"}
        clip_abs = self.advantage_clip_abs

        inter_sum_stats: List[float] = []
        inter_adv_stats: List[float] = []
        task_adv_stats: List[float] = []
        final_adv_stats: List[float] = []
        clipped_count = 0
        floor_hit_count = 0
        override_count = 0
        override_count_by_reason: Dict[str, int] = {r: 0 for r in OVERRIDE_REASONS}

        use_outcome_tie = self.inter_reward_normalization == "outcome_tie"
        is_outcome_tie_group = bool(task_std <= self.outcome_tie_std_threshold)
        inter_mean = 0.0
        inter_std = 0.0
        if use_outcome_tie:
            inter_values = []
            for exps in run_exps.values():
                for exp in exps:
                    details = exp.info.get("reward_details", []) if exp.info else []
                    override_reward = sum(
                        d.get("reward", 0.0) for d in details
                        if d.get("reason") in OVERRIDE_REASONS
                    )
                    if override_reward != 0.0:
                        continue
                    inter_values.append(
                        sum(
                            d.get("reward", 0.0) for d in details
                            if d.get("reason") not in OVERRIDE_REASONS
                        )
                    )
            if inter_values:
                inter_tensor = torch.tensor(inter_values, dtype=torch.float32)
                inter_mean = torch.mean(inter_tensor).item()
                if len(inter_values) > 1:
                    inter_std = torch.std(inter_tensor).item()

        for run_id, exps in run_exps.items():
            traj_length = len(exps)
            final_reward = exps[-1].reward
            adv_task = (final_reward - task_mean) / (task_std + self.epsilon)
            adv_task_val = adv_task.item() if hasattr(adv_task, "item") else float(adv_task)

            for i, exp in enumerate(exps):
                details = exp.info.get("reward_details", []) if exp.info else []

                override_reward = sum(
                    d.get("reward", 0.0) for d in details
                    if d.get("reason") in OVERRIDE_REASONS
                )
                step_override_reasons = {
                    d.get("reason") for d in details
                    if d.get("reason") in OVERRIDE_REASONS and d.get("reward", 0.0) != 0.0
                }
                is_override = override_reward != 0.0

                # Sum all non-override reward_details (no per-reward clipping)
                inter_sum = sum(
                    d.get("reward", 0.0) for d in details
                    if d.get("reason") not in OVERRIDE_REASONS
                )

                if is_override:
                    inter_advantage = 0.0
                    inter_reward_scale_denominator = None
                    final_advantage = override_reward
                    override_count += 1
                    for r in step_override_reasons:
                        override_count_by_reason[r] += 1
                    was_clipped = False
                else:
                    if self.inter_reward_normalization == "task_std":
                        inter_reward_scale_denominator = max(
                            task_std,
                            self.inter_reward_std_floor,
                            self.epsilon,
                        )
                        if task_std < inter_reward_scale_denominator:
                            floor_hit_count += 1
                        inter_advantage = (
                            self.alpha_inter_reward * inter_sum / inter_reward_scale_denominator
                        )
                        raw_advantage = adv_task_val + inter_advantage
                    elif use_outcome_tie:
                        inter_reward_scale_denominator = inter_std + self.epsilon
                        inter_z = (inter_sum - inter_mean) / inter_reward_scale_denominator
                        if is_outcome_tie_group:
                            inter_advantage = self.outcome_tie_inter_reward_alpha * inter_z
                            raw_advantage = inter_advantage
                        else:
                            inter_advantage = self.alpha_inter_reward * inter_z
                            raw_advantage = adv_task_val + inter_advantage

                    else:
                        inter_reward_scale_denominator = None
                        inter_advantage = self.alpha_inter_reward * inter_sum
                        raw_advantage = adv_task_val + inter_advantage

                    final_advantage = max(-clip_abs, min(clip_abs, raw_advantage))
                    was_clipped = (final_advantage != raw_advantage)
                    if was_clipped:
                        clipped_count += 1

                exp.advantages = exp.action_mask * final_advantage  # type: ignore [operator]
                if self.enable_step_norm:
                    exp.advantages /= traj_length
                exp.returns = exp.advantages.clone()

                exp.info = exp.info or {}
                exp.info["advantage_calculation"] = {
                    "step": i,
                    "final_score": exp.info.get("final_score", 0.0),
                    "inter_sum": inter_sum,
                    "task_mean": task_mean,
                    "task_std": task_std,
                    "adv_task": adv_task_val,
                    "alpha_inter_reward": self.alpha_inter_reward,
                    "inter_reward_normalization": self.inter_reward_normalization,
                    "inter_reward_std_floor": self.inter_reward_std_floor,
                    "inter_reward_scale_denominator": inter_reward_scale_denominator,
                    "inter_advantage": inter_advantage,
                    "outcome_tie_std_threshold": self.outcome_tie_std_threshold,
                    "outcome_tie_inter_reward_alpha": self.outcome_tie_inter_reward_alpha,
                    "outcome_tie_is_tie_group": is_outcome_tie_group if use_outcome_tie else False,
                    "outcome_tie_inter_mean": inter_mean if use_outcome_tie else None,
                    "outcome_tie_inter_std": inter_std if use_outcome_tie else None,
                    "override_reward": override_reward,
                    "is_override": is_override,
                    "was_clipped": was_clipped,
                    "final_advantage": final_advantage,
                    "traj_length": traj_length,
                }

                inter_sum_stats.append(inter_sum)
                inter_adv_stats.append(inter_advantage)
                task_adv_stats.append(adv_task_val)
                final_adv_stats.append(final_advantage)

        stats: Dict[str, float] = {}
        if inter_sum_stats:
            stats["inter_sum_mean"] = sum(inter_sum_stats) / len(inter_sum_stats)
            stats["inter_sum_max"] = max(inter_sum_stats)
            stats["inter_sum_min"] = min(inter_sum_stats)
            if len(inter_sum_stats) > 1:
                stats["inter_sum_std"] = torch.std(
                    torch.tensor(inter_sum_stats, dtype=torch.float32)
                ).item()
        if task_adv_stats:
            stats["task_adv_mean"] = sum(task_adv_stats) / len(task_adv_stats)
            stats["task_std"] = task_std
        if inter_adv_stats:
            stats["inter_advantage_mean"] = sum(inter_adv_stats) / len(inter_adv_stats)
            if len(inter_adv_stats) > 1:
                stats["inter_advantage_std"] = torch.std(
                    torch.tensor(inter_adv_stats, dtype=torch.float32)
                ).item()
        if final_adv_stats:
            final_adv_abs_sum = sum(abs(x) for x in final_adv_stats)
            stats["final_advantage_mean"] = sum(final_adv_stats) / len(final_adv_stats)
            if len(final_adv_stats) > 1:
                stats["final_advantage_std"] = torch.std(
                    torch.tensor(final_adv_stats, dtype=torch.float32)
                ).item()
            if final_adv_abs_sum > 0:
                stats["inter_advantage_abs_share"] = (
                    sum(abs(x) for x in inter_adv_stats) / final_adv_abs_sum
                )
                stats["inter_advantage_sum_share"] = sum(inter_adv_stats) / final_adv_abs_sum
            stats["override_step_count"] = override_count
            stats["override_step_ratio"] = override_count / len(final_adv_stats)
            stats["task_std_floor_hit_count"] = floor_hit_count
            stats["task_std_floor_hit_ratio"] = floor_hit_count / len(final_adv_stats)
            stats["outcome_tie_group_count"] = 1.0 if is_outcome_tie_group and use_outcome_tie else 0.0
            stats["outcome_tie_group_ratio"] = 1.0 if is_outcome_tie_group and use_outcome_tie else 0.0
            if use_outcome_tie:
                stats["outcome_tie_inter_mean"] = inter_mean
                stats["outcome_tie_inter_std"] = inter_std
            stats["final_advantage_clipped_count"] = clipped_count
            stats["final_advantage_clipped_ratio"] = clipped_count / len(final_adv_stats)
            for reason, cnt in override_count_by_reason.items():
                stats[f"override_step_count/{reason}"] = cnt
                stats[f"override_step_ratio/{reason}"] = cnt / len(final_adv_stats)

        return run_exps, stats

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], Dict]:
        if len(exps) == 0:
            return [], {}
        cnt = 0
        metric_list = []
        global_stats_list = []

        task_exps = group_by(exps, "task")
        logger.info(f"Advantage groups ({len(task_exps)}): {[(k, len(v)) for k, v in task_exps.items()]}")

        # Pre-compute batch-level last-step reward std when configured
        precomputed_std = None
        if self.std_cal_level == "batch":
            all_laststep_rewards = []
            for task_exp in task_exps.values():
                task_run_exps = group_by(task_exp, "run")
                if _is_all_zero_final_score_group(task_run_exps):
                    continue
                all_laststep_rewards.extend(
                    run_steps[-1].reward for run_steps in task_run_exps.values() if run_steps
                )
            if len(all_laststep_rewards) <= 1:
                precomputed_std = torch.tensor(1.0)
            else:
                precomputed_std = torch.std(torch.tensor(all_laststep_rewards, dtype=torch.float32))

        result_exps = []
        all_zero_group_count = 0
        all_zero_nonzero_local_experience_count = 0
        for task_exp in task_exps.values():
            run_exps = group_by(task_exp, "run")

            is_all_zero_group = _is_all_zero_final_score_group(run_exps)
            if is_all_zero_group:
                all_zero_group_count += 1

            if is_all_zero_group and self.inter_reward_normalization != "outcome_tie":
                run_exps = _filter_nonzero_local_reward_steps(run_exps)
                kept_count = sum(len(step_exps) for step_exps in run_exps.values())
                all_zero_nonzero_local_experience_count += kept_count
                if not run_exps:
                    continue

            last_step_exps = {run_id: step_exps[-1] for run_id, step_exps in run_exps.items()}
            mean, std, metrics = self.calculate_last_step_statistics(
                last_step_exps, precomputed_std=precomputed_std
            )
            metric_list.append(metrics)

            run_exps, group_stats = self.calculate_stepwise_advantages(run_exps, mean, std)
            global_stats_list.append(group_stats)

            for run_id_exps in run_exps.values():
                cnt += len(run_id_exps)
                result_exps.extend(run_id_exps)

        metrics = gather_metrics(metric_list, "group_advantages")
        metrics["experience_count"] = cnt
        metrics["all_zero_group_count"] = all_zero_group_count
        metrics["all_zero_nonzero_local_experience_count"] = all_zero_nonzero_local_experience_count

        if global_stats_list:
            keys = global_stats_list[0].keys()
            for k in keys:
                values = [g[k] for g in global_stats_list if k in g]
                if values:
                    metrics[f"adv_stats/{k}"] = sum(values) / len(values)
            task_std_values = [g["task_std"] for g in global_stats_list if "task_std" in g]
            if task_std_values:
                metrics["adv_stats/task_std/p25"] = _percentile(task_std_values, 0.25)
                metrics["adv_stats/task_std/p50"] = _percentile(task_std_values, 0.50)
                metrics["adv_stats/task_std/p75"] = _percentile(task_std_values, 0.75)
                metrics["adv_stats/task_std/p95"] = _percentile(task_std_values, 0.95)

        # --- Per-group rollout stats (agent, searcher, agent×searcher) and all-1/all-0 ratio ---
        # Collect per-run data keyed by (agent, searcher) from each task group
        def _empty_bucket():
            return {"final_scores": [], "inter_rewards": [], "advantages": [], "reward_stds": []}

        group_buckets = {}   # (agent, searcher) -> bucket
        group_patterns = []  # (pattern, agent, searcher) per task group

        for task_key, task_exp in task_exps.items():
            task_run_exps = group_by(task_exp, "run")
            # Extract agent name from task key (format: "task_id@agent_name")
            agent_name = task_key.rsplit("@", 1)[-1] if "@" in str(task_key) else "_default"
            # Searcher type is consistent within a task group (per-task deterministic)
            searcher_name = None
            for exp_list in task_run_exps.values():
                for e in exp_list:
                    if e.metrics and "_searcher_type" in e.metrics:
                        searcher_name = e.metrics["_searcher_type"]
                        break
                if searcher_name is not None:
                    break
            searcher_name = searcher_name or "_default"

            bucket_key = (agent_name, searcher_name)
            if bucket_key not in group_buckets:
                group_buckets[bucket_key] = _empty_bucket()
            bucket = group_buckets[bucket_key]

            group_final_scores = []
            for run_id, run_steps in task_run_exps.items():
                if not run_steps:
                    continue
                last_exp = run_steps[-1]
                final_score = last_exp.info.get("final_score", last_exp.reward) if last_exp.info else last_exp.reward
                group_final_scores.append(final_score)
                bucket["final_scores"].append(final_score)
                run_inter_total = sum(
                    (e.info.get("intermediate_reward", 0.0) if e.info else 0.0) for e in run_steps
                )
                bucket["inter_rewards"].append(run_inter_total)
                run_advs = [
                    e.advantages.mean().item() for e in run_steps
                    if e.advantages is not None and e.advantages.numel() > 0
                ]
                if run_advs:
                    bucket["advantages"].append(sum(run_advs) / len(run_advs))

            if len(group_final_scores) > 1:
                bucket["reward_stds"].append(
                    torch.std(torch.tensor(group_final_scores, dtype=torch.float32)).item()
                )

            # Classify group pattern
            if group_final_scores:
                if all(s >= 1.0 for s in group_final_scores):
                    group_patterns.append(("all_1", agent_name, searcher_name))
                elif all(s <= 0.0 for s in group_final_scores):
                    group_patterns.append(("all_0", agent_name, searcher_name))
                else:
                    group_patterns.append(("mixed", agent_name, searcher_name))

        def _log_bucket(prefix, data):
            """Log reward/advantage stats for a bucket."""
            if data["final_scores"]:
                metrics[f"{prefix}/final_score/mean"] = sum(data["final_scores"]) / len(data["final_scores"])
            if data["inter_rewards"]:
                metrics[f"{prefix}/intermediate_reward/mean"] = sum(data["inter_rewards"]) / len(data["inter_rewards"])
            if data["reward_stds"]:
                metrics[f"{prefix}/reward_std/mean"] = sum(data["reward_stds"]) / len(data["reward_stds"])
                metrics[f"{prefix}/reward_std/p25"] = _percentile(data["reward_stds"], 0.25)
                metrics[f"{prefix}/reward_std/p50"] = _percentile(data["reward_stds"], 0.50)
                metrics[f"{prefix}/reward_std/p75"] = _percentile(data["reward_stds"], 0.75)
                metrics[f"{prefix}/reward_std/p95"] = _percentile(data["reward_stds"], 0.95)
            if data["advantages"]:
                adv_t = torch.tensor(data["advantages"], dtype=torch.float32)
                metrics[f"{prefix}/advantage/mean"] = torch.mean(adv_t).item()
                if len(data["advantages"]) > 1:
                    metrics[f"{prefix}/advantage/std"] = torch.std(adv_t).item()

        def _merge_buckets(buckets):
            merged = _empty_bucket()
            for b in buckets:
                for k in merged:
                    merged[k].extend(b[k])
            return merged

        # Determine unique agents/searchers to skip redundant metric levels
        unique_agents = set(a for a, _ in group_buckets)
        unique_searchers = set(s for _, s in group_buckets)
        multi_agent = len(unique_agents) > 1
        multi_searcher = len(unique_searchers) > 1

        # --- Per agent×searcher (only when both dimensions vary) ---
        if multi_agent and multi_searcher:
            for (agent, searcher), bucket in group_buckets.items():
                safe_agent = str(agent).rsplit("/", 1)[-1]
                _log_bucket(f"rollout/agent/{safe_agent}/{searcher}", bucket)

        # --- Per agent (merged across searchers) ---
        if multi_agent:
            agent_merged = {}
            for (agent, searcher), bucket in group_buckets.items():
                agent_merged.setdefault(agent, []).append(bucket)
            for agent, buckets in agent_merged.items():
                safe_agent = str(agent).rsplit("/", 1)[-1]
                _log_bucket(f"rollout/agent/{safe_agent}", _merge_buckets(buckets))

        # --- Per searcher (merged across agents) ---
        if multi_searcher:
            searcher_merged = {}
            for (agent, searcher), bucket in group_buckets.items():
                searcher_merged.setdefault(searcher, []).append(bucket)
            for searcher, buckets in searcher_merged.items():
                _log_bucket(f"rollout/searcher/{searcher}", _merge_buckets(buckets))

        # --- Overall (always emitted; doubles as single-agent/single-searcher stats) ---
        _log_bucket("rollout/overall", _merge_buckets(list(group_buckets.values())))

        # --- All-1 / all-0 / mixed ratio ---
        def _log_pattern_ratios(prefix, patterns):
            if not patterns:
                return
            n = len(patterns)
            metrics[f"{prefix}/group_all_1_ratio"] = sum(1 for p in patterns if p == "all_1") / n
            metrics[f"{prefix}/group_all_0_ratio"] = sum(1 for p in patterns if p == "all_0") / n
            metrics[f"{prefix}/group_mixed_ratio"] = sum(1 for p in patterns if p == "mixed") / n

        # Overall
        _log_pattern_ratios("rollout", [p for p, _, _ in group_patterns])

        # Per agent
        if multi_agent:
            agent_pats = {}
            for pat, agent, searcher in group_patterns:
                agent_pats.setdefault(agent, []).append(pat)
            for agent, pats in agent_pats.items():
                safe_agent = str(agent).rsplit("/", 1)[-1]
                _log_pattern_ratios(f"rollout/agent/{safe_agent}", pats)

        # Per searcher
        if multi_searcher:
            searcher_pats = {}
            for pat, agent, searcher in group_patterns:
                searcher_pats.setdefault(searcher, []).append(pat)
            for searcher, pats in searcher_pats.items():
                _log_pattern_ratios(f"rollout/searcher/{searcher}", pats)

        return result_exps, metrics

    def __call__(self, exps, **kwargs):
        return self.process(exps)

    @classmethod
    def compute_in_trainer(cls) -> bool:
        """Whether the advantage should be computed in the trainer loop."""
        return False

    @classmethod
    def default_args(cls) -> Dict:
        """Return the default configuration for this strategy."""
        return {
            "epsilon": 1e-6,
            "enable_step_norm": False,
            "std_cal_level": "group",
            "alpha_inter_reward": 0.1,
            "advantage_clip_abs": 3.0,
        }
