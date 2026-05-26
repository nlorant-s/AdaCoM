"""Shared reward helpers for Context Manager compression outcomes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


DEFAULT_INSUFFICIENT_BUDGET_CONFIG = {
    "enabled": True,
    "tool_budget": 4096,
    "cm_input_limit": 28672,
    "safety_margin": 1024,
    "scale": 6.0,
}

DEFAULT_DEFERRED_OUT_OF_TOKENS_CONFIG = {
    "enabled": True,
    "lookback_steps": 3,
    "rewards_by_distance": {1: -3.0, 2: -2.0, 3: -1.0},
    "only_penalize_insufficient_steps": True,
}

DEFAULT_TERMINAL_OUT_OF_TOKENS_CONFIG = {
    "enabled": True,
    "fail_rollout": True,
    "reward": 0.0,
}


def _section_config(
    reward_config: Dict[str, Any],
    key: str,
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    value = reward_config.get(key, {}) if reward_config else {}
    if value is False:
        value = {"enabled": False}
    if value is True:
        value = {"enabled": True}
    if not isinstance(value, dict):
        value = {}
    return {**defaults, **value}


def get_terminal_out_of_tokens_config(reward_config: Dict[str, Any]) -> Dict[str, Any]:
    return _section_config(
        reward_config,
        "terminal_out_of_tokens",
        DEFAULT_TERMINAL_OUT_OF_TOKENS_CONFIG,
    )


def calculate_budget_penalty(
    compression_metrics: Optional[Dict[str, Any]],
    reward_config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    config = _section_config(
        reward_config,
        "insufficient_budget",
        DEFAULT_INSUFFICIENT_BUDGET_CONFIG,
    )
    if not config.get("enabled", True) or not compression_metrics:
        return None

    after_tokens = int(compression_metrics.get("after_tokens", 0) or 0)
    max_model_len = int(compression_metrics.get("max_model_len", 0) or 0)
    tool_budget = int(config.get("tool_budget", compression_metrics.get("tool_budget", 4096)) or 4096)
    cm_input_limit = int(config.get("cm_input_limit", max_model_len) or max_model_len)
    safety_margin = int(config.get("safety_margin", compression_metrics.get("safety_margin", 1024)) or 1024)
    scale = float(config.get("scale", 6.0))

    if after_tokens <= 0 or max_model_len <= 0:
        return None

    effective_input_limit = min(max_model_len, cm_input_limit) if cm_input_limit > 0 else max_model_len
    required_reserve = tool_budget + safety_margin
    target_after_tokens = max(0, effective_input_limit - required_reserve)
    budget_deficit = max(0, after_tokens - target_after_tokens)
    if budget_deficit <= 0:
        return {
            "enabled": True,
            "triggered": False,
            "after_prompt_tokens": after_tokens,
            "target_after_tokens": target_after_tokens,
            "budget_deficit": 0,
            "required_reserve": required_reserve,
            "cm_input_limit": effective_input_limit,
            "raw_penalty": 0.0,
        }

    deficit_ratio = min(1.0, budget_deficit / required_reserve) if required_reserve > 0 else 1.0
    raw_penalty = -scale * deficit_ratio

    return {
        "enabled": True,
        "triggered": raw_penalty < 0.0,
        "after_prompt_tokens": after_tokens,
        "target_after_tokens": target_after_tokens,
        "budget_deficit": budget_deficit,
        "required_reserve": required_reserve,
        "cm_input_limit": effective_input_limit,
        "raw_penalty": raw_penalty,
    }


def build_insufficient_budget_reward(
    compression_metrics: Optional[Dict[str, Any]],
    reward_config: Dict[str, Any],
    iteration: int,
    step: int,
    text_content: str,
) -> Optional[Dict[str, Any]]:
    penalty = calculate_budget_penalty(compression_metrics, reward_config)
    if not penalty or not penalty.get("triggered"):
        return None

    return {
        "text_content": text_content,
        "reward": penalty["raw_penalty"],
        "reason": "insufficient_budget",
        "iteration": iteration,
        "step": step,
        "after_prompt_tokens": penalty["after_prompt_tokens"],
        "target_after_tokens": penalty["target_after_tokens"],
        "budget_deficit": penalty["budget_deficit"],
        "required_reserve": penalty["required_reserve"],
        "cm_input_limit": penalty["cm_input_limit"],
    }


def build_recent_compression_record(
    step: int,
    compression_metrics: Optional[Dict[str, Any]],
    budget_penalty: Optional[Dict[str, Any]],
    no_change_count: int,
) -> Dict[str, Any]:
    penalty = budget_penalty or {}
    compression_metrics = compression_metrics or {}
    after_tokens = int(
        penalty.get("after_prompt_tokens")
        or compression_metrics.get("after_tokens", 0)
        or 0
    )
    target_after_tokens = int(
        penalty.get("target_after_tokens")
        or compression_metrics.get("target_after_tokens", 0)
        or 0
    )
    return {
        "step": step,
        "after_tokens": after_tokens,
        "target_after_tokens": target_after_tokens,
        "insufficient_budget": bool(penalty.get("triggered", False)),
        "no_change_count": no_change_count,
    }


def build_terminal_out_of_tokens_reward(
    reward_config: Dict[str, Any],
    iteration: int,
    step: Optional[int],
    error_text: str,
) -> Optional[Dict[str, Any]]:
    config = get_terminal_out_of_tokens_config(reward_config)
    if not config.get("enabled", True) or step is None:
        return None
    return {
        "text_content": f"Terminal OutOfTokens: {error_text}",
        "reward": float(config.get("reward", 0.0)),
        "reason": "terminal_out_of_tokens",
        "iteration": iteration,
        "step": step,
    }


def build_deferred_out_of_tokens_rewards(
    recent_compression_steps: List[Dict[str, Any]],
    reward_config: Dict[str, Any],
    iteration: int,
    error_text: str,
) -> List[Dict[str, Any]]:
    config = _section_config(
        reward_config,
        "deferred_out_of_tokens",
        DEFAULT_DEFERRED_OUT_OF_TOKENS_CONFIG,
    )
    if not config.get("enabled", True):
        return []

    lookback_steps = int(config.get("lookback_steps", 3) or 3)
    rewards_by_distance = config.get("rewards_by_distance", {1: -3.0, 2: -2.0, 3: -1.0})
    only_penalize_insufficient_steps = bool(config.get("only_penalize_insufficient_steps", True))

    rewards: List[Dict[str, Any]] = []
    candidates = list(reversed(recent_compression_steps[-lookback_steps:]))
    for distance, record in enumerate(candidates, start=1):
        if only_penalize_insufficient_steps:
            after_tokens = int(record.get("after_tokens", 0) or 0)
            target_after_tokens = int(record.get("target_after_tokens", 0) or 0)
            should_penalize = (
                bool(record.get("insufficient_budget", False))
                or (target_after_tokens > 0 and after_tokens > target_after_tokens)
                or int(record.get("no_change_count", 0) or 0) > 0
            )
            if not should_penalize:
                continue

        reward_value = rewards_by_distance.get(distance, rewards_by_distance.get(str(distance)))
        if reward_value is None:
            continue

        rewards.append({
            "text_content": f"Deferred OutOfTokens attribution: {error_text}",
            "reward": float(reward_value),
            "reason": "deferred_out_of_tokens",
            "iteration": iteration,
            "step": record.get("step"),
            "distance": distance,
            "after_prompt_tokens": record.get("after_tokens"),
            "target_after_tokens": record.get("target_after_tokens"),
            "insufficient_budget": record.get("insufficient_budget", False),
            "no_change_count": record.get("no_change_count", 0),
        })

    return rewards
