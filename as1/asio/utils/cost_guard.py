# -*- coding: utf-8 -*-
"""Per-run token and dollar budget for the frozen agent's API calls.

CLAUDE.md invariant 4: prices are never hardcoded. They are read from a config
file (``examples/dev/anthropic_pricing.yaml`` ships one, checked against
platform.claude.com on the date recorded inside it) and every run must have a
hard dollar cap.

Usage::

    tracker = build_cost_tracker(cost_config, model_name="claude-haiku-4-5")
    ...
    tracker.record_usage(response.usage)      # after each model call
    if tracker.over_budget():                 # abort the rollout, mark it
        ...

Accounting notes:

* agentscope's ``ChatUsage`` carries only ``input_tokens`` / ``output_tokens``,
  dropping the cache breakdown, so cached input is priced as full input. The
  estimate is therefore an **upper bound** whenever prompt caching is on. Raw
  Anthropic usage dicts (with ``cache_creation_input_tokens`` /
  ``cache_read_input_tokens``) are priced properly when they are passed in.
* Claude 4.7 and later use a tokenizer that produces roughly 30% more tokens
  for the same text, so token caps do not carry over between model generations.
* This is a rollout-local guard. Rollouts run in separate Ray workers, so the
  batch total is reconstructed from the per-rollout JSONL at ``log_path``
  rather than from shared memory; ``record_batch_spend`` also keeps a
  process-level running total for the logs.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

TOKENS_PER_MILLION = 1_000_000


class CostBudgetExceeded(Exception):
    """The run hit its configured token or dollar cap."""

    def __init__(self, snapshot: Dict[str, Any]):
        self.snapshot = snapshot
        super().__init__(
            f"cost budget exceeded: ${snapshot.get('cost_usd', 0):.4f} / "
            f"{snapshot.get('total_tokens', 0)} tokens "
            f"(caps: {snapshot.get('max_usd')} usd, {snapshot.get('max_tokens')} tokens)"
        )


@dataclass
class ModelPrice:
    """Dollars per million tokens."""

    input: float = 0.0
    output: float = 0.0
    cache_write_5m: Optional[float] = None
    cache_write_1h: Optional[float] = None
    cache_read: Optional[float] = None

    @classmethod
    def from_config(cls, raw: Dict[str, Any]) -> "ModelPrice":
        return cls(
            input=float(raw.get("input", 0.0)),
            output=float(raw.get("output", 0.0)),
            cache_write_5m=_opt_float(raw.get("cache_write_5m")),
            cache_write_1h=_opt_float(raw.get("cache_write_1h")),
            cache_read=_opt_float(raw.get("cache_read")),
        )


def _opt_float(value):
    return None if value is None else float(value)


def load_pricing(path: str) -> Dict[str, ModelPrice]:
    """Read a pricing file (YAML or JSON) into {model_name: ModelPrice}.

    The file's top level is either the model map itself or a ``models:`` key.
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"pricing file not found: {path!r}")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as e:  # pragma: no cover - depends on the install
            raise ImportError(
                f"reading {path} needs PyYAML; use a .json pricing file instead"
            ) from e
        raw = yaml.safe_load(text) or {}
    else:
        raw = json.loads(text) if text.strip() else {}

    models = raw.get("models", raw) if isinstance(raw, dict) else {}
    return {str(name): ModelPrice.from_config(cfg or {}) for name, cfg in models.items()}


def resolve_price(pricing: Dict[str, ModelPrice], model_name: str) -> Optional[ModelPrice]:
    """Exact match first, then the longest configured id the name starts with.

    Lets one entry (``claude-haiku-4-5``) cover dated ids
    (``claude-haiku-4-5-20251001``).
    """
    if not model_name:
        return None
    name = model_name.lower()
    if name in pricing:
        return pricing[name]
    candidates = [k for k in pricing if name.startswith(k.lower())]
    if not candidates:
        candidates = [k for k in pricing if k.lower() in name]
    if not candidates:
        return None
    return pricing[max(candidates, key=len)]


@dataclass
class UsageTotals:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (self.input_tokens + self.output_tokens
                + self.cache_write_tokens + self.cache_read_tokens)


def extract_usage(usage: Any) -> Dict[str, int]:
    """Normalise an Anthropic / agentscope / OpenAI usage object to ints."""
    if usage is None:
        return {}
    if not isinstance(usage, dict):
        usage = {
            k: getattr(usage, k)
            for k in ("input_tokens", "output_tokens", "prompt_tokens",
                      "completion_tokens", "cache_creation_input_tokens",
                      "cache_read_input_tokens")
            if getattr(usage, k, None) is not None
        }

    def _int(*keys):
        for k in keys:
            v = usage.get(k)
            if isinstance(v, (int, float)):
                return int(v)
        return 0

    return {
        "input_tokens": _int("input_tokens", "prompt_tokens"),
        "output_tokens": _int("output_tokens", "completion_tokens"),
        "cache_write_tokens": _int("cache_creation_input_tokens", "cache_write_tokens"),
        "cache_read_tokens": _int("cache_read_input_tokens", "cache_read_tokens"),
    }


class CostTracker:
    """Counts tokens and dollars for one rollout and enforces the caps."""

    def __init__(
        self,
        pricing: Dict[str, ModelPrice],
        model_name: str = "",
        max_usd: Optional[float] = None,
        max_tokens: Optional[int] = None,
        log_path: Optional[str] = None,
        experiment_logger: Any = None,
        run_id: Any = None,
        batch_id: Any = None,
    ):
        self.pricing = pricing or {}
        self.model_name = model_name or ""
        self.max_usd = None if max_usd is None else float(max_usd)
        self.max_tokens = None if max_tokens is None else int(max_tokens)
        self.log_path = log_path
        self.experiment_logger = experiment_logger
        self.run_id = run_id
        self.batch_id = batch_id
        self.totals = UsageTotals()
        self.cost_usd = 0.0
        self.unpriced_models = set()

    # -- accounting ---------------------------------------------------------
    def record_usage(self, usage: Any, model_name: Optional[str] = None) -> float:
        """Add one call's usage; returns the dollars it cost."""
        fields = extract_usage(usage)
        if not fields:
            return 0.0
        self.totals.calls += 1
        self.totals.input_tokens += fields["input_tokens"]
        self.totals.output_tokens += fields["output_tokens"]
        self.totals.cache_write_tokens += fields["cache_write_tokens"]
        self.totals.cache_read_tokens += fields["cache_read_tokens"]

        price = resolve_price(self.pricing, model_name or self.model_name)
        if price is None:
            # No price configured: still count tokens, and say so once.
            name = model_name or self.model_name
            if name and name not in self.unpriced_models:
                self.unpriced_models.add(name)
                if self.experiment_logger:
                    self.experiment_logger.log_warning(
                        f"COST_GUARD: no pricing entry for {name!r}; "
                        "dollars are not being counted for it"
                    )
            return 0.0

        cost = (
            fields["input_tokens"] * price.input
            + fields["output_tokens"] * price.output
            + fields["cache_write_tokens"] * (price.cache_write_5m
                                              if price.cache_write_5m is not None
                                              else price.input)
            + fields["cache_read_tokens"] * (price.cache_read
                                             if price.cache_read is not None
                                             else price.input)
        ) / TOKENS_PER_MILLION
        self.cost_usd += cost
        return cost

    # -- budget -------------------------------------------------------------
    def over_budget(self) -> bool:
        if self.max_usd is not None and self.cost_usd >= self.max_usd:
            return True
        if self.max_tokens is not None and self.totals.total_tokens >= self.max_tokens:
            return True
        return False

    def check_budget(self) -> None:
        """Raise CostBudgetExceeded if a cap has been hit."""
        if self.over_budget():
            raise CostBudgetExceeded(self.snapshot())

    def snapshot(self) -> Dict[str, Any]:
        snap = asdict(self.totals)
        snap.update({
            "total_tokens": self.totals.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "model": self.model_name,
            "max_usd": self.max_usd,
            "max_tokens": self.max_tokens,
            "over_budget": self.over_budget(),
            "run_id": self.run_id,
            "batch_id": self.batch_id,
        })
        return snap

    # -- reporting ----------------------------------------------------------
    def finalize(self) -> Dict[str, Any]:
        """Persist this rollout's spend and fold it into the batch total."""
        snap = self.snapshot()
        cumulative = record_batch_spend(self.batch_id, self.cost_usd,
                                        self.totals.total_tokens)
        snap["batch_cumulative_usd"] = round(cumulative["cost_usd"], 6)
        snap["batch_cumulative_tokens"] = cumulative["total_tokens"]
        snap["batch_rollouts"] = cumulative["rollouts"]
        if self.log_path:
            try:
                parent = os.path.dirname(self.log_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(snap, ensure_ascii=False) + "\n")
            except OSError as e:
                if self.experiment_logger:
                    self.experiment_logger.log_warning(f"COST_LOG failed: {e}")
        if self.experiment_logger:
            self.experiment_logger.log_info(
                f"COST_GUARD: rollout ${snap['cost_usd']:.4f} "
                f"({snap['total_tokens']} tokens, {snap['calls']} calls); "
                f"batch {self.batch_id} cumulative ${snap['batch_cumulative_usd']:.4f} "
                f"over {snap['batch_rollouts']} rollout(s) in this process"
            )
        return snap


# --- process-level batch accumulator ---------------------------------------
_BATCH_LOCK = threading.Lock()
_BATCH_SPEND: Dict[Any, Dict[str, Any]] = {}


def record_batch_spend(batch_id: Any, cost_usd: float, total_tokens: int) -> Dict[str, Any]:
    with _BATCH_LOCK:
        entry = _BATCH_SPEND.setdefault(
            batch_id, {"cost_usd": 0.0, "total_tokens": 0, "rollouts": 0})
        entry["cost_usd"] += float(cost_usd)
        entry["total_tokens"] += int(total_tokens)
        entry["rollouts"] += 1
        return dict(entry)


def batch_spend(batch_id: Any = None) -> Dict[str, Any]:
    with _BATCH_LOCK:
        if batch_id is None:
            return {k: dict(v) for k, v in _BATCH_SPEND.items()}
        return dict(_BATCH_SPEND.get(batch_id, {"cost_usd": 0.0, "total_tokens": 0, "rollouts": 0}))


def reset_batch_spend() -> None:
    with _BATCH_LOCK:
        _BATCH_SPEND.clear()


# --- construction from run config ------------------------------------------
def build_cost_tracker(
    cost_config: Optional[Dict[str, Any]],
    model_name: str = "",
    experiment_logger: Any = None,
    run_id: Any = None,
    batch_id: Any = None,
) -> Optional[CostTracker]:
    """Build a tracker from ``agent_cost_config``, or None when disabled.

    Config keys: ``enabled`` (default True when the section exists),
    ``pricing_path`` (required), ``max_usd``, ``max_tokens``, ``log_path``.
    """
    cfg = cost_config or {}
    if not cfg or cfg.get("enabled") is False:
        return None
    pricing_path = cfg.get("pricing_path")
    if not pricing_path:
        raise ValueError(
            "agent_cost_config needs pricing_path: prices are read from a "
            "config file, never hardcoded"
        )
    pricing = load_pricing(pricing_path)
    return CostTracker(
        pricing=pricing,
        model_name=model_name,
        max_usd=cfg.get("max_usd"),
        max_tokens=cfg.get("max_tokens"),
        log_path=cfg.get("log_path"),
        experiment_logger=experiment_logger,
        run_id=run_id,
        batch_id=batch_id,
    )
