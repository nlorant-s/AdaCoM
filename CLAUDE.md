# CLAUDE.md — AdaCoM fork (Sonnet 5 / Haiku 4.5 frozen agent)

Read `docs/adacom_sonnet5_spec.md` before touching anything. §12–13 reconcile the paper with this codebase.

## What this repo is
Fork of github.com/luyi256/AdaCoM: Trinity-RFT + vendored AgentScope (`as1/`). An RL-trained
**context manager** (small local LLM, served by vLLM) rewrites the chat history of a **frozen
API agent** between steps. Only the manager is trained (multi-step GRPO). Paper: arXiv 2605.30785.

Our deltas from upstream:
- Frozen agent = Claude via the **direct Anthropic Messages API** (`as1/src/agentscope/model/_anthropic_model.py`),
  **extended thinking ON**. Dev on `claude-haiku-4-5`, experiments on `claude-sonnet-5`.
- The manager may **never modify the latest assistant message** (in-flight thinking block + tool_use);
  its paired tool_result is **rewrite-only**. Implemented in `as1/asio/memory/context_lock.py`,
  wired into `memorymanager.py` (`call_modify`, `perform_modifications`). Tests: `as1/tests/test_context_lock.py`.
- Per-step lineage + pre/post context snapshots are logged (JSONL at `memory_config.lineage_log_path`)
  for a serial-reproduction / drift analysis. Do not remove or "simplify" this logging.

## Key files
| Purpose | Path |
|---|---|
| Manager (apply edits, repair tool pairs, call LLM) | `as1/asio/memory/memorymanager.py` |
| Lock + lineage | `as1/asio/memory/context_lock.py` |
| Live manager prompt | `as1/asio/memory/config/manager_prompts.py` → `MANAGER_1107_STEP_2_PROMPT_UNRESOLVED` (the ~18 other prompts are dead) |
| Agent worker (BCP) | `as1/asio/agent/bcp_worker.py` (`Background_info` constant, `get_model_call_kwargs`, reward wiring) |
| Provider/thinking helpers | `as1/asio/utils/retry.py` (`detect_model_provider`, `get_thinking_kwargs`, `convert_tools_openai_to_anthropic`) |
| Trinity workflow (threads config → worker) | `trinity/common/workflows/envs/browse_comp_plus/bcp_simple_react_workflow.py` |
| Advantage | `trinity/algorithm/advantage_fn/multi_step_grpo_advantage.py` |
| Kept config | `examples/browse_comp_plus/bcp_config_singlenode_8gpu_deepseek.yaml` |

## Invariants — do not break
1. Anthropic message validity: every `tool_use` is followed by a `tool_result` with the same id; the latest
   assistant message's `thinking` block round-trips **verbatim with signature**; `system` is a top-level
   param, never a message. The repair pass in `perform_modifications` handles pairing — keep it.
2. Message ids in manager output are **positional indices** into `_chat_history`, renumbered each round.
   Stable identity lives in `Msg.metadata["cm_uid"]`. Never rely on positional ids across rounds.
3. `temperature` and `top_p` must not both be sent to Claude (already special-cased in `get_model_call_kwargs`).
4. Never hardcode Anthropic pricing, rate limits, or model capabilities — read them from docs.claude.com
   at implementation time and put them in config. Every run must have a hard dollar cap.
5. Advantage mode is a recorded experimental choice (`inter_reward_normalization`: `none` = released
   config; `outcome_tie` ≈ paper §3.2). Do not silently change it.

## Workflow rules
- Small, reviewable commits; one concern each. Run `python as1/tests/test_context_lock.py` (and any new tests) before committing.
- Anything that calls a paid API must be behind a flag and default to a **dry-run / mocked model** in tests.
- No GPU in dev by default: unit tests must run CPU-only without vLLM/Ray. Mark GPU/Unity-only tests clearly.
- When unsure about API behavior, check https://docs.claude.com/en/docs/build-with-claude/extended-thinking
  and https://docs.claude.com/en/api/messages rather than guessing.
- Ask before: changing the manager prompt text, changing reward magnitudes, or touching Trinity internals.

## Environment
- Cluster: UMass Unity (Slurm). Dev: 1× L40S on `gpu-preempt` (≤2h). RL: 4× A100-80GB or 4× L40S on `gpu`
  (`--qos=long` if >48h). Constraint names: `l40s`, `a100-80g`. See spec §13.2.
- Python 3.10+, `bash install.sh` installs trinity-rft + as1. Use a conda env in a workspace, not $HOME.
- Secrets via env vars only (`ANTHROPIC_API_KEY`); never commit `export_template.sh` filled in.
