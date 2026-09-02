# AdaCoM (Sonnet 5 fork) — architecture and forward plan

Companion to `docs/adacom_sonnet5_spec.md` (the research spec) and
`docs/anthropic_api_notes.md` (the sourced API facts). This file describes what
the code actually does after the fork's modifications, and what remains.

---

## 1. The system in one paragraph

Two policies share one loop. A **frozen agent** (`claude-sonnet-5`, never
trained) tries to answer a long-horizon question by alternating private
reasoning with tool calls. A **context manager** (`Qwen3-4B-Instruct`, trained
with multi-step GRPO) sits between the agent's steps and *rewrites the agent's
own chat history* — merging, compressing, deleting messages — before the next
agent call. The manager is the MDP policy: its state is a serialized context,
its action is a JSON edit plan, and the agent is part of the environment's
transition function. Only the manager's tokens receive gradient.

The fork's contribution is running this with the agent's **extended thinking
on**, which the Messages API only permits if the in-flight reasoning block
round-trips verbatim — a constraint the original design has no notion of.

---

## 2. Control loop, concretely

One rollout, as the code executes it:

```
BCPWorker.run_search_task(question)                    as1/asio/agent/bcp_worker.py
│
├── memory.add(task question)                          → MemoryManager._chat_history = [q]
│
└── for iteration in range(max_iters):
    │
    ├── _reasoning()
    │   ├── memory.get_memory()                        the managed context c̃
    │   ├── formatter.format([system_msg, *msgs])      → Anthropic wire format
    │   ├── retry_model_call(model, …,                 ← get_model_call_kwargs():
    │   │       thinking={"type":"adaptive"},             thinking + sampling params
    │   │       output_config={"effort": …})
    │   ├── _record_agent_cost(response)               cost guard; may abort the rollout
    │   └── memory.add(assistant msg)                  thinking + text + tool_use
    │                                                  → no tool_result ⇒ round not closed
    ├── _acting(tool_call)                             search / get_document / finish
    │   └── memory.add(tool_result msg, role="user")   → closes the round
    │       └── MemoryManager.compress()               ← THE MANAGER RUNS HERE
    │           ├── regroup tool pairs (thinking preserved)
    │           ├── call_modify()                      serialize context + locks → manager LLM
    │           │   └── format_msgs_with_locks()       [locked] / [rewrite_only] annotations
    │           └── perform_modifications()
    │               ├── filter_modifications()         drop ops that break the lock
    │               ├── plan_lineage()                 uids consumed → produced
    │               ├── apply ops, repair tool pairs
    │               ├── validate_anthropic_history()   would the API accept this?
    │               └── append one line to lineage.jsonl
    │
    └── _process_memory_result()
        ├── _record_lock_violation_penalty()           format penalty (if configured)
        └── build_* rewards                            degeneration, no-change, budget, …
```

Termination: the agent calls `finish`, `max_iters` is reached, the manager runs
out of tokens, or the cost guard trips. The rollout's `results` dict carries the
answer, `end_reason`, `intermediate_rewards` and a `cost` snapshot.

Above this sits Trinity-RFT: `bcp_simple_react_workflow.py` builds the worker per
task, turns each manager step into an `Experience` with `exp.info["intermediate_reward"]`,
and `multi_step_grpo_advantage.py` combines outcome and process rewards into
per-token advantages.

---

## 3. Component map

| Concern | File | Entry points |
|---|---|---|
| Agent loop, tools, rewards | `as1/asio/agent/bcp_worker.py` | `run_search_task`, `_reasoning`, `_acting`, `_process_memory_result` |
| Context manager | `as1/asio/memory/memorymanager.py` | `add` → `compress` → `call_modify` → `perform_modifications` |
| **Context lock + lineage** | `as1/asio/memory/context_lock.py` | `compute_locks`, `filter_modifications`, `format_msgs_with_locks`, `plan_lineage` |
| **API validity check** | `as1/asio/memory/anthropic_validity.py` | `capture_thinking_snapshot`, `validate_anthropic_history` |
| **Cost guard** | `as1/asio/utils/cost_guard.py` | `build_cost_tracker`, `CostTracker.record_usage/finalize` |
| **Background variants** | `as1/asio/agent/background_info.py` | `resolve_background_info` |
| Provider / thinking params | `as1/asio/utils/retry.py` | `detect_model_provider`, `get_thinking_kwargs`, `get_anthropic_sampling_kwargs` |
| Reward builders | `as1/asio/agent/memory_reward_utils.py` | `build_lock_violation_reward`, `build_*_out_of_tokens_*` |
| Live manager prompt | `as1/asio/memory/config/manager_prompts.py` | `MANAGER_1107_STEP_2_PROMPT_UNRESOLVED` (the other ~18 are dead) |
| Trinity workflow | `trinity/common/workflows/envs/browse_comp_plus/bcp_simple_react_workflow.py` | `_parse_workflow_config`, `run_async` |
| Advantage | `trinity/algorithm/advantage_fn/multi_step_grpo_advantage.py` | `inter_reward_normalization` modes |

Bold rows are new in this fork.

---

## 4. The context lock

**The problem.** With thinking on, the Messages API requires the thinking block
of the *in-flight* tool-use cycle to come back verbatim, signature included. The
manager's whole job is rewriting history. Left alone it will eventually rewrite
the message holding that block, and the next agent call 400s.

**The scope of the lock**, computed fresh each round by `compute_locks`:

| Message | Lock | Rationale |
|---|---|---|
| Latest assistant message | `locked` — no op may touch it | Holds the in-flight thinking block + signature + `tool_use` |
| Its paired `tool_result` | `rewrite_only` — may be compressed, never deleted or merged | Deleting it orphans the `tool_use`, and the repair pass would then delete the assistant message, destroying the thinking block |
| Everything older | free | Only the in-flight block must round-trip; older ones may be stripped |

**Two enforcement points**, deliberately:

1. *Prompt side* — `format_msgs_with_locks` annotates the serialized context with
   a `"lock"` field and appends `LOCK_PROMPT_ADDENDUM` to the prompt, so the
   manager can see what it may not touch and is not blamed for not knowing.
2. *Apply side* — `filter_modifications` drops violating ops before they are
   applied and returns the violations.

Enforcing only at the prompt would trust the policy; enforcing only at apply
time would penalize it for a rule it was never told. Both is the honest setup.

**Violations are a configurable signal.** `memory_config.lock_violation_penalty`
is `None` by default (silent drop — right for SFT/warm-up). Set it and each
round with a dropped op emits a `lock_violation` entry into
`intermediate_rewards`, credited to the manager step that produced it.

---

## 5. Validity checking

`validate_anthropic_history` is a pure function over the post-edit history that
returns every rule the Messages API would reject:

`FIRST_NOT_USER`, `SYSTEM_IN_MESSAGES`, `ORPHAN_TOOL_USE`, `ORPHAN_TOOL_RESULT`,
`TOOL_RESULT_NOT_ADJACENT`, `EMPTY_MESSAGE`, `THINKING_MISSING`,
`THINKING_MODIFIED`, `THINKING_NOT_FIRST`.

The thinking checks compare against a snapshot taken *before* the manager ran —
that is what makes "modified" detectable at all, since a rewritten signature is
still a well-formed block. `memory_config.anthropic_validity_mode` selects
`off` / `log` / `raise`; the dev config raises, so a bad context stops the
rollout loudly instead of 400-ing halfway through.

This is defence in depth, not redundancy: the lock prevents the failures we
predicted, the validator catches the ones we did not — including the two
upstream bugs below, which no lock would have stopped.

---

## 6. Lineage and logging

The repo addresses messages by **positional index**, renumbered every round, so
nothing survives to reconstruct a message's edit history. We stamp a stable
`cm_uid` into `Msg.metadata` and, per round, append one JSONL line containing:

```jsonc
{"round": 3,
 "pre":  [{"cm_uid": …, "role": …, "content": …}, …],   // full context before
 "post": [ … ],                                          // full context after
 "modifications": [ … ],                                 // the ops applied
 "lock_violations": [ … ],
 "validity_violations": [ … ],
 "lineage": [{"consumed_uids": [...], "produced_uid": …, "justification": …,
              "consumed_chars": …, "new_content_chars": …}]}
```

`consumed_uids → produced_uid` is a message's serial-reproduction chain: the
edge list from which the Bartlett-taxonomy analysis is a post-hoc pass over
stored data rather than a live instrumentation problem. **Do not remove or
"simplify" this logging** — it is the thesis's primary corpus, not debug output.

Path resolution (`resolve_lineage_log_path`): explicit `lineage_log_path` >
`lineage_log_dir` (one file per rollout, since G rollouts append concurrently) >
the ExperimentLogger run dir. Prefer `lineage_log_dir`: the run dir only exists
for ~50% of rollouts by the repo's own sampling rule.

---

## 7. Rewards and advantage

Outcome reward: an LLM judge scores the final answer.

Process rewards, summed per manager step into `intermediate_rewards`:

| Reason | Source | Default |
|---|---|---|
| `parsing_failure` | unparseable edit plan | −0.5 |
| `degenerate_generation` | repetition detector | −1.0 |
| `no_change` | rewrite identical to the original | −0.5 × count |
| `insufficient_budget`, `deferred/terminal_out_of_tokens` | context-budget machinery | config |
| search-hit gold/key | BCP-specific, credited to the *preceding* step | +0.6 / +0.3 |
| **`lock_violation`** | ops dropped by the lock | `None` (off) |

`multi_step_grpo_advantage.py` supports three `inter_reward_normalization`
modes: `none` (the released config: `A = z(R) + α·ΣQ`, clipped),
`task_std`, and `outcome_tie` (closest to paper §3.2: z-scored process rewards,
and the *only* signal when all G rollouts share an outcome). **This is an
unmade experimental decision** — currently `none`, untouched.

---

## 8. Cost guard

Every agent call's usage is counted and priced from
`examples/dev/anthropic_pricing.yaml` (rates, source URL, date checked — never
hardcoded). On hitting `max_usd` or `max_tokens` the rollout sets
`abort_rollout` / `abort_reason = "cost_budget_exceeded"`, the loop and the
end-of-run summarizing call stop, and `results` marks it unsuccessful. Each
rollout appends its spend to a JSONL and folds into a process-level batch total.

Two accounting caveats: agentscope's `ChatUsage` drops the cache breakdown, so
with prompt caching on the estimate is an **upper bound**; and Claude 4.7+ uses
a tokenizer producing ~30% more tokens for the same text, so token caps do not
transfer between model generations.

---

## 9. Configuration surface added by this fork

Workflow args (`workflow_args:` in the taskset):

| Key | Meaning |
|---|---|
| `agent_enable_thinking` | Turns on adaptive thinking **and** the context lock |
| `agent_thinking_config: {effort}` | `low`/`medium`/`high`/`xhigh`/`max`; omitted = API default `high` |
| `agent_max_tokens` | Hard ceiling on thinking + response; the SDK default 2048 is too small |
| `agent_cost_config: {pricing_path, max_usd, max_tokens, log_path}` | Cost guard |

`memory_config:`

| Key | Meaning |
|---|---|
| `agent_thinking_enabled` | Threaded in automatically; do not set by hand |
| `lock_violation_penalty` | `None` = silent drop; a float = RL signal |
| `lineage_log_dir` / `lineage_log_path` / `lineage_logging` | Lineage JSONL |
| `anthropic_validity_mode` | `off` / `log` / `raise` |
| `background_info_key` | `bcp` / `mcp` / `anthropic_tool_use` |
| `chat_tokenizer` | Inject a prebuilt tokenizer (tests) |

---

## 10. Testing

69 tests, all CPU-only with no vLLM, Ray, torch, transformers or network:
`python as1/tests/run_all.py`.

`as1/tests/stubs.py` installs minimal stand-ins for agentscope, transformers,
tiktoken and json_repair **only when the real package is missing**, so the same
tests exercise the real objects on a full install. This is what lets
`test_smoke_loop.py` drive three real manager rounds — real `MemoryManager`,
real lock, real repair pass, real validator — against a fake agent, fake tool
and fake manager model, and assert the resulting context is API-valid.

That test is the load-bearing one. It found both upstream bugs.

---

## 11. Two upstream bugs, fixed

Neither could appear on the paper's setup (OpenAI-compatible relays, thinking
off), and neither would have been caught by the lock:

1. **The manager was never invoked.** `MemoryManager.add` returned early on any
   `role == "user"` message. On the direct Anthropic API tool results *must* be
   user-role — the Messages API has no system role in `messages` — and those are
   exactly the messages that close a round. Silent no-op, not a crash: the run
   would have looked like a working ReAct baseline with a manager that never
   fired.
2. **`compress()` destroyed the thinking block.** Its regrouping pass rebuilt
   each assistant message as `[text, tool_use]`, dropping `thinking` every round
   before the manager or the lock ever saw the context — a guaranteed 400 on the
   next agent call.

---

## 12. Status

**Verified** (mocked, CPU): the lock's scope and both enforcement points;
lineage records and JSONL; the validator against synthetic and generated
histories; lock-violation penalties; cost arithmetic and the abort path;
thinking/sampling kwargs for every provider path; three real manager rounds
end-to-end.

**Not verified** (needs a GPU and an API key): that vLLM serves the manager
under this config; that the real Sonnet 5 accepts our generated contexts across
hundreds of steps; real token accounting and dollar burn; whether an untrained
4B manager produces parseable plans often enough to be worth measuring.

---

## 13. Forward plan

### Stage 0 — harness check (cheap, no thinking)
Run the dev config with `agent_model_name: claude-haiku-4-5` and
`agent_enable_thinking: false`. This exercises Trinity, vLLM, the searcher, the
judge and the manager loop with no thinking constraints and the cheapest agent.
**Exit:** 5 rollouts complete, non-zero manager rounds, judge scores land.

### Stage 1 — thinking on, zero 400s
Run `examples/dev/sonnet5_untrained_1gpu.yaml` as written.
**Exit:** a few hundred manager steps with no `MEMORY_VALIDITY` warnings, no
`ContextValidityError`, no API 400s, and lineage JSONL that round-trips.
Then re-run with `anthropic_validity_mode: log` to measure violation *rates*
rather than dying on the first one.

### Stage 2 — measure the noise floor
Before any training: fix 5–10 tasks and run the ReAct baseline (no manager)
several times each. Thinking forbids `temperature`, so the agent is not
reproducible; you need the variance of pass rate to know what an RL gain has to
clear. **This is a prerequisite for any claim, not an optional check.**

### Stage 3 — untrained-manager baseline
The paper's "AdaCoM w/o train". Same config, more tasks. Record: parse-failure
rate, lock-violation rate, mean context length curve, pass rate vs. Stage 2.
If the untrained manager *hurts*, that reproduces the paper's premise.

### Stage 4 — decisions that gate training
- **Advantage mode**: `none` (released) vs `outcome_tie` (paper). Record it in
  the config name; do not switch silently.
- **Lock violations**: leave silent for warm-up, or penalize during RL. The code
  supports both; the choice is `lock_violation_penalty`.
- **SFT warm-up vs. constrained decoding**: vLLM guided decoding against the op
  schema removes JSON-parse and missing-field failures by construction. Pilot it
  before committing to a teacher-generated SFT set.
- **Effort level**: `medium` in dev; `high` (the default) or `xhigh` for
  experiments. It changes both cost and the agent's tool-call behaviour, so fix
  it before the baseline, not after.

### Stage 5 — RL
4× A100-80 or 4× L40S on `gpu` (`--qos=long`); `engine_num: 2`,
`ulysses_sequence_parallel_size: 2`. Compute the agent-token bill against the
current Sonnet 5 rates *before* choosing G, `T_max` and epochs — §6.3 of the
spec, with the pricing file as the input.

### Stage 6 — the actual thesis
The lineage JSONL is the corpus. Code each message's `consumed_uids →
produced_uid` chain for Bartlett-style transformations (levelling, sharpening,
assimilation) and ask whether an RL-trained manager's drift differs
systematically from an untrained one's. Nothing in stages 0–5 is the
contribution; they are what makes the corpus trustworthy.

### Deferred
- `Task` interface for non-BCP tasks (spec §6.2 process-signal hook).
- MCP worker parity: it has the same reward path but no lock-violation penalty.
- Extract op (spec §4.3): on, off, or learned — still undecided for this fork.
