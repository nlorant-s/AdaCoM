# AdaCoM Reimplementation Spec — Sonnet 5 as Frozen Agent

Source: Yi, Lei et al., *Learning Agent-Compatible Context Management for Long-Horizon Tasks* (arXiv 2605.30785, May 2026).
Scope: the RL context-management architecture only. Tasks (BrowseComp-Plus, MCP-Bench-Wiki, or your own) plug in through an interface defined below.

---

## 1. What you are actually building

Two policies, one training loop:

| Component | Paper | Your build |
|---|---|---|
| Frozen agent `A` | Qwen3-Max / Kimi / GLM / DeepSeek via API | `claude-sonnet-5` via Messages API (never trained) |
| Context manager `π_θ` | Qwen3-4B-Instruct, SFT → GRPO | Open-weights model you train locally (size is a decision, see §10) |
| Environment | Tool backends + LLM judge | Your `Task` implementations |
| Trainer | Trinity-RFT | Trinity-RFT / verl / SkyRL / custom (§7) |

The manager is an MDP policy whose "state" is a serialized context and whose "action" is a JSON edit plan. The agent is part of the environment's transition function. That framing decides almost every engineering choice below.

---

## 2. Data model

```python
@dataclass
class Message:
    id: str                 # stable, never reused within a rollout; e.g. "m017"
    role: Literal["user", "assistant"]
    blocks: list[ContentBlock]   # Anthropic content blocks: text | tool_use | tool_result | thinking
    origin: Literal["task", "agent", "env", "manager"]
    born_step: int          # manager step that created it (for drift analysis)
    lineage: list[str]      # ids of messages this one replaced (for drift analysis)

@dataclass
class ContextState:
    system: str             # agent system prompt — NOT visible to / editable by manager
    messages: list[Message] # managed context c̃_t
    id_counter: int

@dataclass
class ModificationOp:        # one δ^(j)_t
    ids: list[str]          # must be consecutive existing ids
    role: Literal["user", "assistant"]
    justification: str      # stripped before agent sees context
    new_content: str        # "" → delete targeted messages

@dataclass
class ManagerStep:           # one (p_t, m_t) sample for training
    rollout_id: str
    t: int
    prompt: str             # P(c_t): serialized context + instruction + token ratio
    raw_action: str         # manager output tokens (what GRPO optimizes)
    parsed: list[ModificationOp] | None
    pre_context: ContextState   # c_t   (log this — it is your Bartlett corpus)
    post_context: ContextState  # c̃_t
    process_reward: float   # Q_{i,t}
    format_ok: bool
```

Note the paper says `role ∈ {SYSTEM, USER, ASSISTANT}` in §3.1 but the actual prompt (Table 9) offers only `"user" | "assistant"`. Follow the prompt. The Messages API has no system role in `messages` anyway.

---

## 3. Control loop

```
c̃_0 = [task_message(q)]
for t in 1..T_max:
    a_t  = Agent(system, c̃_{t-1})            # Sonnet 5 call, tools enabled
    if a_t is finish(answer): break
    o_t  = Env.execute(a_t.tool_calls)        # tool_result blocks
    c_t  = c̃_{t-1} + [a_t, o_t]              # append; assign new ids
    o_t  = Extract(o_t) if len(o_t) > τ_extract   # optional, see §4.3
    p_t  = P(c_t, token_ratio)
    m_t  ~ π_θ(· | p_t)                        # manager call (vLLM)
    c̃_t = Apply(c_t, m_t) → Repair → Validate
    log ManagerStep
R = Judge(q, answer)                           # outcome reward
```

Termination: `finish` tool called, or `T_max` (paper: 35) reached. Paper applies the token penalty when managed context exceeds the cap but does not say whether the rollout continues or is truncated — decide (recommend: continue, penalize, hard-truncate oldest non-task messages only if the *manager's own* window would overflow).

---

## 4. Manager action space and application semantics

### 4.1 Apply(c_t, m_t)
For each op in order:
- Locate messages by `ids`; require they exist and are consecutive.
- `new_content == ""` → remove them.
- Otherwise → replace the span with one new `Message(role=op.role, blocks=[text(new_content)])`, new id, `lineage = ids`.
- Messages not targeted by any op are copied unchanged. Empty op list = no-op.
- Overlapping ids across ops → format penalty, apply none (paper is silent; be strict).

### 4.2 Validity and repair (Sonnet-specific — not in paper)
The paper's agents accept arbitrary message lists. The Messages API does not. Invariants to enforce **after** applying edits:

1. **tool_use ↔ tool_result pairing.** Every `tool_use` block in an assistant message must be immediately followed by a user message containing a `tool_result` with matching `tool_use_id`, and vice versa. If the manager rewrites `a_t` (which contains the tool_use) to plain text but leaves `o_t`, you now have an orphan `tool_result` → 400.
   - **Option A (recommended):** treat `(a_i, o_i)` as an atomic pair for *partial* edits — if either is targeted alone, auto-convert the surviving partner to a text block that describes the call/result. Log that a repair fired.
   - **Option B:** serialize the whole history as text and drop native tool calling. Simpler, but then you are studying a text-ReAct Sonnet, not the tool-native agent. Rejected for a frozen-agent study.
   - **Option C:** constrain the action space so `ids` can only address whole pairs or manager-authored messages. Cleanest, mildly less expressive than the paper.
2. **Message ordering / alternation.** Verify current API behavior for consecutive same-role messages (historically merged; confirm for Sonnet 5 at https://docs.claude.com/en/api/messages). If not merged, insert a minimal separator or merge in Repair.
3. **First message must be `user`.** The task message guarantees this unless the manager deletes it — the prompt forbids that; add a hard guard.
4. **Thinking blocks** — see §5.3. This one changes the design.

### 4.3 Extract (Appendix A, Table 12)
A separate operation the manager also performs: when a tool result exceeds a threshold (paper: 2,000 tokens on MCP-Wiki), chunk it and iteratively refine a "reading note" against the original question before it ever enters the context. This is *serial reproduction inside a single step* — directly relevant to your Bartlett angle, and also a confound if you want to isolate the manager's edit policy. Decide whether it is (a) part of the trained manager, (b) a fixed non-learned preprocessor, or (c) off.

---

## 5. Sonnet 5 API constraints (the new engineering surface)

### 5.1 System prompt
Single top-level `system` parameter, not a message. Keep it fixed and invisible to the manager. The paper's "always preserve the task description" refers to the first user message.

### 5.2 Prompt caching vs. edit locality
Every edit to message `k` invalidates the cache from `k` onward. The paper's learned strategy (state message at index 1, right after the task) is the worst case for caching: it rewrites the second message nearly every step. Options:
- Accept it; budget for uncached input on most steps.
- Place cache breakpoints after `system` + task message only; treat everything else as uncached.
- Experiment: working note at the *end* of context (before the next action) vs. the paper's position. This is a position-vs-recency question your thesis could actually ask.
The paper's "tiered management" for strong agents (intervene rarely, in batches) is also the cache-friendly regime — and Sonnet 5 is the strongest agent anyone has run this on.

### 5.3 Extended / adaptive thinking
If thinking is enabled for the agent, the API requires the thinking block of the *in-flight* tool-use cycle to be passed back unmodified with its signature; altering it or dropping it yields a 400. Consequence: the manager **cannot rewrite `a_t` in the same round it was produced** if `a_t` contains a thinking block — which is exactly what the DeepSeek-style "eager distillation" manager does. Choices:
- Disable thinking on the agent. Clean MDP, matches the paper's agents, weaker Sonnet.
- Enable thinking, and constrain the manager's action space so the most recent assistant message is immutable until the cycle closes (a new non-tool user turn). Older thinking blocks can be stripped.
- Enable thinking, and let `o_t` be edited but never `a_t`. Asymmetric but workable.
Verify current rules at https://docs.claude.com/en/docs/build-with-claude/extended-thinking before committing.

### 5.4 Token counting
The context-length penalty, the token-usage ratio in the manager prompt, and the extract threshold all need agent-tokenizer counts. Use the token-counting endpoint (costs a call) or a calibrated estimator; the manager only needs a ratio, so an estimator with ±5% error is fine for the prompt, but use exact counts for the penalty threshold in logs.

### 5.5 Nondeterminism
Sonnet at any temperature is not bit-reproducible. In GRPO the manager gets credited or blamed for the agent's sampling noise. The paper has this problem too and absorbs it with G=8. Mitigations: `temperature=0` on the agent (reduces, doesn't eliminate), larger G, or seed-matched pairs. Measure the variance of ReAct-baseline pass rates on fixed tasks before training so you know the noise floor.

### 5.6 Rate limits, retries, budget
Rollouts are embarrassingly parallel (G × batch × T_max calls). You need an async orchestrator with per-rollout retry, idempotent step logging (resume mid-rollout after a crash), and a hard dollar cap that aborts training. Check current rate limits and pricing at https://docs.claude.com/en/api/rate-limits and https://claude.com/pricing; do not hardcode.

### 5.7 Context cap
The paper caps every agent's effective input at 32K because the *manager's* window is 32K. Sonnet 5's native window is far larger; the cap becomes an experimental knob rather than a constraint. The Fidelity–Reliability result predicts a strong agent wants a high cap and light intervention — meaning a small manager may be the bottleneck (their own stated limitation).

---

## 6. Training pipeline

### 6.1 SFT warm-up (format only)
Paper: teacher model (GPT-5 / Opus 4.6) generates edit plans while the frozen agent runs on the training split; filter to parseable JSON, <12K chars, ≥20% reduction for `modify` ops; MinHash dedup; ratio-control to ~73% modify / 20% extract / 7% delete+no-change. A few epochs.

**Alternative that may make SFT unnecessary:** constrained JSON decoding in vLLM (guided decoding against the op schema). This eliminates JSON-parse and missing-field failures by construction; only nonexistent-id and non-consecutive-id errors remain. The paper's own finding is that the untrained manager hurts because of *strategy*, not format — so you still need RL, but you may be able to skip SFT or reduce it to a few hundred examples. Worth a pilot.

### 6.2 GRPO (§3.2)
Per query: sample G rollouts (paper G=8). Each rollout yields T_i manager steps.

Rewards:
- Outcome `R_i`: judge score of final answer (binary for deterministic tasks, rubric for open-ended).
- Process `Q_{i,t}` (rule-based, summed):
  - token penalty −0.8 on the step that produced an over-cap context
  - redundant-action penalty −0.4 on the manager step between two identical consecutive tool calls
  - format penalty −0.5 (parse failure, bad ids, missing fields)
  - task-specific positive signals (BrowseComp: +0.6 gold-doc-found, +0.3 key-doc-found, credited to the *preceding* manager step) — exposed as a `Task.process_signals()` hook

Advantages (two-level, then renormalize):
```
A^R_i   = (R_i − μ_R) / (σ_R + ε)            over the G rollouts
A^Q_it  = (Q_it − μ_Q) / (σ_Q + ε)           over all steps of all G rollouts
A_it    = A^R_i + α·A^Q_it                   α = 0.1
Â_it    = (A_it − μ_A) / (σ_A + ε)           over all steps in the group
```
When σ_R = 0 (all rollouts same outcome) the process term is the only signal — this is why they need it.

Objective: PPO-clip over manager-emitted tokens, normalized by total manager tokens Z across the group (token-level mean, not sequence-level).

Hyperparameters (paper, Appendix A.2): lr 5e-6, KL coef 0.006, clip 0.2, no entropy bonus, sampling temp 1.0 (lower degenerated), rollout batch 32 trajectories, train batch 768 manager-step samples, ≤40 epochs, early stop after 20 rollout steps without eval gain, manager ctx 32,768 / max out 4,096.

### 6.3 Cost model (the gap the paper doesn't have)
Agent tokens per rollout ≈ Σ_t |c̃_{t-1}| ≈ T · mean_ctx. With T≈25 and mean_ctx≈8K (tiered regime), that's ~200K input tokens per rollout, ~1.6M per group of 8, ~6.4M per rollout batch of 32 trajectories, plus judge calls. Multiply by the number of rollout batches to convergence (paper: up to 40 epochs over 680 tasks). Compute this against current Sonnet 5 pricing *before* choosing G, T_max, and epochs. Cheap levers: develop and debug the whole pipeline on Haiku 4.5 as the frozen agent, switch to Sonnet 5 for the real runs; cache the stable prefix; reduce T_max; curriculum on shorter tasks.

---

## 7. Framework choice

Requirements: multi-step manager decisions per trajectory with a shared trajectory-level reward, custom rollout that calls an external HTTP API, vLLM-served policy, GRPO with custom advantage computation.

- **Trinity-RFT** (paper's choice): built for exactly this "workflow" pattern; least translation risk; Alibaba ecosystem.
- **verl** agent-loop / **SkyRL** / **rLLM**: all support agentic multi-turn RL with external environments; you'll write the custom two-level advantage either way.
- **TRL GRPOTrainer**: single-turn assumption; you'd fight it.

Recommend decoupling regardless: a **rollout service** (async, API-calling, writes `ManagerStep` records to disk/DB) and a **trainer** that consumes batches. That lets you (a) reuse rollouts for the drift analysis, (b) swap frozen agents, (c) survive crashes.

---

## 8. Logging schema for your thesis

The paper logs enough to compute outcome taxonomies and context-length curves. You need more. Per manager step, persist:
- `pre_context`, `post_context` (full), `raw_action`, `parsed ops`, repairs applied
- exact agent token count pre/post
- which messages were touched (ids, lineage) → lets you reconstruct every message's edit chain (its serial-reproduction history)
- agent's next action and whether it was a repeated tool call
Per rollout: outcome category (correct / incorrect / early give-up <10 / max-iter), judge rationale.

This makes the Bartlett-taxonomy coding a post-hoc pass over stored data rather than a live instrumentation problem.

---

## 9. Gaps and underspecifications in the paper

1. Behavior when managed context exceeds the cap (continue vs. truncate) — unstated.
2. Behavior when the *serialized* `c_t` exceeds the manager's own 32K window — unstated (does P truncate? from where?).
3. Overlapping / non-consecutive ids across ops — only "nonexistent ids" is penalized.
4. Message-ID allocation after merges — unstated; assume fresh ids.
5. Role field mismatch: §3.1 says three roles, prompt offers two.
6. Whether `extract` is a manager action or a fixed preprocessor is blurred: it has its own prompt (Table 12) yet appears in SFT/RL data as an action category.
7. Whether the agent's system prompt is in the manager's view — implied no.
8. Judge model and cost for open-ended rewards during RL (Opus 4.6 as judge for every training rollout is expensive).
9. Agent sampling temperature during rollouts — unstated.
10. The redundant-action penalty credits "the intervening manager action" — for two identical calls at steps t and t+1 that's m_t; confirm no off-by-one.
11. Code link is an anonymized repo; may or may not be live.

---

## 10. Decisions to make before writing code

1. **Thinking on or off for Sonnet 5?** (§5.3) This constrains the action space and is the biggest design fork.
2. **Tool-pair atomicity** — Option A (repair), B (text-serialize), or C (constrained ids)? (§4.2)
3. **Manager model and size.** Paper's own limitation says 4B bottlenecks strong agents. What GPU budget do you have? 4B fits anywhere; 8–14B is the honest choice for a Sonnet-class agent.
4. **Extract**: learned, fixed, or off? (§4.3)
5. **Constrained decoding vs. SFT warm-up.** (§6.1)
6. **Context cap** as fixed constraint (32K, paper-faithful) or as an IV.
7. **Working-note position**: paper's index-1 or end-of-context (cache and recency implications).
8. **Dev agent**: Haiku 4.5 for pipeline development, Sonnet 5 for experiments?
9. **Framework**: Trinity-RFT for fidelity, or verl/SkyRL for your own stack familiarity?
10. **Judge**: which model, and for the Bartlett tasks, what is the outcome reward at all?

---

## 11. Suggested build order

1. `ContextState` + `Apply` + `Repair` + `Validate`, with unit tests against the API's message rules (no model calls yet).
2. Agent adapter for Sonnet 5 with a toy tool; run vanilla ReAct end-to-end; measure baseline variance (§5.5).
3. Manager prompt `P` + serialization; run the *untrained* manager (paper baseline "AdaCoM w/o train") to shake out repairs and logging.
4. Rollout service: async, G rollouts per task, full `ManagerStep` logging, resume, budget cap.
5. Reward module + two-level advantage as pure functions with tests (synthetic rollouts).
6. Trainer integration; overfit on 5 tasks to validate the loop.
7. Then tasks.

---

## 12. Repo reconciliation — github.com/luyi256/AdaCoM (supersedes §7, §10, §11 where they conflict)

The release is a fork of Trinity-RFT with a vendored AgentScope (`as1/`). The context manager lives in `as1/asio/memory/memorymanager.py` (~1.5K lines); workers in `as1/asio/agent/{bcp,mcp}_worker.py`; advantage in `trinity/algorithm/advantage_fn/multi_step_grpo_advantage.py`; kept config `examples/browse_comp_plus/bcp_config_singlenode_8gpu_deepseek.yaml`. Decision: **fork, don't reimplement.** Framework question (§7) is closed.

### What the code settles
| Spec item | Paper | Code |
|---|---|---|
| Message ids | unspecified | **positional integer indices** into the current history, re-indexed every round. No stable ids → lineage for drift analysis must be tracked by you (wrap `perform_modifications`). |
| Tool-pair repair (§4.2) | absent | **already implemented (≈ Option A):** rewriting a `tool_result` keeps it a `tool_result` with replaced `output` (pair stays intact); deleting a `tool_use` marks its result for text-conversion; after applying, a FIFO-by-`call_id` pass deletes orphan `tool_use` msgs and converts orphan `tool_result` msgs to text. Reuse this. |
| Invocation cadence | "before each agent step" | default: every round that contains a `tool_result`; skipped on `finish`. Optional `compress_fre` (fixed) or `compress_fre_min/max` (random) intervals, plus a **forced** call when tokens > `max_model_len − 2·max_response_tokens`. The "tiered deployment" from the limitations section is already a config knob. |
| Agent temperature | unstated | worker default `0` (eval rollouts also `temperature: 0.0`). |
| Two-level advantage (§6.2) | z(R) + α·z(Q), renormalized | kept config leaves `inter_reward_normalization` at default `"none"` → **A = z(R) + α·ΣQ, clipped, no z-scoring of Q and no final renormalization.** The paper's scheme corresponds more closely to the `"outcome_tie"` mode. Paper/code divergence — pick one and record it. |
| Thinking | agents run without | infrastructure exists: `_anthropic_model.py` preserves `thinking` blocks + signature; `agent_enable_thinking` flag (config: `false`); `get_thinking_kwargs` covers DashScope-Claude and Qwen but **check the direct-Anthropic path**. No protection of the latest `a_t` anywhere — that is your addition. |
| Extract op (§4.3) | Table 12 | `extract_from_content` present; threshold/trigger lives in the worker — confirm whether it fires on BCP or only MCP. |
| Extra machinery not in paper | — | manager-output degeneration detection (`detect_repetition_level`, raises/penalizes), `no_change_count`, deferred/terminal out-of-tokens rewards (`memory_reward_utils.py`), scratchpad prompt variants, several prompt variants (`REASONING_PROMPT`, `SELECT_AND_COMPRESS_PROMPT`, …). Identify which prompt is active in the kept config before reading results as "the paper's prompt". |
| Hardware | — | 8×80GB: 4 vLLM engines + 4 trainer GPUs for a 4B manager. |

### Your two concrete modifications

**(a) Thinking-on with immutable latest `a_t`.** Implement as a lock, enforced in two places:
1. *Prompt side:* mark the latest assistant message as `[LOCKED]` in the serialized context (or omit its index from the addressable set) so the manager isn't penalized for edits it cannot know are illegal.
2. *Apply side:* in `perform_modifications`, drop any op whose `ids` include the latest assistant index. Decide whether that's silent (repair) or a format penalty (learning signal). Recommend: silent while SFT/warm-up, penalty during RL.

Scope of the lock, precisely:
- latest `a_t`: immutable (thinking block + signature + tool_use must round-trip verbatim).
- latest `o_t`: **rewrite-only, no delete** — the existing rewrite path keeps it a paired `tool_result`; deletion would orphan `a_t`'s `tool_use` and the repair pass would then delete `a_t` itself, destroying the thinking block.
- older assistant messages: thinking blocks can be stripped or rewritten freely (only the in-flight cycle's block is required). Verify at https://docs.claude.com/en/docs/build-with-claude/extended-thinking, and check whether a manager-inserted `user` text message (working note) counts as ending the tool cycle — it may, since it's a non-`tool_result` user turn.
- strip thinking blocks from the manager's *view* of the context (`format_msgs`) — they're large and not the manager's business. Confirm current behavior.

**(b) Haiku 4.5 dev → Sonnet 5 experiments.** Route through `_anthropic_model.py` directly (not DashScope). Extend `detect_model_provider` / `get_thinking_kwargs` for direct-Anthropic; note `temperature` and `top_p` cannot be sent together to Claude (the worker already special-cases this). Thinking parameterization differs between models — Haiku 4.5 uses `budget_tokens`; check whether Sonnet 5 expects adaptive thinking. Add a per-model cost cap in the worker.

### Revised build order
1. Fork; get the kept BCP config running end-to-end with **manager = untrained Qwen3-4B, agent = Haiku 4.5, thinking off** on ~5 tasks. This validates the whole harness before you touch anything.
2. Add the lock (a); flip thinking on; confirm zero 400s across a few hundred steps.
3. Wrap `perform_modifications` for lineage + pre/post context logging (§8).
4. Pick the advantage mode (`none` vs `outcome_tie`) and document it.
5. Then the `Task` interface for your own tasks.

### Remaining open questions
- Manager size vs. your GPU allocation (4B needs the 8-GPU layout as configured; can you get it on SLURM, and for how long?).
- Which manager prompt variant is live, and does it need Anthropic-specific background text (§5 of Table 8)?
- Extract: on/off for BCP under your fork.
- Advantage mode.

---

## 13. Resolved: live prompt and Unity hardware plan

### 13.1 Which prompt is live
Traced `call_modify` → `self.manager_step_2_prompt` → default param `"manager_step_2_prompt": "MANAGER_1107_STEP_2_PROMPT_UNRESOLVED"` in `as1/asio/memory/config/manager_prompts.py` (line ~2415). Placeholders: `{{full_memory}}` (via `format_msgs(..., strip_content_ids=True)`), `{{token_usage_ratio}}`, `{{Background}}` (filled from the `Background_info` constant in `bcp_worker.py` when `use_bg_info: true`; the kept config sets it).

It is paper Table 9 **plus** three Core-Principles bullets the paper omits: distinguish confirmed evidence from unresolved requirements; avoid final-sounding labels ("Conclusion", "Final answer") unless all constraints are supported; optionally add an "Unresolved constraints / Missing evidence" section. Treat the repo version as canonical. The other ~18 prompts in that file (`MANAGER_1031…`, `_1103_`, `_1104_`, `STEP_1/2`, `RESUM_PROMPT`, scratchpad variants) are dead development history under the kept config.

For your fork: append a `[LOCKED]` convention to this prompt (§12a) and, since Sonnet 5 is the agent, revise `Background_info` to describe *its* tool protocol — the current text describes `search`/`get_document` and document-ID traceability for BCP.

### 13.2 Unity GPU mapping
Kept config assumes 1 node × 8 × 80GB (4 vLLM engines + 4 trainer GPUs, Ulysses SP=4). Unity options (general-access partitions; check current availability with `sinfo`):

| Layout | Where on Unity | Notes |
|---|---|---|
| 8× A100-80GB, one node (paper-faithful) | `gpu-preempt` (133 A100-80, nodes of 1/4/8), `gpu` (32 A100-80, nodes of 4/8) | 8-GPU nodes are scarce on `gpu`; on `gpu-preempt` you get ≤2h and can be killed → rely on `continue_from_checkpoint: true`, `save_interval: 1`, and a resubmit loop. |
| 4× A100-80GB or 4× H100, one node | `gpu` (A100 4-node, H100 4-node) | Set `engine_num: 2`, `ulysses_sequence_parallel_size: 2`. A 4B manager: bf16 weights ~8GB; full-param + Adam states ~64GB total sharded over 2 trainer GPUs = ~32GB each + activations at `max_token_len_per_gpu: 16384` — fits in 80GB. vLLM engine for 4B at 32K ctx is comfortable on 80GB. |
| 4× L40S (48GB) | `gpu` (36), `gpu-preempt` (68), all 4-per-node | Most available large-VRAM option. Trainer at 2 GPUs is tight (~32GB states + activations); drop `max_token_len_per_gpu` to 8192 or enable optimizer offload. vLLM engine fine. Reasonable dev target. |
| H200 (143GB, 8/node) | `gpu-preempt` (8) | Overkill and preemptible; skip. |
| GH200 `arm-gpu` | 4 nodes | aarch64 — vLLM/Ray/Trinity wheel risk. Avoid. |

Partition rules that matter: `gpu` is non-preemptible up to 48h (`--qos=long` beyond); `gpu-preempt` is ≤2h effective and requires checkpointing. A full RL run is tens of hours → `gpu` with `--qos=long`, or `gpu-preempt` with aggressive checkpointing. Disk: the BCP dense index needs ~200GB → use an HPC Workspace scratch allocation, not home/project quota. If you skip BCP (you're not doing tasks yet), that requirement disappears and so does the retriever GPU.

Practical split: develop the harness (Haiku 4.5 agent, untrained manager, thinking on, lock in place) on **1× L40S** interactive/`gpu-preempt` — you only need one vLLM engine and no trainer for that. Move to 4× A100-80 / 4× L40S on `gpu` for RL. Request 8× A100-80 only if the 4-GPU throughput turns out to be the bottleneck — with an API-bound agent it usually isn't; the API is.

Example allocation for the dev phase:
```bash
#SBATCH -p gpu-preempt
#SBATCH -t 02:00:00
#SBATCH --gpus=1
#SBATCH --constraint=l40s
#SBATCH --mem=64G
#SBATCH -c 8
```
and for RL:
```bash
#SBATCH -p gpu
#SBATCH -t 48:00:00          # add --qos=long if longer
#SBATCH --gpus=4
#SBATCH --constraint=a100-80g   # or l40s
#SBATCH --mem=250G
#SBATCH -c 32
```
