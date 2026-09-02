# Anthropic API facts this fork depends on

Checked 2026-09-02 against `platform.claude.com/docs/en/build-with-claude/`
(`docs.claude.com` 302s there): `extended-thinking`, `thinking`, `effort`.
Re-check before a run on a new model — CLAUDE.md invariant 4.

## Thinking parameter shape (the Task 2 decision)

| Agent | Mode | Request |
|---|---|---|
| `claude-haiku-4-5` (dev) | manual extended thinking — the **only** mode on 4.5 and earlier | `thinking={"type":"enabled","budget_tokens":N}` |
| `claude-sonnet-5` (experiments) | adaptive — `"enabled"` returns **400** on 4.7+ | `thinking={"type":"adaptive"}`, depth via `output_config={"effort":...}` |
| 4.6 generation | both; `budget_tokens` deprecated | prefer adaptive |

- `budget_tokens` ≥ 1024 and < `max_tokens` (exception: interleaved thinking).
- Effort levels: `low`, `medium`, `high` (= default = omitting it), `xhigh`, `max`.
  Sonnet 5 supports all five and defaults to `high`.
- Haiku 4.5 has **no interleaved thinking**; the beta header is accepted and ignored.
- Manual mode requires the final assistant turn to *begin* with a thinking block;
  adaptive drops that requirement.

Implemented in `as1/asio/utils/retry.py`: `classify_anthropic_thinking_mode`
picks the mode from the model id (≥4.6 adaptive, 3.7–4.5 budget, older
unsupported, unknown → adaptive because that is the forward-compatible guess),
and the run config supplies `mode` / `budget_tokens` / `effort`.

## Sampling parameters

- Claude 4.7 and later (`claude-sonnet-5`, `claude-opus-4-7/4-8/5`, Fable/Mythos 5.x):
  **any** non-default `temperature`, `top_p` or `top_k` is a 400, thinking or not.
- Older models: with thinking on, `temperature` and `top_k` are incompatible;
  `top_p` allowed only in [0.95, 1].

Consequence for this fork: a thinking-enabled Claude agent sends **no** sampling
params, so the worker's `temperature: 0` no longer applies (`get_anthropic_sampling_kwargs`
returns `{}`). Spec §5.5's variance argument gets stronger — measure the ReAct
baseline noise floor before reading any RL curve.

## Thinking-block round-trip (why the lock exists)

- Every `thinking` block carries a `signature`; pass thinking blocks back
  **complete and unmodified** with tool results.
- Toggling thinking mid-turn does not error: the API silently disables thinking
  for that request and may strip blocks that would make the turn structure
  invalid. So a manager edit can degrade the agent without any 400 to detect it
  — check for `thinking` blocks in the response, don't rely on errors.
- Preservation of *prior*-turn blocks is per model: keep-all on Opus 4.5+,
  Sonnet 4.6+, Fable/Mythos 5.x; **last turn only** on all Haiku through 4.5 and
  earlier Sonnet/Opus, where the API strips older blocks itself. Either way the
  in-flight cycle's block must round-trip — that is what `context_lock.py` protects.

### One caveat that matters for a context-rewriting project

On **Claude Fable 5.1** a thinking block's signature also binds the conversation
*prefix* (system prompt, tools, and every preceding message): change any of them
and the API rejects the request or drops the block. Rewriting history is exactly
what the manager does, so that model is off the table for this design without
the context-editing escape hatches. **Claude Sonnet 5 and Claude Mythos 5.1 check
only the model condition**, not the prefix — our agents are safe. Enforcement of
the prefix condition applies to accounts created on or after 2026-08-31 and is
planned for future models: re-check this before switching agent models.

## Cost / rate limits

Not hardcoded anywhere. Pricing lives in a config file (see Task 6,
`examples/dev/anthropic_pricing.yaml`); rate limits at
`platform.claude.com/docs/en/api/rate-limits`.

## Two upstream bugs the thinking path hit

Both found by the mocked loop test (`as1/tests/test_smoke_loop.py`) and fixed in
`memorymanager.py`. Neither could show up on the paper's setup, which ran
OpenAI-compatible relays without thinking.

1. **User-role tool results never closed a round.** `MemoryManager.add`
   returned early for any `role == "user"` message. On the direct Anthropic API
   tool results *must* be user-role (there is no system role in `messages`), so
   the manager was never invoked at all. Now a user-role message returns early
   only when it carries no `tool_result`.

2. **`compress()` stripped thinking blocks.** Its regrouping pass rebuilt each
   assistant message as `[text, tool_use]`, silently dropping `thinking`. The
   in-flight block was destroyed every round, before the manager or the lock
   ever saw the context — a guaranteed 400 on the next agent call. Thinking
   blocks are now carried through and lead the message holding the first
   `tool_use`.

Related quirk, left alone: that same pass drops an assistant message that has
neither `tool_use` nor `tool_result` blocks. Unreachable in the current loop
(compression only runs on a round closed by a tool result), but worth knowing
if the action space grows.
