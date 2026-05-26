"""Provider-specific response post-processing.

Centralizes every transformation that depends on *how* a given model returns
its response (format quirks, protocol drift, id conventions). Workers should
never reach into ``response.content`` to fix these things — ``retry_model_call``
applies ``normalize_response_content`` on every returned ChatResponse so the
downstream pipeline always sees OpenAI-canonical shape.

Handled quirks:
- ``kimi-k2-thinking`` via DashScope reuses response-local tool_call ids
  (``functions.search:1``) across separate calls → remap to UUIDs.
- ``minimax.MiniMax-M2.7-highspeed`` emits
  ``<minimax:tool_call><invoke name="X"><parameter name="Y">Z</parameter>
  </invoke></minimax:tool_call>`` inside the thinking channel rather than
  using the OpenAI tool_calls protocol → extract as a tool_use block.
- ``gpt-oss`` family emits ``to=functions.<name> json {...}`` in the text
  channel without the tool_calls protocol → extract as a tool_use block.
- ``gpt-oss`` leaks channel markers (``analysis`` prefix,
  ``assistantanalysis``, ``assistantfinal``) into text → strip.
- Reasoning-channel models emit ``{"type": "thinking"|"reasoning"}`` blocks
  → preserve as plain text so the agent can see its own prior reasoning on
  later turns without risking it being parsed as a final answer.
"""
import json
import re
import uuid
from typing import Any, Dict, Optional


_XML_INVOKE_RE = re.compile(
    r'<invoke\s+name=["\']([^"\']+)["\']\s*>(.*?)</invoke>', re.S,
)
_XML_PARAM_RE = re.compile(
    r'<parameter\s+name=["\']([^"\']+)["\']\s*>(.*?)</parameter>', re.S,
)
_GPT_OSS_TOOL_RE = re.compile(
    r"(.*?)(?:assistantanalysis)?\s*to=functions\.([a-zA-Z_][a-zA-Z_0-9]*)\s+\w+(\{.*\})",
    re.DOTALL,
)
_ANALYSIS_PREFIX_RE = re.compile(r"^analysis\s*", re.IGNORECASE)
_OSS_MARKER_RE = re.compile(r"assistantanalysis|assistantfinal", re.IGNORECASE)


def extract_xml_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse a MiniMax-style ``<invoke name="X"><parameter .../></invoke>``
    into ``{"name": str, "input": dict}``. Returns None if no invoke is found."""
    if not text or "<invoke" not in text:
        return None
    m = _XML_INVOKE_RE.search(text)
    if not m:
        return None
    name = m.group(1)
    body = m.group(2)
    params: Dict[str, Any] = {}
    for pm in _XML_PARAM_RE.finditer(body):
        params[pm.group(1)] = pm.group(2).strip()
    return {"name": name, "input": params}


def normalize_tool_use_ids(content: Any) -> None:
    """In-place remap ``tool_use`` block ids to fresh UUIDs.

    OpenAI-compatible APIs only require ``tool_call_id`` to match a
    ``tool_calls[].id`` within the same request — no specific format. UUIDs
    eliminate cross-turn collisions from providers using response-local ids.
    """
    if not isinstance(content, list):
        return
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            block["id"] = str(uuid.uuid4())


def normalize_response_content(content: Any, logger: Any = None) -> Any:
    """Apply every provider-quirk fix to a response's content.

    Returns the normalized content. Non-list content is returned unchanged.
    Callers should reassign ``response.content`` to the result.
    """
    if not isinstance(content, list):
        return content

    # Pass 1: if the response has no native tool_use, try XML recovery from
    # any thinking/text block (minimax). Stops at the first successful match.
    has_tool_use = any(
        isinstance(b, dict) and b.get("type") == "tool_use" for b in content
    )
    if not has_tool_use:
        for block in content:
            if not isinstance(block, dict):
                continue
            src = block.get("thinking") or block.get("text") or ""
            parsed = extract_xml_tool_call(src)
            if parsed:
                content.append({
                    "type": "tool_use",
                    "name": parsed["name"],
                    "id": str(uuid.uuid4()),
                    "input": parsed["input"],
                })
                if logger:
                    logger.log_info(
                        f"Recovered XML tool call from {block.get('type')} "
                        f"block: {parsed['name']}"
                    )
                break

    # Pass 2: rewrite blocks — thinking→text, gpt-oss text-tool-call recovery,
    # channel-marker stripping.
    new_content = []
    for block in content:
        if not isinstance(block, dict):
            new_content.append(block)
            continue
        btype = block.get("type")

        if btype in ("thinking", "reasoning"):
            raw = (
                block.get("thinking")
                or block.get("reasoning")
                or block.get("text")
                or ""
            )
            if raw:
                new_content.append({"type": "text", "text": raw})
            continue

        if btype == "text":
            text = block.get("text", "")
            text = _ANALYSIS_PREFIX_RE.sub("", text)

            match = _GPT_OSS_TOOL_RE.search(text)
            if match:
                thinking_content = match.group(1).strip()
                func_name = match.group(2)
                json_args = match.group(3)
                try:
                    func_args = json.loads(json_args)
                    if thinking_content:
                        new_content.append({"type": "text", "text": thinking_content})
                    new_content.append({
                        "type": "tool_use",
                        "name": func_name,
                        "id": str(uuid.uuid4()),
                        "input": func_args,
                    })
                    if logger:
                        logger.log_info(f"Fixed malformed tool call: {func_name}")
                    continue
                except json.JSONDecodeError:
                    pass  # fall through to plain-text cleanup

            clean_text = _OSS_MARKER_RE.sub("", text).strip()
            if clean_text:
                new_content.append({"type": "text", "text": clean_text})
            continue

        new_content.append(block)

    # Pass 3: ensure every tool_use id is a UUID.
    normalize_tool_use_ids(new_content)
    return new_content
