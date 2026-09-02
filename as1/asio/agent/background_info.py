# -*- coding: utf-8 -*-
"""Background-info variants for the manager prompt's ``{{Background}}`` slot.

The live manager prompt (``MANAGER_1107_STEP_2_PROMPT_UNRESOLVED``) fills
``{{Background}}`` from ``memory_config["background_info"]`` when the run sets
``use_bg_info``. The workers used to hardcode one benchmark-specific text each;
a run now picks one with ``memory_config["background_info_key"]``, or supplies
its own with ``memory_config["background_info"]``.

Nothing here touches the manager prompt itself — only what gets substituted
into it.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# The BrowseComp-Plus text the BCP worker has always used (verbatim).
BCP_BACKGROUND_INFO = """
## Task Background Information

Agent Mission: The agent is tasked with answering complex questions by iteratively searching a database for essential clues and evidence.

Operational Workflow & Tools:

search: Takes a query and returns a list of document snippets.

get_document: Takes a specific document ID and retrieves the full-text content.

Core Requirement (Data Traceability): Each document is identified by a unique ID. To ensure the agent can provide verifiable citations and maintain information provenance, it is critical to preserve the document IDs associated with important information.
"""

# The MCP-Bench text the MCP worker has always used (verbatim).
MCP_BACKGROUND_INFO = """
## Task Background

The agent is responsible for executing complex tasks by leveraging tools across multiple MCP servers.

**Information Retention Strategy:**  
Retain information at a level of detail appropriate to the task at hand.  
- If the task requires generating structured output (e.g., reports, summaries), preserve comprehensive details.  
- Otherwise, retain only key findings and critical clues necessary to complete the task.
"""

# Task-agnostic variant for a frozen Claude agent: describes the Messages API
# message protocol and the context lock, neither of which is benchmark-specific.
ANTHROPIC_TOOL_USE_BACKGROUND_INFO = """
## Task Background Information

Agent Mission: A frozen Claude agent works through a long-horizon task by alternating private reasoning with tool calls, until it has enough evidence to answer and calls the 'finish' tool.

Message Protocol:

An assistant message may contain a `thinking` block (the agent's reasoning, carrying a cryptographic signature), `text` blocks, and one or more `tool_use` blocks.

Every `tool_use` is answered by a `tool_result` in the user message that immediately follows it, matched by id. The two form a pair: a `tool_use` without its `tool_result`, or a `tool_result` without its `tool_use`, is an invalid context.

Locked Messages:

The agent's most recent assistant message holds the reasoning and tool call currently in flight. It is marked `"lock": "locked"` and must not be modified in any way. Its paired `tool_result` is marked `"lock": "rewrite_only"`: its content may be rewritten or compressed, but it must not be deleted or merged with other messages. Older messages carry no lock and may be rewritten or removed freely.

Core Requirement (Evidence Traceability): Preserve the original task, confirmed evidence together with the identifiers needed to cite it (document ids, urls, names), and the requirements that are still unresolved. Tool results from earlier rounds are usually the largest and most compressible part of the context.
"""

BACKGROUND_INFO_VARIANTS: Dict[str, str] = {
    "bcp": BCP_BACKGROUND_INFO,
    "mcp": MCP_BACKGROUND_INFO,
    "anthropic_tool_use": ANTHROPIC_TOOL_USE_BACKGROUND_INFO,
}


def resolve_background_info(
    memory_config: Optional[Dict[str, Any]],
    default_key: str,
) -> str:
    """Pick the background text for this run.

    Precedence: an explicit ``background_info`` string wins, then
    ``background_info_key``, then the worker's ``default_key``. An unknown key
    raises rather than silently describing the wrong agent to the manager.
    """
    cfg = memory_config or {}
    explicit = cfg.get("background_info")
    if isinstance(explicit, str) and explicit.strip():
        return explicit

    key = cfg.get("background_info_key") or default_key
    try:
        return BACKGROUND_INFO_VARIANTS[key]
    except KeyError:
        raise ValueError(
            f"Unknown background_info_key {key!r}; "
            f"expected one of {sorted(BACKGROUND_INFO_VARIANTS)}"
        ) from None
