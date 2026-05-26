"""
Elasticsearch fallback for Wikipedia MCP tools.

When the ToolResultCache misses, this module queries a local Elasticsearch
index built from a Wikipedia dump.  It also logs cache misses to a JSONL
file so a background script can backfill the cache via the real MCP server.

Supports all 9 Wikipedia MCP tools:
  search_wikipedia, get_article, get_summary, get_sections, get_links,
  get_related_topics, summarize_article_for_query, summarize_article_section,
  extract_key_facts
"""

import json
import os
import re
import logging
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from elasticsearch import Elasticsearch
except ImportError:
    Elasticsearch = None  # type: ignore


# ---------------------------------------------------------------------------
# Text parsing helpers (operate on plain-text Wikipedia dump articles)
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)


def _extract_summary(text: str) -> str:
    """Return text before the first section heading (== ... ==)."""
    m = _SECTION_RE.search(text)
    if m:
        return text[: m.start()].strip()
    # No section headers – return first 1500 chars
    return text[:1500].strip()


def _parse_sections(text: str) -> List[Dict[str, Any]]:
    """Parse ``== Section ==`` markers into a nested list."""
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return []

    sections: List[Dict[str, Any]] = []
    for i, m in enumerate(matches):
        level = len(m.group(1)) - 2  # == → 0, === → 1, …
        title = m.group(2)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append({"title": title, "level": level, "text": body, "sections": []})

    # Build nesting
    root: List[Dict[str, Any]] = []
    stack: List[Dict[str, Any]] = []
    for sec in sections:
        while stack and stack[-1]["level"] >= sec["level"]:
            stack.pop()
        if stack:
            stack[-1]["sections"].append(sec)
        else:
            root.append(sec)
        stack.append(sec)
    return root


def _find_section_text(sections: List[Dict[str, Any]], target: str) -> Optional[str]:
    """Recursively find a section by title (case-insensitive) and return its text."""
    target_lower = target.lower()
    for sec in sections:
        if sec["title"].lower() == target_lower:
            return sec["text"]
        found = _find_section_text(sec["sections"], target)
        if found is not None:
            return found
    return None


def _snippet_around_query(text: str, query: str, max_length: int = 250) -> str:
    """Return a snippet of *text* centred on the first occurrence of *query*."""
    idx = text.lower().find(query.lower())
    if idx == -1:
        return text[:max_length] + ("..." if len(text) > max_length else "")
    half = max_length // 2
    start = max(0, idx - half)
    end = min(len(text), idx + len(query) + half)
    snippet = text[start:end]
    if len(snippet) >= max_length:
        snippet = snippet[:max_length]
    return snippet + ("..." if end < len(text) else "")


def _split_sentences(text: str, count: int) -> List[str]:
    """Split text into sentences and return the first *count*."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    result = []
    for s in sentences[:count]:
        if not s.endswith((".", "!", "?")):
            s += "."
        result.append(s)
    return result or ["No content found."]


# ---------------------------------------------------------------------------
# Miss-log writer (thread-safe, append-only JSONL)
# ---------------------------------------------------------------------------

class _MissLogger:
    """Append cache-miss records to a JSONL file."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def log(self, server: str, tool: str, args: dict) -> None:
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "server": server,
            "tool": tool,
            "args": args,
        }
        with self._lock:
            with open(self._path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class WikipediaESFallback:
    """Query a local Elasticsearch Wikipedia index as fallback for MCP tools."""

    def __init__(
        self,
        es_url: Optional[str] = None,
        es_index: Optional[str] = None,
        miss_log_path: Optional[str] = None,
    ) -> None:
        if Elasticsearch is None:
            raise ImportError(
                "elasticsearch package required for ES fallback. "
                "pip install elasticsearch"
            )
        self._es_url = es_url or os.environ.get("WIKIPEDIA_ES_URL", "http://localhost:9200")
        self._es_index = es_index or os.environ.get("WIKIPEDIA_ES_INDEX", "wikipedia_en")
        self._es = Elasticsearch(self._es_url)
        self._miss_logger = _MissLogger(miss_log_path) if miss_log_path else None

        # Dispatch table: MCP tool name → handler method
        self._handlers = {
            "search_wikipedia": self._search_wikipedia,
            "get_article": self._get_article,
            "get_summary": self._get_summary,
            "get_sections": self._get_sections,
            "get_links": self._get_links,
            "get_related_topics": self._get_related_topics,
            "summarize_article_for_query": self._summarize_for_query,
            "summarize_article_section": self._summarize_section,
            "extract_key_facts": self._extract_key_facts,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, tool_name: str, args: dict) -> Optional[str]:
        """Attempt to answer a Wikipedia MCP tool call from ES.

        Returns a JSON string matching the MCP tool output format,
        or ``None`` if ES cannot satisfy the request.
        """
        handler = self._handlers.get(tool_name)
        if handler is None:
            return None
        try:
            result = handler(args)
        except Exception as e:
            logger.warning("ES fallback error for %s(%s): %s", tool_name, args, e)
            result = None

        if result is None and self._miss_logger:
            self._miss_logger.log("Wikipedia", tool_name, args)
        return result

    # ------------------------------------------------------------------
    # ES helpers
    # ------------------------------------------------------------------

    def _get_doc_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Fetch a single document by exact title match, fall back to fuzzy."""
        # Try exact match first
        resp = self._es.search(
            index=self._es_index,
            body={
                "query": {"term": {"title.keyword": title}},
                "size": 1,
            },
        )
        hits = resp["hits"]["hits"]
        if hits:
            return hits[0]["_source"]

        # Fuzzy match
        resp = self._es.search(
            index=self._es_index,
            body={
                "query": {"match": {"title": {"query": title, "fuzziness": "AUTO"}}},
                "size": 1,
            },
        )
        hits = resp["hits"]["hits"]
        if hits and hits[0]["_score"] > 5:
            return hits[0]["_source"]
        return None

    # ------------------------------------------------------------------
    # Tool handlers — each returns a JSON string or None
    # ------------------------------------------------------------------

    def _search_wikipedia(self, args: dict) -> Optional[str]:
        query = args.get("query", "")
        limit = args.get("limit", 10)
        if not query:
            return None
        resp = self._es.search(
            index=self._es_index,
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^3", "text"],
                        "type": "best_fields",
                    }
                },
                "size": limit,
                "_source": ["title", "url", "word_count"],
            },
        )
        results = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            results.append({
                "title": src.get("title", ""),
                "snippet": "",  # ES highlight could be added but not required
                "pageid": int(hit["_id"]) if hit["_id"].isdigit() else 0,
                "wordcount": src.get("word_count", 0),
                "timestamp": "",
            })
        return json.dumps({"query": query, "results": results}, ensure_ascii=False)

    def _get_article(self, args: dict) -> Optional[str]:
        title = args.get("title", "")
        doc = self._get_doc_by_title(title)
        if doc is None:
            return None
        text = doc.get("text", "")
        summary = doc.get("summary", "") or _extract_summary(text)
        sections = _parse_sections(text)
        return json.dumps({
            "title": doc.get("title", title),
            "pageid": doc.get("id", 0),
            "summary": summary,
            "text": text,
            "url": doc.get("url", ""),
            "sections": sections,
            "categories": [],
            "links": [],
            "exists": True,
        }, ensure_ascii=False)

    def _get_summary(self, args: dict) -> Optional[str]:
        title = args.get("title", "")
        doc = self._get_doc_by_title(title)
        if doc is None:
            return None
        text = doc.get("text", "")
        summary = doc.get("summary", "") or _extract_summary(text)
        return json.dumps({"title": doc.get("title", title), "summary": summary}, ensure_ascii=False)

    def _get_sections(self, args: dict) -> Optional[str]:
        title = args.get("title", "")
        doc = self._get_doc_by_title(title)
        if doc is None:
            return None
        sections = _parse_sections(doc.get("text", ""))
        return json.dumps({"title": doc.get("title", title), "sections": sections}, ensure_ascii=False)

    def _get_links(self, args: dict) -> Optional[str]:
        title = args.get("title", "")
        doc = self._get_doc_by_title(title)
        if doc is None:
            return None
        # Plain-text dump has no [[links]], return empty list
        # (backfill script will populate the real cache entry later)
        return json.dumps({"title": doc.get("title", title), "links": []}, ensure_ascii=False)

    def _get_related_topics(self, args: dict) -> Optional[str]:
        title = args.get("title", "")
        limit = args.get("limit", 10)
        doc = self._get_doc_by_title(title)
        if doc is None:
            return None
        # Use more_like_this to find related articles
        try:
            resp = self._es.search(
                index=self._es_index,
                body={
                    "query": {
                        "more_like_this": {
                            "fields": ["title", "text"],
                            "like": doc.get("text", "")[:2000],
                            "min_term_freq": 1,
                            "min_doc_freq": 2,
                            "max_query_terms": 25,
                        }
                    },
                    "size": limit,
                    "_source": ["title", "url", "summary"],
                },
            )
            related = []
            for hit in resp["hits"]["hits"]:
                src = hit["_source"]
                s = src.get("summary", "")
                related.append({
                    "title": src.get("title", ""),
                    "summary": (s[:200] + "...") if len(s) > 200 else s,
                    "url": src.get("url", ""),
                    "type": "link",
                })
            return json.dumps({"title": doc.get("title", title), "related_topics": related}, ensure_ascii=False)
        except Exception:
            return json.dumps({"title": doc.get("title", title), "related_topics": []}, ensure_ascii=False)

    def _summarize_for_query(self, args: dict) -> Optional[str]:
        title = args.get("title", "")
        query = args.get("query", "")
        max_length = args.get("max_length", 250)
        doc = self._get_doc_by_title(title)
        if doc is None:
            return None
        text = doc.get("text", "")
        snippet = _snippet_around_query(text, query, max_length)
        return json.dumps({"title": doc.get("title", title), "query": query, "summary": snippet}, ensure_ascii=False)

    def _summarize_section(self, args: dict) -> Optional[str]:
        title = args.get("title", "")
        section_title = args.get("section_title", "")
        max_length = args.get("max_length", 150)
        doc = self._get_doc_by_title(title)
        if doc is None:
            return None
        text = doc.get("text", "")
        sections = _parse_sections(text)
        sec_text = _find_section_text(sections, section_title)
        if sec_text is None:
            summary = f"Section '{section_title}' not found in article '{title}'."
        else:
            summary = sec_text[:max_length]
            if len(sec_text) > max_length:
                summary += "..."
        return json.dumps({
            "title": doc.get("title", title),
            "section_title": section_title,
            "summary": summary,
        }, ensure_ascii=False)

    def _extract_key_facts(self, args: dict) -> Optional[str]:
        title = args.get("title", "")
        topic = args.get("topic_within_article", "")
        count = args.get("count", 5)
        doc = self._get_doc_by_title(title)
        if doc is None:
            return None
        text = doc.get("text", "")
        if topic:
            sections = _parse_sections(text)
            sec_text = _find_section_text(sections, topic)
            source = sec_text if sec_text else (doc.get("summary", "") or _extract_summary(text))
        else:
            source = doc.get("summary", "") or _extract_summary(text)
        facts = _split_sentences(source, count)
        return json.dumps({
            "title": doc.get("title", title),
            "topic_within_article": topic,
            "facts": facts,
        }, ensure_ascii=False)
