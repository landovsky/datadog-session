#!/usr/bin/env python3
"""
SwagBot Trace Replay Script

Reads pre-recorded scenarios from replay_scenarios.json and injects synthetic
LLMObs traces into Datadog without making real LLM API calls.

Each scenario may contain a full 'spans' list captured from a real trace
(via capture_traces.py). When present, spans are replayed verbatim with their
original content, preserving the exact span hierarchy through recursive
context-manager nesting. When 'spans' is absent or empty, a minimal synthetic
hierarchy is created as a fallback.

Span hierarchy (mirrors swagbot_langgraph_workflow_full.py):
  workflow: swagbot_workflow
    agent:   SwagBot
      agent: Planning Agent
        task: llm  (Intent Analysis)
      agent: Orchestrator Agent
        task: orchestrator  (Agent Dispatch)
      agent: <Specialist(s)>
        retrieval: rag
        task: llm  (Response Generation)
      agent: Response Synthesizer
        task: llm  (Response Synthesis)

Usage:
    ddtrace-run python /app/scripts/replay.py
"""

import json
import logging
import math
import os
import random
import sys
import time
from typing import Dict, List

from ddtrace.llmobs import LLMObs

# ---------------------------------------------------------------------------
# RAG helpers — lazy-imported so replay works even if the app is not installed.
# We add /app to sys.path so swagbot_utils and swagbot_langgraph_workflow_full
# are importable from within the container.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from swagbot_langgraph_workflow_full import _rag_search_raw as _live_rag_search
    _RAG_AVAILABLE = True
except Exception:
    _live_rag_search = None   # type: ignore[assignment]
    _RAG_AVAILABLE = False

# Map span name substrings → agent_types list for _live_rag_search
_SPAN_NAME_TO_AGENT_TYPES = {
    "product specialist": ["product_specialist"],
    "promotion specialist": ["promotion_specialist"],
    "customer service": ["customer_service"],
    "feedback": ["feedback_handler"],
}


def _agent_types_for_retrieval_span(span_name: str) -> List[str]:
    """Infer agent_types from the retrieval span name (e.g. 'Product Specialist Agent Retrieval')."""
    name_lower = span_name.lower()
    for keyword, types in _SPAN_NAME_TO_AGENT_TYPES.items():
        if keyword in name_lower:
            return types
    return []
try:
    from ddtrace.llmobs.utils import Prompt   # ddtrace < 4.x
except ImportError:
    from ddtrace.llmobs import Prompt          # ddtrace >= 4.x
try:
    from ddtrace.llmobs._constants import AGENT_MANIFEST as _AGENT_MANIFEST_KEY
except ImportError:
    _AGENT_MANIFEST_KEY = "_ml_obs.meta.agent_manifest"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCENARIOS_PATH          = os.path.join(os.path.dirname(__file__), "..", "resources", "replay_scenarios.json")
SLEEP_BETWEEN_SCENARIOS = 2.0    # seconds between individual replays
SLEEP_BETWEEN_CYCLES    = 0.0    # no pause between cycles — replay runs continuously

PLANNING_MODEL    = os.getenv("PLANNING_MODEL",    "gemini-3.5-flash-lite")
SPECIALIST_MODEL  = os.getenv("SPECIALIST_MODEL",  "gemini-3.5-flash")
SYNTHESIZER_MODEL = os.getenv("SYNTHESIZER_MODEL", "gemini-3.5-flash")

# Prompt templates (+ RAG variables) are always injected so Datadog evaluators can run.
# REPLAY_EVALUATIONS=1 additionally submits the pre-captured evaluation scores via
# LLMObs.submit_evaluation(), making results appear immediately without waiting for
# Datadog to evaluate. Leave at 0 (default) when students configure their own
# evaluations — Datadog will then run them live on the replayed spans.
REPLAY_EVALUATIONS = os.getenv("REPLAY_EVALUATIONS", "0") == "1"
_RESOURCES_DIR     = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "resources"))
_PROMPT_TEMPLATES: dict = {}   # populated in main()


def _check_skip_condition(condition: str) -> bool:
    """Return True if the named condition is currently satisfied.

    Conditions are evaluated against the live state of resource files so the
    replay self-regulates without requiring an env var or container restart.

    Currently supported:
      "pricing_fixed" — True once products.json contains a price for the
                        Dog Steel Bottle (i.e. the Lab 3 pricing fix was applied).
    """
    if condition == "pricing_fixed":
        try:
            with open(os.path.join(_RESOURCES_DIR, "products.json")) as fh:
                products = json.load(fh).get("products", [])
            return any(
                "priceUsd" in p
                for p in products
                if "steel bottle" in p.get("name", "").lower()
            )
        except Exception:
            return False
    return False

# Fallback category → agent name mapping (used when spans list is empty)
CATEGORY_TO_AGENTS: Dict[str, List[str]] = {
    "product_specialist":  ["Product Specialist"],
    "promotion_specialist": ["Promotion Specialist"],
    "customer_service":    ["Customer Service"],
    "feedback_handler":    ["Feedback Handler"],
    "cross_domain":        ["Customer Service", "Product Specialist"],
}

def _retrieval_output_data(span: dict, input_data: object) -> "list | None":
    """
    Resolve output_data for a retrieval span.

    The Datadog search API only returns output.value (a comma-joined list of document
    names), not the individual documents.  We do a fresh RAG search using the captured
    query so individual documents appear correctly in the replay UI.
    Fallback chain: live RAG → single-doc wrapper → None (no output annotation).
    """
    raw = _parse_io(span.get("output", ""))

    if isinstance(raw, list):
        # Already a proper documents list (future-proof)
        return [d if isinstance(d, dict) else {"text": str(d)} for d in raw if d] or None

    # Non-list: try a live RAG search with the captured query
    query = input_data if isinstance(input_data, str) else ""
    agent_types = _agent_types_for_retrieval_span(span.get("name", ""))
    if _RAG_AVAILABLE and query and agent_types:
        try:
            docs = _live_rag_search(agent_types, query)  # type: ignore[misc]
            if docs:
                return [
                    {"id": str(i), "text": d.get("content", ""),
                     "score": d.get("score", 0), "name": d.get("name", "Document")}
                    for i, d in enumerate(docs)
                ]
        except Exception as exc:
            logger.debug("RAG re-fetch failed for retrieval span: %s", exc)

    # Fallback: extract whatever text is available as a single document
    if isinstance(raw, dict):
        text = raw.get("rag_context", "") or str(raw)
    else:
        text = str(raw) if raw else ""
    return [{"text": text}] if text else None


# LLMObs context manager lookup by span_kind
_LLMOBS_CM = {
    "workflow":  LLMObs.workflow,
    "agent":     LLMObs.agent,
    "task":      LLMObs.task,
    "retrieval": LLMObs.retrieval,
    "llm":       LLMObs.llm,
    "tool":      LLMObs.tool,
    "embedding": LLMObs.embedding,
}


# ---------------------------------------------------------------------------
# Prompt template injection helpers
# ---------------------------------------------------------------------------

def _load_prompt_templates() -> dict:
    """
    Load all prompt templates from resources/prompt-*.txt + prompt-metadata.json.
    Returns a dict keyed by agent_type (e.g. "planning", "synthesizer", ...).
    """
    try:
        with open(os.path.join(_RESOURCES_DIR, "prompt-metadata.json")) as f:
            metadata = json.load(f)
    except Exception as exc:
        logger.warning("Could not load prompt-metadata.json: %s", exc)
        return {}

    templates: dict = {}
    for agent_type, meta in metadata.items():
        # Try both underscore and hyphen variants (metadata keys use underscores,
        # prompt files are named with hyphens, e.g. prompt-product-specialist.txt)
        hyphen_type = agent_type.replace("_", "-")
        for candidate in (agent_type, hyphen_type):
            txt_path = os.path.join(_RESOURCES_DIR, f"prompt-{candidate}.txt")
            try:
                with open(txt_path) as f:
                    template_text = f.read()
                break
            except FileNotFoundError:
                template_text = None
        if template_text is None:
            logger.warning("Prompt file not found for %s — skipping", agent_type)
            template_text = ""
        templates[agent_type] = {
            "id":       meta["id"],
            "version":  meta["version"],
            "template": template_text,
        }
    logger.info("Loaded %d prompt templates: %s", len(templates), list(templates))
    return templates


def _resolve_agent_type(span: dict, span_by_id: dict) -> "str | None":
    """
    Walk the parent chain of an LLM span to determine its prompt-metadata key.

    Captured span hierarchy (parent task name → agent type):
      LLM span
        Intent Analysis (task)       → "planning"
        Response Generation (task)   → specialist type from great-grandparent agent
          llm (task)
            <Specialist Agent> (agent)  ← name used to derive type
        Response Synthesis (task)    → "synthesizer"
    """
    parent = span_by_id.get(span.get("parent_id"))
    if parent is None:
        return None
    parent_name = parent.get("name", "")

    if parent_name == "Intent Analysis":
        return "planning"
    if parent_name == "Response Synthesis":
        return "synthesizer"
    if parent_name == "Response Generation":
        grandparent = span_by_id.get(parent.get("parent_id"))
        if grandparent is None:
            return None
        gg_parent = span_by_id.get(grandparent.get("parent_id"))
        if gg_parent is None:
            return None
        return _specialist_agent_type(gg_parent.get("name", ""))
    return None


def _specialist_agent_type(agent_name: str) -> "str | None":
    """Map a specialist agent span name to its prompt-metadata.json key."""
    name = agent_name.lower()
    if "product" in name:
        return "product-specialist"
    if "customer" in name:
        return "customer-service"
    if "promotion" in name:
        return "promotion-specialist"
    if "feedback" in name:
        return "feedback-handler"
    return None


def _extract_specialist_text(out: object) -> str:
    """
    Extract readable text from a specialist agent span output.

    Handles three output shapes produced by LangGraph specialists:
    - dict with "response"/"output" key (non-tool-call path)
    - dict with "messages" list (tool-call path — find last non-routing message)
    - plain string
    Gemini Pro/Flash may encode message content as a list of typed blocks,
    e.g. [{"type": "text", "text": "..."}]; those are flattened to plain text.
    """
    if isinstance(out, str):
        return out
    if not isinstance(out, dict):
        return ""
    text = out.get("response") or out.get("output") or ""
    if text:
        return text if isinstance(text, str) else ""
    for m in reversed(out.get("messages", [])):
        content = (m[1] if isinstance(m, list) and len(m) > 1
                   else m.get("content", "") if isinstance(m, dict) else "")
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") if isinstance(part, dict) else ""
                for part in content
            ).strip()
        if content and not (content.strip().startswith("{") and "agents_needed" in content):
            return content
    return ""


def _extract_rag_context_from_input(span_input: object) -> str:
    """
    Extract the RAG context injected into the human turn of a specialist LLM span.

    The live workflow builds: HumanMessage("Context from knowledge base:\\n\\n{rag}\\n\\nCustomer question: {q}")
    The captured input is a list of message dicts; we parse the user message to recover {rag}.
    Returns empty string when no RAG context was injected (e.g. planning / orchestrator spans).
    """
    if not isinstance(span_input, list):
        return ""
    for msg in span_input:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if "Context from knowledge base:" not in content:
            return ""
        after_kb = content.split("Context from knowledge base:", 1)[-1]
        if "Customer question:" in after_kb:
            return after_kb.split("Customer question:")[0].strip()
        return after_kb.strip()
    return ""


def _build_prompt_map(all_spans: list, user_query: str) -> dict:
    """
    Pre-compute a span_id → Prompt mapping for every LLM span in the trace.

    - Synthesizer LLM spans: variables={query, context}, rag_*_variables set
      (enables hallucination-detection evaluator)
    - Specialist LLM spans: chat_template + variables={query, context} matching
      the live workflow so Datadog Prompt Tracking sees a consistent template
    - Other LLM spans (planning, orchestrator): variables={query} only

    The specialist RAG context is extracted from the captured LLM input messages,
    mirroring the chat_template built by swagbot_langgraph_workflow_full.py.
    """
    if not _PROMPT_TEMPLATES:
        return {}

    span_by_id = {s["span_id"]: s for s in all_spans}

    # Collect specialist outputs to use as synthesizer `context`
    specialist_outputs: list = []
    for s in all_spans:
        if s.get("span_kind") != "agent":
            continue
        name_l = s.get("name", "").lower()
        if any(k in name_l for k in ("product", "customer", "promotion", "feedback")):
            text = _extract_specialist_text(_parse_io(s.get("output", "")))
            if text:
                specialist_outputs.append(text)
    synth_context = "\n\n".join(specialist_outputs)

    # Human-turn template used by the live workflow when RAG context is present
    _USER_TMPL_WITH_CTX = "Context from knowledge base:\n\n{context}\n\nCustomer question: {query}"

    prompt_map: dict = {}
    for span in all_spans:
        if span.get("span_kind") != "llm":
            continue
        agent_type = _resolve_agent_type(span, span_by_id)
        if agent_type is None:
            continue
        tmpl = _PROMPT_TEMPLATES.get(agent_type)
        if not tmpl:
            continue

        common = {"id": tmpl["id"], "version": tmpl["version"]}

        if agent_type == "planning":
            # Mirrors swagbot_langgraph_workflow_full.py planning chat_template:
            # system = prompt_text, user = "{query}" (no RAG context).
            p = Prompt(
                **common,
                chat_template=[
                    {"role": "system", "content": tmpl["template"]},
                    {"role": "user",   "content": "{query}"},
                ],
                variables={"query": user_query},
            )
        elif agent_type == "synthesizer":
            # Mirrors swagbot_langgraph_workflow_full.py synthesizer chat_template.
            p = Prompt(
                **common,
                chat_template=[
                    {"role": "system", "content": tmpl["template"]},
                    {"role": "user",   "content": "Context:\n{context}\n\nQuestion: {query}"},
                ],
                variables={"query": user_query, "context": synth_context},
                rag_query_variables=["query"],
                rag_context_variables=["context"],
            )
        else:
            # Specialist — extract the RAG context that was actually fed to this
            # specialist LLM call from the captured input messages.
            span_input = _parse_io(span.get("input", ""))
            rag_ctx = _extract_rag_context_from_input(span_input)
            effective_ctx = rag_ctx if rag_ctx else "No relevant products found in knowledge base."
            p = Prompt(
                **common,
                chat_template=[
                    {"role": "system", "content": tmpl["template"]},
                    {"role": "user",   "content": _USER_TMPL_WITH_CTX},
                ],
                variables={"query": user_query, "context": effective_ctx},
                rag_query_variables=["query"],
                rag_context_variables=["context"],
            )

        prompt_map[span["span_id"]] = p
        logger.debug("  Prompt mapped: %s → %s", span.get("name"), agent_type)

    return prompt_map


def _submit_evaluations(llmobs_span: object, span: dict) -> None:
    """
    Re-submit pre-captured evaluation results for a span via LLMObs.submit_evaluation().
    Only called when REPLAY_EVALUATIONS=1.

    Expected format per evaluator entry (Datadog LLMObs API):
      {"eval_metric_type": "score"|"categorical", "value": <number|string>, "status": "OK", ...}
    Entries with status ERROR or WARN (e.g. "Not a root span") are skipped.
    """
    evaluation = span.get("evaluation")
    if not evaluation or not isinstance(evaluation, dict):
        return

    for label, eval_data in evaluation.items():
        if not isinstance(eval_data, dict):
            continue
        # Skip failed/warning evaluations — only submit actual results
        if eval_data.get("status") in ("ERROR", "WARN"):
            continue
        metric_type = eval_data.get("eval_metric_type")
        value       = eval_data.get("value")
        if not metric_type or value is None:
            continue
        try:
            LLMObs.submit_evaluation(
                span_context=llmobs_span,
                label=label,
                metric_type=metric_type,
                value=float(value) if metric_type == "score" else str(value),
            )
            logger.debug("  Evaluation submitted: %s / %s = %s", span.get("name"), label, value)
        except Exception as exc:
            logger.warning("submit_evaluation skipped for %s on %s: %s",
                           label, span.get("name"), exc)


# ---------------------------------------------------------------------------
# Span-tree replay (real captured data)
# ---------------------------------------------------------------------------

def _parse_io(value: object) -> object:
    """
    Span input/output stored in JSON may be a raw string or a JSON-encoded
    list of message dicts.  Return the most useful form for LLMObs.annotate.
    """
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass  # no-dd-sa:python-best-practices/silent-exception
    return value


def _cm_for_span(span: dict):
    """Return the appropriate LLMObs context manager for this span."""
    kind = span.get("span_kind", "task")
    cm_factory = _LLMOBS_CM.get(kind, LLMObs.task)

    name = span.get("name", kind)

    # LLM spans require model_name and model_provider
    if kind == "llm":
        model    = span.get("model_name") or span.get("metadata", {}).get("model", SPECIALIST_MODEL)
        provider = span.get("model_provider") or ("vertexai" if "gemini" in model.lower() else "custom")
        return cm_factory(model_name=model, name=name, model_provider=provider)

    return cm_factory(name=name)


def _replay_span_tree(span: dict, all_spans: list[dict], prompt_map: "dict | None" = None) -> None:
    """
    Recursively replay a span and all its children using LLMObs context managers.
    Children are identified by parent_id == span["span_id"].
    Duration is simulated via time.sleep.

    prompt_map: optional span_id → Prompt mapping built by _build_prompt_map().
    When provided, each LLM span is annotated with the matching Prompt object,
    which enables Datadog evaluators (hallucination-detection, etc.).
    """
    children = sorted(
        [s for s in all_spans if s.get("parent_id") == span.get("span_id") and s.get("span_id") != span.get("span_id")],
        key=lambda s: s.get("start_ns", 0),
    )

    duration_s = span.get("duration_ms", 100) / 1000.0

    with _cm_for_span(span) as llmobs_span:
        # Inject agent manifest for the Execution Graph.
        # The LangGraph integration sets this via span._set_ctx_item(AGENT_MANIFEST, manifest),
        # not via LLMObs.annotate(metadata={"_dd": ...}) — so we must replicate it here.
        if span.get("span_kind") == "agent":
            agent_manifest = span.get("metadata", {}).get("_dd", {}).get("agent_manifest")
            if agent_manifest:
                try:
                    llmobs_span._set_ctx_item(_AGENT_MANIFEST_KEY, agent_manifest)
                except Exception as exc:
                    logger.debug("agent_manifest injection skipped on %s: %s", span.get("name"), exc)

        if children:
            # Distribute self duration across children (they'll sleep themselves)
            for child in children:
                _replay_span_tree(child, all_spans, prompt_map)
        else:
            # Leaf span — simulate its duration
            time.sleep(max(0.01, duration_s))

        # Annotate with original content
        kind        = span.get("span_kind", "")
        input_data  = _parse_io(span.get("input",  ""))
        output_data = _parse_io(span.get("output", ""))

        # LLM spans require message dicts — wrap plain strings
        if kind == "llm":
            if isinstance(input_data, str) and input_data:
                input_data = [{"role": "user", "content": input_data}]
            if isinstance(output_data, str) and output_data:
                output_data = [{"role": "assistant", "content": output_data}]

        # Retrieval spans: resolve from helper (fresh RAG search or fallback)
        if kind == "retrieval":
            output_data = _retrieval_output_data(span, input_data)

        # Preserve the full state dict for all span kinds — Datadog uses "messages" and
        # "workflow_path" in the input for the Execution Graph rendering.

        # Strip all _dd* internal keys — agent manifest is now set via _set_ctx_item above
        metadata = {k: v for k, v in span.get("metadata", {}).items()
                    if not k.startswith("_dd")}

        # Tags: API returns a list ["key:value", ...], LLMObs.annotate expects a dict.
        # Convert and preserve tags like "integration:langgraph" and "source:integration"
        # which Datadog uses to identify LangGraph traces and render the Execution Graph.
        raw_tags = span.get("tags", [])
        if isinstance(raw_tags, dict):
            tags = {k: v for k, v in raw_tags.items()
                    if isinstance(k, str) and isinstance(v, (str, int, float, bool))}
        elif isinstance(raw_tags, list):
            tags = {}
            for t in raw_tags:
                if isinstance(t, str) and ":" in t:
                    k, v = t.split(":", 1)
                    tags[k] = v
        else:
            tags = {}

        # Mark LLM spans as replay so they can be filtered out in LLM Observability
        # (e.g. filter @replay:false to find real production spans for Playground testing)
        if kind == "llm":
            tags["replay"] = "true"

        annotate_kwargs: dict = {}

        # Inject token counts via LLMObs.annotate metrics parameter (ddtrace 4.x)
        if kind == "llm":
            captured_metrics = span.get("metrics", {})
            usage = {k: captured_metrics[k] for k in ("input_tokens", "output_tokens", "total_tokens")
                     if k in captured_metrics}
            if usage:
                annotate_kwargs["metrics"] = usage
        if input_data:
            annotate_kwargs["input_data"] = input_data
        if output_data:
            annotate_kwargs["output_data"] = output_data
        if metadata:
            annotate_kwargs["metadata"] = metadata
        if tags:
            annotate_kwargs["tags"] = tags

        # Set integration/source as APM-level span tags via set_tag().
        # Datadog's backend looks for these at the APM level (not LLMObs meta.tags)
        # to identify LangGraph traces and render the Execution Graph tab.
        # Always hardcode source:integration for LangGraph spans — the LangGraph
        # integration sets this automatically at APM level; it is not captured in
        # the LLMObs tags, so we cannot rely on the captured tag value.
        if tags.get("integration") == "langgraph":
            try:
                llmobs_span.set_tag("integration", "langgraph")
                llmobs_span.set_tag("source", "integration")
            except Exception as exc:
                logger.debug("set_tag skipped on %s: %s", span.get("name"), exc)
        else:
            for apm_key in ("integration", "source"):
                if apm_key in tags:
                    try:
                        llmobs_span.set_tag(apm_key, tags[apm_key])
                    except Exception as exc:
                        logger.debug("set_tag(%s) skipped on %s: %s", apm_key, span.get("name"), exc)

        # Inject prompt template when available (always built for real-span replays)
        if kind == "llm" and prompt_map:
            p = prompt_map.get(span.get("span_id"))
            if p is not None:
                annotate_kwargs["prompt"] = p

        if annotate_kwargs:
            try:
                LLMObs.annotate(span=llmobs_span, **annotate_kwargs)
            except Exception as exc:
                logger.warning("annotate skipped on %s (%s): %s", span.get("name"), kind, exc)

        # Submit pre-captured evaluation scores when REPLAY_EVALUATIONS=1
        if REPLAY_EVALUATIONS:
            _submit_evaluations(llmobs_span, span)

        # Handle error status — mark the ddtrace span as errored so Datadog
        # records it as an error trace and increments the error rate metric.
        if span.get("status") == "error":
            try:
                llmobs_span.error = 1
                error_msg = span.get("error_message", "")
                if error_msg:
                    llmobs_span.set_tag("error.message", error_msg)
                llmobs_span.set_tag("error.type", span.get("error_type", "Exception"))
            except Exception as exc:
                logger.debug("error status injection skipped on %s: %s", span.get("name"), exc)


def replay_from_spans(scenario: dict) -> None:
    """Replay a scenario using its captured span tree."""
    all_spans = scenario["spans"]

    # Find the root span (no parent or parent not in this set)
    span_ids = {s["span_id"] for s in all_spans}
    roots = [s for s in all_spans if s.get("parent_id", "") not in span_ids]

    if not roots:
        logger.warning("  No root span found — falling back to synthetic replay")
        replay_synthetic(scenario)
        return

    # There should be exactly one root (the workflow span)
    root = min(roots, key=lambda s: s.get("start_ns", 0))

    # Always use the original scenario message as user_query and root input.
    # The captured root["input"] may be SDS-redacted (e.g. "[Visa Card Scanner...]")
    # for security scenarios — restoring the original message ensures Datadog's SDS
    # scans and detects sensitive data on every replay cycle.
    user_query = scenario.get("message") or _parse_io(root.get("input", ""))
    prompt_map = _build_prompt_map(all_spans, user_query)
    logger.debug("  Prompt map: %d LLM spans mapped", len(prompt_map))

    # Inject replay filter tags into root span so students can filter in Datadog UI
    root = dict(root)   # shallow copy — do not mutate scenario data
    root["input"] = user_query   # restore original message (may have been SDS-redacted)
    raw_tags = root.get("tags", [])
    replay_extra = [
        "replay:true",
        f"scenario_id:{scenario['id']}",
        f"category:{scenario.get('category', '')}",
    ]
    if isinstance(raw_tags, list):
        root["tags"] = raw_tags + replay_extra
    elif isinstance(raw_tags, dict):
        root["tags"] = {**raw_tags, "replay": "true",
                        "scenario_id": scenario["id"],
                        "category": scenario.get("category", "")}
    else:
        root["tags"] = replay_extra

    _replay_span_tree(root, all_spans, prompt_map)


# ---------------------------------------------------------------------------
# Synthetic fallback replay (no captured spans)
# ---------------------------------------------------------------------------

def _ms(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def _distribute_latency(total_ms: float) -> Dict[str, float]:
    t = max(total_ms, 200)
    return {
        "planning":     t * 0.15 / 1000,
        "orchestrator": t * 0.05 / 1000,
        "specialist":   t * 0.55 / 1000,
        "synthesizer":  t * 0.25 / 1000,
    }


def _estimate_tokens(text: str) -> Dict[str, int]:
    count = max(1, math.ceil(len(text) / 4))
    return {
        "input_tokens":  max(1, count // 3),
        "output_tokens": max(1, count - count // 3),
        "total_tokens":  count,
    }


def _kb_source(agent_name: str) -> str:
    return {
        "Product Specialist":   "resources/products.json",
        "Promotion Specialist": "resources/promotions.json",
        "Customer Service":     "resources/cs_info.json",
        "Feedback Handler":     "resources/faqs.json",
    }.get(agent_name, "resources/faqs.json")


def _synthetic_planning(query: str, agents: list, latency_s: float) -> None:
    routing = json.dumps({"selected_agents": agents, "reasoning": "Based on query intent."})
    with LLMObs.agent(name="Planning Agent") as span:
        with LLMObs.task(name="llm") as t:
            _ms(latency_s)
            LLMObs.annotate(
                span=t,
                input_data=[{"role": "user", "content": query}],
                output_data=[{"role": "assistant", "content": routing}],
                metadata={"task_type": "Intent Analysis", "model": PLANNING_MODEL, **_estimate_tokens(query + routing)},
            )
        LLMObs.annotate(span=span, input_data=query, output_data=routing)


def _synthetic_orchestrator(query: str, agents: list, latency_s: float) -> None:
    dispatch = json.dumps({"dispatched_to": agents})
    with LLMObs.agent(name="Orchestrator Agent") as span:
        with LLMObs.task(name="orchestrator") as t:
            _ms(latency_s)
            LLMObs.annotate(span=t, input_data=query, output_data=dispatch,
                            metadata={"task_type": "Agent Dispatch", "agents_dispatched": agents})
        LLMObs.annotate(span=span, input_data=query, output_data=dispatch)


def _synthetic_specialist(agent_name: str, query: str, response: str, latency_s: float) -> None:
    kb = _kb_source(agent_name)
    rag_out = f"Retrieved data from {kb}"
    with LLMObs.agent(name=agent_name) as span:
        with LLMObs.retrieval(name="rag") as r:
            _ms(latency_s * 0.2)
            LLMObs.annotate(span=r, input_data=query,
                            output_data=[{"text": rag_out, "name": kb, "score": 1.0}])
        with LLMObs.task(name="llm") as t:
            _ms(latency_s * 0.8)
            LLMObs.annotate(
                span=t,
                input_data=[{"role": "user", "content": query}],
                output_data=[{"role": "assistant", "content": response}],
                metadata={"task_type": "Response Generation", "model": SPECIALIST_MODEL,
                          **_estimate_tokens(query + response)},
            )
        LLMObs.annotate(span=span, input_data=query, output_data=response)


def _synthetic_synthesizer(query: str, response: str, latency_s: float) -> None:
    with LLMObs.agent(name="Response Synthesizer") as span:
        with LLMObs.task(name="llm") as t:
            _ms(latency_s)
            LLMObs.annotate(
                span=t,
                input_data=[{"role": "user", "content": query}],
                output_data=[{"role": "assistant", "content": response}],
                metadata={"task_type": "Response Synthesis", "model": SYNTHESIZER_MODEL,
                          **_estimate_tokens(query + response)},
            )
        LLMObs.annotate(span=span, input_data=query, output_data=response)


def replay_synthetic(scenario: dict) -> None:
    """Fallback: create a plausible synthetic span tree."""
    query    = scenario["message"]
    response = scenario.get("output", "")
    category = scenario.get("category", "customer_service")
    latency  = scenario.get("latency_ms", 800)
    agents   = CATEGORY_TO_AGENTS.get(category, ["Customer Service"])
    lat      = _distribute_latency(latency)

    with LLMObs.workflow(name="swagbot_workflow") as wf:
        with LLMObs.agent(name="SwagBot") as root:
            _synthetic_planning(query, agents, lat["planning"])
            _synthetic_orchestrator(query, agents, lat["orchestrator"])
            per = lat["specialist"] / max(len(agents), 1)
            for a in agents:
                _synthetic_specialist(a, query, response, per)
            _synthetic_synthesizer(query, response, lat["synthesizer"])

            LLMObs.annotate(span=root, input_data=query, output_data=response,
                            metadata={"workflow_path": ["planning", "orchestrator"] + agents + ["synthesizer"],
                                      "category": category})
        LLMObs.annotate(span=wf, input_data=query, output_data=response,
                        tags={"replay": "true", "scenario_id": scenario["id"], "category": category})


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------

def replay_scenario(scenario: dict) -> None:
    spans = scenario.get("spans", [])
    if spans:
        logger.info("↩  Replaying (real spans): %s — %s", scenario["id"], scenario["description"])
        replay_from_spans(scenario)
    else:
        logger.info("↩  Replaying (synthetic):  %s — %s", scenario["id"], scenario["description"])
        replay_synthetic(scenario)
    logger.info("   ✅ %s — %dms (simulated)", scenario["id"], scenario.get("latency_ms", 0))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ml_app    = os.getenv("DD_LLMOBS_ML_APP", "swagbot")
    agentless = os.getenv("DD_LLMOBS_AGENTLESS_ENABLED", "0") == "1"
    if not LLMObs.enabled:
        LLMObs.enable(ml_app=ml_app, agentless_enabled=agentless)

    scenarios_path = os.path.normpath(SCENARIOS_PATH)
    if not os.path.exists(scenarios_path):
        logger.error("replay_scenarios.json not found at %s — exiting", scenarios_path)
        return

    with open(scenarios_path) as f:
        scenarios = json.load(f)

    # Prompt templates are always loaded and injected on LLM spans
    global _PROMPT_TEMPLATES
    _PROMPT_TEMPLATES = _load_prompt_templates()
    eval_status = "ENABLED" if REPLAY_EVALUATIONS else "DISABLED (set REPLAY_EVALUATIONS=1 to enable)"
    logger.info("🔖 Prompt templates: %d loaded | Evaluations: %s", len(_PROMPT_TEMPLATES), eval_status)

    real_count = sum(1 for s in scenarios if s.get("spans"))
    logger.info("🔁 SwagBot Trace Replay — %d scenarios (%d with real spans, %d synthetic)",
                len(scenarios), real_count, len(scenarios) - real_count)

    cycle = 0
    while True:
        cycle += 1
        logger.info("── Cycle %d ──────────────────────────────", cycle)

        # Re-read scenarios on every cycle so a file swap (e.g. switching from
        # error scenarios to success scenarios after a prompt fix) takes effect
        # without restarting the container.
        try:
            with open(scenarios_path) as f:
                scenarios = json.load(f)
        except Exception as exc:
            logger.warning("Could not reload scenarios: %s — reusing previous set", exc)

        cycle_scenarios = list(scenarios)
        random.shuffle(cycle_scenarios)

        for scenario in cycle_scenarios:
            # Skip scenario if its skip_condition is now satisfied (e.g. pricing is fixed)
            skip_cond = scenario.get("skip_condition")
            if skip_cond and _check_skip_condition(skip_cond):
                logger.debug("Skipping %s — condition '%s' is satisfied", scenario.get("id"), skip_cond)
                continue
            try:
                replay_scenario(scenario)
            except Exception as exc:
                logger.error("Failed to replay %s: %s", scenario.get("id"), exc)
            time.sleep(SLEEP_BETWEEN_SCENARIOS)

        logger.info("── Cycle %d complete. Sleeping %ss ──────", cycle, SLEEP_BETWEEN_CYCLES)
        time.sleep(SLEEP_BETWEEN_CYCLES)


if __name__ == "__main__":
    main()
