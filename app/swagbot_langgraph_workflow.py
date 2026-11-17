"""
SwagBot LangGraph Workflow — Base version (zero Datadog decorators)

Architecture: an outer StateGraph coordinates seven independent agent subgraphs.
Each agent is a compiled LangGraphPregel added directly as a node — ddtrace detects
these as `agent` spans automatically without any manual annotation.

Datadog span hierarchy (auto-detected by ddtrace via the LangGraph patch):

  workflow: swagbot_workflow
    agent: SwagBot                         ← outer StateGraph
      agent: Planning                      ← 1 LLM call, outputs routing JSON
        task: llm
      agent: Orchestrator                  ← pure dispatch node, no LLM
        task: orchestrator
      agent: Customer Service              ← 1 RAG retrieval + 1 LLM call
        task: rag
        task: llm
      agent: Product Specialist            ← 1 RAG retrieval + 1 LLM call (+ tools)
        task: rag
        task: llm
      agent: Promotion Specialist          ← 1 RAG retrieval + 1 LLM call
        task: rag
        task: llm
      agent: Feedback Handler              ← 1 RAG retrieval + 1 LLM call
        task: rag
        task: llm
      agent: Response Synthesizer          ← 1 LLM call, formats HTML response
        task: llm

File structure:
  1. State / schema definitions
  2. SwagBotWorkflow class  ← start reading here
  3. Tools
  4. RAG utilities
  5. LLM infrastructure (LLMCaller, LLMFactory)
  6. Entry point

Prompts and KB data are loaded from disk on every request — update files
in app/resources/ and changes take effect immediately without restarting.
"""

import html
import json
import operator
import os
import re
import logging
import warnings
from typing import Annotated, Any, Dict, List, Optional, Sequence
from typing_extensions import TypedDict

# Suppress ChatVertexAI deprecation warning from langchain-google-vertexai>=3.2.0.
# ChatVertexAI still works; the suggested replacement (ChatGoogleGenerativeAI) uses
# Google AI Studio API keys instead of service-account auth — incompatible with this lab.
warnings.filterwarnings(
    "ignore",
    message=".*ChatVertexAI.*",
    category=DeprecationWarning,
)

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Send
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool as lc_tool

from swagbot_langgraph_config import LangChainConfig
from swagbot_utils import (
    DocumentUtils, DocumentHandlers, HTMLUtils,
    KnowledgeBaseUtils, SynthesisUtils, PromptTrackingUtils,
)


logger = logging.getLogger(__name__)
config = LangChainConfig()


# ===============================================================================
# STATE
# ===============================================================================

class SwagBotState(TypedDict):
    """Internal state for the outer graph — all fields."""
    input: str
    output: str
    messages: Annotated[List[AnyMessage], add_messages]
    workflow_path: Annotated[List[str], operator.add]


class SwagBotInput(TypedDict):
    """Input schema — input first so Datadog shows it at the top."""
    input: str
    messages: Annotated[List[AnyMessage], add_messages]
    output: str
    workflow_path: Annotated[List[str], operator.add]


class SwagBotOutput(TypedDict):
    """Output schema — output first so Datadog shows it at the top."""
    output: str
    input: str
    messages: Annotated[List[AnyMessage], add_messages]
    workflow_path: Annotated[List[str], operator.add]


class BaseAgentState(MessagesState):
    """
    Base state for all subgraphs. Shares the workflow_path reducer with SwagBotState
    so each agent can append its name and it propagates up to the outer graph.
    """
    workflow_path: Annotated[List[str], operator.add]


class AgentInput(TypedDict):
    """
    Input schema for all subgraphs. Excludes workflow_path so each subgraph
    starts with workflow_path=[] — prevents operator.add accumulation bug.
    """
    messages: Annotated[List[AnyMessage], add_messages]


class SpecialistState(BaseAgentState):
    """State for specialist subgraphs — passes RAG results from rag_node to llm_node."""
    rag_context: Optional[str]


class SynthesizerState(BaseAgentState):
    """
    Synthesizer state — includes output so its llm_node writes directly
    to SwagBotState.output, eliminating the need for a separate finalize node.
    """
    output: str


# ===============================================================================
# SWAGBOT WORKFLOW
# ===============================================================================

class SwagBotWorkflow:
    """
    Builds the outer StateGraph where every node is a compiled subgraph (LangGraphPregel).
    ddtrace skips task spans for Pregel nodes and emits direct `agent` spans instead.
    """

    def __init__(self):
        self.llm_caller = LLMCaller(config)
        planning_llm    = self.llm_caller.get_llm_instance("planning")
        specialist_llm  = self.llm_caller.get_llm_instance("specialist")
        synthesizer_llm = self.llm_caller.get_llm_instance("synthesizer")

        logger.info("🔧 Building Planning agent…")
        self._planning = self._build_planning_agent(planning_llm)
        logger.info("✅ Planning agent ready")

        logger.info("🔧 Building Orchestrator agent…")
        self._orchestrator = self._build_orchestrator_node()
        logger.info("✅ Orchestrator agent ready")

        logger.info("🔧 Building specialist agents…")
        self._customer_service = self._build_specialist_agent(
            llm=specialist_llm,
            agent_types=["customer_service"],
            prompt_fn=lambda: self._load_prompt("customer-service"),
            name="Customer Service Agent",
            tools=[get_order_status],
        )
        self._product_specialist = self._build_specialist_agent(
            llm=specialist_llm,
            agent_types=["product_specialist"],
            prompt_fn=lambda: self._load_prompt("product-specialist"),
            name="Product Specialist Agent",
            tools=[check_product_availability],
        )
        self._promotion_specialist = self._build_specialist_agent(
            llm=specialist_llm,
            agent_types=["promotion_specialist"],
            prompt_fn=lambda: self._load_prompt("promotion-specialist"),
            name="Promotion Specialist Agent",
        )
        self._feedback_handler = self._build_specialist_agent(
            llm=specialist_llm,
            agent_types=["feedback_handler"],
            prompt_fn=lambda: self._load_prompt("feedback-handler"),
            name="Feedback Handler Agent",
        )
        logger.info("✅ Specialist agents ready")

        logger.info("🔧 Building Synthesizer agent…")
        self._synthesizer = self._build_synthesizer_agent(synthesizer_llm)
        logger.info("✅ Synthesizer agent ready")

        logger.info("🔧 Compiling StateGraph…")
        self.app = self._build_graph()
        logger.info("✅ SwagBot workflow initialised — 6 agents as direct subgraph nodes")

    # ── Graph structure ────────────────────────────────────────────────────────

    def _build_graph(self) -> CompiledStateGraph:
        """
        Outer graph: Planning → Orchestrator → specialists (parallel) → Synthesizer.
        SwagBotInput/SwagBotOutput schemas control field display order in Datadog.
        """
        g = StateGraph(SwagBotState, input_schema=SwagBotInput, output_schema=SwagBotOutput)

        g.add_node("Planning",             self._planning)
        g.add_node("Orchestrator",         self._orchestrator)
        g.add_node("Customer Service",     self._customer_service)
        g.add_node("Product Specialist",   self._product_specialist)
        g.add_node("Promotion Specialist", self._promotion_specialist)
        g.add_node("Feedback Handler",     self._feedback_handler)
        g.add_node("Response Synthesizer", self._synthesizer)

        g.add_edge(START, "Planning")
        g.add_edge("Planning", "Orchestrator")
        g.add_conditional_edges(
            "Orchestrator",
            self._dispatch_specialists,
            ["Customer Service", "Product Specialist", "Promotion Specialist", "Feedback Handler"],
        )
        for specialist in ["Customer Service", "Product Specialist",
                           "Promotion Specialist", "Feedback Handler"]:
            g.add_edge(specialist, "Response Synthesizer")
        g.add_edge("Response Synthesizer", END)

        compiled = g.compile()
        compiled.name = "SwagBot"
        return compiled

    # ── Agent builders (in execution order) ───────────────────────────────────

    def _build_planning_agent(self, planning_llm: BaseChatModel) -> CompiledStateGraph:
        """
        Mini StateGraph: single LLM call that outputs routing JSON.
        The LLM is prompted to output: {"agents_needed": ["customer_service", ...]}
        _dispatch_specialists parses this JSON from the resulting AIMessage.
        """
        def planning_llm_node(state: dict) -> dict:
            prompt_text = self._build_planning_prompt()
            # Use only the latest HumanMessage for consistent template fingerprint.
            human_msgs = [m for m in state.get("messages", []) if isinstance(m, HumanMessage)]
            user_query_msg = human_msgs[-1] if human_msgs else HumanMessage(content="")
            messages = [SystemMessage(content=prompt_text), user_query_msg]
            response = planning_llm.invoke(messages)
            return {"messages": [response], "workflow_path": ["planning"]}

        g = StateGraph(BaseAgentState, input_schema=AgentInput)
        g.add_node("llm", planning_llm_node)
        g.add_edge(START, "llm")
        g.add_edge("llm", END)
        compiled = g.compile()
        compiled.name = "Planning Agent"
        return compiled

    def _build_orchestrator_node(self) -> CompiledStateGraph:
        """
        Pure coordination node — zero LLM calls.
        Reads Planning's routing decision from messages and logs the dispatch context.
        """
        def orchestrator_node(state: dict) -> dict:
            agents_needed: List[str] = []
            user_request = ""
            for msg in state.get("messages", []):
                if isinstance(msg, HumanMessage) and not user_request:
                    user_request = (msg.content or "")[:80]
                if isinstance(msg, AIMessage) and msg.content and not agents_needed:
                    try:
                        cleaned = SwagBotWorkflow._extract_json_content(SwagBotWorkflow._get_str_content(msg.content))
                        data = json.loads(cleaned)
                        agents_needed = data.get("agents_needed", [])
                    except (json.JSONDecodeError, TypeError, AttributeError) as exc:
                        logger.debug("Could not parse Planning JSON in orchestrator: %s", exc)
            logger.info("🎯 Orchestrator: %.60r → %s", user_request, agents_needed)
            return {"workflow_path": ["orchestrator"]}

        g = StateGraph(BaseAgentState, input_schema=AgentInput)
        g.add_node("orchestrator", orchestrator_node)
        g.add_edge(START, "orchestrator")
        g.add_edge("orchestrator", END)
        compiled = g.compile()
        compiled.name = "Orchestrator Agent"
        return compiled

    def _build_specialist_agent(
        self,
        llm: BaseChatModel,
        agent_types: List[str],
        prompt_fn,
        name: str,
        tools: Optional[list] = None,
    ) -> CompiledStateGraph:
        """
        Mini StateGraph: 1 deterministic RAG retrieval + 1 LLM call.
        Optionally augmented with tools — if provided, the LLM can call them
        and a second LLM call produces the final answer with tool results.

        Without tools:
          agent: <name>
            task: rag   ← deterministic KB search
            task: llm   ← single LLM call with retrieved context

        With tools:
          agent: <name>
            task: rag   ← deterministic KB search
            task: llm   ← LLM decides whether to call a tool
            task: tools ← tool execution (if called)
            task: llm   ← final LLM call with tool results
        """
        llm_bound = llm.bind_tools(tools) if tools else llm

        def rag_node(state: dict) -> dict:
            user_query = next(
                (m.content for m in state.get("messages", [])
                 if isinstance(m, HumanMessage) and (m.content or "").strip()),
                "",
            )
            if not user_query:
                return {"rag_context": ""}
            docs = _rag_search_raw(agent_types, user_query)
            return {"rag_context": _format_docs_for_llm(docs)}

        def llm_node(state: dict) -> dict:
            prompt_text = prompt_fn()
            messages_in = state.get("messages", [])

            last_msg = messages_in[-1] if messages_in else None
            rag_context = (state.get("rag_context") or "").strip()
            user_query = next(
                (m.content for m in messages_in
                 if isinstance(m, HumanMessage) and (m.content or "").strip()),
                "",
            )
            # Always use the same template structure so prompt tracking sees a single fingerprint
            effective_context = rag_context if rag_context else "No relevant products found in knowledge base."
            human_content = (
                f"Context from knowledge base:\n\n{effective_context}"
                f"\n\nCustomer question: {user_query}"
            )
            messages = [SystemMessage(content=prompt_text), HumanMessage(content=human_content)]
            if isinstance(last_msg, ToolMessage):
                # After a tool call, keep the knowledge-base context and append the tool call
                # + its results. Without the context the LLM invents details (e.g. a price).
                tool_call_idx = max(
                    i for i, m in enumerate(messages_in)
                    if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
                )
                messages += messages_in[tool_call_idx:]

            response = llm_bound.invoke(messages)
            # Only update workflow_path on the final response (not when a tool call is pending)
            has_tool_calls = bool(getattr(response, "tool_calls", None))
            result: dict = {"messages": [response]}
            if not has_tool_calls:
                result["workflow_path"] = [name]
            return result

        def should_call_tools(state: dict) -> str:
            last = state.get("messages", [])[-1] if state.get("messages") else None
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                return "tools"
            return END

        g = StateGraph(SpecialistState, input_schema=AgentInput)
        g.add_node("rag", rag_node)
        g.add_node("llm", llm_node)
        g.add_edge(START, "rag")
        g.add_edge("rag", "llm")

        if tools:
            g.add_node("tools", ToolNode(tools))
            g.add_conditional_edges("llm", should_call_tools, {"tools": "tools", END: END})
            g.add_edge("tools", "llm")
        else:
            g.add_edge("llm", END)

        compiled = g.compile()
        compiled.name = name
        return compiled

    def _build_synthesizer_agent(self, llm: BaseChatModel) -> CompiledStateGraph:
        """
        Synthesizer: formats all specialist text responses as a single HTML answer.
        Writes directly to SynthesizerState.output, which propagates to SwagBotState.output.
        """
        _PROMPT_SIZE_LIMIT = 4000  # chars; v2.0 prompt (~5100) exceeds this, v2.1 (~3300) does not

        def llm_node(state: dict) -> dict:
            from google.api_core.exceptions import ResourceExhausted
            text_msgs = SwagBotWorkflow._extract_text_messages(state.get("messages", []))
            specialist_responses = [m for m in text_msgs if isinstance(m, AIMessage)]
            logger.info("🔀 Synthesizer: formatting %d specialist response(s) as HTML",
                        len(specialist_responses))
            prompt_text = self._build_synthesizer_prompt()
            if config.error_simulation_enabled and len(prompt_text) > _PROMPT_SIZE_LIMIT:
                raise ResourceExhausted(
                    "The model's input context length was exceeded. "
                    f"Prompt size ({len(prompt_text)} chars) exceeds the allowed limit. "
                    "Please reduce the size of the synthesizer prompt."
                )
            original_query = next((m for m in text_msgs if isinstance(m, HumanMessage)), None)
            reference = "\n\n---\n\n".join(
                t for t in (SwagBotWorkflow._get_str_content(m.content) for m in specialist_responses)
                if t.strip()
            )
            user_question = SwagBotWorkflow._get_str_content(original_query.content) if original_query else ""
            # Specialist answers go in the USER turn as explicit context. Passing them as prior
            # ASSISTANT turns made Gemini 3.x re-answer the repeated question and invent
            # missing facts (e.g. a price).
            messages = [
                SystemMessage(content=prompt_text),
                HumanMessage(content=f"Context:\n{reference}\n\nQuestion: {user_question}"),
            ]
            response = llm.invoke(messages)
            output_text = SwagBotWorkflow._get_str_content(response.content)
            # Strip markdown code fences the LLM occasionally wraps around HTML output.
            output_text = re.sub(r'^```(?:html)?\s*\n?', '', output_text.strip())
            output_text = re.sub(r'\n?```\s*$', '', output_text)
            # Convert [PRODUCT_IMAGE:/path] placeholders to <img> tags.
            # Accept any path and normalize to /static/images/{basename} so that
            # LLM-generated paths like /path/to/image.jpg are resolved correctly.
            output_text = re.sub(
                r'\[PRODUCT_IMAGE:([^\]]+)\]',
                lambda m: (
                    f'<img src="/static/images/{html.escape(os.path.basename(m.group(1)), quote=True)}"'
                    f' alt="Product image" style="max-width:180px;display:block;margin:6px 0">'
                ),
                output_text,
            )
            # Remove <img> tags with external (http/https) src — hallucinated by the model.
            output_text = re.sub(
                r'<img\b[^>]*\bsrc=["\']https?://[^"\']*["\'][^>]*/?>',
                '', output_text, flags=re.IGNORECASE,
            )
            # Strip <a href="http(s)://..."> wrappers, keeping their visible text.
            output_text = re.sub(
                r'<a\b[^>]*\bhref=["\']https?://[^"\']*["\'][^>]*>(.*?)</a>',
                r'\1', output_text, flags=re.IGNORECASE | re.DOTALL,
            )
            # Strip <a> wrappers around <img> tags (regardless of href type).
            output_text = re.sub(
                r'<a\b[^>]*>\s*(<img\b[^>]*/?>)\s*</a>',
                r'\1', output_text, flags=re.IGNORECASE | re.DOTALL,
            )
            # Final allowlist sanitization pass — defense in depth against
            # any HTML the regex cleanups above don't anticipate.
            output_text = HTMLUtils.sanitize_llm_html(output_text)
            return {
                "messages": [response],
                "workflow_path": ["synthesizer"],
                "output": output_text,
            }

        g = StateGraph(SynthesizerState, input_schema=AgentInput)
        g.add_node("llm", llm_node)
        g.add_edge(START, "llm")
        g.add_edge("llm", END)
        compiled = g.compile()
        compiled.name = "Response Synthesizer Agent"
        return compiled

    # ── Routing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _dispatch_specialists(state: SwagBotState) -> Sequence[Send]:
        """
        Parse Planning's routing JSON from its AIMessage and fan-out to specialists.
        Falls back to customer_service if the JSON cannot be parsed.
        """
        valid = {"customer_service", "product_specialist", "promotion_specialist", "feedback_handler"}
        agents_needed: List[str] = []

        for msg in state.get("messages", []):
            if isinstance(msg, AIMessage) and msg.content:
                try:
                    cleaned = SwagBotWorkflow._extract_json_content(SwagBotWorkflow._get_str_content(msg.content))
                    data = json.loads(cleaned)
                    found = [a for a in data.get("agents_needed", []) if a in valid]
                    if found:
                        agents_needed = found
                        logger.info("🗺️  Routing from Planning: %s", agents_needed)
                        break
                except (json.JSONDecodeError, TypeError, AttributeError) as exc:
                    logger.debug("Could not parse Planning routing JSON: %s", exc)

        agents_needed = agents_needed or ["customer_service"]

        node_map = {
            "customer_service":     "Customer Service",
            "product_specialist":   "Product Specialist",
            "promotion_specialist": "Promotion Specialist",
            "feedback_handler":     "Feedback Handler",
        }
        sends = [Send(node_map[a], state) for a in agents_needed if a in node_map]
        if not sends:
            sends = [Send("Customer Service", state)]
        logger.info("📤 Dispatching to: %s", [s.node for s in sends])
        return sends

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    @staticmethod
    def _get_str_content(content) -> str:
        """
        Normalise LangChain message content to a plain string.
        Gemini may return content as a list of part-dicts, e.g.
        [{"type": "text", "text": "..."}]. Extract and join text parts.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in content
            )
        return ""

    def _extract_text_messages(messages: list) -> list:
        """
        Return only text-bearing messages safe to pass to any LLM call.
        Keeps HumanMessages and AIMessages with text content (no tool_calls).
        Strips Planning's routing JSON so it does not pollute the Synthesizer's context.
        Handles content that is either a str or a list of part-dicts (Gemini format).
        """
        result = []
        for m in messages:
            text = SwagBotWorkflow._get_str_content(m.content)
            if isinstance(m, HumanMessage) and text.strip():
                result.append(m)
            elif (isinstance(m, AIMessage)
                  and not (m.tool_calls or [])
                  and text.strip()):
                try:
                    cleaned = SwagBotWorkflow._extract_json_content(text)
                    data = json.loads(cleaned)
                    if isinstance(data, dict) and "agents_needed" in data:
                        continue
                except (json.JSONDecodeError, TypeError, AttributeError) as exc:
                    logger.debug("Message is not Planning JSON (expected): %s", exc)
                result.append(m)
        return result

    @staticmethod
    def _extract_json_content(text: str) -> str:
        """Strip markdown code fences (```json ... ``` or ``` ... ```) if present."""
        result = text.strip()
        if result.startswith("```"):
            result = re.sub(r'^```(?:json)?\s*\n?', '', result)
            result = re.sub(r'\n?```\s*$', '', result)
        return result.strip()

    # ── Prompt builders ───────────────────────────────────────────────────────

    @staticmethod
    def _load_prompt(agent_type: str) -> str:
        try:
            return SynthesisUtils.load_agent_prompt(agent_type, _resources_dir())
        except Exception:
            return f"You are a SwagBot {agent_type.replace('-', ' ')} specialist. Help the customer thoroughly."

    @staticmethod
    def _build_planning_prompt() -> str:
        # Returns the on-disk prompt verbatim so Datadog Prompt Tracking sees the
        # same template fingerprint as scripts/replay.py, which also reads the file.
        try:
            return PromptTrackingUtils.load_prompt_with_metadata("planning")["template"]
        except Exception as exc:
            logger.debug("Planning prompt file not found, using fallback: %s", exc)
            return "You are the Planning Agent for SwagBot, a multi-agent e-commerce assistant."

    @staticmethod
    def _build_synthesizer_prompt() -> str:
        # Returns the on-disk prompt verbatim — see _build_planning_prompt note.
        try:
            return SynthesisUtils.load_agent_prompt("synthesizer", _resources_dir())
        except Exception as exc:
            logger.debug("Synthesizer prompt file not found, using fallback: %s", exc)
            return (
                "You are the Response Synthesizer Agent for SwagBot.\n"
                "Deliver specialist responses to the customer as one clear, complete, and well-formatted answer."
            )

    # ── Public interface ──────────────────────────────────────────────────────

    def process_request(self, input: str) -> Dict[str, Any]:
        return process_swagbot_request(input)


# ===============================================================================
# TOOLS
# ===============================================================================

@lc_tool
def get_order_status(order_id: str) -> str:
    """Look up the current status and tracking information for a customer order."""
    mock_orders = {
        "ORD-001": {"status": "Delivered",        "date": "2026-03-20", "carrier": "FedEx", "tracking": "FX123456789"},
        "ORD-002": {"status": "In Transit",       "date": "2026-03-28", "carrier": "UPS",   "tracking": "UP987654321"},
        "ORD-003": {"status": "Processing",       "date": "2026-03-30", "carrier": None,    "tracking": None},
        "ORD-004": {"status": "Out for Delivery", "date": "2026-03-27", "carrier": "DHL",   "tracking": "DH556677"},
    }
    key = order_id.upper().strip()
    order = mock_orders.get(key)
    if not order:
        return f"Order '{order_id}' not found. Please verify the order ID and try again."
    if order["carrier"]:
        return (
            f"Order {key} — {order['status']}. "
            f"Expected: {order['date']}. "
            f"Carrier: {order['carrier']}, tracking: {order['tracking']}."
        )
    return f"Order {key} — {order['status']}. Being prepared for shipment, tracking available shortly."


@lc_tool
def check_product_availability(product_name: str) -> str:
    """Check the current stock availability for a SwagBot product by name."""
    catalog = {
        "dog headphones":          {"status": "Low Stock",    "quantity": 3,   "note": "Only 3 units left — order soon."},
        "dog steel bottle":        {"status": "In Stock",     "quantity": 42,  "note": None},
        "dog t-shirt":             {"status": "In Stock",     "quantity": 85,  "note": "Available in all sizes."},
        "dog t-shirt i love you":  {"status": "Out of Stock", "quantity": 0,   "note": "Restock expected in 2-3 weeks."},
        "dog sweatshirt gray":     {"status": "In Stock",     "quantity": 31,  "note": "Available in S, M, L, XL."},
        "dog sweatshirt black":    {"status": "Low Stock",    "quantity": 2,   "note": "Only 2 units left."},
        "dog notebook":            {"status": "In Stock",     "quantity": 60,  "note": None},
        "dog sko notebook":        {"status": "Out of Stock", "quantity": 0,   "note": "Limited edition — no restock planned."},
        "dog mug":                 {"status": "In Stock",     "quantity": 120, "note": None},
        "dog hoodie":              {"status": "In Stock",     "quantity": 18,  "note": "Available in M, L, XL."},
        "dog plastic bottle":      {"status": "In Stock",     "quantity": 55,  "note": None},
        "dog sticker pack":        {"status": "In Stock",     "quantity": 200, "note": None},
        "dog beanie":              {"status": "Pre-order",    "quantity": 0,   "note": "Ships from April 15, 2026."},
    }
    query = product_name.lower().strip()
    match = catalog.get(query) or next(
        (v for k, v in catalog.items() if query in k or k in query), None
    )
    if not match:
        return f"Product '{product_name}' not found in the catalog. Please check the product name."
    status, qty, note = match["status"], match["quantity"], match["note"]
    result = f"{product_name} — {status}."
    if qty > 0:
        result += f" {qty} units available."
    if note:
        result += f" {note}"
    return result


# ===============================================================================
# RAG UTILITIES
# ===============================================================================

def _resources_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "resources")


def _rag_search_raw(agent_types: List[str], query: str) -> List[Dict[str, Any]]:
    """Load KB sections from disk and return ranked document dicts."""
    try:
        kb = KnowledgeBaseUtils.load_selective_knowledge_base(agent_types, _resources_dir())
        handler = DocumentHandlers()
        words = [w.lower() for w in query.split() if w]
        docs: List[Dict[str, Any]] = []

        for ag in agent_types:
            if ag == "customer_service":
                docs.extend(handler.search_faqs(words, kb.get("faqs", {}), None))
                docs.extend(handler.search_customer_service(words, kb.get("customer_service", {}), None))
            elif ag == "product_specialist":
                docs.extend(handler.search_products(words, kb.get("products", {}), None))
            elif ag == "promotion_specialist":
                docs.extend(handler.search_promotions(words, kb.get("promotions", {}), None))
            elif ag == "feedback_handler":
                docs.extend(handler.search_faqs(words, kb.get("faqs", {}), None))

        return DocumentUtils.rank_and_filter_documents(docs, max_results=6)
    except Exception as exc:
        logger.error("RAG search error for %s: %s", agent_types, exc)
        return []


def _format_docs_for_llm(docs: List[Dict[str, Any]]) -> str:
    """
    Format retrieved documents as an LLM-readable context string.
    Uses [PRODUCT_IMAGE:/path] placeholders instead of <img> tags so the LLM
    cannot hallucinate or modify the image path.  Post-processing in the synthesizer
    converts these placeholders back to <img> tags before returning the response.
    """
    if not docs:
        return "No relevant documents found."
    parts = []
    for i, d in enumerate(docs, 1):
        content = d.get("content", "")
        # Convert "Image: /path" → "[PRODUCT_IMAGE:/path]" so the LLM sees the
        # explicit placeholder it is instructed to preserve verbatim.
        content = re.sub(r'(?m)^Image: (/\S+)', r'[PRODUCT_IMAGE:\1]', content)
        parts.append(f"[{i}] {d.get('name', 'Document')} (relevance {d.get('score', 0):.1f})\n{content}")
    return "\n\n".join(parts)


# ===============================================================================
# LLM INFRASTRUCTURE
# ===============================================================================

class LLMCaller:
    def __init__(self, cfg: LangChainConfig):
        self.config = cfg
        self.platform = cfg.llm_platform
        self._cache: Dict[str, BaseChatModel] = {}

    def get_llm_instance(self, llm_type: str = "specialist") -> BaseChatModel:
        if llm_type not in self._cache:
            getter = {
                "planning":    self.config.get_planning_config,
                "specialist":  self.config.get_specialist_config,
                "synthesizer": self.config.get_synthesizer_config,
            }.get(llm_type, self.config.get_specialist_config)
            model_cfg = getter()
            self._cache[llm_type] = LLMFactory.create(self.platform, model_cfg)
            logger.info("Created %s LLM: %s", llm_type, model_cfg.get("model_id", "unknown"))
        return self._cache[llm_type]


class LLMFactory:
    @staticmethod
    def create(platform: str, model_cfg: Dict[str, Any]) -> BaseChatModel:
        if platform == "vertex":
            return LLMFactory._vertex(model_cfg)
        elif platform == "bedrock":
            return LLMFactory._bedrock(model_cfg)
        elif platform == "openai":
            return LLMFactory._openai(model_cfg)
        elif platform == "azure":
            return LLMFactory._azure(model_cfg)
        raise ValueError(f"Unsupported platform: {platform}")

    @staticmethod
    def _vertex(c: Dict[str, Any]) -> BaseChatModel:
        from langchain_google_vertexai import ChatVertexAI
        return ChatVertexAI(
            model_name=c["model_id"],
            project=c.get("project_id"),
            location=c.get("location", "global"),
            temperature=c.get("temperature", 0.3),
            max_output_tokens=c.get("max_tokens", 1000),
        )

    @staticmethod
    def _bedrock(c: Dict[str, Any]) -> BaseChatModel:
        import boto3
        from langchain_aws import ChatBedrock
        client = boto3.client("bedrock-runtime", region_name=c.get("region_name", "us-east-1"))
        return ChatBedrock(
            client=client,
            model_id=c["model_id"],
            model_kwargs={"temperature": c.get("temperature", 0.3), "max_tokens": c.get("max_tokens", 1000)},
        )

    @staticmethod
    def _openai(c: Dict[str, Any]) -> BaseChatModel:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=c["model_id"],
            api_key=c.get("api_key"),
            temperature=c.get("temperature", 0.3),
            max_tokens=c.get("max_tokens", 1000),
        )

    @staticmethod
    def _azure(c: Dict[str, Any]) -> BaseChatModel:
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            deployment_name=c["model_id"],
            azure_endpoint=c.get("azure_endpoint"),
            api_key=c.get("api_key"),
            api_version=c.get("api_version", "2024-02-15-preview"),
            temperature=c.get("temperature", 0.3),
            max_tokens=c.get("max_tokens", 1000),
        )


# ===============================================================================
# ENTRY POINT
# ===============================================================================

try:
    swagbot_workflow = SwagBotWorkflow()
except Exception as _init_exc:
    import traceback
    logging.getLogger(__name__).critical(
        "❌ FATAL: SwagBotWorkflow initialisation failed: %s\n%s",
        _init_exc,
        traceback.format_exc(),
    )
    raise


def process_swagbot_request(input: str) -> Dict[str, Any]:
    """
    Main entry point — called by swagbot_app.py.
    No Datadog decorators — ddtrace discovers everything automatically
    via the LangGraph patch (ddtrace-run in docker-compose entrypoint).
    """
    try:
        result = swagbot_workflow.app.invoke({
            "input": input,
            "output": "",
            "messages": [HumanMessage(content=input)],
            "workflow_path": [],
        })

        # Synthesizer writes output directly to state via SynthesizerState.output
        final = result.get("output") or next(
            (SwagBotWorkflow._get_str_content(m.content) for m in reversed(result.get("messages", []))
             if isinstance(m, AIMessage) and SwagBotWorkflow._get_str_content(m.content).strip()),
            "No response generated.",
        )
        # Do NOT strip HTML — the UI renders the response as HTML (images, formatting)
        return {"output": final, "input": input, "error": ""}

    except Exception as exc:
        logger.error("Workflow error: %s", exc, exc_info=True)
        raise
