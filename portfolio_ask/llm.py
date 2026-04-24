"""
llm.py — Gemini API integration with tool calling and structured output.

Handles three query modes:
  1. cited_answer: General RAG query → CitedAnswer (free text + sources)
  2. tool_query: Numerical/allocation queries → triggers tool call → structured result
  3. The LLM decides which mode applies based on query content and tool definitions.

Tool-calling flow (Variant B):
  Step 1: Send query + context + tool definitions to Gemini
  Step 2: If Gemini returns a function_call → execute the tool locally
  Step 3: Send tool result back to Gemini → get final natural language answer

Never extract tool outputs with regex. Never ask the LLM to compute arithmetic.
"""

import os
import json
from typing import Any, Dict, List, Optional, Tuple
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool

from portfolio_ask.models import CitedAnswer, PnLSummary, AllocationBreakdown
from portfolio_ask.tools import TOOL_REGISTRY, TOOL_DEFINITIONS
from portfolio_ask.retrieve import retrieve, format_context

# Configure Gemini with API key from environment
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

MODEL_NAME = "gemini-1.5-flash"  # Fast, cheap, supports tool calling and JSON mode

# ── System prompt ──────────────────────────────────────────────────────────────
# Each instruction exists for a specific reason — see comments inline.
SYSTEM_PROMPT = """You are a portfolio analyst assistant for an HNI (High Net-Worth Individual) client.
You have access to the client's portfolio data and recent market news.

STRICT RULES — follow these without exception:

1. ANSWER ONLY FROM PROVIDED CONTEXT.
   Do not use any knowledge from your training data about current prices, recent events,
   or market conditions. If the context does not contain enough information to answer,
   say so explicitly.

2. ALWAYS CITE YOUR SOURCES.
   Every factual claim must reference a [Source N: filename] from the provided context.
   Do not make claims that cannot be traced to a specific source.

3. ADMIT UNCERTAINTY.
   If retrieval context is insufficient, say: "I don't have enough information in the
   provided sources to answer this confidently." Do not guess or extrapolate.

4. NEVER COMPUTE NUMBERS YOURSELF.
   For any P&L, allocation percentage, or financial calculation — these will be provided
   to you via tool results. Do not attempt arithmetic. If a tool result is provided,
   use those exact numbers.

5. USE INDIAN FINANCIAL CONTEXT.
   Amounts are in INR. Use terms like LTCG, STCG, SIP, XIRR correctly as defined
   in the glossary context when relevant.

6. BE CONCISE AND PROFESSIONAL.
   This is a wealth-tech product. Answers should be clear, grounded, and actionable.
   Avoid filler language."""


def _build_gemini_tools() -> List[Tool]:
    """Convert our TOOL_DEFINITIONS into Gemini Tool objects."""
    declarations = []
    for tool_def in TOOL_DEFINITIONS:
        declarations.append(FunctionDeclaration(
            name=tool_def["name"],
            description=tool_def["description"],
            parameters=tool_def["parameters"]
        ))
    return [Tool(function_declarations=declarations)]


def _execute_tool(tool_name: str, tool_args: Dict[str, Any]) -> str:
    """
    Look up and execute a tool from TOOL_REGISTRY.
    Returns a JSON string of the result for sending back to Gemini.

    If the tool name is unknown, returns an error string — the LLM
    will then report inability to compute rather than hallucinating.
    """
    if tool_name not in TOOL_REGISTRY:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    func = TOOL_REGISTRY[tool_name]
    result = func()  # All our tools take no arguments (read from portfolio.json directly)

    # Pydantic model → dict → JSON string
    if hasattr(result, "model_dump"):
        return json.dumps(result.model_dump(), ensure_ascii=False)
    return json.dumps(result, ensure_ascii=False)


def query(user_question: str, k: int = 4) -> Dict[str, Any]:
    """
    Main entry point. Given a user question:
      1. Retrieve relevant context from FAISS
      2. Send to Gemini with tool definitions
      3. If Gemini calls a tool → execute → send result back → get final answer
      4. Parse and return structured response

    Returns a dict with at minimum:
      - 'answer': str
      - 'sources': List[str]
      - 'tool_used': Optional[str]
      - 'raw_data': Optional[dict]  (tool output, if any)
    """
    # Step 1: Retrieve context
    chunks = retrieve(user_question, k=k)
    context = format_context(chunks)
    source_names = [c["source"] for c in chunks]

    # Step 2: Build the prompt
    user_message = f"""Context from portfolio data and market news:

{context}

---

User question: {user_question}

If this question requires financial calculations (P&L, allocation percentages),
use the available tools rather than computing yourself."""

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=SYSTEM_PROMPT,
        tools=_build_gemini_tools()
    )

    # Step 3: First API call
    response = model.generate_content(user_message)
    candidate = response.candidates[0]
    content = candidate.content

    tool_used = None
    raw_data = None

    # Step 4: Check if Gemini wants to call a tool
    function_call = None
    for part in content.parts:
        if hasattr(part, "function_call") and part.function_call.name:
            function_call = part.function_call
            break

    if function_call:
        tool_name = function_call.name
        tool_args = dict(function_call.args) if function_call.args else {}

        # Execute the tool locally — deterministic, not LLM arithmetic
        tool_result_str = _execute_tool(tool_name, tool_args)
        tool_used = tool_name
        raw_data = json.loads(tool_result_str)

        # Step 5: Second API call — send tool result back for narrative generation
        # We start a chat to maintain the tool_call → tool_result conversation structure
        chat = model.start_chat()

        # Replay: first message
        chat.send_message(user_message)

        # Send tool result using Gemini's function response format
        import google.generativeai.types as gtypes
        tool_response = chat.send_message(
            gtypes.content_types.to_contents({
                "role": "function",
                "parts": [{
                    "function_response": {
                        "name": tool_name,
                        "response": {"result": tool_result_str}
                    }
                }]
            })
        )
        final_text = tool_response.text

    else:
        # No tool call — use the direct text response
        final_text = response.text

    return {
        "answer": final_text,
        "sources": source_names,
        "tool_used": tool_used,
        "raw_data": raw_data,
    }