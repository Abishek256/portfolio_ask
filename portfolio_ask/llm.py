"""
llm.py — Groq API integration with native tool calling.

Groq provides free, fast LLM inference. We use the OpenAI-compatible
Groq SDK since Groq's API follows the same interface.

Model: llama-3.3-70b-versatile
This model is explicitly designed for tool calling on Groq's platform.

Tool-calling flow (Variant B):
  Step 1: Send query + context + tool definitions to Groq
  Step 2: If Groq returns a tool_call -> execute the function locally
  Step 3: Send tool result back to Groq -> get final natural language answer

The LLM never computes arithmetic. Tools handle all deterministic computation.
"""

import os
import json
from typing import Any, Dict, List, Optional
from groq import Groq

from portfolio_ask.tools import TOOL_REGISTRY, TOOL_DEFINITIONS
from portfolio_ask.retrieve import retrieve, format_context

MODEL_NAME = "llama-3.3-70b-versatile"

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

7. TRUST TOOL RESULTS COMPLETELY.
   When a tool result is provided, it is sourced directly from the client's own portfolio data.
   Never express doubt about whether tool results apply to the client's portfolio — they always do.

8. BE DIRECT.
   Start answers with the insight itself.
   Do NOT use phrases like:
   - "Based on the provided context"
   - "According to the sources"
   - "The answer is"
   
9. USE TOOLS ONLY WHEN NECESSARY.
   - Use compute_pnl ONLY for portfolio performance, gains/losses, or returns.
   - Use get_sector_allocation ONLY for allocation/exposure questions.
   - DO NOT use tools for news-related or qualitative questions.
   - If the answer can be derived directly from context, DO NOT call a tool.
   
10. DO NOT MIX TOOL OUTPUTS WITH UNRELATED QUESTIONS.
    If the question is about news, do not include portfolio P&L unless explicitly asked."""




def _build_groq_tools() -> List[Dict]:
    """
    Convert TOOL_DEFINITIONS into Groq/OpenAI function calling format.
    Groq follows the same tool schema as OpenAI.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"]
            }
        }
        for tool in TOOL_DEFINITIONS
    ]


def _execute_tool(tool_name: str) -> str:
    """
    Execute a tool from TOOL_REGISTRY.
    Returns a JSON string of the result to send back to Groq.

    If tool name is unknown, returns an error JSON so the LLM
    reports inability to compute rather than hallucinating numbers.
    """
    if tool_name not in TOOL_REGISTRY:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    result = TOOL_REGISTRY[tool_name]()

    if hasattr(result, "model_dump"):
        return json.dumps(result.model_dump(), ensure_ascii=False)
    return json.dumps(result, ensure_ascii=False)


def query(user_question: str, k: int = 4) -> Dict[str, Any]:
    """
    Main entry point. Given a user question:
      1. Retrieve relevant context chunks from FAISS
      2. Send to Groq with tool definitions
      3. If Groq calls a tool -> execute locally -> send result back -> get final answer
      4. Return structured response dict

    Returns:
      {
        'answer': str,
        'sources': List[str],
        'tool_used': Optional[str],
        'raw_data': Optional[dict]
      }
    """
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    # Step 1: Retrieve context
    chunks = retrieve(user_question, k=k)
    context = format_context(chunks)
    # Deduplicate sources while preserving order of first appearance
    seen = set()
    source_names = []
    for c in chunks:
        src = c["source"]
        if src not in seen:
            seen.add(src)
            source_names.append(src)

    user_message = f"""Context from portfolio data and market news:

{context}

---

User question: {user_question}

If this question requires financial calculations (P&L, allocation percentages),
use the available tools rather than computing yourself."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    tools = _build_groq_tools()

    # Step 2: First API call
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    tool_used = None
    raw_data = None

    message = response.choices[0].message

    # Step 3: Check if Groq wants to call a tool
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        tool_name = tool_call.function.name
        tool_result_str = _execute_tool(tool_name)
        tool_used = tool_name
        raw_data = json.loads(tool_result_str)

        # Step 4: Second API call — send tool result back for final narrative
        # Groq requires the assistant message with tool_calls in history
        messages.append(message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result_str
        })

        followup = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages
        )
        final_text = followup.choices[0].message.content
        if not final_text:
            raise RuntimeError(
                f"Groq returned empty text after tool call. "
                f"Tool used: {tool_name}. Full response: {followup}"
            )

    else:
        final_text = message.content
        if not final_text:
            raise RuntimeError(
                f"Groq returned empty text. Full response: {response}"
            )

    return {
        "answer": final_text,
        "sources": source_names,
        "tool_used": tool_used,
        "raw_data": raw_data,
    }