"""
Deterministic computation tools for Variant B (Tool Use).

These functions are NEVER called by the LLM directly.
The flow is:
  1. LLM receives user query + tool definitions
  2. LLM returns a tool_call indicating which function to invoke
  3. This module executes the function with real data
  4. Result is passed back to the LLM for narrative generation

All functions take the raw holdings list from portfolio.json.
All functions return typed Pydantic objects — never raw dicts or strings.

Known limitation: prices in portfolio.json are static. In production,
current_price would be fetched from a live market data API.
"""

import json
from typing import List, Dict, Any
from portfolio_ask.models import HoldingPnL, PnLSummary, SectorAllocation, AllocationBreakdown


def load_holdings(portfolio_path: str = "data/portfolio.json") -> List[Dict[str, Any]]:
    """Load holdings from portfolio.json. Single source of truth."""
    with open(portfolio_path, "r") as f:
        data = json.load(f)
    return data["holdings"]


def compute_pnl(portfolio_path: str = "data/portfolio.json") -> PnLSummary:
    """
    Compute unrealised P&L for every holding and portfolio totals.

    This is the canonical tool for any query involving:
    - Total portfolio return
    - Which stocks are up/down
    - Absolute gain/loss figures

    The LLM must not attempt to compute these numbers itself.
    """
    holdings = load_holdings(portfolio_path)

    holding_results = []
    total_invested = 0.0
    total_current = 0.0

    for h in holdings:
        qty = h["quantity"]
        avg_cost = h["avg_cost"]
        current_price = h["current_price"]

        invested = round(qty * avg_cost, 2)
        current = round(qty * current_price, 2)
        abs_pnl = round(current - invested, 2)
        pnl_pct = round((abs_pnl / invested) * 100, 2) if invested > 0 else 0.0

        holding_results.append(HoldingPnL(
            ticker=h["ticker"],
            name=h["name"],
            quantity=qty,
            avg_cost=avg_cost,
            current_price=current_price,
            invested_value=invested,
            current_value=current,
            absolute_pnl=abs_pnl,
            pnl_pct=pnl_pct
        ))

        total_invested += invested
        total_current += current

    total_pnl = round(total_current - total_invested, 2)
    total_pnl_pct = round((total_pnl / total_invested) * 100, 2) if total_invested > 0 else 0.0

    winners = [h.ticker for h in holding_results if h.absolute_pnl > 0]
    losers = [h.ticker for h in holding_results if h.absolute_pnl < 0]

    return PnLSummary(
        holdings=holding_results,
        total_invested=round(total_invested, 2),
        total_current_value=round(total_current, 2),
        total_absolute_pnl=total_pnl,
        total_pnl_pct=total_pnl_pct,
        winners=winners,
        losers=losers
    )


def get_sector_allocation(portfolio_path: str = "data/portfolio.json") -> AllocationBreakdown:
    """
    Compute sector-wise allocation as percentage of total portfolio value.

    Used for queries like:
    - 'what is my sector allocation?'
    - 'how much of my portfolio is in IT?'
    - 'am I overexposed to banking?'
    """
    holdings = load_holdings(portfolio_path)

    sector_map: Dict[str, Dict] = {}
    total_value = 0.0

    for h in holdings:
        current_value = h["quantity"] * h["current_price"]
        sector = h["sector"]
        total_value += current_value

        if sector not in sector_map:
            sector_map[sector] = {"value": 0.0, "tickers": []}
        sector_map[sector]["value"] += current_value
        if h["ticker"] not in sector_map[sector]["tickers"]:
            sector_map[sector]["tickers"].append(h["ticker"])

    sectors = []
    for sector_name, data in sorted(sector_map.items(), key=lambda x: -x[1]["value"]):
        weight = round((data["value"] / total_value) * 100, 2) if total_value > 0 else 0.0
        sectors.append(SectorAllocation(
            sector=sector_name,
            weight_pct=weight,
            holdings=data["tickers"]
        ))

    return AllocationBreakdown(
        sectors=sectors,
        total_portfolio_value=round(total_value, 2),
        note="Allocation is based on current market prices. Prices are static in this demo.",
        sources=["data/portfolio.json"]
    )


# Registry: maps tool names (as seen by the LLM) to Python functions.
# When the LLM returns tool_call with name "compute_pnl",
# your dispatcher looks up this registry and calls the function.
TOOL_REGISTRY = {
    "compute_pnl": compute_pnl,
    "get_sector_allocation": get_sector_allocation,
}

# Tool definitions sent to Gemini API so the LLM knows what tools exist.
# These follow the Gemini function calling schema.
TOOL_DEFINITIONS = [
    {
        "name": "compute_pnl",
        "description": (
            "Computes the unrealised profit and loss (P&L) for all holdings in the portfolio. "
            "Use this whenever the user asks about returns, gains, losses, profit, P&L, "
            "portfolio performance, or which stocks are up or down."
        ),
        "parameters": {
            "type": "object",
            "properties": {},  # No inputs needed — reads from portfolio.json directly
            "required": []
        }
    },
    {
        "name": "get_sector_allocation",
        "description": (
            "Computes the sector-wise allocation of the portfolio as a percentage of total value. "
            "Use this whenever the user asks about sector exposure, diversification, "
            "concentration, or allocation breakdown."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]