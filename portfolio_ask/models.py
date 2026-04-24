"""
Pydantic schemas for structured outputs.

Every LLM response that carries structured data must be typed here.
The LLM is instructed to return JSON matching these schemas.
Validation is enforced at parse time — bad output raises, never silently passes.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class SectorAllocation(BaseModel):
    """Single sector's weight in the portfolio."""
    sector: str = Field(description="Sector name, e.g. 'Information Technology'")
    weight_pct: float = Field(description="Percentage of total portfolio value, e.g. 18.4")
    holdings: List[str] = Field(description="Ticker symbols belonging to this sector")


class AllocationBreakdown(BaseModel):
    """
    Structured response for allocation queries.
    Example query: 'what is my sector allocation?' or 'what is my tech exposure?'
    """
    sectors: List[SectorAllocation]
    total_portfolio_value: float = Field(description="Total current value in INR")
    note: Optional[str] = Field(
        default=None,
        description="Any caveat or observation about the allocation"
    )
    sources: List[str] = Field(description="Source chunks used to construct this answer")


class HoldingPnL(BaseModel):
    """P&L breakdown for a single holding."""
    ticker: str
    name: str
    quantity: int
    avg_cost: float
    current_price: float
    invested_value: float = Field(description="quantity * avg_cost")
    current_value: float = Field(description="quantity * current_price")
    absolute_pnl: float = Field(description="current_value - invested_value, in INR")
    pnl_pct: float = Field(description="(absolute_pnl / invested_value) * 100")


class PnLSummary(BaseModel):
    """
    Structured response for P&L queries.
    Example query: 'what is my total P&L?' or 'which holdings are at a loss?'

    IMPORTANT: This is always populated by the compute_pnl tool, never by the LLM.
    The LLM only provides the narrative summary around these numbers.
    """
    holdings: List[HoldingPnL]
    total_invested: float = Field(description="Total capital deployed in INR")
    total_current_value: float = Field(description="Total portfolio value at current prices")
    total_absolute_pnl: float = Field(description="Total unrealised P&L in INR")
    total_pnl_pct: float = Field(description="Overall portfolio return percentage")
    winners: List[str] = Field(description="Tickers currently in profit")
    losers: List[str] = Field(description="Tickers currently at a loss")


class CitedAnswer(BaseModel):
    """
    General-purpose cited answer for free-text queries that don't
    require structured numerical output.
    Example query: 'what is the latest news about ADANIENT?'
    """
    answer: str = Field(description="The answer in plain English")
    sources: List[str] = Field(description="Document names/chunks used to answer")
    confidence: str = Field(
        description="'high', 'medium', or 'low' based on source relevance"
    )
    caveat: Optional[str] = Field(
        default=None,
        description="Uncertainty or limitation the user should be aware of"
    )