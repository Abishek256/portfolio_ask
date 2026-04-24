"""
__main__.py — CLI entry point.

Usage:
    python -m portfolio_ask "what is my total P&L?"
    python -m portfolio_ask "what is my IT sector exposure?"
    python -m portfolio_ask "what is the latest news about ADANIENT?"

The --k flag controls how many context chunks are retrieved (default: 4).
    python -m portfolio_ask "what is my exposure to banking?" --k 6
"""

import argparse
import json
import sys
from dotenv import load_dotenv

load_dotenv()  # Load GEMINI_API_KEY from .env before any other import

from portfolio_ask.llm import query


def main():
    parser = argparse.ArgumentParser(
        description="Ask natural language questions about your portfolio."
    )
    parser.add_argument("question", type=str, help="Your question about the portfolio")
    parser.add_argument(
        "--k", type=int, default=4,
        help="Number of context chunks to retrieve (default: 4)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON response instead of formatted text"
    )
    args = parser.parse_args()

    print(f"\nQuestion: {args.question}")
    print("─" * 60)

    result = query(args.question, k=args.k)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print(f"\n{result['answer']}")
    print("\n─" * 60)
    print("Sources:")
    for src in result["sources"]:
        print(f"  • {src}")

    if result.get("tool_used"):
        print(f"\n[Tool used: {result['tool_used']}]")


if __name__ == "__main__":
    main()