"""
CLI entry point
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import sys
from dotenv import load_dotenv

load_dotenv()  # loads GROQ_API_KEY

from portfolio_ask.llm import query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default=None)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.question is None:
        try:
            args.question = input("Enter your question: ").strip()
        except:
            print("\nAborted.")
            sys.exit(0)

    if not args.question:
        print("No question provided.")
        sys.exit(1)

    print(f"\nQuestion: {args.question}")
    print("-" * 60)

    result = query(args.question, k=args.k)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print("\n" + result["answer"])

    print("\n" + "-" * 60)
    print("Sources:")
    for s in result["sources"]:
        print(f"  • {s}")

    if result["tool_used"]:
        print(f"\n[Tool used: {result['tool_used']}]")

    print()


if __name__ == "__main__":
    main()