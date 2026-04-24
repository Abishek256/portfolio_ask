"""
run_evals.py — Execute all evaluation cases and report pass/fail.

Failures are reported, never hidden. A failing test with a clear explanation
is more valuable than a passing test that doesn't catch anything real.

Run via: make eval
"""

import yaml
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from portfolio_ask.llm import query


def check_case(case: dict, result: dict) -> tuple[bool, list[str]]:
    """
    Evaluate a single test case against the result.
    Returns (passed: bool, failures: list of failure descriptions).
    """
    failures = []
    answer_lower = result["answer"].lower()
    sources = result.get("sources", [])
    tool_used = result.get("tool_used")

    # Check expected facts appear in answer
    for fact in case.get("expected_facts", []):
        if fact.lower() not in answer_lower:
            failures.append(f"Missing expected fact: '{fact}'")

    # Check required sources were cited
    for required_src in case.get("must_cite", []):
        if not any(required_src in s for s in sources):
            failures.append(f"Required source not cited: '{required_src}'")

    # Check forbidden sources were NOT cited
    for forbidden_src in case.get("must_not_cite", []):
        if any(forbidden_src in s for s in sources):
            failures.append(f"Forbidden source was cited (retrieval error): '{forbidden_src}'")

    # Check tool was called when expected
    if case.get("tool_expected"):
        if tool_used != case["tool_expected"]:
            failures.append(
                f"Expected tool '{case['tool_expected']}' but got '{tool_used}'"
            )

    return len(failures) == 0, failures


def run_evals():
    cases_path = Path(__file__).parent / "cases.yaml"
    with open(cases_path) as f:
        data = yaml.safe_load(f)

    cases = data["cases"]
    total = len(cases)
    passed = 0

    print(f"\nRunning {total} evaluation cases...\n")
    print("=" * 60)

    for case in cases:
        case_id = case["id"]
        query_text = case["query"]
        intent = case.get("intent", "").strip()

        print(f"\n[{case_id}] {query_text}")
        print(f"Intent: {intent[:80]}...")

        result = query(query_text)
        success, failures = check_case(case, result)

        if success:
            passed += 1
            print("  ✅ PASSED")
        else:
            print("  ❌ FAILED")
            for f in failures:
                print(f"     → {f}")

        print(f"  Tool used: {result.get('tool_used') or 'none'}")
        print(f"  Sources: {result.get('sources', [])}")

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} passed")

    if passed < total:
        print(f"\n{total - passed} test(s) failed. See above for details.")
        print("Failures are expected for edge cases — document them in README.")
    else:
        print("\nAll tests passed.")


if __name__ == "__main__":
    run_evals()