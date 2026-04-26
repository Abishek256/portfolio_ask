# AI_LOG.md
### Ask Your Portfolio — BYLD Wealth Intern Assignment
**Variant B: Tool Use | Role: AI/ML Engineer Intern**

---

This log is written honestly. It documents what I used AI for, what I accepted, what I rejected, where it misled me, and what I figured out myself. It is not a polished summary written after the fact. It reflects the actual journey.

---

## Tools Used

Claude (claude.ai) was my primary tool throughout the entire assignment. I used it for architecture design, code generation, debugging, data design reasoning, and as a reviewer that challenged my decisions before I built anything. The LLM powering the actual RAG system is Google Gemini (gemini-2.0-flash-lite). For embeddings I used sentence-transformers with BGE-small-en-v1.5 running locally, and FAISS as the vector store.

---

## The Journey — Day by Day

### Day 1 — Understanding Before Building (around 3 hours)

Before writing any code, I spent time actually understanding what the assignment was asking for. My first instinct was to start building immediately. I resisted that.

Reading the assignment carefully, I noticed something that I think most people would miss. The grading breakdown shows that AI_LOG.md and retrieval quality together carry 40% of the weight, which is more than code quality, structured output, and reproducibility combined. That changed how I approached everything.

I chose Variant B (Tool Use) over A and C deliberately. Variant A is mostly text processing and reviewers know it can be scaffolded quickly. Variant C sounds impressive but is genuinely hard to finish cleanly in 12 hours without it sprawling. Variant B has a concrete, verifiable outcome: either the tool call fires correctly or it doesn't. A working implementation is unambiguous proof of competence.

I also spent real time on data design before touching code. This was the right call because the data shapes everything downstream.

I asked Claude to challenge every data decision rather than just generate it. This forced me to understand the reasoning behind choices like why ADANIENT should be included, why 6 irrelevant news articles should exist, and why the glossary matters even though the LLM already knows these terms. The answers to all three are in the Significant Prompts section.

One obstacle I hit on Day 1 was wanting to skip the data design phase and jump straight to the RAG pipeline. Claude pushed back and explained that weak data means weak retrieval means weak answers, and that the weakness stays invisible until evaluation. I stayed in data design for the full first session.

---

### Day 2 — Architecture Design and Core Build (around 4 hours)

With the data designed, I mapped the full pipeline before writing code. Every component was justified before any file was created. Five core architectural decisions were locked: paragraph-level chunking, BGE-small-en-v1.5, FAISS over Chroma, k=4 retrieval, and native tool calling for deterministic computation.

Code was written in a deliberate order. Models first, then tools, then ingest, then retriever, then LLM, then CLI. Types before logic before interface. This prevented circular dependency issues and meant I always knew what I was building toward.

The obstacle on Day 2 was finding bugs in AI-generated code. I reviewed every file line by line before running anything and caught three real issues in ingest.py: a division by zero when invested_value is 0, a buffer overwrite bug in the chunking loop that would silently drop paragraphs, and a missing strip() call after buffer merge. These were caught before the first run. If I had trusted the output blindly, they would have caused subtle failures that are very hard to trace back to their source.

---

### Day 3 — Debugging the API (around 3 hours)

This day was almost entirely debugging the Gemini API integration. I'm documenting this in detail because it illustrates a real lesson about AI-generated code and external dependencies.

Claude generated llm.py using google-generativeai, a package Google deprecated in 2025. Runtime error: model not found on v1beta. I migrated to google-genai, the new SDK, but the new SDK kept routing to v1beta with authentication failures and model 404s persisted.

I then abandoned the SDK entirely and rewrote llm.py using requests to call the REST API directly. This is when I flagged a trade-off that I think matters: raw REST means I now own the response contract. If Google changes the response shape, _extract_text() silently returns an empty string instead of crashing. I pushed back on this and changed the function to raise explicitly with the full response printed when the shape is unexpected. Failures should be loud, not silent.

The /v1/ endpoint doesn't support system_instruction or tools. The /v1beta/ endpoint does, and with the key passed explicitly as a query parameter, authentication worked. The final model issue was that my API key only has access to Gemini 2.x models. I listed available models directly via GET /v1beta/models?key=... and identified gemini-2.0-flash-lite as the correct free-tier model. The right workflow is list-then-use, not assume-then-debug.

The lesson from Day 3 is that AI training data has a cutoff. Library deprecations, model availability changes, and API version differences after that cutoff will not be reflected in generated code. Every external dependency requires independent verification.

---

### Day 4 — Evals, README, and This Log (around 2 hours)

Wrote 5 evaluation cases each designed to catch a different class of failure. One tests tool calling, one tests structured output, one tests retrieval discrimination for ADANIENT, one tests intra-sector comparison between INFY and WIPRO which is hallucination-prone, and one tests that glossary definitions come from glossary.md and not from LLM memory.

One test is designed to expose a known failure. The Tata Steel article shares the "Tata" token with Tata Motors and may be incorrectly retrieved for Tata Motors queries. I documented this rather than hiding it.

---

## Significant Prompts

### Prompt 1 — Data Design and the Label Leakage Decision

I asked Claude to design a realistic portfolio with deliberate edge cases that would stress-test naive retrieval. It produced a structured portfolio with two IT stocks, two auto stocks, two banks, ADANIENT as a controversial holding, mutual funds, and a debt fund.

What I rejected: Claude initially included "Note:" lines at the bottom of irrelevant news articles saying things like "This article is included to test retrieval discrimination." I rejected this immediately. Including such annotations introduces label leakage, meaning the model is explicitly told which documents are irrelevant instead of having to figure this out through retrieval. This masks the core RAG failure mode entirely. A system tested with labeled noise is not a system tested at all. I removed all such annotations.

---

### Prompt 2 — Architecture Before Code

I asked Claude to design the full RAG pipeline and justify every component choice before any code was written. It produced a structured breakdown of all five decisions.

What I added that wasn't in the AI output: k=4 is a starting point, not a fixed truth. It will be tuned based on eval results. Claude presented it as a recommendation. I treated it as a tunable parameter. FAISS was chosen over Chroma not because it has more features but because I can explain every line of it. The assignment warns against using libraries you can't explain. Chroma's abstractions would have been a liability in exactly that situation.

---

### Prompt 3 — BGE Query Prefix

I asked Claude to implement the retriever using BGE-small. It produced retrieve.py with the BGE query prefix applied at retrieval time.

What I verified independently: the BGE model card on HuggingFace explicitly documents that a specific prefix string must be prepended to queries during retrieval but not to documents during ingestion. This is model-specific behavior, not a framework convention. Skipping it measurably degrades retrieval quality. I verified this before accepting the code because it's the kind of detail that is easy to get wrong and very hard to debug afterward.

---

### Prompt 4 — Cross-Platform CLI Fix

This one came from me, not AI. The Makefile read command is bash-specific and fails on Windows. I proposed replacing it with python -m portfolio_ask and handling input inside __main__.py using argparse with nargs="?" and Python's input() as fallback. Claude confirmed it was correct. I caught this because I'm working on Windows and the bash read command doesn't error out in PowerShell, it just silently does nothing. That kind of silent platform failure makes submissions non-reproducible for reviewers on different systems.

---

### Prompt 5 — System Prompt Design

I asked Claude to design the system prompt. It produced six rules covering: answer only from context, cite sources, admit uncertainty, never compute numbers, use Indian financial context, be concise.

What I questioned before accepting: Rule 4 (never compute numbers yourself) only works if the tool-calling flow is correctly implemented. If the LLM ignores the tool and computes arithmetic anyway, the rule does nothing. I verified that the tool definitions are correctly structured so the LLM has a clear alternative to arithmetic before treating this rule as meaningful rather than decorative.

---

### Prompt 6 — Silent Failure Mode Pushback

Claude initially wrote _extract_text() with a bare except that returned an empty string on unexpected response shapes. I pushed back on this directly. Silent empty strings propagate through the system and produce empty answers that look like the system worked. That is worse than a crash because it is invisible. The function was changed to raise RuntimeError with the full response printed if the shape is unexpected or if text is empty. This was my pushback, not a suggestion from the AI.

---

## A Bug My AI Introduced

The buffer overwrite bug in ingest.py was the clearest example. The chunking loop had this:

```python
# AI wrote:
if len(para) < 30:
    buffer = para + " "
    continue
```

The correct version is buffer += para + " " with accumulation instead of assignment. The bug would silently discard every short paragraph in a consecutive run except the last one. The system would still run and produce output. There would be no crash. Chunks would just have missing content, retrieval quality would be degraded, and tracing it back to this specific line would be very hard.

I caught it by reading the code line by line before running anything. Not by running it and seeing what happened.

---

## A Design Choice I Made Against AI Suggestion

Keeping ADANIENT in the portfolio. There was an implicit assumption in the AI's framing that safer, less controversial holdings make the system easier to build and demonstrate. ADANIENT generates high-noise, negative-sentiment news, has governance questions, and a volatile price history.

I kept it deliberately. A wealth-tech system that only handles clean, uncontroversial holdings is not really a wealth-tech system. Real HNI portfolios contain high-risk, high-controversy positions. If the system cannot surface risk clearly for ADANIENT while avoiding confusion with unrelated Adani Group entities, that is a real retrieval problem worth knowing about.

The eval case for ADANIENT specifically checks that the credit rating article is retrieved and that the Tata Steel article is not cited. That test exists because I kept ADANIENT. A safer portfolio would have produced a less meaningful test.

---

## Time Split

Reading the assignment and understanding the problem properly took about 45 minutes and was done before any code. Data design took around 1.5 hours, more than planned but worth it. Architecture design on paper took about an hour with no code written during that phase. The core build across all Python files took around 3 hours. Line-by-line bug review took 45 minutes and caught 3 bugs before the first run. API debugging consumed around 3 hours and was entirely unplanned. Writing the eval harness took 45 minutes. README and this log took about 1.5 hours. Total came to just over 12 hours.

Why I hit the ceiling: the API debugging consumed 3 hours that were not budgeted. If I were starting over, I would validate the LLM API connection with a minimal 5-line test script on Day 1 before building anything around it. That is the lesson.

---

## What I Would Do With 2 More Days

Add re-ranking: retrieve k=8, then use a cross-encoder to re-rank to k=4. This would improve precision for queries where semantic similarity alone is not enough.

Add live price fetching to replace static current_price values in portfolio.json. P&L numbers are only as current as the last manual update right now.

Add conversation memory so follow-up questions like "what about its debt levels?" can reference the previous turn.

Investigate the Tata Steel retrieval failure more carefully. The shared "Tata" token between Tata Motors and Tata Steel articles is a name-confusion problem. I would experiment with ticker-level metadata filtering as a pre-retrieval step to eliminate this class of error before it reaches the embedding comparison.
