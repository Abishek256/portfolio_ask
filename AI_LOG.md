# AI_LOG.md
### Ask Your Portfolio — BYLD Wealth Intern Assignment
**Variant B: Tool Use | Role: AI/ML Engineer Intern**

---

This log is written honestly. It documents what I used AI for, what I accepted, what I rejected, where it misled me, and what I figured out myself. It is not a polished summary written after the fact. It reflects the actual journey.

---

## Tools Used

Claude (claude.ai) was my primary tool throughout the entire assignment. I used it for architecture design, code generation, debugging, data design reasoning, and as a reviewer that challenged my decisions before I built anything. The LLM powering the actual RAG system is Groq (llama-3.3-70b-versatile), arrived at after switching through Gemini and OpenAI due to quota and billing issues documented below. For embeddings I used sentence-transformers with BGE-small-en-v1.5 running locally, and FAISS as the vector store.

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

### Day 3 — Debugging the API (around 5 hours, not the planned 3)

This day started with the Gemini integration and ended with Groq working end to end. It was the most chaotic day and I'm documenting it in full because it contains the most honest lessons.

**Gemini — SDK deprecated, then quota zero**

Claude generated llm.py using google-generativeai, a package Google deprecated in 2025. Runtime error: model not found on v1beta. I migrated to google-genai, the new SDK, but the new SDK kept routing to v1beta with authentication failures and model 404s persisted.

I then abandoned the SDK entirely and rewrote llm.py using requests to call the REST API directly. This is when I flagged a trade-off that I think matters: raw REST means I now own the response contract. If Google changes the response shape, _extract_text() silently returns an empty string instead of crashing. I pushed back on this and changed the function to raise explicitly with the full response printed when the shape is unexpected. Failures should be loud, not silent.

After all that, the API returned 429 RESOURCE_EXHAUSTED with quota = 0. Not exhausted — zero allocated. Reset would not have fixed it. Enabling billing on the Google Cloud project also failed because the account had no active billing method that attached correctly.

**OpenAI — insufficient quota**

Switched to OpenAI (gpt-4o-mini). Got insufficient_quota because new OpenAI accounts require prepaid credits even for minimal usage. No free tier that works out of the box for new accounts.

**Groq — final resolution**

Switched to Groq API. Free access, OpenAI-compatible interface, strong tool calling support. The switch required changing one import, one client instantiation, one model name, and one environment variable. No architectural changes. The fact that the system switched providers three times with minimal code changes is a direct result of the design decision to keep the LLM layer thin and replaceable.

First model tried on Groq was llama3-groq-70b-8192-tool-use-preview, which had been decommissioned on January 6, 2025. Checked Groq's official deprecation page and switched to llama-3.3-70b-versatile which is the documented replacement.

The lesson from Day 3 is that AI training data has a cutoff. Library deprecations, model availability changes, API version differences, and provider quota policies after that cutoff will not be reflected in generated code or suggestions. Every external dependency requires independent verification against current documentation.

---

### Day 4 — Integration Testing, Output Polish, Evals, README, and This Log (around 4 hours)

Once Groq was working, the first end to end run confirmed tool calling was firing correctly. compute_pnl triggered on the P&L query and returned the real number. The LLM used the tool result rather than computing arithmetic itself. That was the moment Variant B was actually working.

But the output had problems. Sources were duplicating — the same file appearing three times because multiple chunks from it were retrieved. The LLM was using generic opening phrases like "The latest news about ADANIENT is that..." instead of leading with the fact. The separator line was printing each character on a new line due to Windows encoding handling the Unicode dash character incorrectly.

Fixed all three. Source deduplication using order-preserving set logic. Added Rule 8 to the system prompt to eliminate generic phrasing. Replaced Unicode separator with ASCII.

Then found that compute_pnl was firing on news queries — the tool description wasn't specific enough about when not to use it. Tightened the description with an explicit "Do NOT use this for news queries." Tool routing improved but llama-3.3-70b-versatile still occasionally reaches for compute_pnl when it detects portfolio position context. Accepted this as a model behaviour since the answers were actually more useful with the position data included.

Wrote 5 evaluation cases each designed to catch a different class of failure. One tests tool calling, one tests structured output, one tests retrieval discrimination for ADANIENT, one tests intra-sector comparison between INFY and WIPRO, and one tests that glossary definitions come from glossary.md and not from LLM memory.

The eval harness itself went through two rounds of fixing. First run was 1/5 because expected_facts strings were too literal — checking for exact words the LLM paraphrases rather than for computed values and concepts. Fixed the test cases to check for things like the actual P&L number rather than specific phrasing. Second run was 4/5. The one remaining failure (tc004, INFY vs WIPRO comparison) is a genuine retrieval limitation documented below.

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

### Prompt 7 — Tool Routing Accuracy

After the first successful run I noticed compute_pnl firing on news queries. The LLM was reaching for the tool whenever it detected portfolio position context, even when the query was purely about news.

I strengthened the system prompt with two additions: an explicit instruction to trust tool results completely when they are provided, and a rule to never use tools unless the query genuinely requires computation. I also tightened the tool descriptions in TOOL_DEFINITIONS to make the intended scope clearer. The description for compute_pnl now explicitly says "Do NOT use this for news queries, sector questions, or general portfolio questions."

The improvement was real but not complete. llama-3.3-70b-versatile still occasionally reaches for compute_pnl on queries involving specific portfolio holdings because it interprets "tell me about ADANIENT" as implicitly requiring position data. I decided to accept this rather than fight it further because the answers with position data are genuinely more useful. Documented as intended behaviour.

The lesson here is that tool calling reliability depends heavily on prompt clarity and tool description specificity, not just model capability. The model is not wrong for wanting to provide position context — the description wasn't precise enough about when it shouldn't.

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

## Known Failures I Documented Instead of Hiding

**tc004 — INFY vs WIPRO comparison**

At k=4, portfolio position chunks for both stocks consume most of the retrieval budget. The WIPRO news article makes it in but the INFY news article does not. The LLM answers from position data (P&L figures) rather than quarterly performance news, which is what the query was actually asking about.

Root cause: short comparison queries have stronger semantic overlap with portfolio chunks than with news articles. The fix would be a query rewriter that identifies comparison queries and adjusts retrieval strategy — higher k, or a filter that prioritises news chunks for performance questions.

I left this as a documented failure in the eval harness. A 4/5 with one clearly explained failure is more honest and more impressive than 5/5 with tests designed to always pass.

**Source deduplication was missing initially**

The first version of the output was printing the same source file three times when multiple chunks from it were retrieved. This looked sloppy and was a real output quality issue. Fixed with order-preserving deduplication but the fact that it shipped initially is worth noting.

---

## Time Split

Reading the assignment and understanding the problem properly took about 45 minutes and was done before any code. Data design took around 1.5 hours, more than planned but worth it. Architecture design on paper took about an hour with no code written during that phase. The core build across all Python files took around 3 hours. Line-by-line bug review took 45 minutes and caught 3 bugs before the first run. API debugging consumed around 5 hours across Gemini, OpenAI, and Groq — entirely unplanned. Integration testing, output polish, and eval harness took about 2 hours. README and this log took about 1.5 hours. Total came to around 15 hours.

Why I went over: the API provider chain was the main culprit. Gemini had zero free tier quota allocated, OpenAI required prepaid credits, and Groq's recommended model had been decommissioned. Each switch required diagnosis, research, and implementation. If I were starting over, I would spend 30 minutes on Day 1 testing the LLM connection with a 5-line script before building anything around it. That alone would have saved 2 to 3 hours.

---

## What I Would Do With 2 More Days

Add re-ranking: retrieve k=8, then use a cross-encoder to re-rank to k=4. This would improve precision for queries where semantic similarity alone is not enough, and would directly fix the tc004 failure.

Add live price fetching to replace static current_price values in portfolio.json. P&L numbers are only as current as the last manual update right now.

Add conversation memory so follow-up questions like "what about its debt levels?" can reference the previous turn.

Add a query rewriter as a lightweight preprocessing step that identifies query type — news, position, definition, comparison — and adjusts retrieval strategy accordingly. Different k values and source filters per type would solve the intra-sector comparison failure cleanly.

Investigate the Tata Steel retrieval failure more carefully. The shared "Tata" token between Tata Motors and Tata Steel articles is a name-confusion problem. Ticker-level metadata filtering as a pre-retrieval step would eliminate this class of error before it reaches the embedding comparison.
