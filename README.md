# Ask Your Portfolio

A Python CLI tool that answers natural language questions about an HNI investment portfolio using a RAG pipeline with native tool calling.

**Variant B — Tool Use | Abishek M**

---

## What This Does

```bash
python -m portfolio_ask "what is my total P&L?"
python -m portfolio_ask "what is the latest news about ADANIENT?"
python -m portfolio_ask "what is my sector allocation?"
python -m portfolio_ask "what is XIRR and how does it differ from CAGR?"
```

Each query retrieves relevant chunks from an embedded corpus of portfolio data, market news, and a financial glossary. Numerical queries trigger deterministic Python tools via the LLM's native function calling API — the LLM never computes arithmetic itself. All answers are cited by source.

---

## Setup

```bash
git clone https://github.com/Abishek256/portfolio_ask
cd portfolio_ask

pip install -r requirements.txt

cp .env.example .env
# Add your Groq API key to .env
# Get a free key at: console.groq.com

python -m portfolio_ask.ingest
```

After setup, run any query:

```bash
python -m portfolio_ask "your question here"
```

To run the evaluation harness:

```bash
python evals/run_evals.py
```

---

## Architecture

```
[Data Files] → [Chunker] → [BGE Embedder] → [FAISS Index]
                                                   ↑
[User Query] → [BGE Embedder] → [FAISS Search] ───┘
                                      ↓
                               [Top-4 Chunks]
                                      ↓
                         [LLM + System Prompt]
                              ↑           ↓
                        [Tool Call]   [Cited Answer]
                        (Variant B)
```

**Embedding model:** `BAAI/bge-small-en-v1.5` — retrieval-optimized, not just similarity-optimized. Chosen over MiniLM because BGE is purpose-built for query-to-document matching, not general sentence similarity.

**Vector store:** FAISS FlatIP with normalised embeddings (cosine similarity). Chosen over Chroma because every line of the retrieval pipeline is explicit and explainable — no hidden abstractions.

**Chunking:** Paragraph-level. Each news article paragraph is one retrievable unit. Portfolio holdings are converted to natural language sentences before embedding — raw JSON fields embed poorly compared to readable text.

**Top-k:** k=4 as a starting point. Tunable via `--k` flag based on query type.

**LLM:** `llama-3.3-70b-versatile` via Groq API. Tool calling via native function calling — no regex, no prompt engineering workarounds.

---

## Tool Calling (Variant B)

Two tools are registered and available to the LLM:

**`compute_pnl`** — calculates unrealised P&L for every holding from `portfolio.json`. Returns total invested, current value, absolute P&L, percentage return, and lists of winners and losers.

**`get_sector_allocation`** — calculates sector weights as a percentage of total portfolio value.

The flow for a tool query:
1. LLM receives the question and tool definitions
2. LLM returns a `tool_call` — not an answer
3. The Python function executes deterministically
4. Result is sent back to the LLM
5. LLM generates the final answer using the real numbers

The LLM never computes the numbers. It only decides when to call a tool and how to narrate the result.

---

## Data Design

**Portfolio:** 15 holdings across equity, mutual funds, and a debt fund. Includes two IT stocks (INFY, WIPRO), two banks (HDFCBANK, ICICIBANK), two auto stocks (TATAMOTORS, MARUTI), one high-controversy holding (ADANIENT), and one gilt fund (SBI Magnum). The diversity is deliberate — it forces the retrieval system to discriminate between similar entities rather than pattern-matching on sector alone.

**News:** 20 articles — 14 relevant to portfolio holdings and 6 deliberately irrelevant (Zomato, Bajaj Finance, ITC, Tata Steel, gold prices, US Fed). The irrelevant articles carry no annotations. They are genuine retrieval traps. A system that surfaces them for portfolio queries has a real precision problem that would otherwise stay hidden.

**Glossary:** 20 Indian wealth-tech terms (XIRR, LTCG, STCG, ELSS, Gilt, Duration Risk, etc.) with Indian regulatory context. Included so domain definitions are retrieved from a controlled, citable source rather than relying on implicit LLM knowledge — this keeps answers consistent and traceable.

---

## Hallucination in This System

In a RAG pipeline, hallucination shows up in two ways that are easy to confuse.

The first is retrieval hallucination. The retriever pulls an irrelevant chunk, the LLM treats it as valid context, and produces a confident but wrong answer. The answer looks grounded because it has a citation — but the citation is wrong. This is the harder kind to catch because there is no obvious error signal. The system appears to be working.

The second is generation hallucination. The LLM ignores retrieved context and answers from its training data instead. The system prompt addresses this directly — answer only from provided context, cite every claim, admit uncertainty when context is weak.

Three things in this system reduce hallucination. The system prompt enforces source citation and uncertainty admission. Tools replace LLM arithmetic entirely so numbers are never generated, only narrated. The eval harness includes `must_not_cite` checks that catch retrieval hallucination when irrelevant sources appear in answers.

It does not eliminate hallucination. A confident model can override system prompt instructions. At k=4, the retriever can miss relevant chunks. BGE can assign high similarity to semantically adjacent but factually different content. These are real limitations and they are documented, not papered over.

---

## Evaluation Results

```
tc001 — Total P&L query              ✅ PASSED  (tool: compute_pnl)
tc002 — Sector allocation            ✅ PASSED  (tool: get_sector_allocation)
tc003 — ADANIENT news + risks        ✅ PASSED  (retrieval: correct article, no false citations)
tc004 — INFY vs WIPRO comparison     ❌ FAILED  (known retrieval limitation — documented)
tc005 — XIRR vs CAGR glossary        ✅ PASSED  (source: glossary.md)

Results: 4/5
```

**tc004 failure:** At k=4, portfolio position chunks consume the retrieval budget before the INFY news article is reached. The LLM answers from P&L data instead of quarterly performance news. This is a genuine system limitation, not a test design problem. The fix is a query rewriter that adjusts retrieval strategy for comparison queries — higher k, or a news-priority filter. Documented clearly rather than hidden.

---

## Known Limitations

**Static prices:** `current_price` in `portfolio.json` is manually set. P&L figures are accurate to these values, not live market data. In production this would be replaced by a live NSE/BSE feed.

**No conversation memory:** Every query is independent. Follow-up questions like "what about its debt levels?" have no context from the previous turn.

**k=4 retrieval budget:** For comparison queries involving two entities, position chunks can crowd out news articles. Documented in tc004.

**Groq tool calling stability:** `llama-3.3-70b-versatile` occasionally invokes compute_pnl on queries involving specific holdings even when the query is about news. The answers are usually more useful with position data included, so this was accepted rather than suppressed — but it is a real behaviour worth knowing about.

---

## What I'd Do With 2 More Days

The honest version of this section isn't a wish list — it's the things I know are broken or missing and exactly how I'd fix them.

The tc004 failure would be the first thing I'd address. The fix is re-ranking: retrieve k=8 chunks, then use a cross-encoder to score them against the query and keep the top 4. This separates retrieval recall from retrieval precision and would surface the INFY article that currently gets pushed out by position chunks.

After that, live price fetching. The P&L numbers right now are only as accurate as the last time someone updated portfolio.json manually. Wiring up a Yahoo Finance or NSE API call inside compute_pnl is not a big change architecturally — the tool interface stays the same, only the data source changes.

Conversation memory would make the tool genuinely useful for real back-and-forth analysis. Right now every question starts from zero. Adding a rolling message history so "what about its margins?" works as a follow-up would change the experience significantly.

A query rewriter as a preprocessing step is the thing that would improve retrieval quality the most. Identifying whether a query is asking for news, position data, a definition, or a comparison — and then adjusting k and source filters accordingly — would fix the class of failures that tc004 represents.

Finally, ticker-level metadata filtering to handle name-confusion cases like Tata Steel vs Tata Motors. The shared "Tata" token causes the wrong article to score highly. Filtering by ticker before the embedding search would eliminate this entirely.

---

## Why I Went Over the Time Budget

The assignment scopes for 8 to 12 hours. I came in around 15.

The extra time was almost entirely the API provider chain. Gemini had zero free tier quota allocated — not exhausted, just never provisioned, so waiting for a reset would have done nothing. OpenAI required prepaid credits. Groq's first recommended model had been decommissioned two months earlier.

Each of those required diagnosing the error, researching the current state of the provider, making the switch, and verifying it worked. That chain consumed around 5 hours instead of the planned 2.

The fix for next time is simple and I know exactly what it is: spend 30 minutes on Day 1 running a 5-line test script that calls the LLM API before writing anything that depends on it. If I had done that, I would have found the Gemini quota issue on Day 1 and been on Groq by the end of that session. Every other phase stayed within budget.

---

## Project Structure

```
portfolio_ask/
├── portfolio_ask/
│   ├── __init__.py
│   ├── __main__.py      # CLI entry point
│   ├── ingest.py        # Chunking + embedding + FAISS index builder
│   ├── retrieve.py      # Semantic search
│   ├── llm.py           # Groq API + tool calling
│   ├── models.py        # Pydantic schemas
│   └── tools.py         # compute_pnl, get_sector_allocation
├── data/
│   ├── portfolio.json   # 15 HNI holdings
│   ├── glossary.md      # 20 wealth-tech terms
│   └── news/            # 20 market news articles (14 relevant, 6 traps)
├── evals/
│   ├── cases.yaml       # 5 test cases
│   └── run_evals.py     # Eval runner
├── .env.example
├── .gitignore
├── AI_LOG.md
├── Makefile
└── requirements.txt
```
