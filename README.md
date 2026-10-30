# RAG Agent (Project 1)

Minimal RAG API: drop documents in `docs/`, ingest them into a local Chroma
vector store, then ask questions against them through a FastAPI endpoint
that grounds its answer with an LLM (Gemini by default; Anthropic supported by setting `LLM_PROVIDER=anthropic`).

This is the **tonight's-step** version — just ingestion + grounded Q&A.
Tool calling and the agent loop come after this works end to end.

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
# Windows PowerShell:  $env:GEMINI_API_KEY="your-key"
# Mac/Linux:           export GEMINI_API_KEY=your-key
```

## 1. Add documents

Put `.txt` or `.md` files into `docs/`. Start with your CV, a project
writeup, whatever you have handy — content doesn't matter for testing
the pipeline, only that it's real text you can ask questions about.

## 2. Ingest

```bash
python -m app.ingest
```

This chunks every file in `docs/`, embeds the chunks (Chroma's built-in
local embedding model — no API key needed for this step), and stores
them in `./chroma_db/`.

## 3. Run the API

```bash
uvicorn app.main:app --reload
```

## 4. Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What companies has this person worked at?"}'
```

You should get back an answer plus the source chunks it was grounded in.

## Evaluating retrieval

```bash
python -m app.evaluate          # default: top 4 chunks per question
python -m app.evaluate --k 6    # try a different k
```

`eval_questions.json` holds test questions with the file (and keyword) each
answer should come from. The script queries ChromaDB the same way `/ask`
does, with no LLM calls, and reports source hit rate, keyword hit rate and
mean reciprocal rank, plus what was retrieved for every failure. Re-run it
after any change to chunking, embeddings, or `TOP_K`. Distances are
lower = more similar.

## Results and known limitations

Measured with `python -m app.evaluate` (k=6, 442 chunks from 6 documents,
15 scored answerable questions, 7 unanswerable ones, and 1 known limitation
tracked separately):

| Metric | Result |
|---|---|
| Source hit rate (right file in top 6) | 15/15 |
| Keyword hit rate (right passage in top 6) | 14/14 |
| Mean reciprocal rank | 1.00 |
| Distance cutoff 1.5: answerable kept / unanswerable refused | 15/15 and 6/7 |

One hard question (the exact queue name, see limitations) is tracked separately
and not scored; counting it, keyword hits are 14/15.

The first run scored 12/15 on keyword hits at k=4. The comparison is not
perfectly like for like: I also added a missing document, fixed a wrong test
expectation, raised k, and moved one hard question out of the score.

**What I learned while building it**

- Chunks lose context when they are cut away from the sentence that names
  their topic. One answer ("what I would change today") failed retrieval until
  each item named its project. How source documents are written matters as much
  as the retrieval code.
- Splitting oversized paragraphs at sentence boundaries improved ranking
  (mean reciprocal rank 0.92 to 1.00) but did not fix the keyword failures by
  itself.
- My first keyword check used substring matching, so "rag" matched "storage".
  It now matches whole words. A diagnosis step in the evaluation tells apart
  "text missing from the index", "chunk ranks too low" and "wrong test".
- Raising k from 4 to 6 recovered two near misses (correct chunks at ranks 5
  and 6).

**Known limitations**

- **Exact names are hard for meaning-based search.** "What is the name of the
  queue that triggers the anomaly detection experiment?" fails: the chunk with
  the queue name ranks 48th, behind general anomaly-detection text. It is kept
  in `eval_questions.json` flagged `known_limitation` and reported separately. Next step:
  hybrid search (keyword scoring combined with embeddings).
- **Near-topic questions that the documents cannot answer can pass the
  cutoff.** "What did Theertha do at Siemens?" has a best distance of 1.39,
  below the 1.5 cutoff, so it reaches the language model. Only one such example
  exists in the test set, so I have not tightened the cutoff.
- **The test set is small (23 questions) and written by the author.** Treat the
  numbers as indicative, not as a benchmark.

## What's next (not tonight)

- `app/tools.py` — a function the LLM can call (e.g. query Postgres, hit an API)
- an agent loop in `main.py` that lets Claude decide to call a tool,
  read the result, and decide again before answering
- a tiny eval script: does the answer actually use the retrieved chunks,
  or did the model ignore them?
- `.github/workflows/ci.yml` to run tests on push (Project 3)
