# RAG Agent (Project 1)

![CI](https://github.com/theertha18/rag-agent/actions/workflows/ci.yml/badge.svg)

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
- **`/ask` returns its retrieved chunks as `sources` even when the model says
  the answer is not in them.** The answer text is correct, but the sources list
  then looks like supporting evidence. Planned fix: have the model signal "not
  found" explicitly and return an empty sources list in that case.
- **The test set is small (23 questions) and written by the author.** Treat the
  numbers as indicative, not as a benchmark.

## The agent (tool calling)

`/ask` always does the same thing: search the documents, then answer. The agent
lets the model choose. It is given three tools and a short description of each:

| Tool | Use |
|---|---|
| `search_documents` | explain what was done and how (the RAG search) |
| `find_projects` | exact lists and counts, from a small SQLite database |
| `github_repos` | live data from the GitHub API |

The model never runs anything itself. It replies with a request ("call
`find_projects` with keyword ultrasonic"), the code runs the real function and
sends the result back, and this repeats until the model can answer.

```bash
python -m app.seed_db        # once: builds the project database
python -m app.agent "Which of my projects used ultrasonic sensors?"
python -m app.agent          # three example questions
```

The printout shows every step: which tool the model asked for, what the code
ran, and the final answer. Written in plain Python on purpose, without a
framework, so each part is visible.

**Safety limits**

- The model can only call the listed tools, and only with the arguments their
  descriptions declare. Anything else it sends is dropped.
- The project database is opened read-only and values are passed as query
  parameters, never pasted into SQL.
- A failing tool returns an error message to the model instead of crashing.
- At most 5 rounds of tool use per question. After that the tools are taken
  away and the model must answer with what it already has.

Tests (`python -m unittest tests.test_tools tests.test_agent`) cover the tools,
the argument filtering, the loop and the step limit, using a fake model, so
they run offline.

Plain RAG is enough for most questions about the documents. The agent earns its
extra cost only on questions that need exact lists or live data, or more than
one source.

## Tests and continuous integration

```bash
python -m unittest discover -s tests -t . -v
```

41 offline unit tests cover the tools, the argument filtering, the agent loop
and its step limit. They use a fake model and a temporary database, so they need
no API key and no installed packages beyond the standard library.

Every push and pull request runs the same tests plus a `ruff` lint on GitHub
Actions (`.github/workflows/ci.yml`). The retrieval evaluation is not part of CI
because it needs the private documents.

## What's next

- an `/agent` endpoint in the FastAPI app (today the agent runs from the command line)
- fix: return no `sources` from `/ask` when the model says the answer is not in them
- hybrid search (keyword scoring plus embeddings) for exact names
- Dockerfile, then a Kubernetes deployment
- the same agent rebuilt in LangGraph, for comparison
