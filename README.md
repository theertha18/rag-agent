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

## What's next (not tonight)

- `app/tools.py` — a function the LLM can call (e.g. query Postgres, hit an API)
- an agent loop in `main.py` that lets Claude decide to call a tool,
  read the result, and decide again before answering
- a tiny eval script: does the answer actually use the retrieved chunks,
  or did the model ignore them?
- `.github/workflows/ci.yml` to run tests on push (Project 3)
