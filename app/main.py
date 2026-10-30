"""
Run with: uvicorn app.main:app --reload

One endpoint: POST /ask {"question": "..."}
Retrieves the most relevant chunks from Chroma, then asks the LLM to answer
using only that context. This is the whole "RAG" part — tool calling and
the agent loop are a deliberate next step, not part of this file.
"""

import pathlib

import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.llm import generate

DB_DIR = pathlib.Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "documents"
TOP_K = 6
# Chunks farther than this from the question are dropped. Chosen from the
# evaluation: answerable questions' best matches were all well below it,
# unanswerable ones well above it. Re-check with `python -m app.evaluate`.
MAX_DISTANCE = 1.5

app = FastAPI(title="RAG Agent")

_client = chromadb.PersistentClient(path=str(DB_DIR))


class AskRequest(BaseModel):
    question: str


class SourceChunk(BaseModel):
    source: str
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


def get_collection():
    try:
        return _client.get_collection(COLLECTION_NAME)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. Run `python -m app.ingest` first.",
        )


def build_prompt(question: str, contexts: list[str]) -> str:
    context_block = "\n\n---\n\n".join(contexts)
    return f"""Answer the question using ONLY the context below. \
If the context doesn't contain the answer, say so explicitly rather than guessing.

Context:
{context_block}

Question: {question}"""


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    collection = get_collection()

    results = collection.query(query_texts=[request.question], n_results=TOP_K)
    kept = [
        (doc, meta)
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
        if dist <= MAX_DISTANCE
    ]

    # Nothing close enough: say so without calling the LLM (saves a call and
    # removes the chance of it making something up).
    if not kept:
        return AskResponse(
            answer="I couldn't find anything relevant to that question in the indexed documents.",
            sources=[],
        )

    documents = [doc for doc, _ in kept]
    metadatas = [meta for _, meta in kept]

    prompt = build_prompt(request.question, documents)

    try:
        answer_text = generate(prompt)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

    sources = [
        SourceChunk(source=meta["source"], text=doc)
        for doc, meta in zip(documents, metadatas)
    ]

    return AskResponse(answer=answer_text, sources=sources)


@app.get("/health")
def health():
    return {"status": "ok"}
