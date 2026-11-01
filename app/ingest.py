"""
Run with: python -m app.ingest

Reads every .txt/.md file in docs/, chunks it, and stores the chunks in a
local persistent Chroma collection. Chroma's default embedding function
(all-MiniLM-L6-v2, runs locally via onnxruntime) is used, so this step
needs no API key.
"""

import pathlib

import chromadb

from app.chunking import chunk_text

DOCS_DIR = pathlib.Path(__file__).parent.parent / "docs"
DB_DIR = pathlib.Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "documents"


def load_documents() -> list[tuple[str, str]]:
    """Returns list of (filename, text)."""
    docs = []
    for path in DOCS_DIR.glob("*"):
        if path.suffix.lower() in (".txt", ".md"):
            docs.append((path.name, path.read_text(encoding="utf-8")))
    return docs


def main():
    documents = load_documents()
    if not documents:
        print(f"No .txt/.md files found in {DOCS_DIR}. Add some and re-run.")
        return

    client = chromadb.PersistentClient(path=str(DB_DIR))
    # Fresh collection each run, so re-ingesting doesn't duplicate chunks.
    try:
        client.delete_collection(COLLECTION_NAME)
    # The collection may not exist yet; a real failure will surface at create_collection.
    except Exception:  # noqa: BLE001, S110
        pass
    collection = client.create_collection(COLLECTION_NAME)

    all_ids, all_texts, all_metadatas = [], [], []

    for filename, text in documents:
        chunks = chunk_text(text, source=filename)
        for chunk in chunks:
            all_ids.append(f"{filename}::{chunk.chunk_index}")
            all_texts.append(chunk.text)
            all_metadatas.append({"source": chunk.source, "chunk_index": chunk.chunk_index})

    collection.add(ids=all_ids, documents=all_texts, metadatas=all_metadatas)

    print(f"Ingested {len(all_texts)} chunks from {len(documents)} document(s) into {DB_DIR}")


if __name__ == "__main__":
    main()
