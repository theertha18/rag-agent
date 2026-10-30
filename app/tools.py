"""
Tools the agent will be able to call.

Each tool is a plain Python function that takes simple arguments and returns
plain data (lists and dicts), so it can be tested on its own, with no model
involved. Wiring them to the model (function calling) and the agent loop come
later.

    search_documents  - semantic search over your ingested documents (ChromaDB)
    find_projects     - look up projects in a small SQLite database
    github_repos      - list a user's public repositories via the GitHub API

TOP_K and MAX_DISTANCE mirror app/main.py; keep them in sync until main.py
imports them from here.
"""

import json
import pathlib
import re
import sqlite3
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).parent.parent
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "documents"
DB_PATH = ROOT / "data" / "portfolio.db"

TOP_K = 6
MAX_DISTANCE = 1.5


def search_documents(query: str, k: int = TOP_K, max_distance: float = MAX_DISTANCE,
                     collection=None) -> list[dict]:
    """
    Return up to k chunks from the ingested documents that are close to the
    query, as [{"source", "text", "distance"}] with the closest first.
    Chunks farther than max_distance are dropped, so an empty list means
    "nothing relevant found".

    `collection` can be passed in for testing; by default the real ChromaDB
    collection is opened.
    """
    if collection is None:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(COLLECTION_NAME)

    results = collection.query(query_texts=[query], n_results=k)
    return [
        {"source": meta["source"], "text": doc, "distance": round(dist, 3)}
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
        if dist <= max_distance
    ]


def find_projects(keyword: str = "", limit: int = 10, db_path=DB_PATH) -> list[dict]:
    """
    Look up projects whose name, description or technologies contain the
    keyword as a whole word or phrase (case-insensitive; a plural "s" or "es"
    is allowed, so "microservice" finds "microservices"). Searching "go" finds
    the Go project, not "Django". An empty keyword lists everything. Returns
    [{"name", "kind", "description", "technologies"}].

    The database is opened read-only and the keyword is never part of the SQL;
    matching happens in Python, so a model-supplied value cannot change or
    damage anything.
    """
    db_path = pathlib.Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Project database not found at {db_path}. Run `python -m app.seed_db` first."
        )

    keyword = keyword.strip()
    pattern = (
        re.compile(r"(?<!\w)" + re.escape(keyword) + r"(?:s|es)?(?!\w)", re.IGNORECASE)
        if keyword
        else None
    )

    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT name, kind, description, technologies FROM projects ORDER BY id"
        ).fetchall()
    finally:
        conn.close()

    projects = [dict(row) for row in rows]
    if pattern is not None:
        projects = [
            p for p in projects
            if pattern.search(f"{p['name']} {p['description']} {p['technologies']}")
        ]
    return projects[:limit]


def github_repos(username: str, limit: int = 10, timeout: float = 10.0) -> list[dict]:
    """
    List a user's public, non-fork repositories, most recently updated first,
    as [{"name", "description", "language", "stars", "updated", "url"}].
    Private repositories are not visible through this API. Uses GitHub's
    unauthenticated API (a small hourly rate limit applies).
    """
    url = (
        f"https://api.github.com/users/{urllib.parse.quote(username)}/repos"
        "?per_page=100&sort=updated"
    )
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "rag-agent"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise ValueError(f"GitHub user '{username}' not found.") from e
        if e.code == 403:
            raise RuntimeError("GitHub rate limit reached; try again later.") from e
        raise RuntimeError(f"GitHub returned HTTP {e.code}.") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach GitHub: {e.reason}") from e

    repos = [
        {
            "name": r["name"],
            "description": r.get("description"),
            "language": r.get("language"),
            "stars": r.get("stargazers_count", 0),
            "updated": (r.get("updated_at") or "")[:10],
            "url": r["html_url"],
        }
        for r in data
        if not r.get("fork")
    ]
    return repos[:limit]
