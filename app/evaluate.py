"""
Retrieval evaluation.  Run with:  python -m app.evaluate [--k 6]

For each question in eval_questions.json, query ChromaDB exactly like /ask
does and check:
  1. Source hit   - did a chunk from an expected file come back in the top k?
  2. Keyword hit  - did one of those chunks actually contain an expected
                    keyword? (stricter: right passage, not just right file)
  3. Rank         - how high was the first correct chunk? (1 = best)

No LLM calls and no API key needed, so it runs in a few seconds and can be
re-run after every change to chunking, embeddings, or TOP_K.

Distances: Chroma returns a distance per chunk. LOWER = MORE SIMILAR.
Questions marked "unanswerable" are not scored; their distances are printed
so you can see whether a similarity cutoff could separate good matches from
irrelevant ones.
"""

import argparse
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).parent.parent
QUESTIONS_FILE = ROOT / "eval_questions.json"
DB_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "documents"


def contains_keyword(text: str, keyword: str) -> bool:
    """
    Whole-word, case-insensitive match. Plain substring matching is wrong for
    short keywords: 'rag' would match 'storage', 'average' and 'leverage'.
    """
    pattern = r"(?<!\w)" + re.escape(keyword) + r"(?!\w)"
    return re.search(pattern, text, re.IGNORECASE) is not None


def score_question(question: dict, retrieved: list[dict]) -> dict:
    """
    retrieved: chunks in rank order, each {"source", "text", "distance"}.
    Returns the first rank (1-based) where an expected source / keyword
    was found, or None.
    """
    expected = set(question.get("expected_sources", []))
    keywords = question.get("keywords", [])

    source_rank = None
    keyword_rank = None

    for rank, chunk in enumerate(retrieved, start=1):
        from_expected_file = chunk["source"] in expected

        if from_expected_file and source_rank is None:
            source_rank = rank

        if from_expected_file and keywords and keyword_rank is None:
            if any(contains_keyword(chunk["text"], k) for k in keywords):
                keyword_rank = rank

    return {"source_rank": source_rank, "keyword_rank": keyword_rank}


def retrieve(collection, query: str, k: int) -> list[dict]:
    results = collection.query(query_texts=[query], n_results=k)
    return [
        {"source": meta["source"], "text": doc, "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def diagnose(question: dict, all_chunks: list[dict], deep: list[dict], k: int) -> list[str]:
    """
    Explain WHY a keyword check failed:
      - keyword nowhere in the index  -> missing from docs/ or wrong keyword
      - keyword in the index, but the chunk ranks low -> a retrieval problem
    all_chunks: every chunk in the index. deep: top-N retrieval for the question.
    """
    keywords = question.get("keywords", [])
    expected = set(question.get("expected_sources", []))
    if not keywords:
        return ["no keywords set for this question, nothing to diagnose"]

    holders = [c for c in all_chunks if any(contains_keyword(c["text"], kw) for kw in keywords)]
    if not holders:
        return [
            "NO chunk in the whole index contains an expected keyword.",
            "-> the text is missing from docs/, or the keyword is wrong. "
            "Check with Select-String before touching the retrieval code.",
        ]

    counts: dict[str, int] = {}
    for c in holders:
        counts[c["source"]] = counts.get(c["source"], 0) + 1
    lines = ["keyword appears in: " + ", ".join(f"{s} ({n})" for s, n in sorted(counts.items()))]

    if not any(c["source"] in expected for c in holders):
        lines.append("-> but not in any expected file: maybe expected_sources is wrong.")
        return lines

    for rank, c in enumerate(deep, start=1):
        if c["source"] in expected and any(contains_keyword(c["text"], kw) for kw in keywords):
            lines.append(
                f"closest correct chunk is rank {rank} of {len(deep)} (dist {c['distance']:.3f}); "
                f"the rank-{k} chunk is at dist {deep[k - 1]['distance']:.3f}"
            )
            break
    else:
        lines.append(f"-> no correct chunk anywhere in the top {len(deep)}: the question "
                     "and the chunk are too far apart for this embedding model.")
    return lines


def preview(text: str, n: int = 110) -> str:
    return " ".join(text.split())[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=6,
                        help="chunks retrieved per question (keep equal to TOP_K in app/main.py)")
    parser.add_argument("--deep", type=int, default=50, help="depth used when diagnosing failures")
    parser.add_argument("--max-dist", type=float, default=1.5,
                        help="similarity cutoff to evaluate (chunks farther than this are dropped)")
    args = parser.parse_args()
    k = args.k

    if not QUESTIONS_FILE.exists():
        print(f"Missing {QUESTIONS_FILE}")
        return

    import chromadb

    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        print("No ingested documents. Run `python -m app.ingest` first.")
        return

    known = [q for q in questions if q.get("known_limitation")]
    answerable = [q for q in questions
                  if not q.get("unanswerable") and not q.get("known_limitation")]
    unanswerable = [q for q in questions if q.get("unanswerable")]

    print(f"\nRetrieval evaluation (k={k}): {len(answerable)} scored, "
          f"{len(known)} known limitation(s) tracked separately, "
          f"{len(unanswerable)} unanswerable\n")
    print(f"{'#':<3} {'result':<6} {'src':<4} {'kw':<4} {'best dist':<10} question")
    print("-" * 90)

    source_hits = 0
    keyword_hits = 0
    keyword_total = 0
    reciprocal_ranks = []
    answerable_best = []
    failures = []

    for i, q in enumerate(answerable, start=1):
        retrieved = retrieve(collection, q["question"], k)
        score = score_question(q, retrieved)
        best = retrieved[0]["distance"] if retrieved else float("nan")
        answerable_best.append(best)

        src_ok = score["source_rank"] is not None
        has_kw = bool(q.get("keywords"))
        kw_ok = score["keyword_rank"] is not None

        if src_ok:
            source_hits += 1
            reciprocal_ranks.append(1 / score["source_rank"])
        else:
            reciprocal_ranks.append(0.0)

        if has_kw:
            keyword_total += 1
            if kw_ok:
                keyword_hits += 1

        # A question passes if the right file came back, and (when keywords
        # are given) the right passage did too.
        passed = src_ok and (kw_ok or not has_kw)
        if not passed:
            failures.append((i, q, retrieved))

        src_col = str(score["source_rank"]) if src_ok else "-"
        kw_col = str(score["keyword_rank"]) if kw_ok else ("-" if has_kw else "n/a")
        print(f"{i:<3} {'PASS' if passed else 'FAIL':<6} {src_col:<4} {kw_col:<4} "
              f"{best:<10.3f} {q['question'][:60]}")

    unanswerable_best = []
    if unanswerable:
        print("\nUnanswerable questions (not scored, distances shown for calibration):")
        for q in unanswerable:
            retrieved = retrieve(collection, q["question"], k)
            best = retrieved[0]["distance"] if retrieved else float("nan")
            unanswerable_best.append(best)
            verdict = "refused" if best > args.max_dist else "ANSWERED (bad)"
            print(f"    best dist {best:<8.3f} {verdict:<15} {q['question'][:55]}")

    if known:
        stored_all = collection.get()
        known_chunks = [
            {"source": m["source"], "text": d}
            for d, m in zip(stored_all["documents"], stored_all["metadatas"])
        ]
        print("\nKnown limitations (tracked, not scored):")
        for q in known:
            retrieved = retrieve(collection, q["question"], k)
            score = score_question(q, retrieved)
            ok = score["source_rank"] is not None and (
                score["keyword_rank"] is not None or not q.get("keywords"))
            if ok:
                print(f"    NOW PASSES - consider moving it into the scored set: {q['question'][:60]}")
            else:
                print(f"    still fails (expected): {q['question'][:70]}")
                deep = retrieve(collection, q["question"], min(args.deep, collection.count()))
                for line in diagnose(q, known_chunks, deep, k):
                    print(f"        {line}")

    n = len(answerable)
    print("\n" + "=" * 90)
    print(f"Source hit rate @{k}:  {source_hits}/{n}")
    if keyword_total:
        print(f"Keyword hit rate @{k}: {keyword_hits}/{keyword_total}  "
              f"(right passage, not just right file)")
    print(f"Mean reciprocal rank:  {sum(reciprocal_ranks) / n:.2f}  (1.00 = correct chunk always first)")
    if answerable_best:
        print(f"Avg best distance, answerable:   {sum(answerable_best) / len(answerable_best):.3f}")
    if unanswerable_best:
        print(f"Avg best distance, unanswerable: {sum(unanswerable_best) / len(unanswerable_best):.3f}")
        print("  (If these two are close, a distance cutoff will not separate them well.)")

    if answerable_best or unanswerable_best:
        kept = sum(1 for b in answerable_best if b <= args.max_dist)
        refused = sum(1 for b in unanswerable_best if b > args.max_dist)
        print(f"\nSimilarity cutoff {args.max_dist}: "
              f"answerable questions kept {kept}/{len(answerable_best)}, "
              f"unanswerable questions refused {refused}/{len(unanswerable_best)}")
        if answerable_best:
            print(f"  worst (largest) best-distance among answerable: {max(answerable_best):.3f}")
        if unanswerable_best:
            print(f"  closest (smallest) best-distance among unanswerable: {min(unanswerable_best):.3f}")

    if failures:
        print("\n" + "=" * 90)
        print("FAILURES - what was expected vs what came back\n")
        stored = collection.get()
        all_chunks = [
            {"source": m["source"], "text": d}
            for d, m in zip(stored["documents"], stored["metadatas"])
        ]
        for i, q, retrieved in failures:
            print(f"#{i}  {q['question']}")
            print(f"    expected files:    {q.get('expected_sources')}")
            if q.get("keywords"):
                print(f"    expected keywords: {q['keywords']}")
            for rank, c in enumerate(retrieved, start=1):
                print(f"    {rank}. {c['source']}  (dist {c['distance']:.3f})")
                print(f"       {preview(c['text'])}")
            deep = retrieve(collection, q["question"], min(args.deep, collection.count()))
            print("    DIAGNOSIS:")
            for line in diagnose(q, all_chunks, deep, k):
                print(f"      {line}")
            print()


if __name__ == "__main__":
    main()
