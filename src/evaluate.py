import json
import statistics
import time
from pathlib import Path

from rerank_search import RerankSearcher

QUESTIONS_PATH = Path("data/eval/questions.jsonl")
RESULTS_PATH = Path("data/eval/results.md")
DETAILS_PATH = Path("data/eval/details.jsonl")
K_HIT = 5      # hit-rate@5: is the right card in the top 5?
K_MRR = 10     # MRR@10: how high is it, looking at the top 10?


def find_rank(results, gold_id):
    """Position of the correct card in the results, or None if it's not there."""
    for r in results:
        if r["chunk_id"] == gold_id:
            return r["rank"]
    return None


def evaluate(name, search_fn, questions):
    """Ask one system all the questions and score it."""
    search_fn("warm up question")                  # not timed, not scored

    ranks, times = [], []
    for q in questions:
        start = time.perf_counter()
        results = search_fn(q["question"])
        times.append((time.perf_counter() - start) * 1000)   # milliseconds
        ranks.append(find_rank(results, q["chunk_id"]))

    n = len(questions)
    hits = sum(1 for r in ranks if r is not None and r <= K_HIT)
    mrr = sum(1 / r for r in ranks if r is not None) / n

    return {
        "system": name,
        "hit_rate": hits / n,
        "mrr": mrr,
        "avg_ms": statistics.mean(times),
        "ranks": ranks,
    }


def main():
    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]
    print(f"Loaded {len(questions)} questions\n")

    searcher = RerankSearcher()       # loads every model once
    hybrid = searcher.hybrid          # reuse the parts inside it
    vector = hybrid.vector
    bm25 = hybrid.bm25

    systems = [
        ("Vector only",               lambda q: vector.search(q, k=K_MRR)),
        ("BM25 only",                 lambda q: bm25.search(q, k=K_MRR)),
        ("Hybrid (RRF)",              lambda q: hybrid.search(q, k=K_MRR)),
        ("Hybrid + rerank",           lambda q: searcher.search(q, k=K_MRR)),
    ]

    scores = []
    for name, search_fn in systems:
        print(f"Evaluating: {name}...")
        scores.append(evaluate(name, search_fn, questions))

    # build the comparison table
    lines = [
        f"| System | Hit-rate@{K_HIT} | MRR@{K_MRR} | Avg latency (ms) |",
        "|---|---|---|---|",
    ]
    for s in scores:
        lines.append(f"| {s['system']} | {s['hit_rate']:.0%} | {s['mrr']:.3f} | {s['avg_ms']:.0f} |")
    table = "\n".join(lines)

    print("\n" + table)
    RESULTS_PATH.write_text(f"# Retrieval results ({len(questions)} questions)\n\n{table}\n", encoding="utf-8")

    # save where the right card landed, per question per system, for digging in later
    with DETAILS_PATH.open("w", encoding="utf-8") as out:
        for i, q in enumerate(questions):
            row = {"question": q["question"], "chunk_id": q["chunk_id"]}
            for s in scores:
                row[s["system"]] = s["ranks"][i]
            out.write(json.dumps(row) + "\n")

    print(f"\nSaved table to {RESULTS_PATH} and per-question ranks to {DETAILS_PATH}")


if __name__ == "__main__":
    main()
