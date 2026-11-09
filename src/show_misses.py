import json

SYSTEMS = ["Vector only", "BM25 only", "Hybrid (RRF)", "Hybrid + rerank"]

with open("data/eval/details.jsonl", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

misses = [r for r in rows if r["Hybrid + rerank"] is None or r["Hybrid + rerank"] > 5]
print(f"{len(misses)} misses out of {len(rows)}\n")

for r in misses:
    ranks = "  ".join(f"{s}:{r[s] or '-'}" for s in SYSTEMS)
    print(f"Q: {r['question']}")
    print(f"   answer key: {r['chunk_id']}")
    print(f"   ranks -> {ranks}\n")
