import sys

from bm25_search import BM25Searcher
from vector_search import VectorSearcher

RRF_K = 60          # the 60 in 1 / (60 + rank)
CANDIDATES = 20     # how many results we take from each search


def reciprocal_rank_fusion(result_lists, k=RRF_K):
    """Merge several ranked lists into one, using 1 / (k + rank) points per list."""
    points = {}      # tally sheet: chunk_id -> total points
    cards = {}       # chunk_id -> the card itself (so we can show its text later)
    ranks = {}       # chunk_id -> where it ranked in each list, e.g. {"vector": 3, "bm25": 2}

    for list_name, results in result_lists.items():
        for r in results:
            cid = r["chunk_id"]
            points[cid] = points.get(cid, 0) + 1 / (k + r["rank"])   # add this list's points
            cards[cid] = r
            ranks.setdefault(cid, {})[list_name] = r["rank"]

    # sort card IDs by total points, highest first
    ordered = sorted(points, key=points.get, reverse=True)

    fused = []
    for new_rank, cid in enumerate(ordered, start=1):
        card = cards[cid]
        fused.append({
            "rank": new_rank,
            "score": points[cid],
            "chunk_id": cid,
            "sections": card["sections"],
            "text": card["text"],
            "source_ranks": ranks[cid],
        })
    return fused


class HybridSearcher:
    def __init__(self):
        """Setup, runs once: start both search machines."""
        self.vector = VectorSearcher()
        self.bm25 = BM25Searcher()

    def search(self, question, k=5, candidates=CANDIDATES):
        """Run both searches, merge with RRF, return the top k."""
        result_lists = {
            "vector": self.vector.search(question, k=candidates),
            "bm25": self.bm25.search(question, k=candidates),
        }
        return reciprocal_rank_fusion(result_lists)[:k]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How do I run code after sending a response?"

    searcher = HybridSearcher()
    print(f"\nQuestion: {question}\n")
    for r in searcher.search(question):
        v = r["source_ranks"].get("vector", "-")
        b = r["source_ranks"].get("bm25", "-")
        print(f"#{r['rank']}  points {r['score']:.4f}  {r['chunk_id']}   (meaning #{v}, BM25 #{b})")
        print(f"    sections: {r['sections']}\n")
