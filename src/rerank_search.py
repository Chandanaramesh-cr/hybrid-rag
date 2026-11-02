import sys

from sentence_transformers import CrossEncoder

from hybrid_search import HybridSearcher

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATES = 20


class RerankSearcher:
    def __init__(self):
        """Setup, runs once: start hybrid search and load the reranker."""
        self.hybrid = HybridSearcher()
        self.reranker = CrossEncoder(RERANK_MODEL, max_length=512)

    def search(self, question, k=5, candidates=CANDIDATES):
        """Hybrid search for 20 candidates, then let the reranker pick the best k."""
        cards = self.hybrid.search(question, k=candidates)

        # pair the question with every candidate card, then score each pair
        pairs = [(question, card["text"]) for card in cards]
        scores = self.reranker.predict(pairs)

        for card, score in zip(cards, scores):
            card["hybrid_rank"] = card["rank"]     # remember where hybrid had it
            card["score"] = float(score)

        # re-sort by the reranker's score, highest first
        cards.sort(key=lambda card: card["score"], reverse=True)

        for new_rank, card in enumerate(cards, start=1):
            card["rank"] = new_rank
        return cards[:k]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How do I run code after sending a response?"

    searcher = RerankSearcher()
    print(f"\nQuestion: {question}\n")
    for r in searcher.search(question):
        print(f"#{r['rank']}  score {r['score']:.2f}  {r['chunk_id']}   (hybrid had it at #{r['hybrid_rank']})")
        print(f"    sections: {r['sections']}\n")
