import json
import sys
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
INDEX_PATH = Path("data/processed/vector.index")
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class VectorSearcher:
    def __init__(self):
        """Setup, runs once: load cards, index, and model."""
        with CHUNKS_PATH.open(encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]
        self.index = faiss.read_index(str(INDEX_PATH))
        self.model = SentenceTransformer(MODEL_NAME)

    def search(self, question, k=5):
        """Return the k cards whose meaning is closest to the question."""
        query_vec = self.model.encode([QUERY_PREFIX + question], normalize_embeddings=True)
        scores, rows = self.index.search(query_vec, k)   # FAISS returns scores and row numbers

        results = []
        for rank, (row, score) in enumerate(zip(rows[0], scores[0]), start=1):
            chunk = self.chunks[row]                      # row number -> actual card
            results.append({
                "rank": rank,
                "score": float(score),
                "chunk_id": chunk["chunk_id"],
                "sections": chunk["sections"],
                "text": chunk["text"],
            })
        return results


if __name__ == "__main__":
    # use the question typed after the command, or a default one
    question = " ".join(sys.argv[1:]) or "How do I run code after sending a response?"

    searcher = VectorSearcher()
    print(f"\nQuestion: {question}\n")
    for r in searcher.search(question):
        print(f"#{r['rank']}  score {r['score']:.2f}  {r['chunk_id']}")
        print(f"    sections: {r['sections']}")
        print(f"    {r['text'][:150].replace(chr(10), ' ')}...\n")
