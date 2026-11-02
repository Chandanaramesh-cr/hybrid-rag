import json
import re
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

CHUNKS_PATH = Path("data/processed/chunks.jsonl")


def tokenize(text):
    """Turn text into a list of lowercase words: 'Return a 404!' -> ['return', 'a', '404']"""
    return re.findall(r"\w+", text.lower())


class BM25Searcher:
    def __init__(self):
        """Setup, runs once: load cards and build the word-count index."""
        with CHUNKS_PATH.open(encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]
        tokenized_cards = [tokenize(chunk["text"]) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_cards)

    def search(self, question, k=5):
        """Return the k cards that best match the question's words."""
        scores = self.bm25.get_scores(tokenize(question))   # one score per card, all 227
        top_rows = np.argsort(scores)[::-1][:k]              # sort high to low, keep top k

        results = []
        for rank, row in enumerate(top_rows, start=1):
            chunk = self.chunks[row]
            results.append({
                "rank": rank,
                "score": float(scores[row]),
                "chunk_id": chunk["chunk_id"],
                "sections": chunk["sections"],
                "text": chunk["text"],
            })
        return results


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How do I run code after sending a response?"

    searcher = BM25Searcher()
    print(f"\nQuestion: {question}\n")
    for r in searcher.search(question):
        print(f"#{r['rank']}  score {r['score']:.2f}  {r['chunk_id']}")
        print(f"    sections: {r['sections']}")
        print(f"    {r['text'][:150].replace(chr(10), ' ')}...\n")
