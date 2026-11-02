import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
INDEX_PATH = Path("data/processed/vector.index")
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_chunks():
    """Read chunks.jsonl into a list of dicts. Row i of this list = line i of the file."""
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    chunks = load_chunks()
    texts = [chunk["text"] for chunk in chunks]
    print(f"Loaded {len(chunks)} chunks")

    model = SentenceTransformer(MODEL_NAME)
    print(f"Model reads up to {model.max_seq_length} tokens")

    # turn every card into 384 numbers, scaled to the same length (normalized)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    print(f"Vectors shape: {vectors.shape}")   # (number of cards, numbers per card)

    # build the index and add every vector, in the same order as chunks.jsonl
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(INDEX_PATH))
    print(f"Index holds {index.ntotal} vectors, saved to {INDEX_PATH}")


if __name__ == "__main__":
    main()
