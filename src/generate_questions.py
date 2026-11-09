import json
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
OUT_PATH = Path("data/eval/questions_draft.jsonl")
NUM_QUESTIONS = 50
SEED = 42                # same seed = same 50 cards every run
MIN_WORDS = 80           # skip cards too small to ask a good question about
PAUSE_SECONDS = 6        # stay under the free tier's requests-per-minute limit
MAX_ROUNDS = 4           # how many times to retry the whole model list
ROUND_WAIT = 30          # seconds to wait after a full failed round (grows each round)

LLM_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
]

PROMPT = """Below is a passage from the FastAPI documentation.

Write ONE question that a developer might ask, which this passage answers.

Rules:
- Phrase it the way a real developer would ask it, in your own words.
- Do NOT copy distinctive phrases or section titles from the passage.
- The question must be answerable using this passage alone.
- Reply with only the question, nothing else.

Passage:
{text}"""


def ask_llm(client, prompt):
    """Try every model; if all are busy, wait (longer each time) and try again."""
    for round_num in range(1, MAX_ROUNDS + 1):
        for model in LLM_MODELS:
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                return response.text.strip()
            except (errors.ServerError, errors.ClientError) as e:
                print(f"   {model} failed ({e.code})")
        wait = ROUND_WAIT * round_num                      # 30s, 60s, 90s...
        print(f"   all models busy, waiting {wait}s (round {round_num}/{MAX_ROUNDS})")
        time.sleep(wait)
    raise RuntimeError("Every model failed. Rerun later, finished questions are saved.")


def load_done_ids():
    """Chunk IDs that already have a question saved, so we can skip them."""
    if not OUT_PATH.exists():
        return set()
    with OUT_PATH.open(encoding="utf-8") as f:
        return {json.loads(line)["chunk_id"] for line in f if line.strip()}


def main():
    load_dotenv()
    client = genai.Client(http_options=types.HttpOptions(timeout=60_000))

    with CHUNKS_PATH.open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    candidates = [c for c in chunks if len(c["text"].split()) >= MIN_WORDS]
    random.seed(SEED)
    picked = random.sample(candidates, NUM_QUESTIONS)

    done_ids = load_done_ids()
    print(f"{len(done_ids)} already saved, {NUM_QUESTIONS - len(done_ids)} to go\n")

    with OUT_PATH.open("a", encoding="utf-8") as out:       # "a" = add to the end, don't erase
        for i, chunk in enumerate(picked, start=1):
            if chunk["chunk_id"] in done_ids:
                continue                                      # already have this one
            question = ask_llm(client, PROMPT.format(text=chunk["text"]))
            out.write(json.dumps({"question": question, "chunk_id": chunk["chunk_id"]}) + "\n")
            out.flush()
            print(f"[{i}/{NUM_QUESTIONS}] {question}")
            print(f"         answer key: {chunk['chunk_id']}")
            time.sleep(PAUSE_SECONDS)

    print(f"\nDone. Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
