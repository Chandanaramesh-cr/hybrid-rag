import json
from pathlib import Path

DRAFT_PATH = Path("data/eval/questions_draft.jsonl")
FINAL_PATH = Path("data/eval/questions.jsonl")

# question number -> rewritten question (answer key stays the same)
REWRITES = {
    3:  "how do I give a whole group of routes the same prefix and tags?",
    5:  "how do I let an endpoint return one of two different models?",
    6:  "can a dependency depend on another dependency?",
    7:  "what does a basic dependency function look like?",
    11: "what security tools does FastAPI give me out of the box?",
    18: "if I add dependencies on a router and on a single route, which runs first?",
    26: "what happens if my endpoint throws an error while a yield dependency is open?",
    28: "who is the automatic /docs page meant for?",
    34: "how do I return a user but hide the password field?",
    45: "do I need to set any headers myself for server-sent events?",
}


def main():
    with DRAFT_PATH.open(encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    with FINAL_PATH.open("w", encoding="utf-8") as out:
        for i, item in enumerate(items, start=1):
            if i in REWRITES:
                print(f"{i:>2} OLD: {item['question']}")
                item["question"] = REWRITES[i]
                print(f"   NEW: {item['question']}\n")
            out.write(json.dumps(item) + "\n")

    print(f"Saved {len(items)} questions to {FINAL_PATH} ({len(REWRITES)} rewritten)")


if __name__ == "__main__":
    main()
