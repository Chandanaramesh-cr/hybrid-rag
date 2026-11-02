import sys

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from rerank_search import RerankSearcher

LLM_MODELS = [                  # tried in this order until one works
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]
TIMEOUT_MS = 60_000             # give up on a model after 60 seconds
TOP_K = 5

INSTRUCTIONS = """You answer questions about the FastAPI documentation.

Rules:
1. Use ONLY the numbered sources below. Do not use outside knowledge.
2. After every claim, cite the source it came from, like [1] or [2][3].
3. If the sources do not contain the answer, reply exactly:
   "I don't know based on the provided documentation."
4. Be concise."""


def build_prompt(question, cards):
    """Put the rules, the numbered cards, and the question together into one prompt."""
    sources = []
    for i, card in enumerate(cards, start=1):
        header = f"[{i}] ({card['chunk_id']}: {', '.join(card['sections'])})"
        sources.append(f"{header}\n{card['text']}")

    return (
        f"{INSTRUCTIONS}\n\n"
        f"Sources:\n\n" + "\n\n".join(sources) + "\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


class Answerer:
    def __init__(self):
        """Setup, runs once: start the search funnel and connect to Gemini."""
        load_dotenv()
        self.searcher = RerankSearcher()
        self.client = genai.Client(http_options=types.HttpOptions(timeout=TIMEOUT_MS))

    def ask_llm(self, prompt):
        """Try each model in LLM_MODELS until one answers."""
        for model in LLM_MODELS:
            try:
                print(f"Asking {model}...")
                response = self.client.models.generate_content(model=model, contents=prompt)
                return response.text, model
            except (errors.ServerError, errors.ClientError) as e:
                print(f"   {model} failed ({e.code}), trying the next one")
        raise RuntimeError("Every model in LLM_MODELS failed. Try again in a few minutes.")

    def answer(self, question, k=TOP_K):
        """Find the best cards, then ask Gemini to answer using only those."""
        print("Searching...")
        cards = self.searcher.search(question, k=k)
        prompt = build_prompt(question, cards)
        text, model = self.ask_llm(prompt)
        return text, cards, model


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How do I run code after sending a response?"

    answerer = Answerer()
    text, cards, model = answerer.answer(question)

    print(f"\nQuestion: {question}")
    print(f"(answered by {model})\n")
    print(f"Answer:\n{text}\n")
    print("Sources:")
    for i, card in enumerate(cards, start=1):
        print(f"  [{i}] {card['chunk_id']}")
