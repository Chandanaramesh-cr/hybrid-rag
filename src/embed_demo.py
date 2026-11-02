from sentence_transformers import SentenceTransformer, util

# load the embedding model (downloads ~90 MB the first time)
model = SentenceTransformer("all-MiniLM-L6-v2")

question = "How do I handle errors in FastAPI?"
cards = [
    "Use HTTPException to return error responses to the client.",   # same meaning, different words
    "Background tasks run after the response is sent.",              # different topic
    "Raise an exception when the item is not found.",                # related
]

# turn text into numbers
question_vec = model.encode(question)
card_vecs = model.encode(cards)

print(f"Each text became {len(question_vec)} numbers.")
print(f"First 5 numbers of the question: {question_vec[:5]}\n")

# compare the question to each card
scores = util.cos_sim(question_vec, card_vecs)[0]
for card, score in zip(cards, scores):
    print(f"{score:.2f}  {card}")
