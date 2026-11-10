import time

import streamlit as st

from answer import Answerer

st.set_page_config(page_title="FastAPI Docs Q&A", page_icon="⚡")
st.title("Ask the FastAPI docs")
st.caption(
    "Hybrid BM25 + vector search, RRF fusion, cross-encoder reranking, "
    "and answers from Gemini with citations."
)


@st.cache_resource                 # run once, reuse on every rerun
def load_answerer():
    return Answerer()


with st.spinner("Loading models (first time only)..."):
    answerer = load_answerer()

question = st.text_input(
    "Your question",
    placeholder="How do I run code after sending a response?",
)

if question:
    with st.spinner("Searching and asking Gemini..."):
        start = time.perf_counter()
        try:
            text, cards, model = answerer.answer(question)
        except RuntimeError as e:          # every Gemini model was busy
            st.error(str(e))
            st.stop()
        elapsed = time.perf_counter() - start

    st.subheader("Answer")
    st.markdown(text)
    st.caption(f"Answered by {model} in {elapsed:.1f}s")

    st.subheader("Sources")
    for i, card in enumerate(cards, start=1):
        with st.expander(f"[{i}] {card['chunk_id']}  |  {', '.join(card['sections'])}"):
            st.markdown(card["text"])
