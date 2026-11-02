import html
import re
from pathlib import Path

import tiktoken

HTML_TAGS = r"</?(font|span|div|u|b|i|br|strong|em)\b[^>]*>"
MAX_TOKENS = 400

enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    """How many tokens a piece of text is."""
    return len(enc.encode(text))


def clean_text(text):
    """Remove FastAPI-specific markup and HTML noise from a markdown doc."""
    text = re.sub(r"\s*\{\s*#[^}]*\}", "", text)          # heading anchors { #first-steps }
    text = re.sub(r"\{\*.*?\*\}", "", text)                # code includes {* ...py *}
    text = re.sub(r"^///.*$", "", text, flags=re.M)        # tip box markers /// tip, ///
    text = re.sub(HTML_TAGS, "", text)                     # html formatting tags
    text = html.unescape(text)                             # &apos; -> '   &amp; -> &
    text = re.sub(r"\n{3,}", "\n\n", text)                 # collapse extra blank lines
    return text.strip()


def split_by_headings(text):
    """Split a doc into a list of (heading, section_text) pairs."""
    sections = []
    current_title = "Introduction"
    current_lines = []
    inside_code = False

    for line in text.splitlines():
        if line.strip().startswith("```"):
            inside_code = not inside_code

        is_heading = line.startswith("#") and not inside_code

        if is_heading:
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))

    return [(title, body) for title, body in sections if body]


def merge_sections(sections, max_tokens=MAX_TOKENS):
    """Glue neighboring small sections into cards of up to max_tokens."""
    cards = []
    card_titles = []      # section titles on the current card
    card_text = ""        # text on the current card

    for title, body in sections:
        piece = f"## {title}\n{body}"

        # would adding this piece make the card too big?
        if card_text and count_tokens(card_text + "\n\n" + piece) > max_tokens:
            cards.append({"sections": card_titles, "text": card_text})   # save the full card
            card_titles = []                                              # start a fresh one
            card_text = ""

        card_titles.append(title)
        card_text = piece if not card_text else card_text + "\n\n" + piece

    if card_text:                                                         # save the last card
        cards.append({"sections": card_titles, "text": card_text})

    return cards


if __name__ == "__main__":
    sample = Path("data/raw/fastapi_tutorial/first-steps.md").read_text(encoding="utf-8")
    sections = split_by_headings(clean_text(sample))
    cards = merge_sections(sections)

    print(f"{len(sections)} sections became {len(cards)} cards:\n")
    for i, card in enumerate(cards):
        print(f"Card {i}: {count_tokens(card['text'])} tokens")
        print(f"   sections: {card['sections']}\n")
