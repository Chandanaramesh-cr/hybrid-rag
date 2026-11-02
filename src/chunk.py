import html
import json
import re
from pathlib import Path

import tiktoken

RAW_DIR = Path("data/raw/fastapi_tutorial")
OUT_PATH = Path("data/processed/chunks.jsonl")
HTML_TAGS = r"</?(font|span|div|u|b|i|br|strong|em|dfn|abbr)\b[^>]*>"
MAX_TOKENS = 400
OVERLAP = 50

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


def split_long_section(title, body, max_tokens=MAX_TOKENS, overlap=OVERLAP):
    """If a section is too big, cut it into overlapping pieces. Otherwise leave it alone."""
    tokens = enc.encode(body)
    if len(tokens) <= max_tokens:
        return [(title, body)]

    pieces = []
    step = max_tokens - overlap
    for start in range(0, len(tokens), step):
        window = tokens[start:start + max_tokens]
        pieces.append((title, enc.decode(window)))
        if start + max_tokens >= len(tokens):
            break
    return pieces


def merge_sections(sections, max_tokens=MAX_TOKENS):
    """Glue neighboring small sections into cards of up to max_tokens."""
    cards = []
    card_titles = []
    card_text = ""

    for title, body in sections:
        piece = f"## {title}\n{body}"

        if card_text and count_tokens(card_text + "\n\n" + piece) > max_tokens:
            cards.append({"sections": card_titles, "text": card_text})
            card_titles = []
            card_text = ""

        card_titles.append(title)
        card_text = piece if not card_text else card_text + "\n\n" + piece

    if card_text:
        cards.append({"sections": card_titles, "text": card_text})

    return cards


def process_file(path):
    """Run the full pipeline on one file: clean -> split -> cut big -> glue small."""
    text = clean_text(path.read_text(encoding="utf-8"))
    sections = split_by_headings(text)

    pieces = []
    for title, body in sections:
        pieces.extend(split_long_section(title, body))

    return merge_sections(pieces)


def main():
    files = sorted(RAW_DIR.rglob("*.md"))      # every .md file, including subfolders
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sizes = []

    with OUT_PATH.open("w", encoding="utf-8") as out:
        for path in files:
            source = path.relative_to(RAW_DIR).as_posix()   # e.g. "security/first-steps.md"

            for i, card in enumerate(process_file(path)):
                record = {
                    "chunk_id": f"{source}::{i}",
                    "source": source,
                    "sections": card["sections"],
                    "text": card["text"],
                }
                out.write(json.dumps(record) + "\n")        # one card per line
                sizes.append(count_tokens(card["text"]))

    print(f"Files processed: {len(files)}")
    print(f"Total cards: {len(sizes)}")
    print(f"Tokens per card: smallest {min(sizes)}, biggest {max(sizes)}, "
          f"average {sum(sizes) / len(sizes):.0f}")
    print(f"Cards under 100 tokens: {sum(1 for s in sizes if s < 100)}")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
