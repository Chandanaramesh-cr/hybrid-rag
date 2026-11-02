import html
import re
from pathlib import Path

HTML_TAGS = r"</?(font|span|div|u|b|i|br|strong|em)\b[^>]*>"


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
    current_title = "Introduction"     # label for text before the first heading
    current_lines = []                 # the current "stack of paper"
    inside_code = False                # the switch

    for line in text.splitlines():
        # flip the switch every time we pass a ``` line
        if line.strip().startswith("```"):
            inside_code = not inside_code

        is_heading = line.startswith("#") and not inside_code

        if is_heading:
            # put the current stack away, then start a new one
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)

    # don't forget the last stack
    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))

    # drop sections that ended up empty
    return [(title, body) for title, body in sections if body]


if __name__ == "__main__":
    sample = Path("data/raw/fastapi_tutorial/first-steps.md").read_text(encoding="utf-8")
    cleaned = clean_text(sample)
    sections = split_by_headings(cleaned)

    print(f"Found {len(sections)} sections:\n")
    for title, body in sections:
        print(f"- {title}  ({len(body.split())} words)")
