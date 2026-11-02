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


if __name__ == "__main__":
    sample = Path("data/raw/fastapi_tutorial/first-steps.md").read_text(encoding="utf-8")
    print(clean_text(sample)[:1500])
