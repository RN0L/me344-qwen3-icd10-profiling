#!/usr/bin/env python3
"""Turn ``slides/slides.md`` into a self-contained ``slides/slides.html``.

The markdown is the source of truth and stays readable on GitHub. This produces a deck that
opens in any browser with no toolchain, no network and no extension: images are inlined as
data URIs, so the single HTML file is the whole deliverable. Arrow keys or space to advance;
the print stylesheet puts one slide per page, so Cmd-P → Save as PDF gives a PDF deck.

::

    python3 profiling/analyze.py
    python3 profiling/make_dashboard.py --panels     # writes slides/figures/*.png
    python3 slides/build_slides.py                   # writes slides/slides.html

Only the markdown subset actually used by ``slides.md`` is supported — headings, bullets,
tables, block quotes, images, and inline code/bold/italic. It is a renderer for this deck,
not a general markdown implementation, and it fails loudly rather than silently dropping a
construct it does not know.

Standard library only.
"""

from __future__ import annotations

import argparse
import base64
import html
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

CODE_RE = re.compile(r"`([^`]+)`")
BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
ITALIC_RE = re.compile(r"(?<![A-Za-z0-9_])_([^_]+)_(?![A-Za-z0-9_])")
STAR_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
WIDTH_RE = re.compile(r"w:(\d+)")


def inline(text: str, base_dir: str) -> str:
    """Render inline markdown. Code spans are protected before anything else runs."""
    spans: List[str] = []

    def stash(match: "re.Match[str]") -> str:
        spans.append(match.group(1))
        return "\x00%d\x00" % (len(spans) - 1)

    text = CODE_RE.sub(stash, text)
    text = html.escape(text, quote=False)
    text = BOLD_RE.sub(r"<strong>\1</strong>", text)
    text = ITALIC_RE.sub(r"<em>\1</em>", text)
    text = STAR_ITALIC_RE.sub(r"<em>\1</em>", text)

    for index, span in enumerate(spans):
        text = text.replace("\x00%d\x00" % index, "<code>%s</code>" % html.escape(span, quote=False))
    return text


def embed_image(path: str, base_dir: str) -> str:
    """Read an image and return it as a data URI, so the deck needs no sibling files."""
    full = path if os.path.isabs(path) else os.path.join(base_dir, path)
    if not os.path.exists(full):
        raise SystemExit(
            "slides.md references %s, which does not exist.\n"
            "Run: python3 profiling/make_dashboard.py --panels" % full
        )
    suffix = os.path.splitext(full)[1].lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml", ".gif": "image/gif"}.get(suffix)
    if mime is None:
        raise SystemExit("unsupported image type for %s" % full)
    with open(full, "rb") as handle:
        payload = base64.b64encode(handle.read()).decode("ascii")
    return "data:%s;base64,%s" % (mime, payload)


def render_table(rows: Sequence[str], base_dir: str) -> str:
    """Render a pipe table. A row of dashes marks the header separator."""
    def cells(line: str) -> List[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    parsed = [cells(row) for row in rows]
    body_start = 1
    has_header = True
    if len(parsed) > 1 and all(set(cell) <= set("-: ") and cell for cell in parsed[1]):
        body_start = 2
    else:
        has_header = False
        body_start = 0

    out = ["<table>"]
    if has_header:
        head = "".join("<th>%s</th>" % inline(cell, base_dir) for cell in parsed[0])
        # A table whose header cells are all empty is a layout table, not a data table.
        if any(cell.strip() for cell in parsed[0]):
            out.append("<thead><tr>%s</tr></thead>" % head)
    out.append("<tbody>")
    for row in parsed[body_start:]:
        out.append("<tr>%s</tr>" % "".join("<td>%s</td>" % inline(cell, base_dir) for cell in row))
    out.append("</tbody></table>")
    return "".join(out)


def render_slide(source: str, base_dir: str) -> str:
    """Render one slide's markdown to HTML."""
    lines = source.strip("\n").split("\n")
    out: List[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        if stripped.startswith("## "):
            out.append('<h2>%s</h2>' % inline(stripped[3:], base_dir))
            index += 1
            continue

        if stripped.startswith("# "):
            out.append("<h1>%s</h1>" % inline(stripped[2:], base_dir))
            index += 1
            continue

        if stripped.startswith("> "):
            block = []
            while index < len(lines) and lines[index].strip().startswith("> "):
                block.append(lines[index].strip()[2:])
                index += 1
            out.append("<blockquote>%s</blockquote>" % inline(" ".join(block), base_dir))
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            block = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                block.append(lines[index])
                index += 1
            out.append(render_table(block, base_dir))
            continue

        if stripped.startswith("- "):
            block = []
            while index < len(lines) and lines[index].strip().startswith("- "):
                block.append(inline(lines[index].strip()[2:], base_dir))
                index += 1
            out.append("<ul>%s</ul>" % "".join("<li>%s</li>" % item for item in block))
            continue

        # A line that is only images becomes a figure row.
        if IMAGE_RE.fullmatch(stripped) or re.fullmatch(r"(!\[[^\]]*\]\([^)]+\)\s*)+", stripped):
            figures = []
            for alt, target in IMAGE_RE.findall(stripped):
                width = WIDTH_RE.search(alt)
                style = ' style="max-width:%spx"' % width.group(1) if width else ""
                figures.append('<img src="%s" alt="%s"%s>' % (
                    embed_image(target, base_dir),
                    html.escape(WIDTH_RE.sub("", alt).strip(), quote=True),
                    style,
                ))
            out.append('<div class="figures">%s</div>' % "".join(figures))
            index += 1
            continue

        # Otherwise: a paragraph, joined until a blank line or a block construct.
        block = []
        while index < len(lines):
            current = lines[index].strip()
            if not current or current.startswith(("- ", "> ", "|", "#")) or IMAGE_RE.match(current):
                break
            block.append(current)
            index += 1
        text = inline(" ".join(block), base_dir)
        css = ' class="note"' if block and block[0].startswith("_") else ""
        out.append("<p%s>%s</p>" % (css, text))

    return "\n".join(out)


CSS = """
:root {
  --ink: #16181d; --muted: #5b6472; --rule: #dfe3ea; --bg: #ffffff;
  --accent: #1f6fb4; --warm: #c1442f; --good: #1f8a4c; --code-bg: #f2f4f7;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: #7b828d; color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Helvetica, Arial, sans-serif; }
.deck { display: flex; flex-direction: column; align-items: center; gap: 24px; padding: 28px 16px 80px; }
.slide {
  width: min(1280px, 96vw); aspect-ratio: 16 / 9; background: var(--bg);
  border-radius: 6px; box-shadow: 0 8px 28px rgba(0,0,0,.22);
  padding: 42px 56px 60px; display: none; flex-direction: column; overflow: hidden; position: relative;
}
.slide.active { display: flex; }
h1 { font-size: 40px; line-height: 1.12; margin: 0 0 4px; letter-spacing: -0.6px; }
h2 { font-size: 17px; margin: 0 0 18px; color: var(--accent); text-transform: uppercase;
     letter-spacing: 1.4px; font-weight: 700; }
.slide > h2:first-child { margin-top: 0; }
h1 + h2 { margin-top: 10px; }
p { font-size: 19px; line-height: 1.45; margin: 0 0 12px; }
p.note { font-size: 14.5px; color: var(--muted); font-style: italic; margin-top: auto; margin-bottom: 0; }
ul { margin: 0 0 14px; padding-left: 22px; }
li { font-size: 18.5px; line-height: 1.45; margin-bottom: 9px; }
strong { font-weight: 700; }
em { font-style: italic; }
code { background: var(--code-bg); border-radius: 3px; padding: 1px 5px;
       font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.88em; }
blockquote { margin: 14px 0; padding: 14px 20px; border-left: 5px solid var(--accent);
  background: #f4f8fc; font-size: 21px; line-height: 1.4; font-weight: 600; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 14px; font-size: 16.5px; }
th { text-align: left; font-weight: 700; border-bottom: 2px solid var(--ink); padding: 7px 10px; }
td { border-bottom: 1px solid var(--rule); padding: 7px 10px; vertical-align: top; line-height: 1.35; }
tr td:first-child { color: var(--muted); }
tr td:first-child strong { color: var(--ink); }
.figures { display: flex; gap: 18px; justify-content: center; align-items: center;
           margin: 6px 0 10px; min-height: 230px; flex: 1 1 auto; }
.figures img { max-width: 100%; max-height: 100%; object-fit: contain;
               border: 1px solid var(--rule); border-radius: 4px; }
.pagenum { position: absolute; right: 22px; bottom: 14px; font-size: 13px; color: var(--muted); }
.brand { position: absolute; left: 56px; bottom: 14px; font-size: 13px; color: var(--muted); }
.hud { position: fixed; bottom: 0; left: 0; right: 0; padding: 10px 18px; background: rgba(22,24,29,.9);
  color: #eef1f5; font-size: 13px; display: flex; gap: 18px; align-items: center; justify-content: center; }
.hud button { background: #2b303a; color: #eef1f5; border: 0; border-radius: 4px;
  padding: 5px 12px; font-size: 13px; cursor: pointer; }
.hud button:hover { background: #3a4150; }
@media print {
  html, body { background: #fff; }
  .deck { display: block; padding: 0; gap: 0; }
  .hud { display: none; }
  .slide { display: flex !important; width: 100%; aspect-ratio: auto; height: 100vh;
           box-shadow: none; border-radius: 0; page-break-after: always; break-after: page; }
}
"""

SCRIPT = """
(function () {
  var slides = Array.prototype.slice.call(document.querySelectorAll('.slide'));
  var at = 0;
  function show(next) {
    at = Math.max(0, Math.min(slides.length - 1, next));
    slides.forEach(function (slide, index) { slide.classList.toggle('active', index === at); });
    var counter = document.getElementById('counter');
    if (counter) counter.textContent = (at + 1) + ' / ' + slides.length;
    if (location.hash !== '#' + (at + 1)) history.replaceState(null, '', '#' + (at + 1));
  }
  document.addEventListener('keydown', function (event) {
    if (event.key === 'ArrowRight' || event.key === 'PageDown' || event.key === ' ') { show(at + 1); event.preventDefault(); }
    else if (event.key === 'ArrowLeft' || event.key === 'PageUp') { show(at - 1); event.preventDefault(); }
    else if (event.key === 'Home') { show(0); }
    else if (event.key === 'End') { show(slides.length - 1); }
  });
  document.getElementById('prev').addEventListener('click', function () { show(at - 1); });
  document.getElementById('next').addEventListener('click', function () { show(at + 1); });
  var start = parseInt((location.hash || '#1').slice(1), 10);
  show(isNaN(start) ? 0 : start - 1);
})();
"""


def build(markdown_path: str, out_path: str, title: str, footer: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(markdown_path))
    with open(markdown_path, "r", encoding="utf-8") as handle:
        source = handle.read()

    # Strip the Marp front matter — it is for Marp, not for this renderer.
    if source.startswith("---"):
        end = source.find("\n---", 3)
        if end != -1:
            source = source[end + 4:]

    chunks = [chunk for chunk in re.split(r"\n---+\s*\n", source) if chunk.strip()]
    slides = []
    for number, chunk in enumerate(chunks, start=1):
        slides.append(
            '<section class="slide">\n%s\n<div class="brand">%s</div>'
            '<div class="pagenum">%d / %d</div>\n</section>'
            % (render_slide(chunk, base_dir), html.escape(footer), number, len(chunks))
        )

    document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%s</title>
<style>%s</style>
</head><body>
<div class="deck">
%s
</div>
<div class="hud">
  <button id="prev">&larr; prev</button>
  <span id="counter"></span>
  <button id="next">next &rarr;</button>
  <span>arrow keys / space &middot; print for PDF</span>
</div>
<script>%s</script>
</body></html>
""" % (html.escape(title), CSS, "\n".join(slides), SCRIPT)

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(document)
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--markdown", default=os.path.join(here, "slides.md"))
    parser.add_argument("--out", default=os.path.join(here, "slides.html"))
    parser.add_argument("--title", default="ME344 — LLM fine-tuning across CPU, GPU and TPU")
    parser.add_argument("--footer", default="ME344 final project · Qwen3 + CodiEsp · CPU vs GPU vs TPU")
    args = parser.parse_args(argv)

    path = build(args.markdown, args.out, args.title, args.footer)
    size = os.path.getsize(path) / 1024.0
    print("wrote %s (%.0f KB, self-contained)" % (path, size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
