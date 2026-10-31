"""Build tests/test_sentences.json from the assignment PDF's embedded text layer.

The PDF text layer (read with PyMuPDF) matches the rendered glyphs exactly, whereas pypdf,
the supplied .docx (a PDF-to-Word conversion) and copy-paste all damage the Telugu text.
See plan.txt, REVISION 1 (R1.1, R1.2) for the evidence.

Rules applied to obtain the canonical text (and nothing else):
  * the line fragments of a table cell are concatenated in reading order exactly as stored;
    the text layer already carries real word spaces as characters (e.g. the wrap after
    "parliamentary " keeps its trailing space), so no separator is invented;
  * trailing whitespace of the cell is removed;
  * every other character, including the double spaces in E3 and T3, is kept verbatim.

Usage:
  python tools/build_test_manifest.py --pdf ../FastSpeech2_Assignment.pdf \
      --docx ../FastSpeech2_Assignment.docx --out tests/test_sentences.json
"""
import argparse
import hashlib
import json
import re
import unicodedata
import zipfile

import pymupdf

# (id, language, page index, y-range of the cell's lines, item prefix to strip)
CELLS = [
    ("E1", "en", 0, (600, 620), None),
    ("E2a", "en", 0, (630, 640), "a) "),
    ("E2b", "en", 0, (645, 655), "b) "),
    ("E3", "en", 0, (665, 675), None),
    ("E4", "en", 0, (686, 696), None),
    ("T1", "te", 1, (95, 105), None),
    ("T2a", "te", 1, (118, 126), "a) "),
    ("T2b", "te", 1, (133, 142), "b) "),
    ("T3", "te", 1, (155, 165), None),
    ("T4", "te", 1, (180, 202), None),
]
TEXT_COLUMN_X = 90  # the ID column ends before this x; sentence text starts after it


def pdf_lines(page):
    """Return (y0, x0, x1, text) for every text line on the page."""
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"])
            x0, y0, x1, _ = line["bbox"]
            out.append((y0, x0, x1, text))
    return sorted(out)


def join_cell(parts):
    """Concatenate the line fragments of one cell in reading order (top-to-bottom, then
    left-to-right within a baseline). Fragments within 3 pt vertically share a baseline."""
    ordered = sorted(parts, key=lambda p: (round(p[0] / 3), p[1]))
    return "".join(frag for _y0, _x0, _x1, frag in ordered).rstrip()


def docx_paragraphs(path):
    xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8")
    paras = []
    for p in re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S):
        s = "".join(re.findall(r"<w:t(?: [^>]*)?>([^<]*)</w:t>", p))
        if s.strip():
            paras.append(s)
    return paras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--docx", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    doc = pymupdf.open(args.pdf)
    pages = {i: pdf_lines(doc[i]) for i in range(len(doc))}
    paras = docx_paragraphs(args.docx)
    # In the docx each ID label ("E1", "E2", ...) is its own paragraph, followed by the sentence
    # paragraph(s); E2/T2 are followed by their a) and b) paragraphs.
    labels = [p.strip() for p in paras]
    docx_by_id = {}
    for sid, *_rest in CELLS:
        group, sub = sid[:2], sid[2:]
        pos = labels.index(group)
        docx_by_id[sid] = paras[pos + 1 + (1 if sub == "b" else 0)]

    items = []
    for sid, lang, page, (ylo, yhi), prefix in CELLS:
        parts = [(y, x0, x1, t) for y, x0, x1, t in pages[page] if ylo <= y <= yhi and x0 > TEXT_COLUMN_X]
        if not parts:
            raise SystemExit(f"{sid}: no text found in the expected cell region")
        raw = [{"y": round(y, 1), "x": round(x0, 1), "text": t} for y, x0, _x1, t in parts]
        canonical = join_cell(parts)
        if prefix:
            if not canonical.startswith(prefix):
                raise SystemExit(f"{sid}: expected prefix {prefix!r} in {canonical!r}")
            canonical = canonical[len(prefix):]
        docx_text = docx_by_id[sid].strip()
        if prefix and docx_text.startswith(prefix):
            docx_text = docx_text[len(prefix):]
        if not unicodedata.is_normalized("NFC", canonical):
            raise SystemExit(f"{sid}: canonical text is not NFC")
        items.append({
            "id": sid,
            "lang": lang,
            "canonical": canonical,
            "canonical_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "canonical_codepoints": " ".join(f"{ord(c):04X}" for c in canonical),
            "pdf_pymupdf_raw": raw,
            "docx_text": docx_text,
            "docx_matches_canonical": docx_text == canonical,
            "layout_note": "soft-wrapped across lines in the PDF table cell" if len({r['y'] for r in raw}) > 1 else "",
        })

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump({
            "source": "FastSpeech2_Assignment.pdf text layer (PyMuPDF); see plan.txt REVISION 1",
            "items": items,
        }, f, ensure_ascii=False, indent=2)
        f.write("\n")
    for it in items:
        print(f"{it['id']:4s} docx_match={it['docx_matches_canonical']!s:5s} {it['canonical']!r}")


if __name__ == "__main__":
    main()
