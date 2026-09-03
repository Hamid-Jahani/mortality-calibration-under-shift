"""Check every direct quote in the paper against the extracted literature texts.

The Dowd et al. misquotation (2026-09-03) came from quoting a working paper
against a published citation. This finds any other quote that cannot be
located in the corpus at all -- the mechanical half of that check. A quote
that IS found still needs its attribution read by eye; a quote that is NOT
found is either paraphrase-in-quote-marks, our own coinage, or a defect.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(r"g:/Mortality - Explainable AI")
TXT = ROOT / "literature" / "txt"


def norm(s: str) -> str:
    """Aggressive normalisation: the PDF extractions carry ligatures, curly
    quotes, hyphenation and hard-wrapped lines that a literal match trips on."""
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    s = s.replace("``", '"').replace("''", '"')
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def strip_tex(s: str) -> str:
    s = re.sub(r"\\cite[tp]?\*?(\[[^\]]*\])*\{[^}]*\}", " ", s)
    s = re.sub(r"\\(emph|texttt|textbf|textit)\{([^{}]*)\}", r"\2", s)
    s = re.sub(r"\\[A-Za-z]+\s*", " ", s)
    s = s.replace("~", " ").replace("\\%", "%").replace("\\&", "&")
    s = s.replace("\\_", "_").replace("$", "")
    return s


corpus = {}
for p in sorted(TXT.glob("*.txt")):
    corpus[p.stem] = norm(p.read_text(encoding="utf-8", errors="replace"))
print(f"corpus: {len(corpus)} extracted texts\n")

quote_re = re.compile(r"``(.+?)''", re.S)
rows = []
for tex in sorted((ROOT / "paper" / "sections").glob("*.tex")):
    raw = tex.read_text(encoding="utf-8", errors="replace")
    # join hard-wrapped lines so a quote spanning a line break still matches
    flat = re.sub(r"\s*\n\s*", " ", raw)
    for m in quote_re.finditer(flat):
        q = norm(strip_tex(m.group(1)))
        if len(q.split()) < 3:          # too short to attribute meaningfully
            continue
        hits = [name for name, body in corpus.items() if q in body]
        rows.append((tex.name, q, hits))

found = [r for r in rows if r[2]]
missing = [r for r in rows if not r[2]]
print(f"{len(rows)} quotes of 3+ words: {len(found)} located, {len(missing)} NOT located\n")
print("=" * 78)
print("NOT LOCATED IN ANY EXTRACTED TEXT -- read each by eye")
print("=" * 78)
for f, q, _ in missing:
    print(f"\n[{f}]\n  {q[:300]}")
print("\n" + "=" * 78)
print("LOCATED (quote -> source)")
print("=" * 78)
for f, q, hits in found:
    src = ", ".join(h[:46] for h in hits[:2])
    print(f"\n[{f}] {src}{' +more' if len(hits) > 2 else ''}\n  {q[:150]}")
