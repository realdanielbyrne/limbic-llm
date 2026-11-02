"""Format docs/references.bib into grouped citation strings for the Word document.

Groups and their order come from the section headings and BibTeX keys in
REFERENCES.md, so the document, REFERENCES.md and references.bib stay in step.

Usage:  python docs/src/bib_to_json.py > docs/src/references.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import bibtexparser

ROOT = Path(__file__).resolve().parents[2]

ACCENTS = {
    r'{\"a}': "ä", r'{\"o}': "ö", r'{\"u}': "ü", r"{\'e}": "é", r"{\'a}": "á", r"{\'\i}": "í",
    r"{\v{r}}": "ř", r"{\L}": "Ł", r"\&": "&", r"\_": "_",
}


def clean(s: str) -> str:
    for k, v in ACCENTS.items():
        s = s.replace(k, v)
    s = re.sub(r"\\url\{([^}]*)\}", r"\1", s)
    return s.replace("{", "").replace("}", "").strip()


def authors(raw: str) -> str:
    names = [n.strip() for n in clean(raw).split(" and ")]
    others = names and names[-1] == "others"
    names = [n for n in names if n != "others"]
    fmt = []
    for n in names:
        if "," in n:
            last, first = [p.strip() for p in n.split(",", 1)]
            initials = " ".join(f"{p[0]}." for p in re.split(r"[\s\-]+", first) if p and p[0].isalpha())
            fmt.append(f"{last}, {initials}" if initials else last)
        else:
            fmt.append(n)
    if len(fmt) > 7:
        return ", ".join(fmt[:6]) + ", et al."
    if others:
        return ", ".join(fmt) + ", et al."
    if len(fmt) == 1:
        return fmt[0]
    return ", ".join(fmt[:-1]) + ", & " + fmt[-1]


def cite(e) -> str:
    f = {k: clean(e[k]) for k in e.fields_dict}
    a, y, t = authors(e["author"]), f.get("year", "n.d."), f["title"]
    if e.entry_type == "article":
        venue = f.get("journal", "")
        vol = f.get("volume")
        vp = f" {vol}" + (f"({f['number']})" if f.get("number") else "") if vol else ""
        pages = f", {f['pages'].replace('--', '–')}" if f.get("pages") else ""
        body = f"{venue}{vp}{pages}."
    elif e.entry_type == "inproceedings":
        body = f"In {f.get('booktitle', '')}."
    elif e.entry_type == "book":
        body = f"{f.get('publisher', '')}."
    else:  # misc / software
        body = (f.get("howpublished") or f.get("url") or "") + "."
        if f.get("license"):
            body = body[:-1] + f" ({f['license']})."
    url = f.get("url", "")
    if url and url not in body:
        body += f" {url}"
    return f"{a} ({y}). {t}. {body}".replace("..", ".")


def groups() -> list[tuple[str, list[str]]]:
    out, cur = [], None
    for line in (ROOT / "REFERENCES.md").read_text().splitlines():
        if line.startswith("## "):
            cur = (line[3:].strip(), [])
            out.append(cur)
        elif cur and line.startswith("| `"):
            cur[1].append(re.match(r"\| `([^`]+)`", line).group(1))
    return out


def main() -> None:
    lib = bibtexparser.parse_file(str(ROOT / "docs" / "references.bib"))
    by_key = {e.key: e for e in lib.entries}
    result, used = [], set()
    for title, keys in groups():
        result.append({"group": title, "entries": [{"key": k, "text": cite(by_key[k])} for k in keys]})
        used.update(keys)
    missing = sorted(set(by_key) - used)
    if missing:
        raise SystemExit(f"keys in references.bib but not REFERENCES.md: {missing}")
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
