"""Verify every BibTeX entry against authoritative metadata (anti-hallucination gate).

For each entry the script resolves its identifier and compares title, year and first-author
surname with the registry record:

* ``doi``      -> DOI content negotiation (CSL-JSON; works for Crossref and DataCite DOIs);
* ``eprint`` / arXiv id in ``journal``/``note``/``url`` -> arXiv API.

Statuses::

    OK          identifier resolves; title similarity >= 0.90; year within +-1; author matches
    WARN        resolves, but similarity in [0.75, 0.90), year/author mismatch, or manual check; or a DOI that
                does not resolve (printed but not registered) while the publisher's page at the entry's ``url``
                states the same DOI and a matching title (``citation_doi`` / ``citation_title`` meta tags)
    FAIL        identifier does not resolve (and no such publisher page), or title similarity < 0.75
    UNVERIFIED  no DOI / arXiv id. Allowed only with a field
                ``x-verified = {manual: <who> <YYYY-MM-DD> <how>}`` added by a human.

Exit code is 1 if any entry is FAIL (or UNVERIFIED without ``x-verified``), else 0.

Usage::

    python tools/lit/verify_bib.py docs/lit/*.bib paper/refs.bib [--report out.md]
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search import fetch_arxiv_ids, http_get  # noqa: E402

# --------------------------------------------------------------------------- BibTeX parsing


def parse_bibtex(text: str) -> list[dict]:
    """Minimal BibTeX parser: handles nested braces and quoted values; ignores @comment/@string."""
    entries = []
    i = 0
    while True:
        at = text.find("@", i)
        if at < 0:
            break
        m = re.match(r"@\s*(\w+)\s*([{(])", text[at:])
        if not m:
            i = at + 1
            continue
        etype = m.group(1).lower()
        open_ch = m.group(2)
        close_ch = "}" if open_ch == "{" else ")"
        start = at + m.end()
        depth, j = 1, start
        while j < len(text) and depth:
            if text[j] == open_ch:
                depth += 1
            elif text[j] == close_ch:
                depth -= 1
            j += 1
        body = text[start : j - 1]
        i = j
        if etype in ("comment", "string", "preamble"):
            continue
        key, _, rest = body.partition(",")
        entries.append({"ENTRYTYPE": etype, "ID": key.strip(), **_parse_fields(rest)})
    return entries


def _parse_fields(s: str) -> dict:
    fields: dict[str, str] = {}
    pos = 0
    while pos < len(s):
        m = re.compile(r"\s*([\w\-:]+)\s*=\s*").match(s, pos)
        if not m:
            break
        name = m.group(1).lower()
        pos = m.end()
        if pos >= len(s):
            break
        if s[pos] == "{":
            depth, j = 1, pos + 1
            while j < len(s) and depth:
                depth += {"{": 1, "}": -1}.get(s[j], 0)
                j += 1
            value = s[pos + 1 : j - 1]
            pos = j
        elif s[pos] == '"':
            j = pos + 1
            while j < len(s) and not (s[j] == '"' and s[j - 1] != "\\"):
                j += 1
            value = s[pos + 1 : j]
            pos = j + 1
        else:
            m2 = re.compile(r"([^,]+)").match(s, pos)
            value = m2.group(1).strip() if m2 else ""
            pos = m2.end() if m2 else pos
        fields[name] = re.sub(r"\s+", " ", value).strip()
        comma = s.find(",", pos)
        pos = comma + 1 if comma >= 0 else len(s)
    return fields


# --------------------------------------------------------------------------- normalisation


# Letters without a Unicode decomposition, on both sides of a comparison (registry text and BibTeX commands).
_SPECIAL = {"ø": "o", "Ø": "O", "ł": "l", "Ł": "L", "ı": "i", "ß": "ss", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE",
            "đ": "d", "Đ": "D", "ð": "d", "þ": "th"}
_LATEX_LETTERS = {"o": "o", "O": "O", "l": "l", "L": "L", "i": "i", "j": "j", "ss": "ss", "ae": "ae", "AE": "AE",
                  "oe": "oe", "OE": "OE", "aa": "a", "AA": "A"}
_SYMBOL_ACCENT = re.compile(r"\\['\"`^~=.]\s*\{?\s*([A-Za-z])\s*\}?")  # \'a, \'{a}, {\"u}
_LETTER_ACCENT = re.compile(r"\\[cvHurktdb]\s*\{\s*([A-Za-z])\s*\}|\\[cvHurktdb]\s+([A-Za-z])")  # \c{c}, \v S


def _strip_latex(s: str) -> str:
    """Plain text of a BibTeX value: accented letters (``{\\"u}``, ``\\'{\\i}``, ``\\c{c}``) keep their letter,
    special letters (``\\o``, ``\\ss``) become their transliteration, other commands and braces are dropped."""
    s = s or ""
    for _ in range(2):  # an accent on a special letter (\'{\i}) needs the letter first
        s = _SYMBOL_ACCENT.sub(lambda m: m.group(1), s)
        s = _LETTER_ACCENT.sub(lambda m: m.group(1) or m.group(2), s)
        s = re.sub(r"\\(ss|ae|AE|oe|OE|aa|AA|o|O|l|L|i|j)(?![A-Za-z])", lambda m: _LATEX_LETTERS[m.group(1)], s)
    s = re.sub(r"\\[a-zA-Z]+\s*", " ", s)
    s = s.replace("{", "").replace("}", "").replace("\\", "")
    return s


def norm_title(s: str) -> str:
    s = "".join(_SPECIAL.get(ch, ch) for ch in _strip_latex(s))
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm_title(a), norm_title(b)).ratio()


def first_surname_bib(author_field: str) -> str:
    first = re.split(r"\s+and\s+", author_field or "", maxsplit=1)[0].strip()
    if "," in first:
        surname = first.split(",")[0]
    else:
        surname = first.split()[-1] if first.split() else ""
    return norm_title(surname)


# --------------------------------------------------------------------------- resolvers


def resolve_doi(doi: str) -> dict | None:
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi.strip(), flags=re.I)
    csl = http_get(f"https://doi.org/{doi}", headers={"Accept": "application/vnd.citationstyles.csl+json"})
    if not csl or not isinstance(csl, dict):
        return None
    title = csl.get("title") or ""
    if isinstance(title, list):
        title = title[0] if title else ""
    parts = ((csl.get("issued") or {}).get("date-parts") or [[None]])[0]
    year = parts[0] if parts else None
    authors = csl.get("author") or []
    surname = norm_title((authors[0].get("family") or authors[0].get("literal") or "")) if authors else ""
    return {"title": title, "year": year, "surname": surname, "via": "doi"}


_META = re.compile(r'<meta\s+name="([^"]+)"[^>]*?content="([^"]*)"', flags=re.I)


def _norm_doi(doi: str) -> str:
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", doi.strip(), flags=re.I).replace("\\", "").lower()


def resolve_publisher_page(url: str, doi: str, title: str) -> dict | None:
    """The record of a publisher's article page (Highwire / OJS meta tags), if the page states ``doi``: a DOI printed
    in an article can lag behind its registration. The title is the page's ``citation_title`` closest to ``title``
    (pages give one per language)."""
    html = http_get(url, as_json=False)
    if not html or not isinstance(html, str):
        return None
    meta: dict[str, list[str]] = {}
    for name, content in _META.findall(html):
        meta.setdefault(name.lower(), []).append(content)
    stated = {_norm_doi(d) for d in meta.get("citation_doi", []) + meta.get("dc.identifier.doi", [])}
    titles = meta.get("citation_title", [])
    if _norm_doi(doi) not in stated or not titles:
        return None
    best = max(titles, key=lambda t: similarity(title, t))
    year = re.search(r"\d{4}", (meta.get("citation_date") or meta.get("citation_publication_date") or [""])[0])
    authors = meta.get("citation_author", [])
    surname = norm_title(authors[0].split()[-1]) if authors else ""
    return {"title": best, "year": int(year.group()) if year else None, "surname": surname, "via": "publisher page"}


def find_arxiv_id(entry: dict) -> str | None:
    if entry.get("eprint") and (entry.get("archiveprefix", "").lower() == "arxiv" or re.match(r"^\d{4}\.\d{4,5}", entry["eprint"])):
        return re.sub(r"v\d+$", "", entry["eprint"].strip())
    for field in ("journal", "note", "url", "howpublished"):
        m = re.search(r"arxiv[:./\s]*(?:abs/)?(\d{4}\.\d{4,5})", entry.get(field, ""), flags=re.I)
        if m:
            return m.group(1)
    return None


def resolve_arxiv(arxiv_id: str) -> dict | None:
    recs = fetch_arxiv_ids([arxiv_id])
    if not recs:
        return None
    r = recs[0]
    surname = norm_title(r["authors"][0].split()[-1]) if r["authors"] else ""
    return {"title": r["title"], "year": r["year"], "surname": surname, "via": "arxiv"}


# --------------------------------------------------------------------------- verification


def verify_entry(entry: dict) -> dict:
    res = {"key": entry["ID"], "id": "-", "status": "UNVERIFIED", "sim": None, "notes": []}
    meta = None
    doi = entry.get("doi")
    arxiv_id = find_arxiv_id(entry)
    if doi:
        res["id"] = f"doi:{doi}"
        meta = resolve_doi(doi)
        if meta is None and arxiv_id:
            res["notes"].append("DOI did not resolve; trying arXiv")
    if meta is None and arxiv_id:
        res["id"] = f"arXiv:{arxiv_id}"
        meta = resolve_arxiv(arxiv_id)
    unregistered = False
    if meta is None and doi and entry.get("url"):
        meta = resolve_publisher_page(entry["url"], doi, entry.get("title", ""))
        if meta is not None:
            unregistered = True
            domain = re.sub(r"^https?://([^/]+).*$", r"\1", entry["url"].strip())
            res["notes"].append(f"DOI not registered (does not resolve); the publisher's page {domain} states it and the "
                                "title; re-check before submission")

    if meta is None:
        if doi or arxiv_id:
            res["status"] = "FAIL"
            res["notes"].append("identifier does not resolve")
        elif entry.get("x-verified", "").lower().startswith("manual"):
            res["status"] = "WARN"
            res["notes"].append(f"no identifier; manual: {entry['x-verified']}")
        else:
            res["notes"].append("no DOI/arXiv id; add one or x-verified={manual: who date how}")
        return res

    sim = similarity(entry.get("title", ""), meta["title"])
    res["sim"] = round(sim, 3)
    status = "OK"
    if sim < 0.75:
        status = "FAIL"
        res["notes"].append(f"title mismatch: registry says {meta['title'][:90]!r}")
    elif sim < 0.90:
        status = "WARN"
        res["notes"].append(f"title differs: registry says {meta['title'][:90]!r}")

    year_bib = re.search(r"\d{4}", entry.get("year", "") or entry.get("date", ""))
    if year_bib and meta.get("year"):
        dy = abs(int(year_bib.group()) - int(meta["year"]))
        if dy > 1:
            status = "FAIL" if status == "FAIL" else "WARN"
            res["notes"].append(f"year {year_bib.group()} vs registry {meta['year']}")
    elif not year_bib:
        res["notes"].append("no year in entry")

    sb = first_surname_bib(entry.get("author", ""))
    if sb and meta.get("surname") and sb not in meta["surname"] and meta["surname"] not in sb:
        status = "FAIL" if status == "FAIL" else "WARN"
        res["notes"].append(f"first author {sb!r} vs registry {meta['surname']!r}")
    if unregistered and status == "OK":
        status = "WARN"  # a publisher's page is not a registry: never OK
    res["status"] = status
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bibfiles", nargs="+", type=Path)
    ap.add_argument("--report", type=Path, default=None, help="write a Markdown report here")
    args = ap.parse_args(argv)

    rows = []
    for bf in args.bibfiles:
        if not bf.exists():
            print(f"skip: {bf} not found", file=sys.stderr)
            continue
        for entry in parse_bibtex(bf.read_text(encoding="utf-8")):
            r = verify_entry(entry)
            r["file"] = str(bf)
            rows.append(r)

    counts = {s: sum(r["status"] == s for r in rows) for s in ("OK", "WARN", "FAIL", "UNVERIFIED")}
    lines = [
        "| file | key | id | status | title sim | notes |",
        "|------|-----|----|--------|-----------|-------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['file']} | {r['key']} | {r['id']} | {r['status']} | {r['sim'] if r['sim'] is not None else '-'} "
            f"| {'; '.join(r['notes'])} |"
        )
    summary = f"entries={len(rows)} " + " ".join(f"{k}={v}" for k, v in counts.items())
    print("\n".join(lines))
    print("\n" + summary)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("# verify_bib report\n\n" + "\n".join(lines) + f"\n\n{summary}\n", encoding="utf-8")
    return 1 if counts["FAIL"] or counts["UNVERIFIED"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
