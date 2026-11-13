"""``paper/refs.bib`` from the keys the manuscript cites, copied only from the verified bibliographies of the literature
reviews (``docs/lit/*.bib``, every entry checked by ``tools/lit/verify_bib.py``; CLAUDE.md, golden rule 2). A cited
key that is in none of them stops the script: it has to be found, verified and added to a review first.

    python scripts/paper_refs.py            # writes paper/refs.bib
    python scripts/paper_refs.py --check    # exit 1 if paper/refs.bib is not what the citations give
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "refs.bib"
# Case that BibTeX would lower: protected in the copy (the verified entry itself is not changed).
PROTECT = {"TTT plots: a perl program": "{TTT} plots: a {Perl} program"}


def cited_keys(paths) -> list[str]:
    """The keys of every \\cite-like command, in the order of first use."""
    keys: list[str] = []
    for path in paths:
        text = re.sub(r"(?<!\\)%[^\n]*", "", Path(path).read_text(encoding="utf-8"))
        for group in re.findall(r"\\cite[pt]?\*?(?:\[[^\]]*\])*\{([^}]+)\}", text):
            for key in (k.strip() for k in group.split(",")):
                if key and key not in keys:
                    keys.append(key)
    return keys


def verified_entries() -> dict[str, str]:
    """Key → entry text from ``docs/lit/*.bib`` (the first file that has the key wins)."""
    entries: dict[str, str] = {}
    for path in sorted((ROOT / "docs" / "lit").glob("*.bib")):
        for m in re.finditer(r"(@\w+\{([^,\s]+),.*?)(?=\n@|\Z)", path.read_text(encoding="utf-8"), flags=re.S):
            entries.setdefault(m.group(2).strip(), m.group(1).strip())
    return entries


_ACCENT = {"\u0301": "'", "\u0300": "`", "\u0308": '"', "\u0302": "^", "\u0303": "~", "\u0304": "=",
           "\u0307": ".", "\u0327": "c", "\u030c": "v", "\u030a": "r", "\u0306": "u", "\u030b": "H", "\u0328": "k"}
_LETTER = {"ß": r"{\ss}", "ø": r"{\o}", "Ø": r"{\O}", "ł": r"{\l}", "Ł": r"{\L}", "æ": r"{\ae}", "Æ": r"{\AE}",
           "œ": r"{\oe}", "Œ": r"{\OE}", "å": r"{\aa}", "Å": r"{\AA}", "ı": r"{\i}"}


def _latex_letters(text: str) -> str:
    """Non-ASCII letters as LaTeX accent commands (``Á`` → ``{\\'A}``): BibTeX abbreviates a first name by its first
    byte, which splits a multi-byte UTF-8 letter, but it treats ``{\\'A}`` as one character. A letter is read with all
    its combining marks (NFD), so that no mark is left alone."""
    out = []
    nfd = unicodedata.normalize("NFD", text)
    i = 0
    while i < len(nfd):
        ch, i = nfd[i], i + 1
        marks = ""
        while i < len(nfd) and unicodedata.combining(nfd[i]):
            marks, i = marks + nfd[i], i + 1
        if not marks:
            out.append(_LETTER.get(ch, ch))
        elif ord(ch) < 128 and len(marks) == 1 and marks in _ACCENT:
            cmd = _ACCENT[marks]
            base = "\\i" if ch == "i" else ch  # an accent on i sits on the dotless i
            out.append(f"{{\\{cmd}{base}}}" if not cmd.isalpha() else f"{{\\{cmd}{{{base}}}}}")
        else:
            out.append(unicodedata.normalize("NFC", ch + marks))
    return "".join(out)


def _map_field(entry: str, name: str, func) -> str:
    """Apply ``func`` to the braced value of the field ``name`` (nested braces respected)."""
    m = re.search(rf"\b{name}\s*=\s*\{{", entry, flags=re.I)
    if not m:
        return entry
    depth, i = 1, m.end()
    while i < len(entry) and depth:
        depth += {"{": 1, "}": -1}.get(entry[i], 0)
        i += 1
    return entry[:m.end()] + func(entry[m.end():i - 1]) + entry[i - 1:]


# Proper nouns in the titles of the reviews' bibliographies, which the sentence case of the style would lower.
PROPER_NOUNS = {"Markowitz", "Sharpe", "Ising", "Python", "Perl", "Lagrangian", "Gaussian", "Bayesian", "Pareto",
                "Russia", "Russian"}
# a command, inline maths, or a word (Unicode letters); hyphenated words are read part by part
_TITLE_TOKEN = re.compile(r"\\[A-Za-z]+|\\.|\$[^$]*\$|[^\W\d_]+")


def _protect_token(m: re.Match) -> str:
    token = m.group(0)
    if token.startswith("\\"):
        return token
    if token.startswith("$"):
        return "{" + token + "}"
    caps = sum(c.isupper() for c in token)
    acronym = caps >= 2 or (caps == 1 and not token[0].isupper())  # QUBO, SOCP, PySCIPOpt
    single = len(token) == 1 and token.isupper() and token != "A"  # "part I"
    return "{" + token + "}" if acronym or single or token in PROPER_NOUNS else token


def _case_protected(title: str) -> str:
    """The title as the numbered style needs it: elsarticle-num sets titles in sentence case, so acronyms, CamelCase
    names, single capitals and proper nouns go in braces (braced groups are left as they are); Crossref's HTML italics
    become LaTeX (``1/<i>N</i>`` → ``1/{$N$}``), and line breaks inside the title become spaces."""
    title = re.sub(r"/\s*<i>", "/<i>", title)
    title = re.sub(r"<i>([^\W\d_])</i>", r"$\1$", title)
    title = re.sub(r"</?i>", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    out, i = [], 0
    while i < len(title):
        if title[i] == "{":  # a braced group: copied as it is
            depth, j = 1, i + 1
            while j < len(title) and depth:
                depth += {"{": 1, "}": -1}.get(title[j], 0)
                j += 1
            out.append(title[i:j])
            i = j
            continue
        j = title.find("{", i)
        j = len(title) if j < 0 else j
        out.append(_TITLE_TOKEN.sub(_protect_token, title[i:j]))
        i = j
    return "".join(out)


def normalise(entry: str) -> str:
    """The copy as BibTeX needs it, metadata unchanged: protected case, HTML entities of Crossref decoded (``&amp;`` is
    LaTeX's alignment tab), the title protected from the sentence case of the style, accented author letters as LaTeX
    commands, Crossref's ``@inbook`` chapters as ``@incollection``, page ranges with ``--``, and no ``month`` field
    (``month=July`` is not a BibTeX macro; the numbered style does not print months)."""
    for old, new in PROTECT.items():
        entry = entry.replace(old, new)
    entry = entry.replace("&amp;", r"\&")
    entry = _map_field(entry, "title", _case_protected)
    entry = _map_field(entry, "author", _latex_letters)
    # Crossref's "@inbook" with a booktitle is a chapter with its own title: BibTeX's @incollection
    if re.match(r"@inbook\{", entry, flags=re.I) and re.search(r"\bbooktitle\s*=", entry, flags=re.I):
        entry = re.sub(r"^@inbook\{", "@incollection{", entry, flags=re.I)
    entry = _map_field(entry, "pages", lambda v: v.replace("–", "--"))
    return re.sub(r",\s*month\s*=\s*(\{[^}]*\}|\w+)", "", entry)


def build() -> str:
    tex = [ROOT / "paper" / "main.tex", *sorted((ROOT / "paper" / "sections").glob("*.tex"))]
    keys = cited_keys(tex)
    entries = verified_entries()
    missing = [k for k in keys if k not in entries]
    if missing:
        raise SystemExit(f"cited keys not in the verified bibliographies docs/lit/*.bib: {missing}")
    body = []
    for key in sorted(keys):
        body.append(normalise(entries[key]))
    return ("% paper/refs.bib — generated by scripts/paper_refs.py from the cited keys; entries copied from the verified\n"
            "% bibliographies docs/lit/*.bib (tools/lit/verify_bib.py). Do not edit by hand.\n\n" + "\n\n".join(body) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = build()
    if args.check:
        ok = OUT.exists() and OUT.read_text(encoding="utf-8") == text
        print(f"{OUT.relative_to(ROOT)}: {'up to date' if ok else 'not what the citations give'}")
        return 0 if ok else 1
    OUT.write_text(text, encoding="utf-8")
    print(f"{OUT.relative_to(ROOT)}: {text.count(chr(10) + '@') + text.startswith('@')} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
