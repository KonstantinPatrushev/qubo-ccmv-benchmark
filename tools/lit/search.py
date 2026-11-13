"""Literature search over open scholarly APIs: arXiv, Crossref, Semantic Scholar, OpenAlex.

Used by the ``lit-scout`` agent (see ``.claude/skills/lit-review/SKILL.md``). Every record is
normalised to the same schema so results from different sources can be merged and
de-duplicated::

    {"source", "title", "authors", "year", "venue", "doi", "arxiv", "url", "cited_by", "abstract"}

Sub-commands::

    search "<query>" [--sources arxiv crossref s2 openalex] [--since YEAR] [--limit N] [--out F]
    refs   <DOI|arXiv:ID>   backward snowballing (works cited by the paper)
    cites  <DOI|arXiv:ID>   forward snowballing (works citing the paper)
    bibtex <DOI|arXiv:ID>   BibTeX via DOI content negotiation (Crossref / DataCite)

Only the standard library and ``requests`` are used. No API key is required; optional
environment variables ``S2_API_KEY`` (Semantic Scholar) and ``OPENALEX_MAILTO`` (OpenAlex polite
pool) are honoured when set. Responses are cached under ``tools/lit/.cache/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

USER_AGENT = (
    "cif-lit-scout/0.1 (academic literature review; "
    "https://github.com/KonstantinPatrushev/computing_in_finance)"
)
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
ARXIV_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
S2_FIELDS = "title,year,authors,venue,externalIds,citationCount,abstract,url"
ALL_SOURCES = ("arxiv", "crossref", "s2", "openalex")


# --------------------------------------------------------------------------- HTTP + cache


def _cache_path(key: str) -> Path:
    return CACHE_DIR / (hashlib.sha256(key.encode()).hexdigest()[:32] + ".json")


def http_get(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    as_json: bool = True,
    retries: int = 4,
    use_cache: bool = True,
):
    """GET with on-disk cache and exponential back-off on 429/5xx. Returns parsed body or None."""
    key = json.dumps([url, sorted((params or {}).items()), sorted((headers or {}).items())])
    path = _cache_path(key)
    if use_cache and path.exists():
        payload = json.loads(path.read_text())
        return payload["body"]

    hdrs = {"User-Agent": USER_AGENT, **(headers or {})}
    delay = 2.0
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=hdrs, timeout=30)
        except requests.RequestException as exc:
            if attempt == retries - 1:
                print(f"[http] {url}: {exc}", file=sys.stderr)
                return None
            time.sleep(delay)
            delay *= 2
            continue
        if resp.status_code == 404:
            return None
        if resp.status_code in (429, 500, 502, 503, 504):
            if attempt == retries - 1:
                print(f"[http] {url}: HTTP {resp.status_code} after {retries} tries", file=sys.stderr)
                return None
            time.sleep(delay)
            delay *= 2
            continue
        if not resp.ok:
            print(f"[http] {url}: HTTP {resp.status_code}", file=sys.stderr)
            return None
        body = resp.json() if as_json else resp.text
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"url": url, "params": params, "body": body}))
        return body
    return None


# --------------------------------------------------------------------------- identifiers


def parse_identifier(ident: str) -> tuple[str, str]:
    """Return ``("doi", "10.x/...")`` or ``("arxiv", "1411.4028")`` from free-form input."""
    s = ident.strip()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.I)
    m = re.match(r"^(?:arxiv:|https?://arxiv\.org/abs/)(.+)$", s, flags=re.I)
    if m:
        return "arxiv", re.sub(r"v\d+$", "", m.group(1))
    m = re.match(r"^10\.48550/arxiv\.(.+)$", s, flags=re.I)
    if m:
        return "arxiv", re.sub(r"v\d+$", "", m.group(1))
    if s.startswith("10."):
        return "doi", s
    if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", s) or re.match(r"^[a-z\-]+(\.[A-Z]{2})?/\d{7}$", s):
        return "arxiv", re.sub(r"v\d+$", "", s)
    raise ValueError(f"Cannot parse identifier {ident!r}; use a DOI or arXiv:<id>")


def _record(source: str, **kw) -> dict:
    rec = {
        "source": source,
        "title": "",
        "authors": [],
        "year": None,
        "venue": "",
        "doi": None,
        "arxiv": None,
        "url": "",
        "cited_by": None,
        "abstract": "",
    }
    rec.update({k: v for k, v in kw.items() if v is not None})
    rec["title"] = re.sub(r"\s+", " ", rec["title"] or "").strip()
    if rec["doi"]:
        rec["doi"] = rec["doi"].lower().replace("https://doi.org/", "")
    if rec["abstract"]:
        rec["abstract"] = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", rec["abstract"])).strip()[:400]
    return rec


# --------------------------------------------------------------------------- sources


def search_arxiv(query: str, limit: int, since: int | None) -> list[dict]:
    terms = [t for t in re.split(r"\s+", query.strip()) if t]
    q = " AND ".join(f"all:{t}" for t in terms)
    body = http_get(
        "https://export.arxiv.org/api/query",
        params={"search_query": q, "start": 0, "max_results": limit, "sortBy": "relevance"},
        as_json=False,
    )
    return _parse_arxiv_feed(body, since) if body else []


def _parse_arxiv_feed(body: str, since: int | None = None) -> list[dict]:
    out = []
    root = ET.fromstring(body)
    for e in root.findall("a:entry", ARXIV_NS):
        abs_url = (e.findtext("a:id", default="", namespaces=ARXIV_NS) or "").strip()
        arxiv_id = re.sub(r"v\d+$", "", abs_url.rsplit("/abs/", 1)[-1]) if "/abs/" in abs_url else None
        year = int((e.findtext("a:published", default="0000", namespaces=ARXIV_NS))[:4] or 0) or None
        if since and year and year < since:
            continue
        doi = e.findtext("arxiv:doi", default=None, namespaces=ARXIV_NS)
        jref = e.findtext("arxiv:journal_ref", default="", namespaces=ARXIV_NS) or ""
        out.append(
            _record(
                "arxiv",
                title=e.findtext("a:title", default="", namespaces=ARXIV_NS),
                authors=[
                    a.findtext("a:name", default="", namespaces=ARXIV_NS)
                    for a in e.findall("a:author", ARXIV_NS)
                ],
                year=year,
                venue=jref or "arXiv preprint",
                doi=doi,
                arxiv=arxiv_id,
                url=abs_url,
                abstract=e.findtext("a:summary", default="", namespaces=ARXIV_NS),
            )
        )
    return out


def fetch_arxiv_ids(ids: list[str]) -> list[dict]:
    body = http_get(
        "https://export.arxiv.org/api/query",
        params={"id_list": ",".join(ids), "max_results": len(ids)},
        as_json=False,
    )
    return _parse_arxiv_feed(body) if body else []


def _crossref_item(it: dict) -> dict:
    issued = (it.get("issued") or {}).get("date-parts") or [[None]]
    authors = [
        f"{a.get('family', '')}, {a.get('given', '')}".strip(", ")
        for a in it.get("author", [])
    ]
    return _record(
        "crossref",
        title=(it.get("title") or [""])[0],
        authors=authors,
        year=issued[0][0] if issued and issued[0] else None,
        venue=(it.get("container-title") or [""])[0] or it.get("type", ""),
        doi=it.get("DOI"),
        url=it.get("URL", ""),
        cited_by=it.get("is-referenced-by-count"),
        abstract=it.get("abstract", ""),
    )


def search_crossref(query: str, limit: int, since: int | None) -> list[dict]:
    params = {
        "query.bibliographic": query,
        "rows": limit,
        "select": "DOI,title,author,issued,container-title,is-referenced-by-count,type,URL,abstract",
    }
    if since:
        params["filter"] = f"from-pub-date:{since}"
    body = http_get("https://api.crossref.org/works", params=params)
    return [_crossref_item(it) for it in ((body or {}).get("message") or {}).get("items", [])]


def fetch_crossref_doi(doi: str) -> dict | None:
    body = http_get(f"https://api.crossref.org/works/{doi}")
    return _crossref_item(body["message"]) if body and "message" in body else None


def _s2_headers() -> dict:
    key = os.environ.get("S2_API_KEY")
    return {"x-api-key": key} if key else {}


def _s2_item(p: dict) -> dict:
    ext = p.get("externalIds") or {}
    return _record(
        "s2",
        title=p.get("title") or "",
        authors=[a.get("name", "") for a in p.get("authors") or []],
        year=p.get("year"),
        venue=p.get("venue") or "",
        doi=ext.get("DOI"),
        arxiv=ext.get("ArXiv"),
        url=p.get("url") or "",
        cited_by=p.get("citationCount"),
        abstract=p.get("abstract") or "",
    )


def search_s2(query: str, limit: int, since: int | None) -> list[dict]:
    params = {"query": query, "limit": min(limit, 100), "fields": S2_FIELDS}
    if since:
        params["year"] = f"{since}-"
    body = http_get(
        "https://api.semanticscholar.org/graph/v1/paper/search", params=params, headers=_s2_headers()
    )
    return [_s2_item(p) for p in (body or {}).get("data", []) or []]


def _s2_paper_id(kind: str, value: str) -> str:
    return f"DOI:{value}" if kind == "doi" else f"ARXIV:{value}"


def s2_links(ident: str, direction: str, limit: int) -> list[dict]:
    kind, value = parse_identifier(ident)
    endpoint = "references" if direction == "refs" else "citations"
    body = http_get(
        f"https://api.semanticscholar.org/graph/v1/paper/{_s2_paper_id(kind, value)}/{endpoint}",
        params={"fields": S2_FIELDS, "limit": min(limit, 1000)},
        headers=_s2_headers(),
    )
    key = "citedPaper" if direction == "refs" else "citingPaper"
    return [_s2_item(row[key]) for row in (body or {}).get("data", []) or [] if row.get(key)]


def _openalex_params(extra: dict) -> dict:
    mailto = os.environ.get("OPENALEX_MAILTO")
    return {**extra, **({"mailto": mailto} if mailto else {})}


def _openalex_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    positions = sorted((pos, word) for word, poss in inv.items() for pos in poss)
    return " ".join(word for _, word in positions)


def _openalex_item(w: dict) -> dict:
    src = ((w.get("primary_location") or {}).get("source") or {}) or {}
    ids = w.get("ids") or {}
    arxiv = None
    for loc in w.get("locations") or []:
        lurl = (loc.get("landing_page_url") or "") if loc else ""
        m = re.search(r"arxiv\.org/abs/([^\s?#]+)", lurl)
        if m:
            arxiv = re.sub(r"v\d+$", "", m.group(1))
            break
    return _record(
        "openalex",
        title=w.get("display_name") or "",
        authors=[
            (a.get("author") or {}).get("display_name", "") for a in w.get("authorships") or []
        ],
        year=w.get("publication_year"),
        venue=src.get("display_name") or "",
        doi=(ids.get("doi") or w.get("doi") or None),
        arxiv=arxiv,
        url=w.get("id", ""),
        cited_by=w.get("cited_by_count"),
        abstract=_openalex_abstract(w.get("abstract_inverted_index")),
    )


def search_openalex(query: str, limit: int, since: int | None) -> list[dict]:
    params = {"search": query, "per-page": min(limit, 200)}
    if since:
        params["filter"] = f"from_publication_date:{since}-01-01"
    body = http_get("https://api.openalex.org/works", params=_openalex_params(params))
    return [_openalex_item(w) for w in (body or {}).get("results", [])]


def openalex_work(ident: str) -> dict | None:
    kind, value = parse_identifier(ident)
    if kind == "doi":
        return http_get(f"https://api.openalex.org/works/doi:{value}", params=_openalex_params({}))
    body = http_get(
        "https://api.openalex.org/works",
        params=_openalex_params({"filter": f"doi:10.48550/arxiv.{value}"}),
    )
    results = (body or {}).get("results") or []
    return results[0] if results else None


def openalex_links(ident: str, direction: str, limit: int) -> list[dict]:
    work = openalex_work(ident)
    if not work:
        return []
    if direction == "cites":
        wid = work["id"].rsplit("/", 1)[-1]
        body = http_get(
            "https://api.openalex.org/works",
            params=_openalex_params({"filter": f"cites:{wid}", "per-page": min(limit, 200)}),
        )
        return [_openalex_item(w) for w in (body or {}).get("results", [])]
    refs = [r.rsplit("/", 1)[-1] for r in work.get("referenced_works", [])][:limit]
    out: list[dict] = []
    for i in range(0, len(refs), 50):
        chunk = "|".join(refs[i : i + 50])
        body = http_get(
            "https://api.openalex.org/works",
            params=_openalex_params({"filter": f"openalex_id:{chunk}", "per-page": 50}),
        )
        out.extend(_openalex_item(w) for w in (body or {}).get("results", []))
    return out


SEARCHERS = {
    "arxiv": search_arxiv,
    "crossref": search_crossref,
    "s2": search_s2,
    "openalex": search_openalex,
}


# --------------------------------------------------------------------------- merge + output


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9а-яё]+", " ", (t or "").lower()).strip()


def merge_records(records: list[dict]) -> list[dict]:
    """De-duplicate by DOI, arXiv id, then normalised title; keep the richest fields."""
    merged: list[dict] = []
    index: dict[str, int] = {}
    for rec in records:
        keys = [
            k
            for k in (
                f"doi:{rec['doi']}" if rec.get("doi") else None,
                f"arxiv:{rec['arxiv']}" if rec.get("arxiv") else None,
                f"title:{_norm_title(rec['title'])}" if rec.get("title") else None,
            )
            if k
        ]
        hit = next((index[k] for k in keys if k in index), None)
        if hit is None:
            merged.append({**rec, "sources": [rec["source"]]})
            hit = len(merged) - 1
        else:
            cur = merged[hit]
            cur["sources"] = sorted(set(cur["sources"]) | {rec["source"]})
            for field in ("doi", "arxiv", "venue", "year", "abstract", "url"):
                if not cur.get(field) and rec.get(field):
                    cur[field] = rec[field]
            if (rec.get("cited_by") or 0) > (cur.get("cited_by") or 0):
                cur["cited_by"] = rec["cited_by"]
            if len(rec.get("authors") or []) > len(cur.get("authors") or []):
                cur["authors"] = rec["authors"]
        for k in keys:
            index[k] = hit
    return merged


def print_table(records: list[dict]) -> None:
    for i, r in enumerate(records, 1):
        ident = f"doi:{r['doi']}" if r.get("doi") else (f"arXiv:{r['arxiv']}" if r.get("arxiv") else "-")
        first = (r.get("authors") or ["?"])[0]
        more = " et al." if len(r.get("authors") or []) > 1 else ""
        cites = r.get("cited_by")
        print(
            f"{i:3d}. [{r.get('year') or '????'}] {r['title'][:110]}\n"
            f"     {first}{more} | {r.get('venue') or '-'} | {ident} | cited_by={cites if cites is not None else '?'}"
            f" | via {','.join(r.get('sources', [r['source']]))}"
        )


def write_jsonl(records: list[dict], out: Path, meta: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({**meta, **r}, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- bibtex


def bibtex_for(ident: str) -> str | None:
    kind, value = parse_identifier(ident)
    doi = value if kind == "doi" else f"10.48550/arXiv.{value}"
    text = http_get(
        f"https://doi.org/{doi}",
        headers={"Accept": "application/x-bibtex; charset=utf-8"},
        as_json=False,
    )
    if text and text.lstrip().startswith("@"):
        return text.strip()
    if kind == "arxiv":
        recs = fetch_arxiv_ids([value])
        if recs:
            r = recs[0]
            authors = " and ".join(r["authors"])
            key = re.sub(r"[^a-z]", "", (r["authors"][0].split()[-1] if r["authors"] else "anon").lower())
            return (
                f"@misc{{{key}{r['year']}arxiv,\n  title = {{{r['title']}}},\n  author = {{{authors}}},\n"
                f"  year = {{{r['year']}}},\n  eprint = {{{value}}},\n  archivePrefix = {{arXiv}},\n"
                f"  doi = {{10.48550/arXiv.{value}}}\n}}"
            )
    return None


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="keyword search across sources")
    s.add_argument("query")
    s.add_argument("--sources", nargs="+", default=list(ALL_SOURCES), choices=ALL_SOURCES)
    s.add_argument("--since", type=int, default=None, help="earliest publication year")
    s.add_argument("--limit", type=int, default=15, help="max results per source")
    s.add_argument("--out", type=Path, default=None, help="append merged records to this JSONL")
    s.add_argument("--json", action="store_true", help="print JSON instead of a table")

    for name in ("refs", "cites"):
        p = sub.add_parser(name, help=f"{'backward' if name == 'refs' else 'forward'} snowballing")
        p.add_argument("ident")
        p.add_argument("--limit", type=int, default=50)
        p.add_argument("--sources", nargs="+", default=["s2", "openalex"], choices=("s2", "openalex"))
        p.add_argument("--out", type=Path, default=None)
        p.add_argument("--json", action="store_true")

    b = sub.add_parser("bibtex", help="BibTeX for a DOI or arXiv id")
    b.add_argument("ident", nargs="+")

    args = ap.parse_args(argv)

    if args.cmd == "bibtex":
        rc = 0
        for ident in args.ident:
            bib = bibtex_for(ident)
            if bib:
                print(bib + "\n")
            else:
                print(f"% NOT FOUND: {ident}", file=sys.stderr)
                rc = 1
        return rc

    if args.cmd == "search":
        records: list[dict] = []
        for src in args.sources:
            try:
                found = SEARCHERS[src](args.query, args.limit, args.since)
            except Exception as exc:  # one failing API must not kill the whole search
                print(f"[{src}] failed: {exc}", file=sys.stderr)
                found = []
            print(f"[{src}] {len(found)} records", file=sys.stderr)
            records.extend(found)
        meta = {"query": args.query, "retrieved": time.strftime("%Y-%m-%d")}
    else:
        records = []
        for src in args.sources:
            try:
                fn = s2_links if src == "s2" else openalex_links
                found = fn(args.ident, args.cmd, args.limit)
            except Exception as exc:
                print(f"[{src}] failed: {exc}", file=sys.stderr)
                found = []
            print(f"[{src}] {len(found)} records", file=sys.stderr)
            records.extend(found)
        meta = {"query": f"{args.cmd}:{args.ident}", "retrieved": time.strftime("%Y-%m-%d")}

    merged = merge_records(records)
    merged.sort(key=lambda r: (-(r.get("cited_by") or 0), -(r.get("year") or 0)))
    if args.json:
        print(json.dumps(merged, ensure_ascii=False, indent=1))
    else:
        print_table(merged)
    if args.out:
        write_jsonl(merged, args.out, meta)
        print(f"appended {len(merged)} records to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
