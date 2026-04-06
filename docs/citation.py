#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
import time
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
import requests

ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_BATCH = f"{S2_BASE}/paper/batch"
S2_SEARCH = f"{S2_BASE}/paper/search"

OA_WORKS = "https://api.openalex.org/works"
CR_WORKS = "https://api.crossref.org/works"


def load_items(path: str) -> List[Dict[str, Any]]:
    """Load JSON array/object or JSONL."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []

    # JSON first
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for k in ("papers", "items", "data"):
                if k in data and isinstance(data[k], list):
                    return data[k]
            return [data]
    except json.JSONDecodeError:
        pass

    # JSONL fallback
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def dump_items(path: str, items: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def extract_arxiv_id(url: str) -> Optional[str]:
    if not url:
        return None
    m = ARXIV_RE.search(url)
    return m.group(1) if m else None


def title_similarity(a: str, b: str) -> float:
    a = (a or "").lower().strip()
    b = (b or "").lower().strip()
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def chunked(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def s2_headers(api_key: Optional[str]) -> Dict[str, str]:
    h = {"User-Agent": "paper-citation-fetcher/1.0"}
    if api_key:
        h["x-api-key"] = api_key
    return h


def s2_batch_lookup(session: requests.Session, arxiv_ids: List[str], api_key: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """
    Batch query Semantic Scholar by ARXIV:<id>.
    Returns map arxiv_id -> paper_json
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not arxiv_ids:
        return out

    fields = "title,year,citationCount,externalIds,url"
    for batch in chunked([f"ARXIV:{x}" for x in arxiv_ids], 100):
        r = session.post(
            S2_BATCH,
            params={"fields": fields},
            json={"ids": batch},
            headers=s2_headers(api_key),
            timeout=30,
        )
        if r.status_code != 200:
            continue
        data = r.json()
        if not isinstance(data, list):
            continue
        for rec in data:
            if not isinstance(rec, dict):
                continue
            ext = rec.get("externalIds") or {}
            ax = ext.get("ArXiv") or ext.get("arXiv") or ext.get("ARXIV")
            if ax:
                out[str(ax)] = rec
        time.sleep(0.2)
    return out


def s2_search_by_title(session: requests.Session, title: str, api_key: Optional[str]) -> Optional[Dict[str, Any]]:
    if not title:
        return None
    params = {
        "query": title,
        "limit": 5,
        "fields": "title,year,citationCount,externalIds,url",
    }
    r = session.get(S2_SEARCH, params=params, headers=s2_headers(api_key), timeout=30)
    if r.status_code != 200:
        return None
    candidates = (r.json().get("data") or [])
    best = None
    best_score = 0.0
    for c in candidates:
        score = title_similarity(title, c.get("title", ""))
        if score > best_score:
            best_score = score
            best = c
    return best if best_score >= 0.72 else None


def openalex_search(session: requests.Session, title: str, arxiv_id: Optional[str], mailto: Optional[str]) -> Optional[Dict[str, Any]]:
    params = {"per_page": 5}
    if mailto:
        params["mailto"] = mailto

    # Prefer arXiv search if available
    if arxiv_id:
        params["search"] = f"arXiv:{arxiv_id}"
        r = session.get(OA_WORKS, params=params, headers={"User-Agent": "paper-citation-fetcher/1.0"}, timeout=30)
        if r.status_code == 200:
            res = r.json().get("results") or []
            if res:
                return res[0]

    if not title:
        return None

    params["search"] = title
    r = session.get(OA_WORKS, params=params, headers={"User-Agent": "paper-citation-fetcher/1.0"}, timeout=30)
    if r.status_code != 200:
        return None
    res = r.json().get("results") or []
    if not res:
        return None

    # pick best title match
    best = None
    best_score = 0.0
    for w in res:
        score = title_similarity(title, w.get("title", ""))
        if score > best_score:
            best_score = score
            best = w
    return best if best_score >= 0.72 else res[0]


def crossref_cited_by_count(session: requests.Session, doi: str, mailto: Optional[str]) -> Optional[int]:
    if not doi:
        return None
    # Crossref recommends identifying via UA / mailto; and supports select for smaller payload.  [oai_citation:1‡www.crossref.org](https://www.crossref.org/documentation/cited-by/?utm_source=chatgpt.com)
    headers = {"User-Agent": f"paper-citation-fetcher/1.0 (mailto:{mailto or 'unknown'})"}
    params = {"select": "DOI,is-referenced-by-count,title"}
    r = session.get(f"{CR_WORKS}/{requests.utils.quote(doi)}", params=params, headers=headers, timeout=30)
    if r.status_code != 200:
        return None
    msg = r.json().get("message") or {}
    return msg.get("is-referenced-by-count")


def choose_best_count(rec: Dict[str, Any]) -> Tuple[Optional[int], Optional[str]]:
    s2 = (rec.get("citations", {}) or {}).get("semantic_scholar") or {}
    oa = (rec.get("citations", {}) or {}).get("openalex") or {}
    cr = (rec.get("citations", {}) or {}).get("crossref") or {}

    if isinstance(s2.get("citationCount"), int):
        return s2["citationCount"], "semantic_scholar"
    if isinstance(oa.get("cited_by_count"), int):
        return oa["cited_by_count"], "openalex"
    if isinstance(cr.get("is_referenced_by_count"), int):
        return cr["is_referenced_by_count"], "crossref"
    return None, None


def write_csv(path: str, items: List[Dict[str, Any]]) -> None:
    cols = [
        "label", "title", "url",
        "citation_count", "citation_source",
        "semantic_scholar_citationCount",
        "openalex_cited_by_count",
        "crossref_is_referenced_by_count",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for it in items:
            c = it.get("citations") or {}
            s2 = c.get("semantic_scholar") or {}
            oa = c.get("openalex") or {}
            cr = c.get("crossref") or {}
            w.writerow({
                "label": it.get("label"),
                "title": it.get("title"),
                "url": it.get("url"),
                "citation_count": it.get("citation_count"),
                "citation_source": it.get("citation_source"),
                "semantic_scholar_citationCount": s2.get("citationCount"),
                "openalex_cited_by_count": oa.get("cited_by_count"),
                "crossref_is_referenced_by_count": cr.get("is_referenced_by_count"),
            })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input json/jsonl file")
    ap.add_argument("--out", dest="outp", required=True, help="output json file (with citations)")
    ap.add_argument("--csv", dest="csvp", default="", help="optional output csv")
    ap.add_argument("--semantic_api_key", default="", help="optional Semantic Scholar API key")
    ap.add_argument("--openalex", action="store_true", help="also query OpenAlex (recommended)")
    ap.add_argument("--crossref", action="store_true", help="also query Crossref when DOI is known (optional)")
    ap.add_argument("--mailto", default="", help="mailto for OpenAlex/Crossref polite usage")
    ap.add_argument("--sleep", type=float, default=0.15, help="sleep between per-item fallbacks (seconds)")
    args = ap.parse_args()

    items = load_items(args.inp)
    sess = requests.Session()

    # 1) batch Semantic Scholar via arXiv IDs
    arxiv_ids = []
    for it in items:
        ax = extract_arxiv_id((it.get("url") or "").strip())
        if ax:
            arxiv_ids.append(ax)

    s2_map = s2_batch_lookup(sess, sorted(set(arxiv_ids)), args.semantic_api_key or None)

    enriched: List[Dict[str, Any]] = []
    for it in tqdm(items, desc="Processing records", total=len(items)):
        rec = dict(it)
        rec.setdefault("citations", {})

        title = (rec.get("title") or "").strip()
        url = (rec.get("url") or "").strip()
        ax = extract_arxiv_id(url)

        # Semantic Scholar (batch hit first)
        s2 = s2_map.get(ax) if ax else None
        if not s2 and title:
            s2 = s2_search_by_title(sess, title, args.semantic_api_key or None)
            time.sleep(args.sleep)

        if s2:
            rec["citations"]["semantic_scholar"] = {
                "citationCount": s2.get("citationCount"),
                "year": s2.get("year"),
                "url": s2.get("url"),
                "externalIds": s2.get("externalIds"),
                "title_matched": s2.get("title"),
            }
        else:
            rec["citations"]["semantic_scholar"] = None

        # OpenAlex (optional)
        if args.openalex:
            oa = openalex_search(sess, title, ax, args.mailto or None)
            time.sleep(args.sleep)
            if oa:
                rec["citations"]["openalex"] = {
                    "cited_by_count": oa.get("cited_by_count"),
                    "id": oa.get("id"),
                    "doi": oa.get("doi") or "",
                    "title_matched": oa.get("title"),
                }
            else:
                rec["citations"]["openalex"] = None

        # Crossref (optional, needs DOI)
        if args.crossref:
            doi = ""
            # try DOI from Semantic Scholar externalIds, then OpenAlex
            s2_ext = (rec.get("citations", {}).get("semantic_scholar") or {}).get("externalIds") or {}
            doi = s2_ext.get("DOI") or s2_ext.get("doi") or ""
            if not doi:
                doi = (rec.get("citations", {}).get("openalex") or {}).get("doi") or ""
                doi = doi.replace("https://doi.org/", "").strip()

            if doi:
                cr_cnt = crossref_cited_by_count(sess, doi, args.mailto or None)
                time.sleep(args.sleep)
                rec["citations"]["crossref"] = {"doi": doi, "is_referenced_by_count": cr_cnt}
            else:
                rec["citations"]["crossref"] = None

        # unified best count
        best, src = choose_best_count(rec)
        rec["citation_count"] = best
        rec["citation_source"] = src

        enriched.append(rec)

    dump_items(args.outp, enriched)
    if args.csvp:
        write_csv(args.csvp, enriched)

    print(f"Done. {len(enriched)} records -> {args.outp}")


if __name__ == "__main__":
    main()