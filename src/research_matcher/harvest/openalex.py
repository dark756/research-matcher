"""OpenAlex harvest: Queries A (main), B (proceedings journals), C (keyword supplement). Plan §4."""
from __future__ import annotations

import json
import re
import sqlite3
import time
import unicodedata
import zlib
from typing import Any, Callable, Iterator

from ..config import load_settings, load_subdomains, load_universities
from ..http_client import HttpClient
from .abstracts import abstract_of

SELECT = ("id,doi,ids,title,publication_year,publication_date,type,language,primary_location,locations,"
          "authorships,abstract_inverted_index,primary_topic,topics,keywords,referenced_works,related_works,"
          "cited_by_count,fwci,citation_normalized_percentile,open_access,best_oa_location,is_retracted")

SOURCE_RANK = {"A": 0, "B": 1, "C": 2}
ARXIV_DOI = re.compile(r"10\.48550/arxiv\.(.+)$", re.I)
ARXIV_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/([^\s?#]+?)(?:\.pdf)?(?:v\d+)?$", re.I)


def short_id(url: str | None) -> str | None:
    return url.rsplit("/", 1)[-1] if url else None


def norm_title(title: str | None) -> str:
    t = unicodedata.normalize("NFKD", title or "").encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", t).split())


def strip_doi(doi: str | None) -> str | None:
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.I).lower() if doi else None


def derive_arxiv_id(work: dict[str, Any]) -> str | None:
    doi = strip_doi(work.get("doi"))
    if doi and (m := ARXIV_DOI.search(doi)):
        return re.sub(r"v\d+$", "", m.group(1))
    for loc in [work.get("primary_location") or {}, *(work.get("locations") or [])]:
        for u in (loc.get("landing_page_url"), loc.get("pdf_url")):
            if u and (m := ARXIV_URL.search(u)):
                return m.group(1)
    return None


def filter_a(cfg: dict[str, Any], inst_ids: list[str]) -> str:
    return ",".join([
        f"authorships.institutions.lineage:{'|'.join(inst_ids)}",
        f"from_publication_date:{cfg['from_publication_date']}",
        f"type:{'|'.join(cfg['types'])}",
        f"primary_topic.field.id:{cfg['field']}",
        "is_retracted:false", "is_paratext:false",
    ])


def filter_b(cfg: dict[str, Any], inst_ids: list[str]) -> str:
    return ",".join([
        f"authorships.institutions.lineage:{'|'.join(inst_ids)}",
        f"from_publication_date:{cfg['from_publication_date']}",
        "type:article",
        f"locations.source.id:{'|'.join(cfg['proceedings_sources'])}",
        "is_retracted:false", "is_paratext:false",
    ])


def filter_c(cfg: dict[str, Any], inst_ids: list[str], query: str) -> str:
    return ",".join([
        f"authorships.institutions.lineage:{'|'.join(inst_ids)}",
        f"from_publication_date:{cfg['from_publication_date']}",
        f"type:{'|'.join(cfg['types'])}",
        f"primary_topic.field.id:!{cfg['field']}",
        f"title_and_abstract.search:{query}",
        "is_retracted:false", "is_paratext:false",
    ])


def iter_pages(http: HttpClient, filt: str, per_page: int, max_pages: int | None = None) -> Iterator[list[dict]]:
    cursor, n = "*", 0
    while cursor:
        res = http.openalex("/works", {"filter": filt, "per_page": per_page, "cursor": cursor, "select": SELECT})
        results = res.get("results", [])
        if not results:
            return
        yield results
        n += 1
        if max_pages and n >= max_pages:
            return
        cursor = (res.get("meta") or {}).get("next_cursor")


def parse_work(w: dict[str, Any], source: str, target_ids: set[str], type_override: str | None = None) -> dict:
    """Flatten an OpenAlex work to a `works` row + authorship rows + work_universities rows."""
    work_id = short_id(w["id"])
    prim = w.get("primary_location") or {}
    src = prim.get("source") or {}
    best = w.get("best_oa_location") or {}
    oa = w.get("open_access") or {}
    pt = w.get("primary_topic") or {}
    abstract = abstract_of(w)
    doi = strip_doi(w.get("doi"))
    arxiv = derive_arxiv_id(w)
    venue_type = src.get("type")  # conference / journal / repository ...
    topics = [{"id": short_id(t.get("id")), "name": t.get("display_name"), "score": t.get("score"),
               "subfield": (t.get("subfield") or {}).get("display_name"),
               "field": (t.get("field") or {}).get("display_name")} for t in w.get("topics") or []]
    keywords = [{"name": k.get("display_name"), "score": k.get("score")} for k in w.get("keywords") or []]
    pct = w.get("citation_normalized_percentile") or {}
    pdf = oa.get("oa_url") or best.get("pdf_url")
    row = {
        "work_id": work_id, "doi": doi, "arxiv_id": arxiv,
        "title": w.get("title"), "title_norm": norm_title(w.get("title")),
        "year": w.get("publication_year"), "publication_date": w.get("publication_date"),
        "type": type_override or w.get("type"), "type_raw": w.get("type"), "language": w.get("language"),
        "venue_name": src.get("display_name"), "venue_source_id": short_id(src.get("id")), "venue_type": venue_type,
        "doi_url": f"https://doi.org/{doi}" if doi else None, "openalex_url": w["id"],
        "arxiv_abs_url": f"https://arxiv.org/abs/{arxiv}" if arxiv else None,
        "landing_page_url": prim.get("landing_page_url"), "oa_pdf_url": pdf,
        "abstract": abstract, "abstract_source": "openalex" if abstract else None,
        "abstract_missing": 0 if abstract else 1,
        "primary_topic": pt.get("display_name"), "subfield": (pt.get("subfield") or {}).get("display_name"),
        "field": (pt.get("field") or {}).get("display_name"),
        "topics_json": json.dumps(topics), "keywords_json": json.dumps(keywords),
        "referenced_works_json": json.dumps([short_id(x) for x in w.get("referenced_works") or []]),
        "related_works_json": json.dumps([short_id(x) for x in w.get("related_works") or []]),
        "cited_by_count": w.get("cited_by_count"), "fwci": w.get("fwci"),
        "top_10pct": 1 if pct.get("is_in_top_10_percent") else 0,
        "is_oa": 1 if oa.get("is_oa") else 0, "oa_status": oa.get("oa_status"), "license": best.get("license"),
        "harvest_source": source, "harvested_at": time.time(),
        "raw_json": zlib.compress(json.dumps(w).encode()),
    }
    subfield_id = short_id((pt.get("subfield") or {}).get("id"))

    authorships, per_uni = [], {}
    for idx, a in enumerate(w.get("authorships") or []):
        au = a.get("author") or {}
        # Authors OpenAlex couldn't disambiguate have no id; keep them under a synthetic stable id so their
        # university attribution (and professor candidacy) isn't lost.
        aid = short_id(au.get("id")) or f"X:{work_id}:{idx}"
        insts = a.get("institutions") or []
        lineage = {short_id(x) for i in insts for x in (i.get("lineage") or [i.get("id")]) if x}
        unis = sorted(lineage & target_ids)
        authorships.append({
            "work_id": work_id, "author_id": aid, "name": au.get("display_name"), "orcid": au.get("orcid"),
            "position_idx": idx, "author_position": a.get("author_position"),
            "is_corresponding": 1 if a.get("is_corresponding") else 0,
            "inst_ids_json": json.dumps(sorted(lineage)),
            "raw_affiliation": "; ".join(a.get("raw_affiliation_strings") or [])[:1000],
        })
        for u in unis:
            d = per_uni.setdefault(u, {"first": aid, "last": aid})
            d["last"] = aid
    work_unis = [{"work_id": work_id, "inst_id": u, "lead_author_id": d["first"], "senior_author_id": d["last"]}
                 for u, d in per_uni.items()]
    return {"row": row, "authorships": authorships, "work_unis": work_unis, "subfield_id": subfield_id}


def store_work(conn: sqlite3.Connection, parsed: dict, refresh: bool = False) -> bool:
    """Insert a parsed work. Existing works keep their (higher-priority) harvest source unless `refresh`."""
    row = parsed["row"]
    existing = conn.execute("SELECT harvest_source FROM works WHERE work_id=?", (row["work_id"],)).fetchone()
    if existing and SOURCE_RANK.get(existing[0], 9) <= SOURCE_RANK[row["harvest_source"]]:
        if not refresh:
            return False
        row["harvest_source"] = existing[0]  # refresh re-parses but never downgrades the source label
    cols = list(row)
    conn.execute(f"INSERT OR REPLACE INTO works ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                 [row[c] for c in cols])
    wid = row["work_id"]
    conn.execute("DELETE FROM works_fts WHERE work_id=?", (wid,))
    kws = " ".join(k["name"] for k in json.loads(row["keywords_json"]) if k.get("name"))
    conn.execute("INSERT INTO works_fts(title, keywords, abstract, work_id) VALUES (?,?,?,?)",
                 (row["title"] or "", kws, row["abstract"] or "", wid))
    conn.execute("DELETE FROM authorships WHERE work_id=?", (wid,))
    conn.execute("DELETE FROM work_universities WHERE work_id=?", (wid,))
    for a in parsed["authorships"]:
        conn.execute("INSERT OR IGNORE INTO authors(author_id, name, orcid) VALUES (?,?,?)",
                     (a["author_id"], a["name"], a["orcid"]))
        conn.execute("INSERT OR REPLACE INTO authorships VALUES (?,?,?,?,?,?,?)",
                     (a["work_id"], a["author_id"], a["position_idx"], a["author_position"],
                      a["is_corresponding"], a["inst_ids_json"], a["raw_affiliation"]))
    for u in parsed["work_unis"]:
        conn.execute("INSERT OR REPLACE INTO work_universities(work_id, inst_id, senior_author_id, lead_author_id) "
                     "VALUES (?,?,?,?)", (u["work_id"], u["inst_id"], u["senior_author_id"], u["lead_author_id"]))
    return True


def harvest_query(http: HttpClient, conn: sqlite3.Connection, filt: str, source: str, *, target_ids: set[str],
                  per_page: int, type_override: str | None = None, exclude_subfields: set[str] | None = None,
                  max_pages: int | None = None, refresh: bool = False,
                  on_page: Callable[[int], None] | None = None) -> dict[str, int]:
    stats = dict(seen=0, stored=0, skipped_existing=0, no_target_uni=0, excluded_subfield=0)
    for page in iter_pages(http, filt, per_page, max_pages):
        conn.execute("BEGIN")
        try:
            for w in page:
                stats["seen"] += 1
                p = parse_work(w, source, target_ids, type_override)
                if not p["work_unis"]:
                    stats["no_target_uni"] += 1
                    continue
                if exclude_subfields and p["subfield_id"] in exclude_subfields:
                    stats["excluded_subfield"] += 1
                    continue
                if store_work(conn, p, refresh):
                    stats["stored"] += 1
                else:
                    stats["skipped_existing"] += 1
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        if on_page:
            on_page(stats["seen"])
    return stats


def run_harvest(http: HttpClient, conn: sqlite3.Connection, *, do_b: bool | None = None, do_c: bool | None = None,
                max_pages: int | None = None, refresh: bool = False,
                log: Callable[[str], None] = print) -> dict[str, dict[str, int]]:
    cfg = load_settings().harvest
    inst_ids = [u.openalex_id for u in load_universities()]
    targets = set(inst_ids)
    per_page = cfg["per_page"]
    excl = {str(x) for x in cfg.get("exclude_subfields") or []}
    do_b = cfg["include_proceedings_journals"] if do_b is None else do_b
    do_c = cfg["keyword_supplement"] if do_c is None else do_c
    credits_start = http.credits_remaining()
    out: dict[str, dict[str, int]] = {}
    started = time.time()

    def prog(label):
        return lambda n: log(f"  {label}: {n} works processed") if n % 2000 < per_page else None

    log("Query A (main set)…")
    out["A"] = harvest_query(http, conn, filter_a(cfg, inst_ids), "A", target_ids=targets, per_page=per_page,
                             exclude_subfields=excl, max_pages=max_pages, refresh=refresh, on_page=prog("A"))
    if do_b:
        log("Query B (proceedings journals)…")
        out["B"] = harvest_query(http, conn, filter_b(cfg, inst_ids), "B", target_ids=targets, per_page=per_page,
                                 type_override="conference-paper", exclude_subfields=excl, max_pages=max_pages,
                                 refresh=refresh)
    if do_c:
        for sd in load_subdomains():
            if not sd.get("openalex_query"):
                continue
            log(f"Query C ({sd['id']})…")
            out[f"C:{sd['id']}"] = harvest_query(
                http, conn, filter_c(cfg, inst_ids, sd["openalex_query"]), "C", target_ids=targets,
                per_page=per_page, exclude_subfields=excl, max_pages=max_pages or 3, refresh=refresh)
    credits_end = http.credits_remaining()
    out["_meta"] = {"credits_used": (credits_start - credits_end) if credits_start and credits_end else -1,
                    "seconds": int(time.time() - started)}
    conn.execute("INSERT INTO runs VALUES (?,?,?,?,?)",
                 ("harvest", started, time.time(), json.dumps({"max_pages": max_pages, "refresh": refresh}),
                  json.dumps(out)))
    return out
