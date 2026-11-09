"""Semantic Scholar enrichment, pass 1 (plan §2.1): /paper/batch, <=500 ids, through the global 1 req/s limiter."""
from __future__ import annotations

import json
import re
import sqlite3
import time
import zlib
from typing import Any, Callable

from ..config import load_settings
from ..http_client import HttpClient, HttpError

FAKE_VENUES = re.compile(r"^(arxiv|arxiv\.org|arxiv e-prints|corr|ssrn|research square|preprint)\b", re.I)


def s2_id(doi: str | None, arxiv_id: str | None) -> str | None:
    if arxiv_id:
        return f"ARXIV:{arxiv_id}"
    if doi:
        return f"DOI:{doi}"
    return None


def is_real_venue(name: str | None) -> bool:
    return bool(name) and not FAKE_VENUES.match(name.strip())


def apply_s2(conn: sqlite3.Connection, work_id: str, rec: dict[str, Any]) -> None:
    ext = rec.get("externalIds") or {}
    pv = rec.get("publicationVenue") or {}
    venue = pv.get("name") or rec.get("venue")
    venue = venue if is_real_venue(venue) else None
    tldr = (rec.get("tldr") or {}).get("text")
    pdf = (rec.get("openAccessPdf") or {}).get("url")
    row = conn.execute("SELECT abstract, arxiv_id, venue_name, venue_type, oa_pdf_url FROM works WHERE work_id=?",
                       (work_id,)).fetchone()
    abstract, arxiv, venue_name, venue_type, oa_pdf = row
    new_abs = rec.get("abstract")
    fill_abs = not abstract and bool(new_abs)
    # displayed venue prefers S2 when OpenAlex only knows the repository (arXiv)
    if venue and (venue_type == "repository" or not is_real_venue(venue_name)):
        venue_name = venue
    conn.execute(
        """UPDATE works SET s2_status='ok', s2_fetched_at=?, s2_paper_id=?, s2_corpus_id=?, dblp_key=?,
           arxiv_id=COALESCE(arxiv_id, ?), venue_s2=?, venue_s2_type=?, venue_s2_aliases=?, venue_name=?,
           tldr=?, s2_citations=?, s2_influential_citations=?, oa_pdf_url=COALESCE(oa_pdf_url, ?),
           s2_raw_json=? WHERE work_id=?""",
        (time.time(), rec.get("paperId"), str(ext["CorpusId"]) if ext.get("CorpusId") else None, ext.get("DBLP"),
         ext.get("ArXiv"), venue, pv.get("type"), json.dumps(pv.get("alternate_names") or []), venue_name, tldr,
         rec.get("citationCount"), rec.get("influentialCitationCount"), pdf,
         zlib.compress(json.dumps(rec).encode()), work_id))
    if ext.get("ArXiv") and not arxiv:
        conn.execute("UPDATE works SET arxiv_abs_url=? WHERE work_id=?",
                     (f"https://arxiv.org/abs/{ext['ArXiv']}", work_id))
    if fill_abs:
        conn.execute("UPDATE works SET abstract=?, abstract_source='s2', abstract_missing=0 WHERE work_id=?",
                     (new_abs, work_id))
        conn.execute("UPDATE works_fts SET abstract=? WHERE work_id=?", (new_abs, work_id))


def run_pass1(http: HttpClient, conn: sqlite3.Connection, *, max_batches: int | None = None,
              retry_failed: bool = True, log: Callable[[str], None] = print) -> dict[str, int]:
    cfg = load_settings().semantic_scholar
    states = ("failed",) if retry_failed else ()
    marks = ",".join("?" * len(states))
    where = "s2_status IS NULL" + (f" OR s2_status IN ({marks})" if states else "")
    rows = conn.execute(f"SELECT work_id, doi, arxiv_id FROM works WHERE {where} ORDER BY work_id", states).fetchall()
    todo = [(w, s2_id(d, a)) for w, d, a in rows]
    no_id = [w for w, i in todo if i is None]
    todo = [(w, i) for w, i in todo if i]
    for w in no_id:
        conn.execute("UPDATE works SET s2_status='no_id' WHERE work_id=?", (w,))
    size = cfg["batch_size"]
    stats = dict(ok=0, not_found=0, failed=0, no_id=len(no_id), requests=0)
    log(f"S2 pass 1: {len(todo)} ids → {-(-len(todo) // size)} requests at ≤1 req/s ({len(no_id)} works have no id)")
    for n, i in enumerate(range(0, len(todo), size)):
        if max_batches and n >= max_batches:
            break
        chunk = todo[i:i + size]
        try:
            res = http.s2_post("/paper/batch", {"fields": cfg["pass1_fields"]}, {"ids": [x[1] for x in chunk]})
        except HttpError as e:
            log(f"  batch {n}: {e} → marked failed (resumable)")
            conn.execute("BEGIN")
            for w, _ in chunk:
                conn.execute("UPDATE works SET s2_status='failed', s2_fetched_at=? WHERE work_id=?", (time.time(), w))
            conn.execute("COMMIT")
            stats["failed"] += len(chunk)
            continue
        stats["requests"] += 1
        conn.execute("BEGIN")
        for (w, _), rec in zip(chunk, res):
            if rec is None:
                conn.execute("UPDATE works SET s2_status='not_found', s2_fetched_at=? WHERE work_id=?",
                             (time.time(), w))
                stats["not_found"] += 1
            else:
                apply_s2(conn, w, rec)
                stats["ok"] += 1
        conn.execute("COMMIT")
        if n % 10 == 0:
            log(f"  batch {n + 1}: ok={stats['ok']} not_found={stats['not_found']}")
    return stats
