"""SQLite FTS5 keyword retrieval (BM25) over canonical works: title x3, keywords x2, abstract x1 (plan §8.4)."""
from __future__ import annotations

import re
import sqlite3

STOP = {"a", "an", "the", "of", "for", "and", "or", "in", "on", "to", "with", "by", "via", "using", "is", "are", "from"}


def fts_query(text: str) -> str:
    """Turn free text into an OR query of quoted tokens (safe against FTS5 syntax characters)."""
    toks = [t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-@+.]*", text) if t.lower() not in STOP and len(t) > 1]
    return " OR ".join(f'"{t.rstrip(".-")}"' for t in dict.fromkeys(toks))


def bm25_search(conn: sqlite3.Connection, text: str, k: int = 100) -> list[tuple[str, float]]:
    """Returns [(work_id, score)] best first (score = -bm25, higher is better). Canonical works only."""
    q = fts_query(text)
    if not q:
        return []
    rows = conn.execute(
        "SELECT f.work_id, bm25(works_fts, 3.0, 2.0, 1.0, 0.0) AS s FROM works_fts f JOIN works w ON w.work_id=f.work_id "
        "WHERE works_fts MATCH ? AND w.canonical_id=w.work_id ORDER BY s LIMIT ?", (q, k)).fetchall()
    return [(r[0], -r[1]) for r in rows]


def rebuild_fts(conn: sqlite3.Connection) -> int:
    """Rebuild the index from `works` (title, keywords, abstract); picks up abstracts filled in by S2."""
    import json

    conn.execute("BEGIN")
    conn.execute("DELETE FROM works_fts")
    n = 0
    for r in conn.execute("SELECT work_id, title, abstract, keywords_json FROM works").fetchall():
        kws = " ".join(k["name"] for k in json.loads(r["keywords_json"] or "[]") if k.get("name"))
        conn.execute("INSERT INTO works_fts(title, keywords, abstract, work_id) VALUES (?,?,?,?)",
                     (r["title"] or "", kws, r["abstract"] or "", r["work_id"]))
        n += 1
    conn.execute("COMMIT")
    return n
