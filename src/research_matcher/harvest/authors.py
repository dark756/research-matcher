"""OpenAlex author batches (plan §4.6): h-index / works_count for senior & lead authors at target universities."""
from __future__ import annotations

import json
import sqlite3
from typing import Callable

from ..http_client import HttpClient


def pending_author_ids(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        """SELECT DISTINCT a FROM (
             SELECT senior_author_id AS a FROM work_universities UNION SELECT lead_author_id FROM work_universities)
           WHERE a IS NOT NULL AND a NOT LIKE 'X:%' AND a IN (SELECT author_id FROM authors WHERE h_index IS NULL)
           ORDER BY a""").fetchall()
    return [r[0] for r in rows]


def fetch_authors(http: HttpClient, conn: sqlite3.Connection, batch: int = 100,
                  max_batches: int | None = None, log: Callable[[str], None] = print) -> int:
    ids = pending_author_ids(conn)
    log(f"Author batches: {len(ids)} senior/lead authors need stats ({-(-len(ids) // batch)} calls)")
    done = 0
    for n, i in enumerate(range(0, len(ids), batch)):
        if max_batches and n >= max_batches:
            break
        chunk = ids[i:i + batch]
        res = http.openalex("/authors", {
            "filter": "openalex_id:" + "|".join(chunk), "per_page": batch,
            "select": "id,display_name,orcid,summary_stats,last_known_institutions,works_count"})
        seen = set()
        conn.execute("BEGIN")
        for a in res.get("results", []):
            aid = a["id"].rsplit("/", 1)[-1]
            seen.add(aid)
            conn.execute(
                "UPDATE authors SET name=?, orcid=COALESCE(?, orcid), h_index=?, works_count=?, "
                "last_known_insts_json=? WHERE author_id=?",
                (a.get("display_name"), a.get("orcid"), (a.get("summary_stats") or {}).get("h_index") or 0,
                 a.get("works_count"),
                 json.dumps([(x.get("display_name"), x.get("id", "").rsplit("/", 1)[-1])
                             for x in a.get("last_known_institutions") or []]), aid))
        for aid in set(chunk) - seen:  # merged/unknown author ids: mark so we don't re-query forever
            conn.execute("UPDATE authors SET h_index=0 WHERE author_id=? AND h_index IS NULL", (aid,))
        conn.execute("COMMIT")
        done += len(chunk)
    return done
