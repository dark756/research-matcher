"""Cross-source dedupe: DOI → arXiv ID → S2 paperId → fuzzy title (>=95) + first-author surname.

Rows are never deleted. Every work gets `canonical_id`; the canonical row lists the others in `merged_ids_json`.
Downstream stages should use `WHERE canonical_id = work_id`. Prefers conference-paper over preprint.
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict

from rapidfuzz import fuzz

from ..harvest.csrankings import name_norm
from .semantic_scholar import is_real_venue


class UnionFind:
    def __init__(self) -> None:
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        self.p[self.find(a)] = self.find(b)


def surname(name: str | None) -> str:
    toks = name_norm(name or "").split()
    return toks[-1] if toks else ""


def canonical_rank(r: sqlite3.Row) -> tuple:
    return (r["type"] != "conference-paper", not is_real_venue(r["venue_name"]), r["abstract"] is None,
            r["publication_date"] or "9999", r["work_id"])


def run_dedupe(conn: sqlite3.Connection, title_threshold: int = 95) -> dict[str, int]:
    rows = conn.execute("SELECT work_id, doi, arxiv_id, s2_paper_id, title_norm, year, type, venue_name, abstract, "
                        "publication_date FROM works").fetchall()
    first = dict(conn.execute(
        "SELECT a.work_id, au.name FROM authorships a JOIN authors au USING(author_id) WHERE a.position_idx=0"
    ).fetchall())
    uf = UnionFind()
    reasons: dict[str, str] = {}
    for key in ("doi", "arxiv_id", "s2_paper_id"):
        seen: dict[str, str] = {}
        for r in rows:
            v = r[key]
            if not v:
                continue
            if key == "doi" and v.startswith("10.48550/arxiv."):
                continue  # same thing as arxiv_id; avoids double counting
            if v in seen:
                uf.union(r["work_id"], seen[v])
                reasons[r["work_id"]] = key
            else:
                seen[v] = r["work_id"]
    # fuzzy title within first-author-surname blocks
    blocks: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for r in rows:
        s = surname(first.get(r["work_id"]))
        if s and len(r["title_norm"] or "") > 15:
            blocks[s].append(r)
    for items in blocks.values():
        for i, a in enumerate(items):
            for b in items[i + 1:]:
                if uf.find(a["work_id"]) == uf.find(b["work_id"]) or abs((a["year"] or 0) - (b["year"] or 0)) > 1:
                    continue
                if fuzz.ratio(a["title_norm"], b["title_norm"]) >= title_threshold:
                    uf.union(a["work_id"], b["work_id"])
                    reasons[b["work_id"]] = "title"
    groups: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for r in rows:
        groups[uf.find(r["work_id"])].append(r)
    merged = 0
    conn.execute("BEGIN")
    for members in groups.values():
        best = min(members, key=canonical_rank)
        others = [m["work_id"] for m in members if m["work_id"] != best["work_id"]]
        for m in members:
            conn.execute("UPDATE works SET canonical_id=? WHERE work_id=?", (best["work_id"], m["work_id"]))
        conn.execute("UPDATE works SET merged_ids_json=? WHERE work_id=?", (json.dumps(others), best["work_id"]))
        if others:
            merged += len(others)
            # canonical absorbs missing ids/universities from its duplicates
            for o in others:
                conn.execute("UPDATE works SET arxiv_id=COALESCE(arxiv_id,(SELECT arxiv_id FROM works WHERE work_id=?)),"
                             " doi=COALESCE(doi,(SELECT doi FROM works WHERE work_id=?)) WHERE work_id=?",
                             (o, o, best["work_id"]))
                conn.execute("INSERT OR IGNORE INTO work_universities(work_id, inst_id, senior_author_id, "
                             "lead_author_id) SELECT ?, inst_id, senior_author_id, lead_author_id "
                             "FROM work_universities WHERE work_id=?", (best["work_id"], o))
    conn.execute("COMMIT")
    return {"works": len(rows), "canonical": len(groups), "duplicates": merged,
            "by_title": sum(1 for v in reasons.values() if v == "title")}


def sample_groups(conn: sqlite3.Connection, n: int = 30) -> list[dict]:
    out = []
    for r in conn.execute("SELECT work_id, merged_ids_json FROM works WHERE merged_ids_json IS NOT NULL "
                          "AND merged_ids_json != '[]' ORDER BY RANDOM() LIMIT ?", (n,)):
        ids = [r["work_id"], *json.loads(r["merged_ids_json"])]
        out.append({"ids": ids, "titles": [conn.execute("SELECT title||' ['||type||', '||COALESCE(venue_name,'?')||']' "
                                                        "FROM works WHERE work_id=?", (i,)).fetchone()[0] for i in ids]})
    return out
