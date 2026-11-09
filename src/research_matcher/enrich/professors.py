"""Professor verification against CSRankings (plan §5.2)."""
from __future__ import annotations

import sqlite3
from collections import defaultdict

from rapidfuzz import fuzz, process

from ..config import load_universities
from ..harvest.csrankings import name_norm


def _orcid(o: str | None) -> str:
    return (o or "").rsplit("/", 1)[-1].strip().upper()


def _initial_match(a: str, b: str) -> bool:
    """'j smith' vs 'john smith': same surname and one given name is just the initial of the other."""
    ta, tb = a.split(), b.split()
    if len(ta) < 2 or len(tb) < 2 or ta[-1] != tb[-1]:
        return False
    fa, fb = ta[0], tb[0]
    return fa[0] == fb[0] and (len(fa) == 1 or len(fb) == 1)


class FacultyIndex:
    def __init__(self, conn: sqlite3.Connection):
        self.by_aff: dict[str, dict[str, int]] = defaultdict(dict)  # affiliation -> name_norm -> rowid
        self.orcids: dict[tuple[str, str], int] = {}
        for r in conn.execute("SELECT rowid, name_norm, affiliation, orcid FROM faculty"):
            self.by_aff[r["affiliation"]][r["name_norm"]] = r["rowid"]
            if _orcid(r["orcid"]):
                self.orcids[(r["affiliation"], _orcid(r["orcid"]))] = r["rowid"]

    def match(self, aff: str, name: str | None, orcid: str | None) -> int | None:
        if _orcid(orcid) and (aff, _orcid(orcid)) in self.orcids:
            return self.orcids[(aff, _orcid(orcid))]
        n = name_norm(name or "")
        names = self.by_aff.get(aff, {})
        if not n or not names:
            return None
        if n in names:
            return names[n]
        best = process.extractOne(n, names.keys(), scorer=fuzz.ratio, score_cutoff=90)
        if best:
            return names[best[0]]
        for cand in names:
            if _initial_match(n, cand):
                return names[cand]
        return None


def run_professors(conn: sqlite3.Connection) -> dict[str, float | int]:
    idx = FacultyIndex(conn)
    unis = {u.openalex_id: u.csrankings_name for u in load_universities()}
    authors = {r["author_id"]: r for r in conn.execute("SELECT author_id, name, orcid, h_index FROM authors")}
    cache: dict[tuple[str, str], int | None] = {}
    total = verified = 0
    conn.execute("BEGIN")
    conn.execute("UPDATE faculty SET matched_author_id=NULL")
    for r in conn.execute("SELECT work_id, inst_id, senior_author_id FROM work_universities").fetchall():
        a = authors.get(r["senior_author_id"])
        key = (r["senior_author_id"], r["inst_id"])
        if key not in cache:
            cache[key] = idx.match(unis[r["inst_id"]], a["name"], a["orcid"]) if a else None
            if cache[key]:
                conn.execute("UPDATE faculty SET matched_author_id=? WHERE rowid=?", (r["senior_author_id"], cache[key]))
        ok = cache[key] is not None
        h = (a["h_index"] or 0) if a else 0
        conf = "high" if ok else ("medium" if h >= 15 else "low")
        conn.execute("UPDATE work_universities SET professor_verified=?, attribution_confidence=? "
                     "WHERE work_id=? AND inst_id=?", (int(ok), conf, r["work_id"], r["inst_id"]))
        total += 1
        verified += ok
    conn.execute("COMMIT")
    return {"pairs": total, "verified": verified, "verified_pct": round(100 * verified / max(total, 1), 1)}
