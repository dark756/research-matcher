"""Professor profiles (plan §8.5): recency-weighted centroid of abstract vectors per (senior author, target university)."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from datetime import date

import numpy as np

HALF_LIFE_YEARS = 1.0


def recency_weight(pub_date: str | None, today: date | None = None, half_life: float = HALF_LIFE_YEARS) -> float:
    today = today or date.today()
    try:
        d = date.fromisoformat((pub_date or "")[:10])
    except ValueError:
        return 0.5
    age_years = max((today - d).days, 0) / 365.25
    return float(0.5 ** (age_years / half_life))


def build_profiles(conn: sqlite3.Connection, ids: list[str], abs_dense: np.ndarray) -> dict[str, int]:
    """One profile per (senior author, university) over all canonical papers; unresolved authors (X:...) are skipped
    because OpenAlex gave them no persistent id, so they cannot be pooled across papers."""
    idx = {w: i for i, w in enumerate(ids)}
    meta = {r["work_id"]: r for r in conn.execute(
        "SELECT work_id, publication_date, subdomain_cluster, extracted_at FROM works WHERE canonical_id=work_id")}
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in conn.execute("SELECT work_id, inst_id, senior_author_id FROM work_universities "
                          "WHERE senior_author_id IS NOT NULL AND senior_author_id NOT LIKE 'X:%'"):
        if r["work_id"] in idx:
            groups[(r["senior_author_id"], r["inst_id"])].append(r["work_id"])
    conn.execute("BEGIN")
    conn.execute("DELETE FROM professor_profiles")
    for (aid, inst), works in groups.items():
        w = np.array([recency_weight(meta[x]["publication_date"]) for x in works], dtype=np.float32)
        vecs = abs_dense[[idx[x] for x in works]]
        c = (vecs * w[:, None]).sum(axis=0)
        c = c / (np.linalg.norm(c) or 1.0)
        counts = Counter(meta[x]["subdomain_cluster"] for x in works if meta[x]["extracted_at"] and meta[x]["subdomain_cluster"])
        conn.execute("INSERT INTO professor_profiles VALUES (?,?,?,?,?)",
                     (aid, inst, c.astype(np.float32).tobytes(), json.dumps(dict(counts)), len(works)))
    conn.execute("COMMIT")
    return {"profiles": len(groups), "papers_covered": len({x for ws in groups.values() for x in ws})}


def load_centroid(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)
