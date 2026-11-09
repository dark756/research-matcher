"""GitHub links found in abstracts / TL;DRs (regex). Hugging Face lookup for survivors comes in M4."""
from __future__ import annotations

import re
import sqlite3

GITHUB = re.compile(r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", re.I)


def find_github(text: str | None) -> list[str]:
    out = []
    for owner, repo in GITHUB.findall(text or ""):
        repo = re.sub(r"(\.git|\.)+$", "", repo)
        if repo:
            url = f"https://github.com/{owner}/{repo}"
            if url not in out:
                out.append(url)
    return out


def run_code_links(conn: sqlite3.Connection) -> dict[str, int]:
    n = 0
    conn.execute("BEGIN")
    conn.execute("DELETE FROM code_links WHERE source='abstract_regex'")
    for r in conn.execute("SELECT work_id, abstract, tldr FROM works").fetchall():
        urls = find_github(f"{r['abstract'] or ''} {r['tldr'] or ''}")
        for u in urls:
            conn.execute("INSERT INTO code_links VALUES (?,?,?,?)", (r["work_id"], u, "abstract_regex", 0.8))
        if urls:
            n += 1
            conn.execute("UPDATE works SET has_code=1, code_source='abstract_regex', code_confidence=0.8 "
                         "WHERE work_id=?", (r["work_id"],))
    conn.execute("COMMIT")
    return {"papers_with_code": n}
