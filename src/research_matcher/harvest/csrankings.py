"""CSRankings faculty CSV: download once, cache, load into `faculty`."""
from __future__ import annotations

import csv
import io
import sqlite3
import unicodedata
import re

import httpx

from ..config import load_settings, load_universities


def name_norm(name: str) -> str:
    """Accent-stripped lowercase name; drops DBLP disambiguation digits (e.g. 'Wei Wang 0001')."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"\s+\d{4}$", "", s)
    return " ".join(re.sub(r"[^a-z\s]", " ", s).split())


def download_csv() -> str:
    s = load_settings()
    cache = s.path("data_dir") / "cache" / "csrankings.csv"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    cache.parent.mkdir(parents=True, exist_ok=True)
    r = httpx.get(s.csrankings_url, timeout=60, follow_redirects=True)
    r.raise_for_status()
    cache.write_text(r.text, encoding="utf-8")
    return r.text


def load_faculty(conn: sqlite3.Connection) -> dict[str, int]:
    """Replace `faculty` rows; return {short: count} for the 17 target universities."""
    text = download_csv()
    wanted = {u.csrankings_name: u.short for u in load_universities()}
    counts = {u.short: 0 for u in load_universities()}
    conn.execute("BEGIN")
    conn.execute("DELETE FROM faculty")
    for row in csv.DictReader(io.StringIO(text)):
        aff = row.get("affiliation", "")
        if aff not in wanted:
            continue
        conn.execute(
            "INSERT INTO faculty(name,name_norm,affiliation,homepage,scholar_id,orcid) VALUES (?,?,?,?,?,?)",
            (row["name"], name_norm(row["name"]), aff, row.get("homepage", ""),
             row.get("scholarid", ""), row.get("orcid", "")),
        )
        counts[wanted[aff]] += 1
    conn.execute("COMMIT")
    return counts
