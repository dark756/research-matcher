"""Venue hints from abstract text ("Accepted at ICLR 2026") and display-venue resolution (S2 > hint)."""
from __future__ import annotations

import re
import sqlite3

from .semantic_scholar import is_real_venue

HINT = re.compile(
    r"\b(?i:accepted|to appear|published)\s+(?i:at|in|to|by)\s+(?i:the\s+)?((?:[A-Z][A-Za-z&\-]{1,15}\s?'?\d{2,4}|"
    r"[A-Z]{3,10}(?:\s?'?\d{2,4})?)(?:\s+[A-Z][A-Za-z]+){0,3})")


def venue_hint(abstract: str | None) -> str | None:
    if not abstract:
        return None
    m = HINT.search(abstract)
    return m.group(1).strip(" .,;") if m else None


def run_venue_hints(conn: sqlite3.Connection) -> dict[str, int]:
    n_hint = n_disp = 0
    conn.execute("BEGIN")
    for r in conn.execute("SELECT work_id, abstract, venue_name, venue_type, venue_s2 FROM works"):
        hint = venue_hint(r["abstract"])
        if hint:
            n_hint += 1
        name = r["venue_name"]
        if (r["venue_type"] == "repository" or not is_real_venue(name)) and not r["venue_s2"] and hint:
            name = hint
            n_disp += 1
        conn.execute("UPDATE works SET venue_hint=?, venue_name=? WHERE work_id=?", (hint, name, r["work_id"]))
    conn.execute("COMMIT")
    return {"hints": n_hint, "display_from_hint": n_disp}
