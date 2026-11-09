"""Rebuild plain-text abstracts from OpenAlex `abstract_inverted_index`."""
from __future__ import annotations

from typing import Any


def rebuild_abstract(inverted: dict[str, list[int]] | None) -> str | None:
    """Return the abstract text, or None if the index is missing/empty."""
    if not inverted:
        return None
    positions: dict[int, str] = {}
    for word, idxs in inverted.items():
        for i in idxs:
            positions[i] = word
    if not positions:
        return None
    # Join in position order; gaps (shouldn't occur) are simply skipped.
    text = " ".join(positions[i] for i in sorted(positions))
    text = " ".join(text.split())
    return text or None


def abstract_of(work: dict[str, Any]) -> str | None:
    return rebuild_abstract(work.get("abstract_inverted_index"))
