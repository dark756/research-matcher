"""Entity normalisation (datasets / baselines / metrics / hardware) via config/aliases.yaml plus generic cleanup."""
from __future__ import annotations

import re
from functools import lru_cache

from ..config import load_aliases


def clean(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[\"'`()\[\]{}]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .,;:")


@lru_cache
def _alias_maps() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for kind, table in load_aliases().items():
        m: dict[str, str] = {}
        for canon, variants in (table or {}).items():
            m[clean(canon)] = canon
            for v in variants:
                m[clean(v)] = canon
        out[kind] = m
    return out


def normalize(kind: str, raw: str) -> str:
    """kind ∈ datasets / baselines / metrics / hardware. Returns the canonical name or the cleaned string."""
    c = clean(raw)
    return _alias_maps().get(kind, {}).get(c, c)
