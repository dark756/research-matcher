"""Candidate generation -> re-rank -> university-diverse selection (plan §8.4), and the precomputed `pairs` table."""
from __future__ import annotations

import json
import sqlite3
from typing import Callable

import numpy as np

from ..config import load_settings
from ..index.fts import bm25_search
from ..similarity.diversify import select_by_mode
from ..similarity.scorer import benchmark_eligible, combine, why
from ..similarity.signals import SIGNALS, Corpus


def candidate_set(conn: sqlite3.Connection, corpus: Corpus, store, i: int, *, subdomain: str | None,
                  exclude_unis: frozenset = frozenset(), n_abs: int = 150, n_comp: int = 100, n_bm25: int = 100,
                  extra_flt: dict | None = None, keep=None) -> set[int]:
    """Union of dense-abstract top-150, composite top-100, BM25 top-100, same micro-topic and citation neighbours,
    all restricted to `subdomain` (when given) and to papers sharing no university in `exclude_unis`."""
    A = corpus.meta[i]
    flt: dict = {"must": [], "must_not": []}
    if subdomain:
        flt["must"].append({"key": "subdomain", "eq": subdomain})
    if exclude_unis:
        flt["must_not"].append({"key": "target_universities", "any": sorted(exclude_unis)})
    if extra_flt:  # extra payload filters (ask/chat: universities, countries, years, types, has_code)
        flt["must"] += extra_flt.get("must", [])
        flt["must_not"] += extra_flt.get("must_not", [])
    pos = corpus.pos
    cands: set[int] = set()
    for h in store.search("all_papers", "abstract", corpus.abs[i], n_abs + 1, flt):
        cands.add(pos_of(corpus, h))
    if corpus.has_comp[i]:
        cflt = {"must": flt["must"], "must_not": flt["must_not"]}
        for h in store.search("relevant_papers", "composite", corpus.comp[i], n_comp + 1, cflt):
            cands.add(pos_of(corpus, h))
    kw = " ".join(sorted(A["keywords"])[:8])
    for w, _ in bm25_search(conn, f"{A['title']} {kw}", n_bm25 + 1):
        if w in pos:
            cands.add(pos[w])
    for w in A["refs"] | A["related"]:
        if w in pos:
            cands.add(pos[w])
    cands.discard(i)
    return {j for j in cands if admissible(corpus, i, j, subdomain, exclude_unis) and (keep is None or keep(j))}


def pos_of(corpus: Corpus, hit) -> int:
    """Hit -> corpus index via the work_id in its payload (point ids differ between collections)."""
    return corpus.pos[hit.payload["work_id"]]


def admissible(corpus: Corpus, i: int, j: int, subdomain: str | None, exclude_unis: frozenset) -> bool:
    if i == j:
        return False
    if subdomain and corpus.meta[j]["subdomain"] != subdomain:
        return False
    if exclude_unis and (corpus.unis[j] & exclude_unis):
        return False
    return bool(corpus.unis[j])


def topic_members(corpus: Corpus) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for j, m in enumerate(corpus.meta):
        if m["micro"] is not None and m["micro"] >= 0:
            out.setdefault(m["micro"], []).append(j)
    return out


def match_paper(conn, corpus: Corpus, store, i: int, *, weights: dict | None = None, k: int = 20, subdomain: str | None = "auto",
                tops: dict[int, list[int]] | None = None, diversify: bool = True, extra_flt: dict | None = None,
                keep=None, per_uni: int | None = None, min_unis: int | None = None, mode: str = "diverse") -> dict:
    """Cross-university matches for paper i. subdomain='auto' = the paper's own subdomain (hard filter);
    None = free search (soft x0.85 multiplier on subdomain mismatch)."""
    cfg = load_settings().similarity
    A = corpus.meta[i]
    sub = A["subdomain"] if subdomain == "auto" else subdomain
    if sub == "none":
        sub = None  # unextracted/none papers: no subdomain to enforce
    excl = corpus.unis[i]
    cands = candidate_set(conn, corpus, store, i, subdomain=sub, exclude_unis=excl, extra_flt=extra_flt, keep=keep)
    if tops and A["micro"] is not None and A["micro"] >= 0:
        cands |= {j for j in tops.get(A["micro"], []) if admissible(corpus, i, j, sub, excl) and (keep is None or keep(j))}
    js = np.array(sorted(cands), dtype=int)
    S = corpus.signals(i, js)
    score = combine(S, weights)
    if sub is None:
        same = np.array([corpus.meta[j]["subdomain"] == A["subdomain"] for j in js])
        score = np.where(same, score, score * cfg["subdomain_mismatch_penalty"])
    order = np.argsort(-score)
    js, S, score = js[order], S[order], score[order]
    note = None
    if diversify and len(js):
        top = min(len(js), 300)
        sel, note = select_by_mode(mode, score[:top], corpus.abs[js[:top]], [corpus.unis[j] for j in js[:top]], k=k,
                                   lam=cfg["mmr_lambda"], per_uni=per_uni or cfg["per_university_limit"],
                                   min_unis=min_unis or cfg["min_universities"], tau=cfg["tau_close"])
    else:
        sel = list(range(min(k, len(js))))
    res = []
    for r in sel:
        j = int(js[r])
        sh = corpus.shared(i, j)
        res.append({"j": j, "work_id": corpus.ids[j], "score": float(score[r]), "signals": S[r], "shared": sh,
                    "eligible": benchmark_eligible(sh), "why": why(S[r], weights)})
    return {"i": i, "matches": res, "note": note, "n_candidates": len(js)}


def build_pairs(conn: sqlite3.Connection, corpus: Corpus, store, *, weights: dict | None = None, k: int = 20,
                only_relevant: bool = True, log: Callable[[str], None] = print) -> dict:
    """Top-k cross-university matches for every paper of the 3 target subdomains -> `pairs`."""
    anchors = [i for i, m in enumerate(corpus.meta) if m["subdomain"] != "none"] if only_relevant else list(range(len(corpus.ids)))
    tops = topic_members(corpus)
    conn.execute("BEGIN")
    conn.execute("DELETE FROM pairs")
    conn.execute("COMMIT")
    n_pairs, short = 0, 0
    for n, i in enumerate(anchors, 1):
        r = match_paper(conn, corpus, store, i, weights=weights, k=k, tops=tops)
        if r["note"]:
            short += 1
        conn.execute("BEGIN")
        for m in r["matches"]:
            sig = {s: (None if v != v else round(float(v), 4)) for s, v in zip(SIGNALS, m["signals"])}
            conn.execute("INSERT OR REPLACE INTO pairs VALUES (?,?,?,?,?)",
                         (corpus.ids[i], m["work_id"], m["score"], json.dumps(sig), json.dumps(m["shared"])))
            n_pairs += 1
        conn.execute("COMMIT")
        if n % 100 == 0:
            log(f"  {n}/{len(anchors)} anchors, {n_pairs} pairs")
    return {"anchors": len(anchors), "pairs": n_pairs, "anchors_with_fewer_than_min_unis": short}
