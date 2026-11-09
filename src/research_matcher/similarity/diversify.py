"""University-diverse selection: MMR with a per-university cap and a minimum number of universities (plan §8.4)."""
from __future__ import annotations

import numpy as np


MODES = ("diverse", "one_per_uni", "overall", "one_uni")


def select_by_mode(mode: str, scores: np.ndarray, vecs: np.ndarray, unis: list[frozenset], *, k: int = 20, lam: float = 0.75,
                   per_uni: int = 3, min_unis: int = 3, tau: float = 0.0) -> tuple[list[int], str | None]:
    """Selection toggle (candidates must be sorted by score, descending):
    diverse      MMR with a per-university cap and a minimum number of universities (default, plan 8.4)
    one_per_uni  the best paper of each university, best universities first; never more than one per university
    overall      plain top-k by score, no university constraint
    one_uni      prefer a single university: the one with the highest summed top-k scores supplies as many papers as it has;
                 the rest, if any, are filled by overall score"""
    n = len(scores)
    if n == 0:
        return [], "no candidates"
    if mode == "overall":
        return list(range(min(k, n))), None
    if mode == "one_per_uni":
        seen: set[str] = set()
        sel: list[int] = []
        for i in range(n):
            if len(sel) >= k:
                break
            if unis[i] - seen:                       # brings at least one university not yet represented
                sel.append(i)
                seen |= unis[i]
        note = None
        if len(sel) < k:
            note = f"only {len(seen)} universit{'y' if len(seen) == 1 else 'ies'} with a match above the relevance floor; one paper each"
        return sel, note
    if mode == "one_uni":
        tot: dict[str, float] = {}
        cnt: dict[str, int] = {}
        for i in range(n):
            for u in unis[i]:
                if cnt.get(u, 0) < k:
                    tot[u] = tot.get(u, 0.0) + float(scores[i])
                    cnt[u] = cnt.get(u, 0) + 1
        lead = max(tot, key=tot.get)
        sel = [i for i in range(n) if lead in unis[i]][:k]
        note = f"preferring {lead}"
        if len(sel) < k:
            chosen = set(sel)
            rest = [i for i in range(n) if i not in chosen][:k - len(sel)]
            note += f"; it has only {len(sel)} matching paper(s), the rest are the best from other universities"
            sel += rest
        return sel, note
    return select_diverse(scores, vecs, unis, k=k, lam=lam, per_uni=per_uni, min_unis=min_unis, tau=tau)


def select_diverse(scores: np.ndarray, vecs: np.ndarray, unis: list[frozenset], *, k: int = 20, lam: float = 0.75,
                   per_uni: int = 3, min_unis: int = 3, tau: float = 0.0) -> tuple[list[int], str | None]:
    """Returns (indices into the candidate arrays in selection order, note).
    MMR = lam*score - (1-lam)*max cosine to already-selected. Every university of a paper counts against its cap.
    After k picks, if fewer than min_unis universities are represented, keeps picking papers that add a NEW university
    but only while their score >= tau; if still short, says so instead of padding."""
    n = len(scores)
    if n == 0:
        return [], "no candidates"
    chosen: list[int] = []
    count: dict[str, int] = {}
    left = set(range(n))
    maxsim = np.zeros(n)

    def pick(pool: set[int], need_new: bool) -> int | None:
        best, best_v = None, -1e9
        seen = {u for c in chosen for u in unis[c]}
        for i in pool:
            if any(count.get(u, 0) >= per_uni for u in unis[i]):
                continue
            if need_new and not (unis[i] - seen):
                continue
            if need_new and scores[i] < tau:
                continue
            v = lam * scores[i] - (1 - lam) * maxsim[i]
            if v > best_v:
                best, best_v = i, v
        return best

    def add(i: int) -> None:
        chosen.append(i)
        left.discard(i)
        for u in unis[i]:
            count[u] = count.get(u, 0) + 1
        sims = vecs @ vecs[i]
        np.maximum(maxsim, sims, out=maxsim)

    while len(chosen) < k:
        i = pick(left, False)
        if i is None:
            break
        add(i)
    note = None
    while len({u for c in chosen for u in unis[c]}) < min_unis:
        i = pick(left, True)
        if i is None:
            break
        add(i)
    n_unis = len({u for c in chosen for u in unis[c]})
    if n_unis < min_unis:
        note = f"only {n_unis} universit{'y' if n_unis == 1 else 'ies'} with a close match (score >= {tau:.2f}); not padding with weaker ones"
    return chosen, note
