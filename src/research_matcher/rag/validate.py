"""Step 3c: validate the gemma3 answer against the context (plan §9.4.4). Sentences that cite a non-existent [P#] or name a
dataset / baseline / professor / university that is not in the context are marked ⚠. Nothing is silently dropped."""
from __future__ import annotations

import re
from functools import lru_cache

from ..config import load_universities
from .resolve import _n

CITE = re.compile(r"[\[(]\s*(P\d+(?:\s*(?:[,;&]|\band\b)\s*P\d+)*)\s*[\])]")
GENERIC = {"random", "accuracy", "latency", "throughput", "baseline", "default", "none", "other", "recall", "precision", "memory", "cache"}


@lru_cache(maxsize=1)
def _vocab_cache(conn_id: int):
    return {}


def vocab(conn) -> dict:
    """Known entity / professor / university names (normalised) -> kind."""
    v = _vocab_cache(id(conn))
    if v:
        return v
    for kind, e in conn.execute("SELECT DISTINCT kind, normalized FROM entities WHERE kind IN ('datasets','baselines')"):
        k = _n(e)
        if len(k) >= 4 and k not in GENERIC:
            v.setdefault(k, kind)
    for (name,) in conn.execute("SELECT DISTINCT a.name FROM authors a JOIN work_universities w ON w.senior_author_id=a.author_id"):
        k = _n(name)
        if k.count(" ") >= 1 and len(k) >= 7:
            v.setdefault(k, "professor")
    for u in load_universities():
        for a in [u.short, u.name, *u.aliases]:
            k = _n(a)
            if len(k) >= 3 and k not in {"col", "cam", "imp", "oxf"}:
                v[k] = "university:" + u.short
    return v


def allowed_terms(ctx: list[dict], question: str, plan_text: str) -> set[str]:
    terms = set()
    for p in ctx:
        for u in p["universities"]:
            terms.add("university:" + u)
        for n in p["professors"].values():
            terms.add(_n(n))
        for k in ("datasets", "baselines"):
            terms |= {_n(x) for x in p[k]}
    q = " " + _n(question + " " + plan_text) + " "
    text = " " + _n(" ".join(f"{p['title']} {p['abstract']} {p['problem'] or ''} {p['method'] or ''}" for p in ctx)) + " "
    return terms | {q, text}


def _found(sentence: str, v: dict) -> list[tuple[str, str]]:
    toks = _n(sentence).split()
    out = []
    for n in (4, 3, 2, 1):
        for i in range(len(toks) - n + 1):
            g = " ".join(toks[i:i + n])
            if g in v:
                out.append((g, v[g]))
    return out


def validate(conn, answer: str, ctx: list[dict], question: str, plan_text: str) -> tuple[str, dict]:
    """Returns (annotated answer, report)."""
    n_ctx = len(ctx)
    ok_ids = {f"P{i}" for i in range(1, n_ctx + 1)}
    v = vocab(conn)
    allowed = allowed_terms(ctx, question, plan_text)
    qtext = " ".join(a for a in allowed if a.startswith(" "))
    sentences = re.split(r"(?<=[.!?])\s+|\n+", answer)
    out, bad_ids, bad_terms, uncited = [], [], [], 0
    for s in sentences:
        if not s.strip():
            continue
        marks = []
        cited = [x.strip() for m in CITE.finditer(s) for x in re.split(r"[,;&]|\band\b", m.group(1))]
        if not cited and len(s.split()) > 6 and not s.lstrip().startswith(("#", "**Note")):
            uncited += 1
        wrong = [c for c in cited if c not in ok_ids]
        if wrong:
            bad_ids += wrong
            marks.append("unknown " + ",".join(wrong))
        for term, kind in dict.fromkeys(_found(CITE.sub(" ", s), v)):
            key = kind if kind.startswith("university:") else term
            if key in allowed or f" {term} " in qtext:
                continue
            bad_terms.append(term)
            marks.append(f"{term!r} not in context")
        out.append(s + (f" ⚠[{'; '.join(marks)}]" if marks else ""))
    report = {"invalid_ids": bad_ids, "unsupported_terms": bad_terms, "uncited_sentences": uncited, "sentences": len(out)}
    return "\n".join(out) if "\n" in answer else " ".join(out), report
