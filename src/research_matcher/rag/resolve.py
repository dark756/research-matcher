"""Deterministic post-processing of a QueryPlan (plan §9.2): universities / countries via the alias table, entities via
aliases.yaml, paper_ref via ID regex -> previous results -> FTS title match, professor names via the author table."""
from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

from rapidfuzz import fuzz, process

from ..config import load_geo, load_settings, load_subdomains, load_universities
from ..similarity.diversify import MODES
from ..extract.normalize import normalize
from ..index.fts import bm25_search
from .query_plan import QueryPlan

NEGATION = re.compile(r"\b(drop|exclude|excluding|except|without|not|other than|outside|besides|no)\b", re.I)
AMBIGUOUS_ALIASES = {"col", "cam", "imp", "oxf", "gt", "uw"}   # too short / common to scan for in free text


@dataclass
class Resolved:
    plan: QueryPlan
    include: frozenset = frozenset()          # university short names
    exclude: frozenset = frozenset()
    countries: frozenset = frozenset()        # ISO codes
    subdomain: str | None = None              # None = no subdomain preference
    subdomain_hard: bool = False              # True when the user (or a flag) named it; inferred ones are soft
    anchor: int | None = None                 # corpus index of the referenced paper
    prof: tuple | None = None                 # (author_id, inst_id, name)
    mode: str = "diverse"                     # selection mode, see similarity.diversify.MODES
    warnings: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ universities / countries
@lru_cache
def uni_alias_map() -> dict[str, str]:
    m: dict[str, str] = {}
    for u in load_universities():
        for a in [u.short, u.name, *u.aliases]:
            m[_n(a)] = u.short
    return m


@lru_cache
def country_alias_map() -> dict[str, str]:
    m = {}
    for code, words in load_geo()["country_aliases"].items():
        m[code.lower()] = code
        for w in words:
            m[_n(w)] = code
    return m


def _n(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s.lower())).strip()


def _split(items: list[str]) -> list[str]:
    out = []
    for it in items:
        out += [p.strip() for p in re.split(r"[/,&;]|\band\b|\bvs\.?\b|\bor\b", it) if p.strip()]
    return out


def resolve_places(texts: list[str]) -> tuple[set[str], set[str], list[str]]:
    """Free-text names -> (university shorts, country codes, warnings). Unknown names produce a warning."""
    unis, countries, warn = set(), set(), []
    regions = {k.lower(): v for k, v in load_geo()["regions"].items()}
    for t in _split(texts):
        k = _n(t)
        if k in uni_alias_map():
            unis.add(uni_alias_map()[k])
        elif k in country_alias_map():
            countries.add(country_alias_map()[k])
        elif k in regions:
            countries |= set(regions[k])
        else:
            hit = process.extractOne(k, list(uni_alias_map()), scorer=fuzz.ratio, score_cutoff=88)
            if hit:
                unis.add(uni_alias_map()[hit[0]])
            else:
                warn.append(f"unknown university/country {t!r} ignored")
    return unis, countries, warn


def scan_unis(question: str) -> set[str]:
    """Fallback when the planner found no university: spot short names (upper-case) and distinctive aliases in the text."""
    found = set()
    for u in load_universities():
        if re.search(rf"\b{re.escape(u.short)}\b", question):
            found.add(u.short)
            continue
        for a in u.aliases:
            if len(a) >= 4 and a not in AMBIGUOUS_ALIASES and re.search(rf"\b{re.escape(a)}\b", question, re.I):
                found.add(u.short)
    return found


def uni_countries() -> dict[str, str]:
    return {u.short: u.country for u in load_universities()}


# ------------------------------------------------------------------ papers
ID_PATTERNS = {"p": re.compile(r"\bP(\d{1,2})\b"), "w": re.compile(r"\bW\d{6,}\b"),
               "doi": re.compile(r"\b10\.\d{4,9}/[^\s\"'<>,;]+"), "arxiv": re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b")}


def canonical(conn: sqlite3.Connection, work_id: str) -> str | None:
    r = conn.execute("SELECT canonical_id FROM works WHERE work_id=?", (work_id,)).fetchone()
    return (r[0] or work_id) if r else None


def resolve_paper(conn: sqlite3.Connection, ref: str, last_ids: list[str]) -> tuple[str | None, str | None]:
    """ref -> (canonical work_id, warning)."""
    ref = (ref or "").strip()
    if not ref:
        return None, None
    m = re.fullmatch(r"[Pp](\d{1,2})", ref)
    if m:
        n = int(m.group(1))
        if 1 <= n <= len(last_ids):
            return last_ids[n - 1], None
        return None, f"P{n} is not in the previous results"
    if re.fullmatch(r"W\d{6,}", ref):
        return canonical(conn, ref), None if canonical(conn, ref) else f"{ref} not found"
    m = ID_PATTERNS["doi"].search(ref)
    if m:
        r = conn.execute("SELECT canonical_id, work_id FROM works WHERE lower(doi)=?", (m.group(0).lower().rstrip(".)"),)).fetchone()
        if r:
            return r[0] or r[1], None
    m = ID_PATTERNS["arxiv"].search(ref)
    if m:
        r = conn.execute("SELECT canonical_id, work_id FROM works WHERE arxiv_id=?", (re.sub(r"v\d+$", "", m.group(0)),)).fetchone()
        if r:
            return r[0] or r[1], None
    hits = bm25_search(conn, ref, 5)
    if hits:
        title = conn.execute("SELECT title FROM works WHERE work_id=?", (hits[0][0],)).fetchone()[0]
        sim = fuzz.partial_ratio(_n(ref), _n(title))
        return hits[0][0], None if sim >= 80 else f"closest title match for {ref!r}: {title!r}"
    return None, f"no paper found for {ref!r}"


# ------------------------------------------------------------------ professors
_PROF: dict[int, list[tuple]] = {}


def _prof_index(conn: sqlite3.Connection) -> list[tuple]:
    if id(conn) not in _PROF:
        rows = conn.execute("SELECT p.author_id, p.inst_id, a.name, p.n_papers FROM professor_profiles p "
                            "JOIN authors a USING(author_id)").fetchall()
        short = {u.openalex_id: u.short for u in load_universities()}
        _PROF[id(conn)] = [(r[0], short.get(r[1], r[1]), r[2], r[3], _n(r[2])) for r in rows]
    return _PROF[id(conn)]


def resolve_professor(conn: sqlite3.Connection, name: str, unis: set[str]) -> tuple[tuple | None, str | None]:
    idx = _prof_index(conn)
    k = _n(name)
    if not k:
        return None, None
    pool = [r for r in idx if not unis or r[1] in unis] or idx
    names = [r[4] for r in pool]
    hits = process.extract(k, names, scorer=fuzz.token_sort_ratio, score_cutoff=88, limit=5)
    if not hits:
        return None, f"no professor profile matches {name!r}"
    cand = [pool[h[2]] for h in hits if h[1] >= hits[0][1] - 1]
    best = max(cand, key=lambda r: r[3])
    warn = None if hits[0][1] >= 95 else f"closest professor match for {name!r}: {best[2]}"
    return (best[0], best[1], best[2]), warn


# ------------------------------------------------------------------ main entry
def resolve(conn: sqlite3.Connection, corpus, plan: QueryPlan, question: str, *, last_ids: list[str] | None = None,
            overrides: dict | None = None) -> Resolved:
    """Apply deterministic fixes and CLI overrides; returns the fully resolved query."""
    ov = {k: v for k, v in (overrides or {}).items() if v not in (None, [], ())}
    plan = plan.model_copy(deep=True)
    for k in ("intent", "subdomain", "year_from", "year_to", "has_code", "min_universities", "k"):
        if k in ov:
            setattr(plan, k, ov[k])
    if "types" in ov:
        plan.types = list(ov["types"])
    if "datasets" in ov:
        plan.datasets = list(ov["datasets"])
    warns: list[str] = []

    inc, cty, w1 = resolve_places(plan.universities_include + list(ov.get("uni", [])))
    exc, cty2, w2 = resolve_places(plan.universities_exclude + list(ov.get("exclude_uni", [])))
    c3, c4, w3 = resolve_places(plan.countries)
    cty |= c3 | c4
    exc_c = cty2
    warns += w1 + w2 + w3
    if not inc and not exc and not NEGATION.search(question):
        inc = scan_unis(question)
    cty = {c for c in cty}
    if exc_c:
        exc |= {u for u, c in uni_countries().items() if c in exc_c}

    # subdomain: explicit id in the question text wins, then plan / flag
    sub, hard = (plan.subdomain if plan.subdomain != "any" else None), False
    for s in load_subdomains():
        if s["id"] != "none" and s["id"].lower() in question.lower():
            sub = s["id"]
    if sub:
        hard = True

    # intent safety nets
    ql = question.lower()
    ref = plan.paper_ref
    if not ref:
        for key in ("w", "doi", "arxiv", "p"):
            m = ID_PATTERNS[key].search(question)
            if m:
                ref = m.group(0)
                break
    if "intent" not in ov:
        if ref and plan.intent not in ("similar_to_paper", "paper_detail", "compare_unis"):
            plan.intent = "similar_to_paper"
        if re.search(r"micro-?topics?", ql) and plan.intent not in ("cross_uni_topics",):
            plan.intent = "cross_uni_topics"
        if plan.intent == "similar_to_paper" and not ref and plan.semantic_text:
            plan.intent = "topic_across_unis"
        if plan.intent == "compare_unis" and len(inc) < 2:
            plan.intent = "topic_across_unis"
    plan.paper_ref = ref
    for kind in ("datasets", "baselines", "metrics"):
        setattr(plan, kind, sorted({normalize(kind, x) for x in getattr(plan, kind)}))

    mode = ov.get("mode") or load_settings().rag.get("select_mode", "diverse")
    if "mode" not in ov and re.search(r"one (paper )?(per|from each|for each) (university|uni)\b|each universit", ql):
        mode = "one_per_uni"
    if mode not in MODES:
        warns.append(f"unknown selection mode {mode!r}; using diverse")
        mode = "diverse"
    r = Resolved(plan=plan, mode=mode, include=frozenset(inc), exclude=frozenset(exc), countries=frozenset(cty),
                 subdomain=sub, subdomain_hard=hard, warnings=warns)
    if plan.intent in ("similar_to_paper", "paper_detail"):
        wid, w = resolve_paper(conn, ref or plan.semantic_text, last_ids or [])
        if w:
            warns.append(w)
        r.anchor = corpus.pos.get(wid) if wid else None
        if wid and r.anchor is None:
            warns.append(f"{wid} is not a canonical paper in the store")
    if plan.intent == "similar_professors":
        pr, w = resolve_professor(conn, plan.professor or "", set(inc))
        if w:
            warns.append(w)
        r.prof = pr
    return r
