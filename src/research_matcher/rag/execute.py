"""Step 2 of the RAG engine: deterministic execution of a resolved plan (plan §9.3). No LLM calls in this module."""
from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
from rapidfuzz import fuzz

from ..config import load_settings, load_universities
from ..index.fts import bm25_search
from ..match.pairs import match_paper, topic_members
from ..similarity.diversify import select_by_mode
from ..similarity.scorer import benchmark_eligible, combine, default_weights, why
from ..similarity.signals import SIGNALS, Corpus, sparse_norm, title_tokens
from .resolve import Resolved, uni_countries

COL = {s: k for k, s in enumerate(SIGNALS)}


@dataclass
class Result:
    intent: str
    R: Resolved
    items: list[dict] = field(default_factory=list)      # ranked papers; P# = position + 1
    extra: dict = field(default_factory=dict)            # intent-specific tables (professors, stats, topics, ...)
    notes: list[str] = field(default_factory=list)
    ms: dict = field(default_factory=dict)


class Engine:
    """Everything the ask/chat commands need in memory: SQLite, corpus vectors, vector store, weights, models."""

    def __init__(self, conn: sqlite3.Connection | None = None, *, llm: bool = True):
        from ..db import connect, init_schema
        from ..index.vector_store import open_store
        from ..ollama_client import OllamaClient

        self.conn = conn or connect()
        init_schema(self.conn)
        self.corpus = Corpus.load(self.conn)
        self.store = open_store()
        gw = load_settings().path("data_dir") / "gold" / "weights.json"
        self.weights = json.loads(gw.read_text()) if gw.exists() else default_weights()
        self.client = OllamaClient() if llm else None
        self.cfg = load_settings().rag
        self.sim = load_settings().similarity
        self._bge = None
        self._protos = None
        self._tops = None

    # -- query embedding (bge-m3 on CPU, plan §9.6) ------------------------------------------------
    def embed(self, text: str) -> dict:
        if self._bge is None:
            from ..embed.bge import BGE

            self._bge = BGE(self.cfg.get("embed_device", "cpu"))
        dense, lex = self._bge.encode([text], sparse=True, batch_size=1)
        sp = lex[0]
        return {"dense": dense[0], "sparse": sp, "self": float(sum(v * v for v in sp.values())),
                "tokens": title_tokens(text)}

    def infer_subdomain(self, qd: np.ndarray) -> str | None:
        """Prototype cosine of the query (>= the pre-filter τ_c) -> subdomain, else None."""
        from ..embed.prefilter import emb_dir, meaning_scores

        if self._protos is None:
            z = np.load(emb_dir() / "protos.npz")
            self._protos = ([str(c) for c in z["clusters"]], z["protos"], z["owner"].tolist())
        clusters, protos, owner = self._protos
        m = meaning_scores(qd[None, :], protos, owner, len(clusters))[0]
        j = int(np.argmax(m))
        tau = load_settings().prefilter["tau"].get(clusters[j], 0.58)
        return clusters[j] if m[j] >= tau else None

    @property
    def tops(self) -> dict:
        if self._tops is None:
            self._tops = topic_members(self.corpus)
        return self._tops


# ------------------------------------------------------------------ filters from the plan
def make_filters(R: Resolved, extra_exclude: frozenset = frozenset()):
    """-> (keep(j) predicate, store filter spec). Both express the same constraints."""
    p = R.plan
    ucty = uni_countries()
    exc = R.exclude | extra_exclude
    must, must_not = [], []
    if R.subdomain and R.subdomain_hard:
        must.append({"key": "subdomain", "eq": R.subdomain})
    if R.include:
        must.append({"key": "target_universities", "any": sorted(R.include)})
    if R.countries:
        must.append({"key": "countries", "any": sorted(R.countries)})
    if exc:
        must_not.append({"key": "target_universities", "any": sorted(exc)})
    if p.year_from is not None or p.year_to is not None:
        must.append({"key": "year", "gte": p.year_from, "lte": p.year_to})
    if p.types:
        must.append({"key": "type", "any": list(p.types)})
    if p.has_code:
        must.append({"key": "has_code", "eq": True})
    spec = {"must": must, "must_not": must_not}

    def keep_with(corpus: Corpus):
        def keep(j: int) -> bool:
            m, us = corpus.meta[j], corpus.unis[j]
            if not us:
                return False
            if R.subdomain and R.subdomain_hard and m["subdomain"] != R.subdomain:
                return False
            if R.include and not (us & R.include):
                return False
            if exc and (us & exc):
                return False
            if R.countries and not ({ucty.get(u) for u in us} & R.countries):
                return False
            y = m["year"]
            if p.year_from is not None and (y is None or y < p.year_from):
                return False
            if p.year_to is not None and (y is None or y > p.year_to):
                return False
            if p.types and m["type"] not in p.types:
                return False
            if p.has_code and not m["has_code"]:
                return False
            return True

        return keep

    return keep_with, spec


# ------------------------------------------------------------------ query-vs-paper signals
def query_signals(corpus: Corpus, q: dict, js: np.ndarray, ents: dict[str, list[str]]) -> np.ndarray:
    """(n,14) signal matrix between a query and papers js: s1, s2, s3, s4, s5, s7, s9 and entity overlap (s10-s12).
    Paper-paper-only signals (s8 topics, s13 coupling, s14 citation link, s15 SPECTER2) stay NaN."""
    js = np.asarray(js, dtype=int)
    out = np.full((len(js), len(SIGNALS)), np.nan)
    if not len(js):
        return out
    qd = q["dense"]
    out[:, COL["dense_abstract"]] = np.clip(corpus.abs[js] @ qd, 0, 1)
    out[:, COL["title_dense"]] = np.clip(corpus.title[js] @ qd, 0, 1)
    m = corpus.has_comp[js]
    out[m, COL["dense_composite"]] = np.clip(corpus.comp[js[m]] @ qd, 0, 1)
    qtok = set(q["tokens"].split())
    for r, j in enumerate(js):
        B = corpus.meta[j]
        parts = []
        if B["tt"] and q["tokens"]:
            parts.append(fuzz.token_set_ratio(q["tokens"], B["tt"]) / 100)
        sp = sparse_norm(q["sparse"], corpus.sparse_title.get(j), q["self"], corpus.self_title.get(j, 0))
        if sp == sp:
            parts.append(sp)
        if parts:
            out[r, COL["title_lexical"]] = float(np.mean(parts))
        s5 = [sparse_norm(q["sparse"], corpus.sparse_abs.get(j), q["self"], corpus.self_abs.get(j, 0))]
        if corpus.has_comp[j]:
            s5.append(sparse_norm(q["sparse"], corpus.sparse_comp.get(j), q["self"], corpus.self_comp.get(j, 0)))
        s5 = [x for x in s5 if x == x]
        if s5:
            out[r, COL["sparse_lexical"]] = float(np.mean(s5))
        cent = corpus.topic_cent.get(B["micro"]) if B["micro"] is not None and B["micro"] >= 0 else None
        if cent is not None:
            out[r, COL["micro_topic"]] = float(np.clip(cent @ qd, 0, 1))
        if B["keywords"] and qtok:
            kt = {w for k in B["keywords"] for w in k.split()}
            out[r, COL["keywords"]] = min(len(qtok & kt) / len(qtok), 1.0)
        if B["extracted"]:
            for kind in ("datasets", "baselines", "metrics"):
                if ents.get(kind):
                    out[r, COL[kind]] = len(B[kind] & set(ents[kind])) / len(ents[kind])
    return out


def _entity_bonus(S: np.ndarray, ents: dict) -> np.ndarray:
    cols = [COL[k] for k in ("datasets", "baselines", "metrics") if ents.get(k)]
    if not cols:
        return np.zeros(len(S))
    return 0.15 * np.nan_to_num(np.nanmean(np.where(np.isnan(S[:, cols]), 0.0, S[:, cols]), axis=1))


def _item(eng: Engine, j: int, score: float, S_row: np.ndarray, ents: dict, anchor: int | None = None) -> dict:
    A = eng.corpus.meta[j]
    matched = {k: sorted(A[k] & set(ents.get(k, []))) for k in ("datasets", "baselines", "metrics") if A[k] & set(ents.get(k, []))}
    sh = eng.corpus.shared(anchor, j) if anchor is not None else matched
    return {"j": j, "work_id": eng.corpus.ids[j], "score": float(score), "signals": S_row, "shared": sh,
            "eligible": benchmark_eligible(sh), "why": why(S_row, eng.weights)}


# ------------------------------------------------------------------ retrieval for a text query
def topic_retrieve(eng: Engine, R: Resolved, text: str, *, extra_exclude: frozenset = frozenset(),
                   include_override: frozenset | None = None, n_abs: int = 150, n_comp: int = 100, n_bm25: int = 100):
    """Candidates (dense abstract ∪ composite ∪ BM25) -> query-vs-paper signals -> combined score.
    Returns (js, S, score) sorted by score, descending."""
    corpus = eng.corpus
    R2 = R
    if include_override is not None:
        R2 = Resolved(plan=R.plan, include=include_override, exclude=R.exclude, countries=R.countries,
                      subdomain=R.subdomain, subdomain_hard=R.subdomain_hard)
    keep_with, spec = make_filters(R2, extra_exclude)
    keep = keep_with(corpus)
    q = eng.embed(text)
    ents = {k: getattr(R.plan, k) for k in ("datasets", "baselines", "metrics")}
    cands: set[int] = set()
    pos = corpus.pos
    for h in eng.store.search("all_papers", "abstract", q["dense"], n_abs, spec):
        cands.add(pos[h.payload["work_id"]])
    for h in eng.store.search("relevant_papers", "composite", q["dense"], n_comp, spec):
        cands.add(pos[h.payload["work_id"]])
    for w, _ in bm25_search(eng.conn, text, n_bm25):
        if w in pos:
            cands.add(pos[w])
    js = np.array(sorted(j for j in cands if keep(j)), dtype=int)
    S = query_signals(corpus, q, js, ents)
    score = combine(S, eng.weights) + _entity_bonus(S, ents)
    soft = None
    if not R.subdomain:
        soft = eng.infer_subdomain(q["dense"])
        if soft:
            same = np.array([corpus.meta[j]["subdomain"] == soft for j in js], dtype=bool)
            score = np.where(same, score, score * eng.sim["subdomain_mismatch_penalty"])
    order = np.argsort(-score)
    return js[order], S[order], score[order], ents, soft


def _select(eng: Engine, js, S, score, ents, *, k, per_uni, min_unis, diversify=True, anchor=None, mode="diverse"):
    tau = float(eng.cfg.get("tau_query", 0.40))
    floor = float(eng.cfg.get("floor_query", 0.33))
    n_all = len(js)
    best = float(score[0]) if n_all else 0.0
    keep = int((score >= floor).sum()) if floor else n_all     # scores are sorted descending
    js, S, score = js[:keep], S[:keep], score[:keep]
    top = min(len(js), 300)
    if not top:
        return [], ("no papers match those filters" if not n_all else
                    f"nothing in the store is close to that query (best score {best:.2f} < floor {floor:.2f})")
    if diversify:
        sel, note = select_by_mode(mode, score[:top], eng.corpus.abs[js[:top]], [eng.corpus.unis[j] for j in js[:top]],
                                   k=k, lam=eng.sim["mmr_lambda"], per_uni=per_uni, min_unis=min_unis, tau=tau)
    else:
        sel, note = list(range(min(k, top))), None
    items = [_item(eng, int(js[r]), score[r], S[r], ents, anchor) for r in sel]
    items.sort(key=lambda it: -it["score"])
    return items, note


# ------------------------------------------------------------------ intents
def run_topic(eng: Engine, R: Resolved) -> Result:
    p = R.plan
    text = p.semantic_text or ""
    res = Result("topic_across_unis", R)
    if not text.strip():
        res.notes.append("empty search text")
        return res
    js, S, score, ents, soft = topic_retrieve(eng, R, text)
    if soft:
        res.notes.append(f"subdomain inferred: {soft} (soft preference, not a filter)")
    res.items, note = _select(eng, js, S, score, ents, k=p.k, per_uni=p.per_university_limit, min_unis=p.min_universities, mode=R.mode)
    if note:
        res.notes.append(note)
    return res


def run_similar(eng: Engine, R: Resolved) -> Result:
    p = R.plan
    res = Result("similar_to_paper", R)
    if R.anchor is None:
        res.notes.append("could not resolve the paper; give a W-id, DOI, arXiv id, title fragment or P#")
        return res
    i = R.anchor
    keep_with, spec = make_filters(R)
    sub = R.subdomain if R.subdomain else "auto"
    r = match_paper(eng.conn, eng.corpus, eng.store, i, weights=eng.weights, k=p.k, subdomain=sub, tops=eng.tops,
                    extra_flt={"must": [c for c in spec["must"] if c["key"] != "subdomain"], "must_not": spec["must_not"]},
                    keep=keep_with(eng.corpus), per_uni=p.per_university_limit, min_unis=p.min_universities, mode=R.mode)
    for m in r["matches"]:
        m["anchor"] = i
    res.items = r["matches"]
    res.extra["anchor"] = i
    if r["note"]:
        res.notes.append(r["note"])
    return res


def run_compare(eng: Engine, R: Resolved) -> Result:
    p = R.plan
    res = Result("compare_unis", R)
    unis = sorted(R.include)
    per = max(1, min(5, math.ceil(p.k / max(len(unis), 1))))
    chosen: list[dict] = []
    for u in unis:
        js, S, score, ents, _ = topic_retrieve(eng, R, p.semantic_text, include_override=frozenset({u}))
        items, note = _select(eng, js, S, score, ents, k=per, per_uni=per, min_unis=1)
        if not items:
            res.notes.append(f"{u}: no matching papers")
        chosen += items
    chosen.sort(key=lambda it: -it["score"])
    res.items = chosen
    # shared-entity table across the compared universities
    table: dict[str, dict[str, dict[str, set]]] = {k: defaultdict(lambda: defaultdict(set)) for k in ("datasets", "baselines", "metrics")}
    for idx, it in enumerate(chosen):
        A = eng.corpus.meta[it["j"]]
        for u in eng.corpus.unis[it["j"]] & R.include:
            for k in table:
                for e in A[k]:
                    table[k][e][u].add(idx + 1)
    res.extra["shared_table"] = {k: sorted(((e, {u: sorted(ps) for u, ps in d.items()}) for e, d in v.items() if len(d) >= 2),
                                           key=lambda x: -len(x[1]))[:8] for k, v in table.items()}
    return res


def senior_authors(conn: sqlite3.Connection, work_ids: list[str]) -> dict[tuple[str, str], tuple]:
    """(work_id, uni short) -> (author_id, name, professor_verified)."""
    short = {u.openalex_id: u.short for u in load_universities()}
    out = {}
    for s in range(0, len(work_ids), 500):
        part = work_ids[s:s + 500]
        q = ",".join("?" * len(part))
        for r in conn.execute(f"SELECT w.work_id, w.inst_id, w.senior_author_id, w.professor_verified, a.name FROM work_universities w "
                              f"LEFT JOIN authors a ON a.author_id=w.senior_author_id WHERE w.work_id IN ({q})", part):
            out[(r[0], short.get(r[1], r[1]))] = (r[2], r[4], bool(r[3]))
    return out


def run_professors(eng: Engine, R: Resolved) -> Result:
    p = R.plan
    res = Result("find_professors", R)
    js, S, score, ents, soft = topic_retrieve(eng, R, p.semantic_text)
    tau = float(eng.cfg.get("tau_query", 0.40))
    top = [(j, s, S[r]) for r, (j, s) in enumerate(zip(js[:80], score[:80])) if s >= tau * 0.9]
    sa = senior_authors(eng.conn, [eng.corpus.ids[j] for j, _, _ in top])
    agg: dict[tuple[str, str], dict] = {}
    for j, s, row in top:
        for u in eng.corpus.unis[j]:
            if R.include and u not in R.include:
                continue
            a = sa.get((eng.corpus.ids[j], u))
            if not a or not a[0] or not a[1]:
                continue
            e = agg.setdefault((a[0], u), {"author_id": a[0], "name": a[1], "uni": u, "verified": a[2], "score": 0.0, "papers": []})
            e["score"] += float(s)
            e["verified"] = e["verified"] or a[2]
            e["papers"].append((j, float(s), row))
    profs = sorted(agg.values(), key=lambda e: -e["score"])[:p.k]
    seen, items = set(), []
    for e in profs:
        for j, s, row in sorted(e["papers"], key=lambda x: -x[1])[:2]:
            if j not in seen:
                seen.add(j)
                items.append(_item(eng, j, s, row, ents))
    items.sort(key=lambda it: -it["score"])
    rank = {it["j"]: n + 1 for n, it in enumerate(items)}
    for e in profs:
        e["refs"] = sorted(rank[j] for j, _, _ in e["papers"] if j in rank)
        e["n"] = len(e["papers"])
        del e["papers"]
    res.items, res.extra["professors"] = items, profs
    if not profs:
        res.notes.append("no professors found with a close match")
    return res


def run_similar_professors(eng: Engine, R: Resolved) -> Result:
    p = R.plan
    res = Result("similar_professors", R)
    if not R.prof:
        res.notes.append(f"could not find a professor profile for {p.professor!r}")
        return res
    aid0, inst0, name0 = R.prof
    short = {u.openalex_id: u.short for u in load_universities()}
    rows = eng.conn.execute("SELECT author_id, inst_id, centroid, subdomain_counts_json, n_papers FROM professor_profiles").fetchall()
    prof = {(r[0], short.get(r[1], r[1])): r for r in rows}
    me = prof.get((aid0, inst0))
    if me is None:
        res.notes.append("profile missing")
        return res
    c0 = np.frombuffer(me[2], dtype=np.float32)
    sub0 = json.loads(me[3] or "{}")
    # micro-topics and papers per (senior author, uni)
    mt: dict[tuple, set] = defaultdict(set)
    wk: dict[tuple, list[int]] = defaultdict(list)
    for r in eng.conn.execute("SELECT w.senior_author_id, w.inst_id, w.work_id FROM work_universities w WHERE w.senior_author_id IS NOT NULL"):
        j = eng.corpus.pos.get(r[2])
        if j is None:
            continue
        k = (r[0], short.get(r[1], r[1]))
        wk[k].append(j)
        mu = eng.corpus.meta[j]["micro"]
        if mu is not None and mu >= 0:
            mt[k].add(mu)
    ucty = uni_countries()
    own_unis = {u for (a, u) in prof if a == aid0}
    min_n = int(eng.cfg.get("min_prof_papers", 2))
    keep_with, _ = make_filters(R)
    names = {r[0]: r[1] for r in eng.conn.execute("SELECT author_id, name FROM authors")}
    out = []
    for (aid, u), r in prof.items():
        if aid == aid0 or u in own_unis or u in R.exclude or r[4] < min_n:
            continue
        if R.include and u not in R.include:
            continue
        if R.countries and ucty.get(u) not in R.countries:
            continue
        c = np.frombuffer(r[2], dtype=np.float32)
        cos = float(np.clip(c @ c0, 0, 1))
        sub = json.loads(r[3] or "{}")
        keys = set(sub) | set(sub0)
        dot = sum(sub.get(k, 0) * sub0.get(k, 0) for k in keys)
        den = math.sqrt(sum(v * v for v in sub.values()) * sum(v * v for v in sub0.values()))
        overlap = dot / den if den else 0.5
        if R.subdomain and sub.get(R.subdomain, 0) == 0:
            continue
        shared_mt = len(mt[(aid, u)] & mt[(aid0, inst0)])
        jac = shared_mt / max(len(mt[(aid, u)] | mt[(aid0, inst0)]), 1)
        out.append({"author_id": aid, "name": names.get(aid, aid), "uni": u, "n": r[4], "cos": cos,
                    "overlap": overlap, "shared_topics": shared_mt, "score": cos * (0.5 + 0.5 * overlap) + 0.05 * jac})
    out.sort(key=lambda e: -e["score"])
    out = out[:p.k]
    items = []
    kp = keep_with(eng.corpus)
    for e in out:  # each match is shown with the paper closest to the query professor's centroid
        all_js = wk[(e["author_id"], e["uni"])]
        js = [j for j in all_js if kp(j)] or all_js
        if js:
            best = max(js, key=lambda j: float(eng.corpus.abs[j] @ c0))
            sc = float(eng.corpus.abs[best] @ c0)
            it = _item(eng, best, sc, np.full(len(SIGNALS), np.nan), {})
            it["why"] = [("professor profile", round(e["cos"], 2))]
            it["prof_of"] = e["name"]
            items.append(it)
            e["ref"] = len(items)
    res.items, res.extra["professors"] = items, out
    res.extra["query_professor"] = {"name": name0, "uni": inst0, "n": me[4], "subdomains": sub0}
    return res


def run_entity_stats(eng: Engine, R: Resolved, question: str) -> Result:
    p = R.plan
    res = Result("entity_stats", R)
    ql = (question + " " + p.semantic_text).lower()
    kinds = [k for k, w in (("datasets", ("dataset", "benchmark")), ("baselines", ("baseline",)), ("metrics", ("metric",)))
             if any(x in ql for x in w)] or ["datasets", "baselines", "metrics"]
    keep = make_filters(R)[0](eng.corpus)
    cnt: dict[str, Counter] = {k: Counter() for k in kinds}
    unis: dict[tuple, set] = defaultdict(set)
    for w, kind, e in eng.conn.execute("SELECT work_id, kind, normalized FROM entities"):
        j = eng.corpus.pos.get(w)
        if kind in cnt and j is not None and keep(j):
            cnt[kind][e] += 1
            unis[(kind, e)] |= eng.corpus.unis[j]
    n_papers = sum(1 for j, m in enumerate(eng.corpus.meta) if m["extracted"] and keep(j))
    res.extra["stats"] = {k: [(e, n, sorted(unis[(k, e)])) for e, n in c.most_common(12)] for k, c in cnt.items()}
    res.extra["n_papers"] = n_papers
    return res


def run_cross_topics(eng: Engine, R: Resolved) -> Result:
    p = R.plan
    res = Result("cross_uni_topics", R)
    rows = []
    for r in eng.conn.execute("SELECT topic_id, subdomain, label, description, size, universities_json FROM micro_topics ORDER BY size DESC"):
        u = set(json.loads(r[5] or "[]"))
        if R.subdomain and r[1] != R.subdomain:
            continue
        if not R.subdomain and r[1] == "none":
            continue
        if len(u) < p.min_universities or (R.include and not (u & R.include)):
            continue
        rows.append({"topic_id": r[0], "subdomain": r[1], "label": r[2], "description": r[3], "size": r[4], "unis": sorted(u)})
    mem = eng.tops
    for r in rows:
        ms = mem.get(r["topic_id"], [])
        if ms:
            cent = eng.corpus.abs[ms].mean(axis=0)
            cent /= np.linalg.norm(cent) or 1.0
            order = np.argsort(-(eng.corpus.abs[ms] @ cent))[:2]
            r["titles"] = [eng.corpus.meta[ms[i]]["title"] for i in order]
    res.extra["topics"] = rows
    if not rows:
        res.notes.append(f"no micro-topic spans >= {p.min_universities} universities with these filters")
    return res


def run_detail(eng: Engine, R: Resolved) -> Result:
    res = Result("paper_detail", R)
    if R.anchor is None:
        res.notes.append("could not resolve the paper")
        return res
    res.items = [_item(eng, R.anchor, 1.0, np.full(len(SIGNALS), np.nan), {})]
    res.items[0]["why"] = []
    return res


def execute(eng: Engine, R: Resolved, question: str) -> Result:
    i = R.plan.intent
    if i == "similar_to_paper":
        return run_similar(eng, R)
    if i == "compare_unis":
        return run_compare(eng, R)
    if i == "find_professors":
        return run_professors(eng, R)
    if i == "similar_professors":
        return run_similar_professors(eng, R)
    if i == "entity_stats":
        return run_entity_stats(eng, R, question)
    if i == "cross_uni_topics":
        return run_cross_topics(eng, R)
    if i == "paper_detail":
        return run_detail(eng, R)
    return run_topic(eng, R)


# ------------------------------------------------------------------ DB-backed display fields
def load_details(conn: sqlite3.Connection, work_ids: list[str]) -> dict[str, dict]:
    """Every value shown to the user (university, professor, links, entities) is read from SQLite here, never from the LLM."""
    out: dict[str, dict] = {}
    if not work_ids:
        return out
    q = ",".join("?" * len(work_ids))
    for r in conn.execute(f"SELECT work_id, title, year, type, venue_name, abstract, tldr, subdomain_cluster, extracted_at, "
                          f"micro_topic_label, doi_url, arxiv_abs_url, oa_pdf_url, openalex_url, has_code, cited_by_count "
                          f"FROM works WHERE work_id IN ({q})", work_ids):
        out[r["work_id"]] = {"work_id": r["work_id"], "title": r["title"], "year": r["year"], "type": r["type"],
                             "venue": r["venue_name"], "abstract": r["abstract"] or r["tldr"] or "",
                             "subdomain": (r["subdomain_cluster"] or "none") if r["extracted_at"] else "none",
                             "topic": r["micro_topic_label"], "doi": r["doi_url"], "arxiv": r["arxiv_abs_url"],
                             "pdf": r["oa_pdf_url"], "openalex": r["openalex_url"], "has_code": bool(r["has_code"]),
                             "cited_by": r["cited_by_count"], "unis": [], "profs": {}, "github": [],
                             "datasets": [], "baselines": [], "metrics": [], "sub_problem": None, "method": None}
    for (w, u), (aid, name, ver) in senior_authors(conn, work_ids).items():
        out[w]["unis"].append(u)
        out[w]["profs"][u] = (name, ver)
    for w, d in out.items():
        d["unis"] = sorted(d["unis"])
    for r in conn.execute(f"SELECT work_id, kind, normalized FROM entities WHERE work_id IN ({q}) ORDER BY normalized", work_ids):
        if r[1] in ("datasets", "baselines", "metrics"):
            out[r[0]][r[1]].append(r[2])
    for r in conn.execute(f"SELECT work_id, url FROM code_links WHERE work_id IN ({q})", work_ids):
        out[r[0]]["github"].append(r[1])
    for r in conn.execute(f"SELECT work_id, sub_problem, core_method FROM extractions WHERE error IS NULL AND work_id IN ({q})", work_ids):
        out[r[0]]["sub_problem"], out[r[0]]["method"] = r[1], r[2]
    return out
