"""Pairwise similarity signals s1..s15 (plan §8.1). Each is in [0,1]; NaN = not available for that pair."""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field

import numpy as np
from rapidfuzz import fuzz

from ..embed.prefilter import emb_dir, load_embeddings

SIGNALS = ["dense_abstract", "dense_composite", "title_dense", "title_lexical", "sparse_lexical", "micro_topic",
           "openalex_topics", "keywords", "datasets", "baselines", "metrics", "biblio_coupling", "citation_link",
           "specter2"]
STOP = {"a", "an", "the", "of", "for", "and", "or", "in", "on", "to", "with", "by", "via", "using", "is", "are", "from",
        "towards", "toward", "at", "as", "its", "their", "into", "through", "based", "approach", "new"}


def title_tokens(t: str) -> str:
    return " ".join(w for w in re.findall(r"[a-z0-9][a-z0-9\-+.]*", (t or "").lower()) if w not in STOP)


def sparse_norm(a: dict, b: dict, aa: float, bb: float) -> float:
    """bge-m3 lexical score, normalised by the self-scores so it lies in [0,1]."""
    if not a or not b or aa <= 0 or bb <= 0:
        return float("nan")
    if len(a) > len(b):
        a, b = b, a
    s = sum(w * b[t] for t, w in a.items() if t in b)
    return float(min(s / (aa * bb) ** 0.5, 1.0))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else float("nan")


def weighted_jaccard(a: dict, b: dict) -> float:
    if not a or not b:
        return float("nan")
    keys = a.keys() | b.keys()
    num = sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    den = sum(max(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    return num / den if den else float("nan")


@dataclass
class Corpus:
    """In-memory features for every canonical work. Built once per process (~150 MB of vectors)."""
    ids: list[str]
    pos: dict[str, int]
    abs: np.ndarray
    title: np.ndarray
    comp: np.ndarray                  # (n, d), zero rows where not extracted
    has_comp: np.ndarray              # bool
    spec: np.ndarray                  # (n, 768), zero rows where missing
    has_spec: np.ndarray
    meta: list[dict] = field(default_factory=list)
    unis: list[frozenset] = field(default_factory=list)
    topic_cent: dict[int, np.ndarray] = field(default_factory=dict)
    sparse_abs: dict[int, dict] = field(default_factory=dict)
    sparse_title: dict[int, dict] = field(default_factory=dict)
    sparse_comp: dict[int, dict] = field(default_factory=dict)
    self_abs: dict[int, float] = field(default_factory=dict)
    self_title: dict[int, float] = field(default_factory=dict)
    self_comp: dict[int, float] = field(default_factory=dict)

    # ------------------------------------------------------------------ build
    @classmethod
    def load(cls, conn: sqlite3.Connection) -> "Corpus":
        ids, abs_d, title_d = load_embeddings()
        pos = {w: i for i, w in enumerate(ids)}
        n = len(ids)
        d = emb_dir()
        comp = np.zeros_like(abs_d)
        has_comp = np.zeros(n, dtype=bool)
        cids = json.loads((d / "composite_ids.json").read_text())
        cd = np.load(d / "composite_dense.npy")
        for w, v in zip(cids, cd):
            comp[pos[w]] = v
            has_comp[pos[w]] = True
        spec = np.zeros((n, 768), dtype=np.float32)
        has_spec = np.zeros(n, dtype=bool)
        sp = d / "specter2_ids.json"
        if sp.exists():
            sids = json.loads(sp.read_text())
            sv = np.load(d / "specter2.npy")
            sv = sv / np.maximum(np.linalg.norm(sv, axis=1, keepdims=True), 1e-9)
            for w, v in zip(sids, sv):
                if w in pos:
                    spec[pos[w]] = v
                    has_spec[pos[w]] = True
        c = cls(ids, pos, abs_d, title_d, comp, has_comp, spec, has_spec)
        c._load_meta(conn)
        c._load_sparse(conn)
        c.reload_topics(conn)
        return c

    def _load_meta(self, conn: sqlite3.Connection) -> None:
        ents: dict[str, dict[str, set]] = {}
        for r in conn.execute("SELECT work_id, kind, normalized FROM entities"):
            ents.setdefault(r[0], {}).setdefault(r[1], set()).add(r[2])
        short = {r[0]: r[1] for r in conn.execute("SELECT openalex_id, short FROM institutions")}
        unis: dict[str, set] = {}
        for r in conn.execute("SELECT work_id, inst_id FROM work_universities"):
            unis.setdefault(r[0], set()).add(short.get(r[1], r[1]))
        rows = {r["work_id"]: r for r in conn.execute(
            "SELECT work_id, title, year, abstract, tldr, subdomain_cluster, extracted_at, micro_topic_id, primary_topic, "
            "topics_json, keywords_json, referenced_works_json, related_works_json, has_code, type, venue_name "
            "FROM works WHERE canonical_id=work_id")}
        ex = {r[0]: (r[1], r[2]) for r in conn.execute("SELECT work_id, sub_problem, core_method FROM extractions WHERE error IS NULL")}
        for w in self.ids:
            r = rows[w]
            e = ents.get(w, {})
            topics = {t["id"]: float(t.get("score") or 0) for t in json.loads(r["topics_json"] or "[]") if t.get("id")}
            kws = {k["name"].lower() for k in json.loads(r["keywords_json"] or "[]") if k.get("name")}
            refs = set(json.loads(r["referenced_works_json"] or "[]"))
            rel = set(json.loads(r["related_works_json"] or "[]"))
            ttl = r["title"] or ""
            self.meta.append({
                "work_id": w, "title": ttl, "tt": title_tokens(ttl), "year": r["year"], "abstract": r["abstract"],
                "tldr": r["tldr"], "subdomain": (r["subdomain_cluster"] or "none") if r["extracted_at"] else "none",
                "extracted": bool(r["extracted_at"]), "micro": r["micro_topic_id"], "primary_topic": r["primary_topic"],
                "topics": topics, "keywords": kws, "refs": refs, "related": rel, "has_code": bool(r["has_code"]),
                "datasets": e.get("datasets", set()), "baselines": e.get("baselines", set()), "metrics": e.get("metrics", set()),
                "sub_problem": ex.get(w, (None, None))[0], "method": ex.get(w, (None, None))[1],
                "type": r["type"], "venue": r["venue_name"]})
            self.unis.append(frozenset(unis.get(w, set())))

    def _load_sparse(self, conn: sqlite3.Connection) -> None:
        tabs = {"abstract": (self.sparse_abs, self.self_abs), "title": (self.sparse_title, self.self_title),
                "composite": (self.sparse_comp, self.self_comp)}
        for r in conn.execute("SELECT work_id, field, lexical_json FROM sparse_vecs"):
            i = self.pos.get(r[0])
            if i is None or r[1] not in tabs:
                continue
            v = json.loads(r[2] or "{}")
            tabs[r[1]][0][i] = v
            tabs[r[1]][1][i] = float(sum(x * x for x in v.values()))

    def reload_topics(self, conn: sqlite3.Connection) -> None:
        """Re-read micro-topic ids/centroids after `match` assigned them."""
        self.topic_cent = {}
        for r in conn.execute("SELECT topic_id, centroid FROM micro_topics"):
            if r[1]:
                self.topic_cent[r[0]] = np.frombuffer(r[1], dtype=np.float32)
        for r in conn.execute("SELECT work_id, micro_topic_id FROM works WHERE canonical_id=work_id"):
            i = self.pos.get(r[0])
            if i is not None:
                self.meta[i]["micro"] = r[1]

    # ---------------------------------------------------------------- signals
    def signals(self, i: int, js: np.ndarray | list[int], *, query_vec: dict | None = None) -> np.ndarray:
        """(len(js), 14) matrix of signals between paper i and each paper in js (NaN = unavailable)."""
        js = np.asarray(js, dtype=int)
        out = np.full((len(js), len(SIGNALS)), np.nan, dtype=np.float64)
        if not len(js):
            return out
        col = {s: k for k, s in enumerate(SIGNALS)}
        out[:, col["dense_abstract"]] = np.clip(self.abs[js] @ self.abs[i], 0, 1)
        out[:, col["title_dense"]] = np.clip(self.title[js] @ self.title[i], 0, 1)
        if self.has_comp[i]:
            m = self.has_comp[js]
            out[m, col["dense_composite"]] = np.clip(self.comp[js[m]] @ self.comp[i], 0, 1)
        if self.has_spec[i]:
            m = self.has_spec[js]
            out[m, col["specter2"]] = np.clip(self.spec[js[m]] @ self.spec[i], 0, 1)
        A = self.meta[i]
        ca = self.topic_cent.get(A["micro"]) if A["micro"] is not None and A["micro"] >= 0 else None
        for r, j in enumerate(js):
            B = self.meta[j]
            # s4 title lexical: token_set_ratio + bge-m3 sparse on titles
            parts = []
            if A["tt"] and B["tt"]:
                parts.append(fuzz.token_set_ratio(A["tt"], B["tt"]) / 100)
            sp = sparse_norm(self.sparse_title.get(i), self.sparse_title.get(j), self.self_title.get(i, 0), self.self_title.get(j, 0))
            if sp == sp:
                parts.append(sp)
            if parts:
                out[r, col["title_lexical"]] = float(np.mean(parts))
            # s5 sparse lexical (abstract; composite too when both extracted)
            s5 = [sparse_norm(self.sparse_abs.get(i), self.sparse_abs.get(j), self.self_abs.get(i, 0), self.self_abs.get(j, 0))]
            if self.has_comp[i] and self.has_comp[j]:
                s5.append(sparse_norm(self.sparse_comp.get(i), self.sparse_comp.get(j), self.self_comp.get(i, 0), self.self_comp.get(j, 0)))
            s5 = [x for x in s5 if x == x]
            if s5:
                out[r, col["sparse_lexical"]] = float(np.mean(s5))
            # s7 micro-topic
            if ca is not None and B["micro"] is not None and B["micro"] >= 0:
                if A["micro"] == B["micro"]:
                    out[r, col["micro_topic"]] = 1.0
                elif B["micro"] in self.topic_cent:
                    out[r, col["micro_topic"]] = float(np.clip(ca @ self.topic_cent[B["micro"]], 0, 1))
            # s8 OpenAlex topics (+ primary-topic bonus), s9 keywords
            wj = weighted_jaccard(A["topics"], B["topics"])
            if wj == wj:
                bonus = 0.2 if A["primary_topic"] and A["primary_topic"] == B["primary_topic"] else 0.0
                out[r, col["openalex_topics"]] = min(wj + bonus, 1.0)
            out[r, col["keywords"]] = jaccard(A["keywords"], B["keywords"])
            for k in ("datasets", "baselines", "metrics"):
                out[r, col[k]] = jaccard(A[k], B[k])
            # s13 bibliographic coupling, s14 direct citation link
            ra, rb = A["refs"], B["refs"]
            if ra and rb:
                out[r, col["biblio_coupling"]] = len(ra & rb) / (len(ra) * len(rb)) ** 0.5
            if ra or rb or A["related"] or B["related"]:
                wa, wb = A["work_id"], B["work_id"]
                out[r, col["citation_link"]] = float(wb in ra or wa in rb or wb in A["related"] or wa in B["related"])
        return out

    def shared(self, i: int, j: int) -> dict[str, list[str]]:
        A, B = self.meta[i], self.meta[j]
        d = {k: sorted(A[k] & B[k]) for k in ("datasets", "baselines", "metrics")}
        d["keywords"] = sorted(A["keywords"] & B["keywords"])[:6]
        return {k: v for k, v in d.items() if v}
