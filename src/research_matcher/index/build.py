"""Stage 5 `index`: entity normalisation, composite vectors, Qdrant collections, professor profiles, gate checks."""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Callable

import numpy as np

from ..embed.prefilter import emb_dir, load_embeddings
from ..extract.normalize import normalize


def renormalize_entities(conn: sqlite3.Connection) -> dict[str, int]:
    """Recompute `normalized` from `raw` with the current aliases.yaml (extract stored them at extraction time)."""
    rows = conn.execute("SELECT work_id, kind, raw FROM entities").fetchall()
    conn.execute("BEGIN")
    conn.execute("DELETE FROM entities")
    for r in rows:
        conn.execute("INSERT OR IGNORE INTO entities VALUES (?,?,?,?)",
                     (r["work_id"], r["kind"], normalize(r["kind"], r["raw"]), r["raw"]))
    conn.execute("COMMIT")
    return {"before": len(rows), "after": conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]}


def composite_text(sub_problem: str, method: str, ents: dict[str, list[str]]) -> str:
    """'Problem / Method / Baselines / Datasets / Metrics' text embedded for the composite similarity signal."""
    parts = [f"Problem: {sub_problem}", f"Method: {method}"]
    for label, kind in (("Baselines", "baselines"), ("Datasets", "datasets"), ("Metrics", "metrics")):
        if ents.get(kind):
            parts.append(f"{label}: {', '.join(ents[kind])}")
    return ". ".join(p.rstrip(".") for p in parts) + "."


def embed_composites(conn: sqlite3.Connection, bge) -> list[str]:
    """Embed composites for every extracted paper (GPU); sparse weights go to sparse_vecs(field='composite')."""
    ents: dict[str, dict[str, list[str]]] = {}
    for r in conn.execute("SELECT work_id, kind, raw FROM entities WHERE kind IN ('baselines','datasets','metrics')"):
        ents.setdefault(r["work_id"], {}).setdefault(r["kind"], []).append(r["raw"])
    rows = conn.execute("SELECT work_id, sub_problem, core_method FROM extractions WHERE error IS NULL ORDER BY work_id").fetchall()
    ids = [r["work_id"] for r in rows]
    texts = [composite_text(r["sub_problem"] or "", r["core_method"] or "", ents.get(r["work_id"], {})) for r in rows]
    dense, lex = bge.encode(texts, max_length=256, batch_size=48)
    conn.execute("BEGIN")
    for w, l in zip(ids, lex):
        conn.execute("INSERT OR REPLACE INTO sparse_vecs VALUES (?,?,?)", (w, "composite", json.dumps(l)))
    conn.execute("COMMIT")
    d = emb_dir()
    np.save(d / "composite_dense.npy", dense)
    (d / "composite_ids.json").write_text(json.dumps(ids))
    return ids


def build_relevant(conn, store, log: Callable[[str], None] = print) -> tuple[int, int]:
    """Qdrant all_papers (refreshed payloads) and relevant_papers (extracted papers, + composite vector)."""
    from .vector_store import build_collection, build_all_papers, paper_payloads

    ids, abs_d, title_d = load_embeddings()
    n_all = build_all_papers(conn, store, ids, abs_d, title_d)
    log(f"  all_papers: {n_all} points")
    pos = {w: i for i, w in enumerate(ids)}
    d = emb_dir()
    cids = json.loads((d / "composite_ids.json").read_text())
    comp = np.load(d / "composite_dense.npy")
    sel = [pos[w] for w in cids]
    payloads = paper_payloads(conn, cids)
    n_rel = build_collection(store, "relevant_papers", cids,
                             {"abstract": abs_d[sel], "title": title_d[sel], "composite": comp}, payloads)
    log(f"  relevant_papers: {n_rel} points")
    return n_all, n_rel


def check(conn: sqlite3.Connection, store, n_queries: int = 200, log: Callable[[str], None] = print) -> dict:
    """M5 gate: point counts vs SQLite, payload-filter smoke tests, p95 search latency on all_papers."""
    res: dict = {}
    n_canon = conn.execute("SELECT COUNT(*) FROM works WHERE canonical_id=work_id").fetchone()[0]
    n_ext = conn.execute("SELECT COUNT(*) FROM extractions WHERE error IS NULL").fetchone()[0]
    res["counts"] = {"all_papers": (store.count("all_papers"), n_canon), "relevant_papers": (store.count("relevant_papers"), n_ext)}
    ids, abs_d, _ = load_embeddings()
    rng = np.random.default_rng(0)
    qs = abs_d[rng.choice(len(ids), n_queries, replace=False)]
    cross = {"must_not": [{"key": "target_universities", "any": ["ETH", "CMU"]}]}
    filt = {"must": [{"key": "subdomain", "eq": "Edge_RAG_Vector_Systems"}, {"key": "year", "gte": 2026}]}
    lat = {"unfiltered": [], "filtered": []}
    for i, q in enumerate(qs):
        t = time.perf_counter()
        store.search("all_papers", "abstract", q, 150)
        lat["unfiltered"].append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        store.search("all_papers", "abstract", q, 150, filt if i % 2 else cross)
        lat["filtered"].append((time.perf_counter() - t) * 1000)
    res["p95_ms"] = {k: round(float(np.percentile(v, 95)), 1) for k, v in lat.items()}
    smoke = {}
    q0 = qs[0]
    r = store.search("all_papers", "abstract", q0, 50, filt)
    smoke["subdomain+year filter"] = (len(r), all(p.payload["subdomain"] == "Edge_RAG_Vector_Systems" and p.payload["year"] >= 2026 for p in r))
    r = store.search("all_papers", "abstract", q0, 50, cross)
    smoke["must_not ETH,CMU"] = (len(r), all(not ({"ETH", "CMU"} & set(p.payload["target_universities"])) for p in r))
    r = store.search("relevant_papers", "composite", np.load(emb_dir() / "composite_dense.npy")[0], 50,
                     {"must": [{"key": "has_code", "eq": True}]})
    smoke["has_code"] = (len(r), all(p.payload["has_code"] for p in r))
    r = store.search("relevant_papers", "abstract", q0, 50, {"must": [{"key": "target_universities", "any": ["NUS", "NTU"]}]})
    smoke["universities any(NUS,NTU)"] = (len(r), all({"NUS", "NTU"} & set(p.payload["target_universities"]) for p in r))
    res["smoke"] = smoke
    return res
