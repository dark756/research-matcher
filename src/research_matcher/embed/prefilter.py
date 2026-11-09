"""Stage 3 pre-filter (plan §7): embed every canonical paper, score keywords + prototype similarity, keep survivors."""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import load_settings, load_subdomains

CHUNK = 2000


def emb_dir() -> Path:
    d = load_settings().path("data_dir") / "embeddings"
    (d / "chunks").mkdir(parents=True, exist_ok=True)
    return d


def paper_text(r: sqlite3.Row) -> str:
    """Title + abstract; abstract-less papers are embedded by title + keywords + topics + tldr (plan §2)."""
    if r["abstract"]:
        return f"{r['title']}. {r['abstract']}"
    kws = " ".join(k["name"] for k in json.loads(r["keywords_json"] or "[]") if k.get("name"))
    tops = " ".join(t["name"] for t in json.loads(r["topics_json"] or "[]")[:3] if t.get("name"))
    return f"{r['title']}. {kws}. {tops}. {r['tldr'] or ''}".strip()


def canonical_ids(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute("SELECT work_id FROM works WHERE canonical_id=work_id ORDER BY work_id")]


def embed_all(conn: sqlite3.Connection, bge, log: Callable[[str], None] = print) -> int:
    """Resumable: each chunk of 2000 papers is written to data/embeddings/chunks/ and its sparse weights to SQLite."""
    d = emb_dir()
    ids = canonical_ids(conn)
    (d / "ids.json").write_text(json.dumps(ids))
    n_chunks = -(-len(ids) // CHUNK)
    for c in range(n_chunks):
        fa, ft = d / "chunks" / f"abstract_{c:04d}.npy", d / "chunks" / f"title_{c:04d}.npy"
        if fa.exists() and ft.exists():
            continue
        chunk = ids[c * CHUNK:(c + 1) * CHUNK]
        rows = {r["work_id"]: r for r in conn.execute(
            f"SELECT work_id,title,abstract,keywords_json,topics_json,tldr FROM works "
            f"WHERE work_id IN ({','.join('?' * len(chunk))})", chunk)}
        dense_a, lex_a = bge.encode([paper_text(rows[i]) for i in chunk], max_length=512, batch_size=24)
        dense_t, lex_t = bge.encode([rows[i]["title"] or "" for i in chunk], max_length=64, batch_size=128)
        conn.execute("BEGIN")
        for i, la, lt in zip(chunk, lex_a, lex_t):
            conn.execute("INSERT OR REPLACE INTO sparse_vecs VALUES (?,?,?)", (i, "abstract", json.dumps(la)))
            conn.execute("INSERT OR REPLACE INTO sparse_vecs VALUES (?,?,?)", (i, "title", json.dumps(lt)))
        conn.execute("COMMIT")
        np.save(ft, dense_t)
        np.save(fa, dense_a)  # written last: its presence marks the chunk complete
        log(f"  embedded chunk {c + 1}/{n_chunks}")
    for kind in ("abstract", "title"):
        arr = np.concatenate([np.load(d / "chunks" / f"{kind}_{c:04d}.npy") for c in range(n_chunks)])
        np.save(d / f"{kind}_dense.npy", arr)
    return len(ids)


def load_embeddings() -> tuple[list[str], np.ndarray, np.ndarray]:
    d = emb_dir()
    return json.loads((d / "ids.json").read_text()), np.load(d / "abstract_dense.npy"), np.load(d / "title_dense.npy")


def keyword_score(title: str, abstract: str, lexicon: list[str]) -> int:
    """Distinct lexicon patterns hit; a title hit counts double."""
    s = 0
    for pat in lexicon:
        rx = re.compile(pat, re.I)
        if rx.search(title or ""):
            s += 2
        elif rx.search(abstract or ""):
            s += 1
    return s


def prototype_matrix(bge) -> tuple[list[str], np.ndarray, list[int]]:
    """Embed every subdomain prototype; returns (cluster ids, matrix [P,1024], cluster index per row)."""
    clusters, texts, owner = [], [], []
    for sd in load_subdomains():
        if not sd["prototypes"]:
            continue
        clusters.append(sd["id"])
        for p in sd["prototypes"]:
            texts.append(p)
            owner.append(len(clusters) - 1)
    dense, _ = bge.encode(texts, max_length=128, batch_size=16, sparse=False)
    return clusters, dense, owner


def meaning_scores(abs_dense: np.ndarray, protos: np.ndarray, owner: list[int], n_clusters: int) -> np.ndarray:
    """[n, n_clusters] max cosine to each cluster's prototypes (vectors are L2-normalised)."""
    sims = abs_dense @ protos.T
    out = np.zeros((abs_dense.shape[0], n_clusters), dtype=np.float32)
    owner_arr = np.asarray(owner)
    for c in range(n_clusters):
        out[:, c] = sims[:, owner_arr == c].max(axis=1)
    return out


def keep_rule(meaning: float, kw: int, tau: float, kw_min: int) -> bool:
    return meaning >= tau or (kw >= kw_min and meaning >= tau - 0.05)


def score_all(conn: sqlite3.Connection, clusters: list[str], meaning: np.ndarray, ids: list[str],
              tau: dict[str, float], kw_min: int) -> dict[str, dict]:
    lex = {sd["id"]: sd["lexicon"] for sd in load_subdomains()}
    rows = {r["work_id"]: r for r in conn.execute("SELECT work_id, title, abstract FROM works WHERE canonical_id=work_id")}
    out = {}
    for i, wid in enumerate(ids):
        r = rows[wid]
        kws = {c: keyword_score(r["title"], r["abstract"], lex[c]) for c in clusters}
        dense = {c: float(meaning[i, j]) for j, c in enumerate(clusters)}
        passing = [c for c in clusters if keep_rule(dense[c], kws[c], tau[c], kw_min)]
        pool = passing or clusters
        best = max(pool, key=lambda c: dense[c])
        out[wid] = {"kw": kws, "dense": dense, "best": best, "keep": bool(passing)}
    return out


def write_prefilter(conn: sqlite3.Connection, scores: dict[str, dict]) -> int:
    conn.execute("BEGIN")
    conn.execute("DELETE FROM prefilter")
    for wid, s in scores.items():
        conn.execute("INSERT INTO prefilter VALUES (?,?,?,?,?)",
                     (wid, json.dumps(s["kw"]), json.dumps(s["dense"]), s["best"], int(s["keep"])))
    conn.execute("COMMIT")
    return sum(s["keep"] for s in scores.values())
