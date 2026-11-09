"""VectorStore interface with an embedded-Qdrant implementation (plan §10.2). A NumPy fallback can be swapped in
behind the same interface if the M5 latency gate (p95 < 300 ms on all_papers) fails."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..config import load_settings, load_universities


class VectorStore(Protocol):
    def recreate(self, collection: str, vectors: list[str]) -> None: ...
    def upsert(self, collection: str, ids: list[int], vectors: dict[str, np.ndarray], payloads: list[dict]) -> None: ...
    def count(self, collection: str) -> int: ...


@dataclass
class Hit:
    id: int
    score: float
    payload: dict


# Backend-neutral filter spec:
#   {"must": [cond, ...], "must_not": [cond, ...]}  where cond is one of
#   {"key": k, "eq": v} | {"key": k, "any": [v, ...]} | {"key": k, "gte": x, "lte": y}
# List-valued payload fields (e.g. target_universities) match if ANY element matches.
def spec_to_qdrant(spec: dict | None):
    if not spec:
        return None
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range

    def cond(c: dict):
        if "eq" in c:
            return FieldCondition(key=c["key"], match=MatchValue(value=c["eq"]))
        if "any" in c:
            return FieldCondition(key=c["key"], match=MatchAny(any=c["any"]))
        return FieldCondition(key=c["key"], range=Range(gte=c.get("gte"), lte=c.get("lte")))

    return Filter(must=[cond(c) for c in spec.get("must", [])] or None,
                  must_not=[cond(c) for c in spec.get("must_not", [])] or None)


class NumpyStore:
    """Exact flat index: float32 matrices in memory, payload filters as boolean masks built from inverted indexes.
    Persists to data/vectors/<collection>/. Used because local-mode Qdrant has no payload indexes (filtered p95 > 300 ms)."""

    def __init__(self, path: str | None = None):
        self.root = Path(path) if path else load_settings().path("data_dir") / "vectors"
        self.root.mkdir(parents=True, exist_ok=True)
        self._col: dict[str, dict] = {}
        self._pending: dict[str, dict] = {}

    # --- build --------------------------------------------------------------
    def recreate(self, collection: str, vectors: list[str]) -> None:
        self._pending[collection] = {"names": vectors, "ids": [], "vecs": {v: [] for v in vectors}, "payloads": []}
        self._col.pop(collection, None)

    def upsert(self, collection: str, ids: list[int], vectors: dict[str, np.ndarray], payloads: list[dict]) -> None:
        pend = self._pending[collection]
        pend["ids"] += ids
        pend["payloads"] += payloads
        for k, v in vectors.items():
            pend["vecs"][k].append(np.asarray(v, dtype=np.float32))

    def finalize(self, collection: str) -> None:
        pend = self._pending.pop(collection)
        d = self.root / collection
        d.mkdir(parents=True, exist_ok=True)
        for k, chunks in pend["vecs"].items():
            np.save(d / f"vec_{k}.npy", np.concatenate(chunks))
        (d / "meta.json").write_text(json.dumps({"names": pend["names"], "ids": pend["ids"], "payloads": pend["payloads"]}))
        self._col.pop(collection, None)

    # --- query ---------------------------------------------------------------
    def _load(self, collection: str) -> dict:
        if collection not in self._col:
            d = self.root / collection
            meta = json.loads((d / "meta.json").read_text())
            self._col[collection] = {"ids": np.asarray(meta["ids"]), "payloads": meta["payloads"], "inv": {}, "num": {},
                                     "mask_cache": {},
                                     "vecs": {k: np.load(d / f"vec_{k}.npy") for k in meta["names"]}}
        return self._col[collection]

    def count(self, collection: str) -> int:
        return len(self._load(collection)["ids"])

    def _inv(self, c: dict, key: str) -> dict:
        if key not in c["inv"]:
            inv: dict = {}
            n = len(c["payloads"])
            for i, p in enumerate(c["payloads"]):
                vals = p.get(key)
                for v in (vals if isinstance(vals, list) else [vals]):
                    if v is not None and not isinstance(v, (dict, list)):
                        inv.setdefault(v, np.zeros(n, dtype=bool))[i] = True
            c["inv"][key] = inv
        return c["inv"][key]

    def _num(self, c: dict, key: str) -> np.ndarray:
        if key not in c["num"]:
            c["num"][key] = np.array([float(p[key]) if isinstance(p.get(key), (int, float)) else np.nan
                                      for p in c["payloads"]])
        return c["num"][key]

    def _cond_mask(self, c: dict, cond: dict) -> np.ndarray:
        key = cond["key"]
        n = len(c["payloads"])
        if "eq" in cond:
            return self._inv(c, key).get(cond["eq"], np.zeros(n, dtype=bool))
        if "any" in cond:
            m = np.zeros(n, dtype=bool)
            for v in cond["any"]:
                m |= self._inv(c, key).get(v, np.zeros(n, dtype=bool))
            return m
        x = self._num(c, key)
        m = ~np.isnan(x)
        if cond.get("gte") is not None:
            m &= x >= cond["gte"]
        if cond.get("lte") is not None:
            m &= x <= cond["lte"]
        return m

    def _mask(self, c: dict, spec: dict | None) -> np.ndarray | None:
        if not spec:
            return None
        ck = json.dumps(spec, sort_keys=True)
        if ck not in c["mask_cache"]:
            m = np.ones(len(c["payloads"]), dtype=bool)
            for cond in spec.get("must", []):
                m &= self._cond_mask(c, cond)
            for cond in spec.get("must_not", []):
                m &= ~self._cond_mask(c, cond)
            c["mask_cache"][ck] = m
        return c["mask_cache"][ck]

    def search(self, collection: str, using: str, vector: np.ndarray, limit: int, flt: dict | None = None) -> list[Hit]:
        c = self._load(collection)
        m = self._mask(c, flt)
        V = c["vecs"][using]
        idx = np.nonzero(m)[0] if m is not None else None
        scores = (V[idx] if idx is not None else V) @ np.asarray(vector, dtype=np.float32)
        k = min(limit, len(scores))
        if k == 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        real = idx[top] if idx is not None else top
        return [Hit(int(c["ids"][i]), float(sc), c["payloads"][i]) for i, sc in zip(real, scores[top])]

    def close(self) -> None:
        pass


def open_store():
    """Backend from config/settings.yaml -> index.backend ("numpy" default, or "qdrant")."""
    backend = (load_settings().index or {}).get("backend", "numpy")
    return QdrantStore() if backend == "qdrant" else NumpyStore()


class QdrantStore:
    def __init__(self, path: str | None = None):
        from qdrant_client import QdrantClient

        self.client = QdrantClient(path=path or str(load_settings().path("qdrant")))

    def recreate(self, collection: str, vectors: list[str]) -> None:
        from qdrant_client.models import Distance, VectorParams

        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(collection, vectors_config={
            v: VectorParams(size=1024, distance=Distance.COSINE) for v in vectors})

    def upsert(self, collection: str, ids: list[int], vectors: dict[str, np.ndarray], payloads: list[dict]) -> None:
        from qdrant_client.models import PointStruct

        pts = [PointStruct(id=ids[i], vector={k: v[i].tolist() for k, v in vectors.items()}, payload=payloads[i])
               for i in range(len(ids))]
        self.client.upsert(collection, pts, wait=True)

    def count(self, collection: str) -> int:
        return self.client.count(collection, exact=True).count

    def search(self, collection: str, using: str, vector: np.ndarray, limit: int, flt: dict | None = None) -> list[Hit]:
        pts = self.client.query_points(collection, query=vector.tolist(), using=using, limit=limit,
                                       query_filter=spec_to_qdrant(flt), with_payload=True).points
        return [Hit(p.id, float(p.score), p.payload) for p in pts]

    def finalize(self, collection: str) -> None:
        pass

    def close(self) -> None:
        self.client.close()


def paper_payloads(conn: sqlite3.Connection, ids: list[str]) -> dict[str, dict]:
    """Payload for every canonical paper (plan §10.2). Subdomain is the FINAL label for extracted papers, else "none"."""
    short = {u.openalex_id: u.short for u in load_universities()}
    country = {u.openalex_id: u.country for u in load_universities()}
    unis: dict[str, list[str]] = {}
    pis: dict[str, list[str]] = {}
    verified: dict[str, list[str]] = {}
    for r in conn.execute("SELECT w.work_id, w.inst_id, w.professor_verified, a.name FROM work_universities w "
                          "LEFT JOIN authors a ON a.author_id = w.senior_author_id"):
        unis.setdefault(r[0], []).append(r[1])
        if r[3]:
            pis.setdefault(r[0], []).append(r[3])
            if r[2]:
                verified.setdefault(r[0], []).append(r[3])
    ents: dict[str, dict[str, list[str]]] = {}
    for r in conn.execute("SELECT work_id, kind, normalized FROM entities"):
        ents.setdefault(r[0], {}).setdefault(r[1], []).append(r[2])
    code: dict[str, list[str]] = {}
    for r in conn.execute("SELECT work_id, url FROM code_links"):
        code.setdefault(r[0], []).append(r[1])
    out = {}
    for r in conn.execute(
            "SELECT work_id, title, year, type, venue_name, has_code, cited_by_count, subdomain_cluster, extracted_at, "
            "micro_topic_id, doi_url, arxiv_abs_url, oa_pdf_url, needs_review FROM works WHERE canonical_id=work_id"):
        w, u = r["work_id"], unis.get(r["work_id"], [])
        e = ents.get(w, {})
        out[w] = {
            "work_id": w, "title": r["title"], "year": r["year"], "type": r["type"], "venue": r["venue_name"],
            "target_universities": [short[i] for i in u], "countries": sorted({country[i] for i in u}),
            "senior_pis": pis.get(w, []), "verified_pis": verified.get(w, []),
            "subdomain": (r["subdomain_cluster"] or "none") if r["extracted_at"] else "none",
            "micro_topic_id": r["micro_topic_id"], "has_code": bool(r["has_code"]), "github_urls": code.get(w, []),
            "cited_by_count": r["cited_by_count"], "extracted": bool(r["extracted_at"]),
            "needs_review": bool(r["needs_review"]), "datasets": e.get("datasets", []),
            "baselines": e.get("baselines", []), "metrics": e.get("metrics", []),
            "doi_url": r["doi_url"], "arxiv_url": r["arxiv_abs_url"], "pdf_url": r["oa_pdf_url"]}
    return {w: out[w] for w in ids}


def build_collection(store: QdrantStore, name: str, ids: list[str], vectors: dict[str, np.ndarray],
                     payloads: dict[str, dict], batch: int = 1000) -> int:
    """Create `name` with the given named vectors; point id = position in `ids` (see point_id_map)."""
    store.recreate(name, list(vectors))
    for s in range(0, len(ids), batch):
        sl = slice(s, s + batch)
        store.upsert(name, list(range(s, min(s + batch, len(ids)))), {k: v[sl] for k, v in vectors.items()},
                     [payloads[w] for w in ids[sl]])
    store.finalize(name)
    return store.count(name)


def build_all_papers(conn: sqlite3.Connection, store: QdrantStore, ids: list[str], abs_dense: np.ndarray,
                     title_dense: np.ndarray) -> int:
    return build_collection(store, "all_papers", ids, {"abstract": abs_dense, "title": title_dense},
                            paper_payloads(conn, ids))


def point_id_map(ids: list[str]) -> dict[str, int]:
    return {w: i for i, w in enumerate(ids)}
