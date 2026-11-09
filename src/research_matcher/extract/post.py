"""Post-extraction enrichment for pre-filter survivors: Hugging Face GitHub links and S2 pass 2 (SPECTER2 vectors)."""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Callable

import numpy as np

from ..config import load_settings
from ..http_client import HttpClient, HttpError
from ..embed.prefilter import emb_dir

HF_API = "https://huggingface.co/api/papers/{}"


def survivors(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT w.work_id, w.arxiv_id, w.s2_paper_id, w.has_code FROM prefilter p JOIN works w USING(work_id) "
                        "WHERE p.keep=1 ORDER BY w.work_id").fetchall()


def run_hf(http: HttpClient, conn: sqlite3.Connection, *, rps: float = 5.0, log: Callable[[str], None] = print) -> dict[str, int]:
    """HF /api/papers/{arxiv_id} -> githubRepo. Only survivors that have an arXiv id and no code link yet."""
    conn.execute("CREATE TABLE IF NOT EXISTS hf_checked (work_id TEXT PRIMARY KEY, status TEXT, checked_at REAL)")
    done = {r[0] for r in conn.execute("SELECT work_id FROM hf_checked")}
    todo = [r for r in survivors(conn) if r["arxiv_id"] and r["work_id"] not in done]
    stats = dict(checked=0, found=0, not_on_hf=0, errors=0)
    log(f"HF code lookup: {len(todo)} survivors with an arXiv id")
    gap = 1.0 / rps
    for n, r in enumerate(todo, 1):
        status = "none"
        try:
            data = http.get_json(HF_API.format(r["arxiv_id"]))
            repo = (data or {}).get("githubRepo")
            if repo:
                status = "found"
                conn.execute("BEGIN")
                conn.execute("INSERT INTO code_links VALUES (?,?,?,?)", (r["work_id"], repo, "hf", 0.95))
                conn.execute("UPDATE works SET has_code=1, code_source=COALESCE(code_source,'hf'), "
                             "code_confidence=MAX(COALESCE(code_confidence,0),0.95) WHERE work_id=?", (r["work_id"],))
                conn.execute("COMMIT")
                stats["found"] += 1
        except HttpError as e:
            if e.status == 404:
                status = "not_on_hf"
                stats["not_on_hf"] += 1
            else:
                stats["errors"] += 1
                continue  # not marked checked: retried next run
        except Exception:  # noqa: BLE001 network hiccup; retry next run
            stats["errors"] += 1
            continue
        conn.execute("INSERT OR REPLACE INTO hf_checked VALUES (?,?,?)", (r["work_id"], status, time.time()))
        stats["checked"] += 1
        if n % 250 == 0:
            log(f"  {n}/{len(todo)} found={stats['found']}")
        time.sleep(gap)
    return stats


def run_specter(http: HttpClient, conn: sqlite3.Connection, *, log: Callable[[str], None] = print) -> dict[str, int]:
    """S2 pass 2: embedding.specter_v2 for survivors (<=500 ids/request through the global 1 req/s limiter)."""
    cfg = load_settings().semantic_scholar
    rows = [r for r in survivors(conn) if r["s2_paper_id"]]
    out_dir = emb_dir()
    vecs: dict[str, list[float]] = {}
    path = out_dir / "specter2.json"
    if path.exists():
        vecs = json.loads(path.read_text())
    todo = [r for r in rows if r["work_id"] not in vecs]
    size = cfg["batch_size"]
    stats = dict(survivors_with_s2=len(rows), already=len(vecs), fetched=0, missing=0, failed=0)
    log(f"S2 pass 2: {len(todo)} survivors -> {-(-len(todo) // size)} requests")
    for i in range(0, len(todo), size):
        chunk = todo[i:i + size]
        try:
            res = http.s2_post("/paper/batch", {"fields": "embedding.specter_v2"}, {"ids": [r["s2_paper_id"] for r in chunk]})
        except HttpError as e:
            log(f"  batch failed: {e}")
            stats["failed"] += len(chunk)
            continue
        for r, rec in zip(chunk, res):
            v = ((rec or {}).get("embedding") or {}).get("vector")
            if v:
                vecs[r["work_id"]] = v
                stats["fetched"] += 1
            else:
                stats["missing"] += 1
        path.write_text(json.dumps(vecs))
    ids = sorted(vecs)
    if ids:
        np.save(out_dir / "specter2.npy", np.asarray([vecs[i] for i in ids], dtype=np.float32))
        (out_dir / "specter2_ids.json").write_text(json.dumps(ids))
    stats["total_vectors"] = len(ids)
    return stats
