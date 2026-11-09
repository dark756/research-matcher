"""`run-all` and `status`: run every stage in order, one subprocess per stage (frees VRAM between GPU stages), log to `runs`,
and diff table counts against the previous snapshot so a rerun can be compared with the earlier results (plan §10)."""
from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from .config import load_settings

# (name, CLI args, GPU?, quick-mode args that reuse saved artefacts; None = skipped in quick mode)
STAGES: list[tuple[str, list[str], bool, list[str] | None]] = [
    ("init", ["init"], False, ["init", "--skip-bge", "--skip-csrankings"]),
    ("harvest", ["harvest"], False, None),
    ("enrich", ["enrich"], False, None),
    ("prefilter", ["prefilter"], True, ["prefilter", "--skip-embed", "--skip-index"]),
    ("extract", ["extract"], True, ["extract"]),
    ("extract-post", ["extract-post"], False, ["extract-post"]),
    ("index", ["index"], True, ["index", "--skip-embed"]),
    ("match", ["match"], True, ["match", "--skip-topics"]),
    ("report", ["report"], True, ["report", "--no-draft"]),
]
NAMES = [s[0] for s in STAGES]

SNAPSHOT_SQL = {
    "institutions": "SELECT COUNT(*) FROM institutions",
    "works": "SELECT COUNT(*) FROM works",
    "canonical_works": "SELECT COUNT(*) FROM works WHERE canonical_id=work_id",
    "work_universities": "SELECT COUNT(*) FROM work_universities",
    "faculty": "SELECT COUNT(*) FROM faculty",
    "prefilter_kept": "SELECT COUNT(*) FROM prefilter WHERE keep=1",
    "extractions_ok": "SELECT COUNT(*) FROM extractions WHERE error IS NULL",
    "entities": "SELECT COUNT(*) FROM entities",
    "reported_results": "SELECT COUNT(*) FROM reported_results",
    "sparse_vecs": "SELECT COUNT(*) FROM sparse_vecs",
    "professor_profiles": "SELECT COUNT(*) FROM professor_profiles",
    "micro_topics": "SELECT COUNT(*) FROM micro_topics",
    "pairs": "SELECT COUNT(*) FROM pairs",
    "clusters": "SELECT COUNT(*) FROM clusters",
    "http_cache": "SELECT COUNT(*) FROM http_cache",
}


def snapshot(conn: sqlite3.Connection) -> dict[str, int]:
    out = {}
    for k, sql in SNAPSHOT_SQL.items():
        try:
            out[k] = conn.execute(sql).fetchone()[0]
        except sqlite3.Error:
            out[k] = -1
    return out


def diff(before: dict, after: dict) -> list[tuple[str, int, int]]:
    return [(k, before.get(k, -1), after[k]) for k in after if before.get(k) != after[k]]


def select_stages(start: str | None, stop: str | None, skip: list[str]) -> list[str]:
    for n in [start, stop, *skip]:
        if n and n not in NAMES:
            raise ValueError(f"unknown stage {n!r}; choose from {', '.join(NAMES)}")
    a = NAMES.index(start) if start else 0
    b = NAMES.index(stop) if stop else len(NAMES) - 1
    if a > b:
        raise ValueError("--from comes after --to")
    return [n for n in NAMES[a:b + 1] if n not in skip]


def fresh(conn_path: Path, data_dir: Path) -> Path:
    """Start from an empty data/: the old DB is moved to papers.db.bak-<ts> (nothing is deleted), and the new DB inherits only
    http_cache + rate_limits so API responses are not re-fetched. Derived files (embeddings, vectors, qdrant) are removed.
    Gold labels, config, notes and data/cache are kept."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    bak = conn_path.with_name(f"{conn_path.name}.bak-{ts}")
    for ext in ("", "-wal", "-shm"):
        p = Path(str(conn_path) + ext)
        if p.exists():
            p.rename(Path(str(bak) + ext))
    new = sqlite3.connect(conn_path)
    new.execute("ATTACH DATABASE ? AS old", (str(bak),))
    for t in ("http_cache", "rate_limits"):
        sql = new.execute("SELECT sql FROM old.sqlite_master WHERE type='table' AND name=?", (t,)).fetchone()
        if sql:
            new.execute(sql[0])
            new.execute(f"INSERT INTO main.{t} SELECT * FROM old.{t}")
    new.commit()
    new.execute("DETACH DATABASE old")
    new.close()
    for d in ("embeddings", "vectors", "qdrant_storage"):
        shutil.rmtree(data_dir / d, ignore_errors=True)
    return bak


def run_stage(args: list[str]) -> int:
    return subprocess.call([sys.executable, "-m", "research_matcher.cli", *args])


def run_all(conn: sqlite3.Connection, *, start: str | None = None, stop: str | None = None, skip: list[str] | None = None,
            quick: bool = False, dry_run: bool = False, log=print) -> dict:
    names = select_stages(start, stop, skip or [])
    by = {s[0]: s for s in STAGES}
    before = snapshot(conn)
    results = []
    for n in names:
        _, full, gpu, qk = by[n]
        args = qk if quick else full
        if args is None:
            log(f"== {n}: skipped in --quick mode (network stage)")
            results.append({"stage": n, "args": None, "rc": None})
            continue
        log(f"== {n}: research-matcher {' '.join(args)}{'  [GPU]' if gpu else ''}")
        if dry_run:
            results.append({"stage": n, "args": args, "rc": None})
            continue
        t0 = time.time()
        rc = run_stage(args)
        t1 = time.time()
        conn.execute("INSERT INTO runs VALUES (?,?,?,?,?)", (n, t0, t1, json.dumps({"args": args, "rc": rc}), json.dumps(snapshot(conn))))
        results.append({"stage": n, "args": args, "rc": rc, "seconds": round(t1 - t0)})
        try:  # GPU stages never overlap with gemma3 in VRAM (plan §10.3)
            from .ollama_client import OllamaClient

            OllamaClient().unload()
        except Exception:  # noqa: BLE001
            pass
        if rc != 0:
            log(f"stage {n} failed (exit {rc}); fix it and resume with --from {n}")
            break
    after = snapshot(conn)
    return {"stages": results, "before": before, "after": after, "changed": diff(before, after),
            "ok": all(r["rc"] in (0, None) for r in results)}


def status_rows(conn: sqlite3.Connection) -> tuple[dict[str, int], list[tuple]]:
    last = conn.execute("SELECT stage, MAX(finished), COUNT(*) FROM runs GROUP BY stage").fetchall()
    return snapshot(conn), [(r[0], time.strftime("%Y-%m-%d %H:%M", time.localtime(r[1])), r[2]) for r in last]
