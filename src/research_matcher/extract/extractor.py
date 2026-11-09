"""Stage 4: gemma3 structured extraction on pre-filter survivors (resumable)."""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Callable

from pydantic import ValidationError

from ..config import load_settings
from ..ollama_client import OllamaClient
from .grounding import ground
from .normalize import normalize
from .prompts import body_system_prompt, classify_messages, user_prompt
from .schema import Classification, ExtractionBody, PaperExtraction, body_schema, classify_schema

KINDS = {"datasets": "datasets_or_benchmarks_used", "baselines": "baseline_algorithms_compared",
         "metrics": "evaluation_metrics", "hardware": "hardware_mentioned"}


def _call(client: OllamaClient, msgs: list[dict], schema: dict, model_cls, predicts: tuple[int, ...]):
    """Structured call with one retry at a larger num_predict. Returns (parsed | None, raw, error)."""
    cfg = load_settings().extract
    raw, err = "", None
    for predict in predicts:
        opts = {"temperature": cfg["temperature"], "seed": cfg["seed"], "num_ctx": cfg["num_ctx"], "num_predict": predict}
        try:
            raw = client.chat(msgs, schema=schema, options=opts)
            return model_cls.model_validate_json(raw), raw, None
        except (ValidationError, json.JSONDecodeError) as e:
            err = f"invalid JSON: {str(e)[:200]}"
        except Exception as e:  # noqa: BLE001  (HTTP errors etc.)
            return None, raw, f"{type(e).__name__}: {str(e)[:200]}"
    return None, raw, err


def extract_one(client: OllamaClient, title: str, abstract: str | None, tldr: str | None) -> tuple[PaperExtraction | None, int, str, str | None]:
    """Two calls: entities (+ retry at 1024 tokens), then a short few-shot subdomain classification.
    Returns (grounded extraction | None, dropped_items, raw_json, error)."""
    cfg = load_settings().extract
    body, raw, err = _call(client, [{"role": "system", "content": body_system_prompt()},
                                    {"role": "user", "content": user_prompt(title, abstract, tldr)}],
                           body_schema(), ExtractionBody, (cfg["num_predict"], 1024))
    if body is None:
        return None, 0, raw, err
    cls, raw2, err2 = _call(client, classify_messages(title, abstract), classify_schema(), Classification, (64, 128))
    if cls is None:
        return None, 0, raw + raw2, err2
    ex = PaperExtraction(**body.model_dump(), **cls.model_dump())
    ex, dropped = ground(ex, f"{title}. {abstract or ''} {tldr or ''}")
    return ex, dropped, json.dumps({"body": json.loads(raw), "classification": json.loads(raw2)}), None


def final_labels(ex, prefilter_best: str | None, kw: dict | None = None) -> dict:
    """Consolidate the LLM label with lexicon evidence from the pre-filter (plan §10.1). `ex` needs .subdomain_cluster
    and .confidence. "none" stays "none"; a positive label needs >=1 lexicon hit for that subdomain, otherwise it is
    demoted to "none" and flagged. (Flipping to another subdomain by lexicon was tried and removed: the SysML lexicon
    contains generic terms like "reinforcement learning", which relabelled unrelated RL papers.)
    Disagreement with the pre-filter without high confidence also sets needs_review. `llm_label` keeps the raw answer."""
    kw = kw or {}
    label = ex.subdomain_cluster
    agree = label == prefilter_best
    demoted = label != "none" and kw.get(label, 0) < 1
    final = "none" if demoted else label
    needs_review = demoted or not (agree or ex.confidence == "high")
    return {"subdomain_cluster": final, "labels_agree": int(agree), "needs_review": int(needs_review)}


def relabel_all(conn: sqlite3.Connection) -> dict[str, int]:
    """Re-apply final_labels to stored LLM output (no model calls)."""
    from types import SimpleNamespace

    changed = 0
    conn.execute("BEGIN")
    for r in conn.execute("SELECT w.work_id, w.llm_label, w.confidence, w.subdomain_cluster, p.best_cluster, p.kw_scores_json "
                          "FROM works w JOIN prefilter p USING(work_id) WHERE w.extracted_at IS NOT NULL").fetchall():
        fl = final_labels(SimpleNamespace(subdomain_cluster=r["llm_label"], confidence=r["confidence"]),
                          r["best_cluster"], json.loads(r["kw_scores_json"]))
        changed += fl["subdomain_cluster"] != r["subdomain_cluster"]
        conn.execute("UPDATE works SET subdomain_cluster=?, labels_agree=?, needs_review=? WHERE work_id=?",
                     (fl["subdomain_cluster"], fl["labels_agree"], fl["needs_review"], r["work_id"]))
        conn.execute("UPDATE extractions SET subdomain_cluster=?, needs_review=? WHERE work_id=?",
                     (fl["subdomain_cluster"], fl["needs_review"], r["work_id"]))
    conn.execute("COMMIT")
    return {"changed": changed}


def store(conn: sqlite3.Connection, work_id: str, ex: PaperExtraction | None, dropped: int, raw: str, err: str | None,
          prefilter_best: str | None, kw: dict | None = None) -> None:
    cfg, model = load_settings().extract, load_settings().models["llm"]
    now = time.time()
    conn.execute("BEGIN")
    if ex is None:
        conn.execute("INSERT OR REPLACE INTO extractions(work_id, model, prompt_version, raw_json, error) VALUES (?,?,?,?,?)",
                     (work_id, model, cfg["prompt_version"], raw, err))
        conn.execute("COMMIT")
        return
    fl = final_labels(ex, prefilter_best, kw)
    conn.execute(
        "INSERT OR REPLACE INTO extractions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (work_id, model, cfg["prompt_version"], raw, int(ex.is_relevant_cs_systems_or_ml), ex.subdomain_cluster,
         ex.sub_problem_solved, ex.core_algorithm_or_architecture, ex.confidence, fl["needs_review"], dropped, None))
    conn.execute("DELETE FROM entities WHERE work_id=?", (work_id,))
    conn.execute("DELETE FROM reported_results WHERE work_id=?", (work_id,))
    for kind, field in KINDS.items():
        for raw_item in getattr(ex, field):
            conn.execute("INSERT OR IGNORE INTO entities VALUES (?,?,?,?)",
                         (work_id, kind, normalize(kind, raw_item), raw_item))
    for r in ex.reported_results:
        conn.execute("INSERT INTO reported_results VALUES (?,?,?,?,?)", (work_id, r.metric, r.value, r.compared_to, r.dataset))
    conn.execute(
        "UPDATE works SET subdomain_cluster=?, llm_label=?, labels_agree=?, is_relevant=?, confidence=?, needs_review=?, "
        "llm_model=?, prompt_version=?, extracted_at=? WHERE work_id=?",
        (fl["subdomain_cluster"], ex.subdomain_cluster, fl["labels_agree"], int(ex.is_relevant_cs_systems_or_ml),
         ex.confidence, fl["needs_review"], model, cfg["prompt_version"], now, work_id))
    conn.execute("COMMIT")


def run_extract(conn: sqlite3.Connection, *, limit: int | None = None, redo: bool = False,
                log: Callable[[str], None] = print) -> dict[str, int]:
    cfg = load_settings().extract
    client = OllamaClient()
    done_clause = "" if redo else (" AND p.work_id NOT IN (SELECT work_id FROM extractions "
                                   f"WHERE error IS NULL AND prompt_version='{cfg['prompt_version']}')")
    # most plausible first, so an interrupted run still has the best candidates
    rows = conn.execute(
        f"SELECT p.work_id, p.best_cluster, p.kw_scores_json, w.title, w.abstract, w.tldr FROM prefilter p JOIN works w USING(work_id) "
        f"WHERE p.keep=1{done_clause} ORDER BY p.work_id").fetchall()
    if limit:
        rows = rows[:limit]
    stats = dict(total=len(rows), ok=0, failed=0, dropped_items=0)
    log(f"extract: {len(rows)} papers to process with {client.model} (prompt {cfg['prompt_version']})")
    t0 = time.time()
    for n, r in enumerate(rows, 1):
        ex, dropped, raw, err = extract_one(client, r["title"], r["abstract"], r["tldr"])
        store(conn, r["work_id"], ex, dropped, raw, err, r["best_cluster"], json.loads(r["kw_scores_json"]))
        stats["ok" if ex else "failed"] += 1
        stats["dropped_items"] += dropped
        if n % 25 == 0 or n == len(rows):
            rate = n / (time.time() - t0)
            log(f"  {n}/{len(rows)}  ok={stats['ok']} failed={stats['failed']}  {rate:.2f}/s  eta {(len(rows) - n) / rate / 60:.0f} min")
    return stats
