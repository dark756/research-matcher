"""M7 gate: run the evaluation questions through the real pipeline and score them (plan §12 M7)."""
from __future__ import annotations

import json
import re

from rich.console import Console
from rich.table import Table

from ..config import ROOT
from .chat import Session, run_question
from .execute import Engine, make_filters


def _matches(text: str, pats: list[str]) -> bool:
    return any(re.search(p, text, re.I) for p in pats)


def run_eval(path: str, *, answer: bool, show: bool, console: Console) -> dict:
    qs = json.loads((ROOT / path).read_text(encoding="utf-8"))
    eng = Engine()
    t = Table("#", "intent", "top-8 hit", "unis", "oracle unis", "invalid ids", "unsupported", "s", "question", show_lines=False)
    hits = n_topic = topic_ok = invalid = unsupported = 0
    rows = []
    for q in qs:
        out = run_question(eng, q["q"], Session(), answer=answer, console=console, show=False)
        res, dets = out.result, out.dets
        top8 = res.items[:8]
        kind = q.get("kind", "papers")
        pats = q.get("expect", [])
        if kind == "stats":
            hit = sum(len(v) for v in res.extra.get("stats", {}).values()) >= 3
        elif kind == "topics":
            hit = bool(res.extra.get("topics"))
        elif kind == "profs":
            hit = len(res.extra.get("professors", [])) >= 3
        else:
            hit = any(_matches(f"{dets[it['work_id']]['title']} {dets[it['work_id']]['abstract']}", pats) for it in top8)
        unis = {u for it in top8 for u in dets[it["work_id"]]["unis"]}
        # oracle: universities that have ANY paper matching the expectation under the same filters (exempts "when they exist")
        keep = make_filters(res.R)[0](eng.corpus)
        oracle = {u for j, m in enumerate(eng.corpus.meta) if pats and keep(j) and _matches(f"{m['title']} {m['abstract'] or ''}", pats)
                  for u in eng.corpus.unis[j]}
        ids, terms = len(out.validation.get("invalid_ids", [])), len(out.validation.get("unsupported_terms", []))
        invalid += ids
        unsupported += terms
        hits += hit
        topic = q.get("topic", False)
        ok_u = None
        if topic:
            n_topic += 1
            ok_u = len(unis) >= 3 or len(oracle) < 3
            topic_ok += ok_u
        rows.append((q, res, hit, len(unis), len(oracle), ids, terms, out))
        t.add_row(str(q["id"]), res.intent, "✅" if hit else "❌", f"{len(unis)}" + ("" if ok_u in (None, True) else " ❌"), str(len(oracle)),
                  str(ids), str(terms), f"{out.plan_s + out.exec_s + out.answer_s:.0f}", q["q"][:60])
        if show:
            for k, it in enumerate(top8, 1):
                d = dets[it["work_id"]]
                console.print(f"    P{k} [{','.join(d['unis'])}] {d['title'][:90]}")
    console.print(t)
    n = len(qs)
    g = Table("M7 gate", "value", "required")
    g.add_row("questions with ≥1 expected paper in top-8", f"{hits}/{n} ({100 * hits / n:.0f}%)", "≥ 80%")
    g.add_row("topic queries returning ≥3 universities (when they exist)", f"{topic_ok}/{n_topic} ({100 * topic_ok / max(n_topic, 1):.0f}%)", "≥ 90%")
    g.add_row("invalid [P#] in answers", str(invalid), "0")
    g.add_row("unsupported entity/professor/university terms (⚠ marked)", str(unsupported), "marked, not counted")
    g.add_row("uni / professor / link fields from DB", "by construction (execute.load_details)", "all")
    console.print(g)
    return {"hit_rate": hits / n, "topic_ok": topic_ok / max(n_topic, 1), "invalid_ids": invalid, "unsupported": unsupported}
