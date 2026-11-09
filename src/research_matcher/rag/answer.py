"""Step 3b: gemma3 synthesis over the top papers (plan §9.4.3)."""
from __future__ import annotations

import json
import re

from ..config import load_settings

SYSTEM = """You are a research assistant that compares papers across universities. You are given a question, a query plan and up to 12 papers as JSON, each with an id like "P3".
Rules:
- Compare the approaches ACROSS universities; say which university does what.
- Cite after EVERY claim with square brackets exactly like [P3] or [P1][P4], using only the ids provided. Never invent ids.
- Name shared datasets and baselines only if they appear in the papers' fields.
- Use only the information in the JSON. For anything else say "not in the data". Do not use outside knowledge about results or numbers.
- Be concise: 4-8 sentences or a short bullet list. Do not repeat the paper table."""


def trim(text: str, words: int = 180) -> str:
    w = (text or "").split()
    return " ".join(w[:words]) + (" …" if len(w) > words else "")


def build_context(res, dets: dict, n: int | None = None) -> list[dict]:
    n = n or int(load_settings().rag.get("context_papers", 8))
    ctx = []
    for k, it in enumerate(res.items[:n], 1):
        d = dets[it["work_id"]]
        ctx.append({"id": f"P{k}", "title": d["title"], "universities": d["unis"],
                    "professors": {u: v[0] for u, v in d["profs"].items() if v[0]}, "year": d["year"],
                    "abstract": trim(d["abstract"]), "problem": d["sub_problem"], "method": d["method"],
                    "datasets": d["datasets"], "baselines": d["baselines"], "metrics": d["metrics"],
                    "has_code": d["has_code"]})
    return ctx


def synthesize(eng, question: str, res, dets: dict) -> tuple[str, list[dict]]:
    ctx = build_context(res, dets)
    if not ctx:
        return "", ctx
    cfg = load_settings().rag
    plan = {k: v for k, v in res.R.plan.model_dump().items() if v not in (None, [], "", "any")}
    user = f"Question: {question}\nPlan: {json.dumps(plan)}\nPapers:\n{json.dumps(ctx, ensure_ascii=False)}"
    opts = {"temperature": cfg["temperature"], "top_p": cfg["top_p"], "repeat_penalty": cfg["repeat_penalty"],
            "num_ctx": cfg["num_ctx"], "num_predict": cfg["num_predict"], "seed": 42}
    text = eng.client.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], options=opts)
    m = getattr(eng.client, "last", {}) or {}
    used = (m.get("prompt_eval_count") or 0) + (m.get("eval_count") or 0)
    warn = []
    if m.get("done_reason") == "length":
        warn.append(f"answer hit the {cfg['num_predict']}-token output limit and may be cut off (raise rag.num_predict)")
    if used > 0.9 * cfg["num_ctx"]:
        warn.append(f"prompt+answer used {used} of {cfg['num_ctx']} tokens; the start of the prompt may have been dropped "
                    "(raise rag.num_ctx or lower rag.context_papers)")
    elif used > 0.75 * cfg["num_ctx"]:
        warn.append(f"context window is {100 * used // cfg['num_ctx']}% full ({used}/{cfg['num_ctx']} tokens)")
    res.extra["answer_warnings"] = warn
    res.extra["answer_tokens"] = used
    return re.sub(r"\n{3,}", "\n\n", text.strip()), ctx
