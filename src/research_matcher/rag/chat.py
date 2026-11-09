"""Orchestration of one question (plan -> resolve -> execute -> render -> answer -> validate -> log) and the `chat` REPL."""
from __future__ import annotations

import json
import time
import webbrowser
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from rich.console import Console
from rich.markdown import Markdown

from ..config import ROOT, load_settings
from ..similarity.signals import SIGNALS
from .answer import synthesize
from .execute import Engine, Result, execute, load_details
from .query_plan import make_plan
from .render import SIGNAL_NAMES, render
from .resolve import resolve
from .validate import validate

ANSWER_INTENTS = {"topic_across_unis", "similar_to_paper", "compare_unis", "find_professors", "similar_professors"}


@dataclass
class Session:
    last_plan: dict | None = None
    last: Result | None = None
    last_ids: list[str] = field(default_factory=list)
    last_question: str = ""
    last_answer: str = ""
    overrides: dict = field(default_factory=dict)


@dataclass
class Outcome:
    result: Result
    dets: dict
    answer: str = ""
    validation: dict = field(default_factory=dict)
    ctx: list[dict] = field(default_factory=list)
    plan_s: float = 0.0
    exec_s: float = 0.0
    answer_s: float = 0.0


def run_question(eng: Engine, question: str, sess: Session | None = None, *, overrides: dict | None = None, answer: bool = True,
                 console: Console | None = None, show: bool = True) -> Outcome:
    sess = sess or Session()
    ov = {**sess.overrides, **(overrides or {})}
    t0 = time.time()
    plan, warn = make_plan(eng.client, question, sess.last_plan)
    t1 = time.time()
    R = resolve(eng.conn, eng.corpus, plan, question, last_ids=sess.last_ids, overrides=ov)
    if warn:
        R.warnings.insert(0, warn)
    res = execute(eng, R, question)
    t2 = time.time()
    dets = load_details(eng.conn, [it["work_id"] for it in res.items])
    out = Outcome(res, dets, plan_s=t1 - t0, exec_s=t2 - t1)
    console = console or Console()
    if show:
        render(res, dets, eng, console)
    if answer and res.items and res.intent in ANSWER_INTENTS and eng.client is not None:
        t3 = time.time()
        text, ctx = synthesize(eng, question, res, dets)
        text, rep = validate(eng.conn, text, ctx, question, R.plan.semantic_text)
        out.answer, out.validation, out.ctx, out.answer_s = text, rep, ctx, time.time() - t3
        if show:
            for w in res.extra.get("answer_warnings", []):
                console.print(f"[yellow]⚠ {w}[/]")
            console.print("\n[bold]Synthesis[/]")
            console.print(Markdown(text))
            bad = rep["invalid_ids"] or rep["unsupported_terms"]
            console.print(f"[dim]validator: {rep['sentences']} sentences, invalid ids {rep['invalid_ids'] or 'none'}, "
                          f"unsupported terms {rep['unsupported_terms'] or 'none'}, uncited {rep['uncited_sentences']}"
                          f"{'  ⚠ marked above' if bad else ''}[/]")
    if show:
        console.print(f"[dim]plan {out.plan_s:.1f}s · retrieval {out.exec_s:.1f}s" + (f" · answer {out.answer_s:.1f}s" if out.answer_s else "") + "[/]")
    eng.conn.execute("INSERT INTO query_log VALUES (?,?,?,?,?,?)",
                     (time.time(), question, json.dumps(R.plan.model_dump()), json.dumps([it["work_id"] for it in res.items]),
                      out.answer, json.dumps(out.validation)))
    sess.last_plan, sess.last, sess.last_question, sess.last_answer = R.plan.model_dump(), res, question, out.answer
    if res.items:
        sess.last_ids = [it["work_id"] for it in res.items]
    return out


# ------------------------------------------------------------------ REPL
HELP = ("[bold]chat commands[/]: :plan  :why P3  :open P3  :filters  :mode  :save  :reset  :help  :quit\n"
        "Follow-ups refine the last query: 'only with code', 'drop UCL', 'more like P2', 'compare P1 and P4', 'now 2026 only'.")


def _pref(arg: str, sess: Session) -> int | None:
    a = arg.strip().upper().lstrip("P")
    if a.isdigit() and sess.last and 1 <= int(a) <= len(sess.last.items):
        return int(a) - 1
    return None


def command(eng: Engine, line: str, sess: Session, console: Console) -> bool:
    """Handles ':' commands; returns False to quit."""
    cmd, _, arg = line[1:].partition(" ")
    cmd = cmd.lower()
    if cmd in ("q", "quit", "exit"):
        return False
    if cmd == "help":
        console.print(HELP)
    elif cmd == "mode":
        from ..similarity.diversify import MODES

        if arg.strip() in MODES:
            sess.overrides["mode"] = arg.strip()
            console.print(f"selection mode = {arg.strip()} (applies to the next question)")
        else:
            cur = sess.overrides.get('mode', load_settings().rag.get('select_mode', 'diverse'))
            console.print(f"usage: :mode " + " | ".join(MODES) + f"   current: {cur}")
    elif cmd == "reset":
        sess.last_plan, sess.last, sess.last_ids, sess.overrides = None, None, [], {}
        console.print("session reset")
    elif cmd == "plan":
        console.print_json(json.dumps(sess.last_plan, indent=1)) if sess.last_plan else console.print("no query yet")
    elif cmd == "filters":
        r = sess.last.R if sess.last else None
        if not r:
            console.print("no query yet")
        else:
            console.print({"subdomain": r.subdomain, "include": sorted(r.include), "exclude": sorted(r.exclude), "countries": sorted(r.countries),
                           "year": [r.plan.year_from, r.plan.year_to], "types": r.plan.types, "has_code": r.plan.has_code,
                           "mode": r.mode, "min_universities": r.plan.min_universities, "per_university_limit": r.plan.per_university_limit, "k": r.plan.k})
    elif cmd == "why":
        n = _pref(arg, sess)
        if n is None:
            console.print("usage: :why P3")
        else:
            it = sess.last.items[n]
            console.print(f"[bold]P{n + 1}[/] score {it['score']:.3f}")
            for s, v in zip(SIGNALS, it["signals"]):
                w = eng.weights.get(s, 0.0)
                console.print(f"  {SIGNAL_NAMES.get(s, s):16} {'  n/a' if v != v else f'{v:5.2f}'}   weight {w:.2f}" + (f"   contribution {w * v:.3f}" if v == v else ""))
            if it.get("shared"):
                console.print(f"  shared: {it['shared']}")
    elif cmd == "open":
        n = _pref(arg, sess)
        if n is None:
            console.print("usage: :open P3")
        else:
            d = load_details(eng.conn, [sess.last.items[n]["work_id"]])[sess.last.items[n]["work_id"]]
            links = [x for x in (d["doi"], d["arxiv"], d["pdf"], *d["github"]) if x]
            for x in links:
                console.print(x)
            if links:
                webbrowser.open(links[0])
    elif cmd == "save":
        if not sess.last:
            console.print("nothing to save")
        else:
            d = load_details(eng.conn, [it["work_id"] for it in sess.last.items])
            p = ROOT / "notes" / f"{date.today().isoformat()}.md"
            p.parent.mkdir(exist_ok=True)
            lines = [f"## {sess.last_question}", f"_plan_: `{json.dumps(sess.last_plan)}`", ""]
            for n, it in enumerate(sess.last.items, 1):
                x = d[it["work_id"]]
                lines.append(f"- P{n} **{x['title']}** — {', '.join(x['unis'])} ({x['year']}) {x['doi'] or x['arxiv'] or ''}")
            if sess.last_answer:
                lines += ["", sess.last_answer]
            with open(p, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n\n")
            console.print(f"saved to {p}")
    else:
        console.print(f"unknown command :{cmd}  (:help)")
    return True


def repl(eng: Engine, console: Console | None = None, answer: bool = True) -> None:
    console = console or Console()
    sess = Session()
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory

        hist = load_settings().path("data_dir") / "chat_history.txt"
        ps = PromptSession(history=FileHistory(str(hist)))
        ask = lambda: ps.prompt("you> ")  # noqa: E731
    except Exception:  # noqa: BLE001
        ask = lambda: input("you> ")  # noqa: E731
    console.print(HELP)
    while True:
        try:
            line = ask().strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line.startswith(":"):
            if not command(eng, line, sess, console):
                break
            continue
        try:
            run_question(eng, line, sess, answer=answer, console=console)
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]error: {type(e).__name__}: {e}[/]")
