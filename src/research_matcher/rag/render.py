"""Step 3a: deterministic output (plan §9.4.1-2). Every value is read from SQLite through execute.load_details."""
from __future__ import annotations

from collections import defaultdict

from rich.console import Console
from rich.table import Table

from ..similarity.scorer import benchmark_eligible
from ..similarity.signals import SIGNALS

SIGNAL_NAMES = {"dense_abstract": "meaning", "dense_composite": "problem/method", "title_dense": "title", "title_lexical": "title words",
                "sparse_lexical": "rare terms", "micro_topic": "topic", "openalex_topics": "OpenAlex topics", "keywords": "keywords",
                "datasets": "datasets", "baselines": "baselines", "metrics": "metrics", "biblio_coupling": "shared refs",
                "citation_link": "citation link", "specter2": "SPECTER2"}


def why_line(item: dict, det: dict, eng=None) -> str:
    parts = [f"{SIGNAL_NAMES.get(s, s)} {v:.2f}" if s != "professor profile" else f"{s} {v:.2f}" for s, v in item["why"]]
    sh = item.get("shared") or {}
    ent = [e for k in ("datasets", "baselines") for e in sh.get(k, [])]
    if ent:
        parts.append("shared: " + ", ".join(ent[:4]))
    if item.get("eligible"):
        parts.append("benchmark-comparable")
    return " · ".join(parts)


def prof_cell(det: dict, uni: str | None = None) -> str:
    out = []
    for u in ([uni] if uni else []) + [x for x in det["profs"] if x != uni]:
        if u in det["profs"] and det["profs"][u][0]:
            n, ver = det["profs"][u]
            out.append(f"{n}{'✔' if ver else '?'}" + (f" ({u})" if len(det["profs"]) > 1 else ""))
    return "\n".join(out) or "—"


def link_cell(det: dict) -> str:
    links = [x for x in (det["doi"], det["arxiv"], det["pdf"]) if x]
    gh = det["github"][:2]
    return "\n".join(dict.fromkeys(links[:2] + gh)) or (det["openalex"] or "—")


def summary(items: list[dict], dets: dict, eng) -> dict:
    """Deterministic cross-university summary block."""
    unis = sorted({u for it in items for u in dets[it["work_id"]]["unis"]})
    by_ent: dict[tuple, set] = defaultdict(set)
    for it in items:
        d = dets[it["work_id"]]
        for k in ("datasets", "baselines", "metrics"):
            for e in d[k]:
                by_ent[(k, e)] |= set(d["unis"])
    shared = {k: sorted([(e, sorted(u)) for (kk, e), u in by_ent.items() if kk == k and len(u) >= 2], key=lambda x: -len(x[1]))[:6]
              for k in ("datasets", "baselines", "metrics")}
    pairs = []
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            ja, jb = items[a]["j"], items[b]["j"]
            if eng.corpus.unis[ja] & eng.corpus.unis[jb]:
                continue
            sh = eng.corpus.shared(ja, jb)
            if benchmark_eligible(sh):
                pairs.append((a + 1, b + 1, sh))
    return {"universities": unis, "shared": shared, "eligible_pairs": pairs[:8]}


def render_papers(items: list[dict], dets: dict, eng, console: Console, title: str | None = None) -> None:
    groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for n, it in enumerate(items, 1):
        us = dets[it["work_id"]]["unis"] or ["?"]
        groups[us[0]].append((n, it))
    order = sorted(groups, key=lambda u: -max(it["score"] for _, it in groups[u]))
    t = Table(title=title, show_lines=True, expand=True)
    for c, w in (("P#", 3), ("Professor", 16), ("Title", 38), ("Year · Venue", 14), ("Subdomain › topic", 20),
                 ("Datasets / Baselines", 18), ("Links", 28), ("Why matched", 28)):
        t.add_column(c, width=w, overflow="fold", no_wrap=False)
    for u in order:
        t.add_row(f"[bold cyan]{u}[/]", "", f"[bold cyan]{len(groups[u])} paper(s)[/]", "", "", "", "", "")
        for n, it in groups[u]:
            d = dets[it["work_id"]]
            ents = "; ".join(x for x in (", ".join(d["datasets"][:3]), ", ".join(d["baselines"][:3])) if x) or "—"
            venue = (d["type"] or "")[:4] + (f" · {d['venue']}" if d["venue"] else "")
            t.add_row(f"P{n}", prof_cell(d, u), d["title"], f"{d['year']} · {venue}",
                      f"{d['subdomain']}" + (f" › {d['topic']}" if d["topic"] else ""), ents, link_cell(d), why_line(it, d))
    console.print(t)


def render_summary(items, dets, eng, console: Console) -> None:
    s = summary(items, dets, eng)
    console.print(f"[bold]Universities covered ({len(s['universities'])}):[/] {', '.join(s['universities']) or '—'}")
    for k, v in s["shared"].items():
        if v:
            console.print(f"  shared {k}: " + "; ".join(f"{e} ({', '.join(u)})" for e, u in v))
    if s["eligible_pairs"]:
        console.print("  benchmark-comparable pairs: " + "; ".join(
            f"P{a}–P{b} ({', '.join(x for k in ('datasets', 'baselines', 'metrics') for x in sh.get(k, [])[:2])})"
            for a, b, sh in s["eligible_pairs"]))


def render(res, dets: dict, eng, console: Console) -> None:
    p, R = res.R.plan, res.R
    flt = []
    if R.subdomain:
        flt.append(f"subdomain={R.subdomain}")
    if R.include:
        flt.append("only " + "/".join(sorted(R.include)))
    if R.exclude:
        flt.append("not " + "/".join(sorted(R.exclude)))
    if R.countries:
        flt.append("countries " + "/".join(sorted(R.countries)))
    if p.year_from or p.year_to:
        flt.append(f"year {p.year_from or ''}–{p.year_to or ''}")
    if p.has_code:
        flt.append("with code")
    if p.types:
        flt.append("/".join(p.types))
    flt.append(f"mode={R.mode}")
    console.print(f"[dim]intent={p.intent}  query={p.semantic_text or p.paper_ref or p.professor or ''!r}  "
                  f"{'  '.join(flt)}[/]")
    for w in R.warnings:
        console.print(f"[yellow]⚠ {w}[/]")
    ex = res.extra
    if res.intent == "entity_stats":
        console.print(f"[bold]Entities across {ex['n_papers']} extracted papers[/]")
        for kind, rows in ex["stats"].items():
            t = Table(title=kind, show_lines=False)
            t.add_column("name"); t.add_column("papers", justify="right"); t.add_column("universities")
            for e, n, us in rows:
                t.add_row(e, str(n), ", ".join(us))
            console.print(t)
    elif res.intent == "cross_uni_topics":
        t = Table(title=f"Micro-topics spanning ≥{p.min_universities} universities", show_lines=True, expand=True)
        for c in ("ID", "Subdomain", "Topic", "Papers", "Universities", "Central papers"):
            t.add_column(c, overflow="fold")
        for r in ex["topics"]:
            t.add_row(str(r["topic_id"]), r["subdomain"], f"{r['label']}\n[dim]{r['description'] or ''}[/]", str(r["size"]),
                      f"{len(r['unis'])}: {', '.join(r['unis'])}", "\n".join(r.get("titles", [])))
        console.print(t)
    else:
        if "professors" in ex:
            if "query_professor" in ex:
                q = ex["query_professor"]
                console.print(f"[bold]Professors similar to {q['name']} ({q['uni']}, {q['n']} papers)[/]")
            t = Table(show_lines=False)
            for c in ("Professor", "Uni", "Score", "Evidence"):
                t.add_column(c)
            for e in ex["professors"]:
                ev = (f"profile cosine {e['cos']:.2f}, subdomain overlap {e['overlap']:.2f}, {e['shared_topics']} shared topic(s), {e['n']} papers"
                      if "cos" in e else f"{e['n']} matching paper(s) " + ",".join(f"P{r}" for r in e.get("refs", [])))
                t.add_row(e["name"] + ("✔" if e.get("verified") else ""), e["uni"], f"{e['score']:.2f}", ev + (f" → P{e['ref']}" if "ref" in e else ""))
            console.print(t)
        if res.items:
            render_papers(res.items, dets, eng, console)
            if res.intent in ("topic_across_unis", "similar_to_paper", "compare_unis", "find_professors"):
                render_summary(res.items, dets, eng, console)
        if "shared_table" in ex:
            for k, rows in ex["shared_table"].items():
                if rows:
                    console.print(f"  shared {k} across the compared universities: " + "; ".join(
                        f"{e} ({', '.join(f'{u}:' + ','.join(f'P{x}' for x in ps) for u, ps in d.items())})" for e, d in rows))
    for n in res.notes:
        console.print(f"[yellow]ℹ {n}[/]")
