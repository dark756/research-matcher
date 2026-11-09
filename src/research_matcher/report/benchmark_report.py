"""`report`: BENCHMARK_CANDIDATES.md for the top cross-university clusters (plan §10). Tables and claims are read from
SQLite verbatim; only the 'proposed benchmark' and '3050 stress-test plan' paragraphs are drafted by gemma3 (labelled as such,
cited with [P#] and checked by the same validator as `ask`)."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import date

from ..config import ROOT
from ..rag.execute import load_details
from ..rag.validate import validate
from ..similarity.scorer import benchmark_eligible

SYSTEM = """You help plan a small benchmark comparing papers from different universities. You get a cluster of related papers as JSON with ids like "P2".
Write two short sections in Markdown:
### Proposed benchmark
3-5 bullets: the common task, the datasets/baselines/metrics that several papers share (only names present in the JSON), and what a fair comparison needs. Cite [P#] (square brackets, ids provided only) after every claim.
### Stress-test plan for an RTX 3050 Laptop (6 GB VRAM)
3-5 bullets: which papers' methods look feasible to rerun on 6 GB (use the 'hardware' and 'method' fields; say "not in the data" if unknown), what to shrink (dataset size, batch, model size), and which numbers to compare. Cite [P#].
Use only the JSON. No outside knowledge about results."""


def rank_clusters(conn, corpus, top: int) -> list[dict]:
    out = []
    for r in conn.execute("SELECT cluster_id, subdomain, members_json, universities_json, score FROM clusters"):
        members = json.loads(r[2])
        idx = [corpus.pos[m] for m in members if m in corpus.pos]
        pairs = elig = 0
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                if corpus.unis[idx[a]] & corpus.unis[idx[b]]:
                    continue
                pairs += 1
                elig += benchmark_eligible(corpus.shared(idx[a], idx[b]))
        out.append({"id": r[0], "subdomain": r[1], "members": members, "unis": json.loads(r[3]), "score": r[4],
                    "pairs": pairs, "eligible": elig})
    out.sort(key=lambda c: (-c["eligible"], -len(c["unis"]), -c["score"]))
    return out[:top]


def claims(conn, members: list[str], limit: int = 8) -> list[tuple]:
    q = ",".join("?" * len(members))
    rows = conn.execute(f"SELECT work_id, metric, value, compared_to, dataset FROM reported_results WHERE work_id IN ({q})", members).fetchall()
    rows.sort(key=lambda r: -((r[3] is not None) + (r[4] is not None)))
    return rows[:limit]


def hardware(conn, members: list[str]) -> dict[str, list[str]]:
    q = ",".join("?" * len(members))
    out: dict[str, list[str]] = defaultdict(list)
    for w, e in conn.execute(f"SELECT work_id, normalized FROM entities WHERE kind='hardware' AND work_id IN ({q})", members):
        out[w].append(e)
    return out


def cluster_section(eng, c: dict, n: int, draft: bool) -> tuple[str, dict]:
    conn, corpus = eng.conn, eng.corpus
    members = sorted(c["members"], key=lambda m: -corpus.meta[corpus.pos[m]]["year"] if corpus.meta[corpus.pos[m]]["year"] else 0)
    dets = load_details(conn, members)
    hw = hardware(conn, members)
    labels = Counter(d["topic"] for d in dets.values() if d["topic"])
    label = labels.most_common(1)[0][0] if labels else c["subdomain"]
    ref = {m: f"P{i}" for i, m in enumerate(members, 1)}
    lines = [f"## {n}. {label} (cluster {c['id']})", "",
             f"*Cluster {c['id']} · {c['subdomain']} · {len(members)} papers · {len(c['unis'])} universities ({', '.join(c['unis'])}) · "
             f"mean link score {c['score']:.2f} · {c['eligible']}/{c['pairs']} cross-university pairs are benchmark-comparable*", "",
             "| Ref | University | Professor | Title | Year | Datasets | Baselines | Links |", "|---|---|---|---|---|---|---|---|"]
    for m in members:
        d = dets[m]
        prof = "; ".join(f"{v[0]}{'✔' if v[1] else '?'} ({u})" for u, v in d["profs"].items() if v[0]) or "—"
        links = " ".join(f"[{k}]({v})" for k, v in (("doi", d["doi"]), ("arXiv", d["arxiv"]), ("code", (d["github"] or [None])[0])) if v)
        lines.append(f"| {ref[m]} | {'/'.join(d['unis'])} | {prof} | {d['title']} | {d['year']} | {', '.join(d['datasets'][:4]) or '—'} | "
                     f"{', '.join(d['baselines'][:4]) or '—'} | {links or '—'} |")
    by_ent: dict[tuple, set] = defaultdict(set)
    for m in members:
        for k in ("datasets", "baselines", "metrics"):
            for e in dets[m][k]:
                by_ent[(k, e)] |= {ref[m]}
    lines += ["", "**Shared across papers**"]
    any_shared = False
    for k in ("datasets", "baselines", "metrics"):
        sh = sorted(((e, p) for (kk, e), p in by_ent.items() if kk == k and len(p) >= 2), key=lambda x: -len(x[1]))[:6]
        if sh:
            any_shared = True
            lines.append(f"- {k}: " + "; ".join(f"{e} ({', '.join(sorted(p, key=lambda x: int(x[1:])))})" for e, p in sh))
    if not any_shared:
        lines.append("- no dataset, baseline or metric is named by two or more papers (extracted from abstracts only)")
    cl = claims(conn, members)
    lines += ["", "**Claims to verify** (reported in the abstracts, taken verbatim from the extraction; unverified)"]
    lines += [f"- {ref[w]}: {m} = {v}" + (f" vs {cmp}" if cmp else "") + (f" on {ds}" if ds else "") for w, m, v, cmp, ds in cl] or ["- none extracted"]
    report = {}
    if draft:
        ctx = [{"id": ref[m], "title": dets[m]["title"], "universities": dets[m]["unis"], "year": dets[m]["year"],
                "problem": dets[m]["sub_problem"], "method": dets[m]["method"], "datasets": dets[m]["datasets"],
                "baselines": dets[m]["baselines"], "metrics": dets[m]["metrics"], "hardware": hw.get(m, []),
                "has_code": dets[m]["has_code"]} for m in members]
        text = eng.client.chat([{"role": "system", "content": SYSTEM},
                                {"role": "user", "content": json.dumps(ctx, ensure_ascii=False)}],
                               options={"temperature": 0.2, "top_p": 0.9, "repeat_penalty": 1.1, "num_ctx": 8192, "num_predict": 700, "seed": 42})
        ctx_v = [{**p, "abstract": "", "professors": {}} for p in ctx]
        text, report = validate(conn, text, ctx_v, "", "")
        text = re.sub(r"\s*\n\s*(\[P\d+(?:[,;&\s]+P\d+)*\])", r" \1", text)   # keep citations on their sentence's line
        lines += ["", "> Drafted by gemma3 from the fields above; sentences marked ⚠ failed the citation / name check.", "", text]
    return "\n".join(lines) + "\n", report


def build_report(eng, *, top: int = 10, draft: bool = True, out_path=None, log=print) -> dict:
    clusters = rank_clusters(eng.conn, eng.corpus, top)
    parts = [f"# Benchmark candidates", "",
             f"Generated {date.today().isoformat()} from {len(clusters)} cross-university clusters (clusters = papers linked at score ≥ "
             f"{eng.sim.get('tau_strong')}; ranked by number of benchmark-comparable cross-university pairs, then universities, then score). "
             "Papers, professors, links, datasets, baselines and claims come straight from the database; claims are parsed from abstracts and "
             "must be checked against the papers before use. ✔ = professor verified against CSRankings, ? = not verified.", ""]
    bad = 0
    for n, c in enumerate(clusters, 1):
        log(f"cluster {n}/{len(clusters)} ({len(c['members'])} papers)")
        sec, rep = cluster_section(eng, c, n, draft)
        bad += len(rep.get("invalid_ids", [])) + len(rep.get("unsupported_terms", []))
        parts.append(sec)
    path = out_path or ROOT / "BENCHMARK_CANDIDATES.md"
    path.write_text("\n".join(parts), encoding="utf-8")
    return {"path": str(path), "clusters": len(clusters), "flagged_terms_or_ids": bad}
