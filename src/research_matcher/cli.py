"""research-matcher CLI. M0 implements `init` and `status`; later stages are stubs until their milestone."""
from __future__ import annotations

import json
import sys

import typer
from rich.console import Console

for _s in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252 and crash on ✅ / ≈
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")
from rich.table import Table

app = typer.Typer(help="Cross-university CS paper store + local RAG engine", no_args_is_help=True)
console = Console()
OK, BAD = "[green]✅[/green]", "[red]❌[/red]"


def _check_cuda() -> tuple[bool, str]:
    try:
        import torch

        if not torch.cuda.is_available():
            return False, f"torch {torch.__version__}: CUDA not available"
        return True, f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def _check_bge(download: bool) -> tuple[bool, str]:
    """Ensure bge-m3 weights are in the HF cache (downloads ~2.3 GB on first use)."""
    try:
        from huggingface_hub import snapshot_download

        from .config import load_settings

        repo = load_settings().models["embed"]
        if not download:
            try:
                path = snapshot_download(repo, local_files_only=True)
                return True, f"cached at {path}"
            except Exception:  # noqa: BLE001
                return False, "not downloaded (run `init` without --skip-bge)"
        path = snapshot_download(
            repo, allow_patterns=["*.json", "*.model", "*.txt", "pytorch_model.bin", "sentencepiece.bpe.model",
                                  "1_Pooling/*", "colbert_linear.pt", "sparse_linear.pt"],
        )
        return True, f"cached at {path}"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


@app.command()
def init(skip_bge: bool = typer.Option(False, help="Don't download bge-m3 weights"),
         skip_csrankings: bool = typer.Option(False, help="Don't download CSRankings")) -> None:
    """Verify environment, create the schema, load CSRankings (M0 gate)."""
    from .config import load_secrets, load_settings, load_universities
    from .db import connect, init_schema, seed_institutions
    from .http_client import HttpClient
    from .ollama_client import OllamaClient

    settings, secrets = load_settings(), load_secrets()
    conn = connect()
    init_schema(conn)
    n = seed_institutions(conn)
    console.print(f"{OK} schema ready at {settings.path('db')} ({n} institutions seeded)")

    # Secrets present (never print values)
    for name, val in [("OPENALEX_API_KEY", secrets.openalex_api_key), ("OPENALEX_EMAIL", secrets.openalex_email),
                      ("S2_API_KEY", secrets.s2_api_key)]:
        console.print(f"{OK if val else BAD} .env.local {name} {'set' if val else 'MISSING'}")

    # 17 ROR -> OpenAlex IDs
    http = HttpClient(conn)
    unis = load_universities()
    ids = "|".join(u.openalex_id for u in unis)
    try:
        res = http.openalex("/institutions", {"filter": f"openalex_id:{ids}", "per_page": 50,
                                              "select": "id,ror,display_name"})
        found = {r["id"].rsplit("/", 1)[-1]: r["ror"].rsplit("/", 1)[-1] for r in res["results"]}
        bad = [u.short for u in unis if found.get(u.openalex_id) != u.ror]
        console.print(f"{OK if not bad else BAD} ROR→OpenAlex: {len(unis) - len(bad)}/{len(unis)} verified"
                      + (f" (mismatch: {', '.join(bad)})" if bad else ""))
    except Exception as e:  # noqa: BLE001
        console.print(f"{BAD} OpenAlex institution check failed: {type(e).__name__}: {e}")
    rem = http.credits_remaining()
    console.print(f"{OK if rem is not None else BAD} OpenAlex credits remaining: {rem if rem is not None else 'unknown'}")

    # Ollama
    h = OllamaClient().health()
    caps = ", ".join(h.get("capabilities", []))
    console.print(f"{OK if h.get('ok') else BAD} Ollama {settings.models['llm']}"
                  + (f" (capabilities: {caps})" if caps else f" {h.get('error', '')}"))

    # CUDA
    ok, msg = _check_cuda()
    console.print(f"{OK if ok else BAD} CUDA: {msg}")

    # bge-m3
    ok, msg = _check_bge(download=not skip_bge)
    console.print(f"{OK if ok else BAD} bge-m3: {msg}")

    # CSRankings
    if not skip_csrankings:
        from .harvest.csrankings import load_faculty

        try:
            counts = load_faculty(conn)
            total = sum(counts.values())
            missing = [k for k, v in counts.items() if v == 0]
            console.print(f"{OK if not missing else BAD} CSRankings: {total} faculty across "
                          f"{len(counts) - len(missing)}/{len(counts)} universities")
        except Exception as e:  # noqa: BLE001
            console.print(f"{BAD} CSRankings failed: {type(e).__name__}: {e}")
    http.close()


@app.command()
def harvest(
    skip_b: bool = typer.Option(False, "--skip-b", help="Skip Query B (proceedings journals)"),
    skip_c: bool = typer.Option(False, "--skip-c", help="Skip Query C (keyword supplement)"),
    skip_authors: bool = typer.Option(False, "--skip-authors", help="Skip author stats batches"),
    max_pages: int = typer.Option(0, help="Limit pages per query (0 = all; use for a dry run)"),
    refresh: bool = typer.Option(False, help="Re-parse and overwrite works already stored"),
) -> None:
    """(M1) Harvest OpenAlex Queries A + B + C, authorships, work_universities, author stats, FTS."""
    from .db import connect, init_schema, seed_institutions
    from .harvest.authors import fetch_authors
    from .harvest.openalex import run_harvest
    from .http_client import HttpClient

    conn = connect()
    init_schema(conn)
    seed_institutions(conn)
    http = HttpClient(conn)
    try:
        res = run_harvest(http, conn, do_b=not skip_b, do_c=not skip_c, max_pages=max_pages or None,
                          refresh=refresh, log=console.print)
        t = Table("query", "seen", "stored", "already had", "no target uni", "excluded")
        for k, v in res.items():
            if k != "_meta":
                t.add_row(k, str(v["seen"]), str(v["stored"]), str(v["skipped_existing"]),
                          str(v["no_target_uni"]), str(v["excluded_subfield"]))
        console.print(t)
        if not skip_authors:
            fetch_authors(http, conn, log=console.print)
        console.print(f"OpenAlex credits used by queries: {res['_meta']['credits_used']}; "
                      f"remaining: {http.credits_remaining()}")
        harvest_check(conn)
    finally:
        http.close()


@app.command("harvest-check")
def harvest_check_cmd() -> None:
    """(M1 gate) Summarise stored works against the plan's expectations."""
    from .db import connect

    harvest_check(connect())


def harvest_check(conn) -> None:
    def q(sql: str) -> int:
        return conn.execute(sql).fetchone()[0]

    total = q("SELECT COUNT(*) FROM works")
    t = Table("check", "value", "expected")
    t.add_row("works total", str(total), "≈ 36,435 + 643 + ~206 (±1%, minus overlaps)")
    for src, exp in [("A", "≈ 36,435"), ("B", "≈ 643"), ("C", "≈ 206")]:
        t.add_row(f"  source {src}", str(q(f"SELECT COUNT(*) FROM works WHERE harvest_source='{src}'")), exp)
    no_uni = q("SELECT COUNT(*) FROM works WHERE work_id NOT IN (SELECT work_id FROM work_universities)")
    t.add_row("works with no target uni", str(no_uni), "0")
    with_abs = q("SELECT COUNT(*) FROM works WHERE abstract IS NOT NULL")
    t.add_row("works with abstract", f"{with_abs} ({100 * with_abs / max(total, 1):.1f}%)", "≈ 88.7%")
    t.add_row("fts rows", str(q("SELECT COUNT(*) FROM works_fts")), str(total))
    t.add_row("authors / with stats", f"{q('SELECT COUNT(*) FROM authors')} / "
              f"{q('SELECT COUNT(*) FROM authors WHERE h_index IS NOT NULL')}", "senior+lead filled")
    pre = q("SELECT COUNT(*) FROM works WHERE type='preprint'")
    pre_ax = q("SELECT COUNT(*) FROM works WHERE type='preprint' AND arxiv_id IS NOT NULL")
    t.add_row("preprints with arXiv id", f"{pre_ax} / {pre}", "M2 gate ≥ 90%")
    console.print(t)


@app.command()
def enrich(
    skip_s2: bool = typer.Option(False, "--skip-s2", help="Skip Semantic Scholar pass 1"),
    max_batches: int = typer.Option(0, help="Limit S2 batches (0 = all)"),
    spot_check: int = typer.Option(30, help="Print N random merged duplicate groups for manual review"),
) -> None:
    """(M2) S2 pass 1 → dedupe → venue → CSRankings verification → abstract GitHub links."""
    from .db import connect, init_schema
    from .enrich.code_links import run_code_links
    from .enrich.dedupe import run_dedupe, sample_groups
    from .enrich.professors import run_professors
    from .enrich.semantic_scholar import run_pass1
    from .enrich.venue_hints import run_venue_hints
    from .http_client import HttpClient

    conn = connect()
    init_schema(conn)
    http = HttpClient(conn)
    try:
        if not skip_s2:
            console.print(f"S2 pass 1: {run_pass1(http, conn, max_batches=max_batches or None, log=console.print)}")
    finally:
        http.close()
    console.print(f"venue: {run_venue_hints(conn)}")
    console.print(f"dedupe: {run_dedupe(conn)}")
    console.print(f"professors: {run_professors(conn)}")
    console.print(f"code links: {run_code_links(conn)}")
    enrich_check(conn)
    if spot_check:
        console.print(f"\n[bold]Dedupe spot-check ({spot_check} random merged groups)[/bold]")
        for g in sample_groups(conn, spot_check):
            console.print(" • " + "\n   ≡ ".join(g["titles"]))


@app.command("enrich-check")
def enrich_check_cmd() -> None:
    """(M2 gate) Summarise enrichment quality."""
    from .db import connect

    enrich_check(connect())


def enrich_check(conn) -> None:
    def q(sql: str) -> int:
        return conn.execute(sql).fetchone()[0]

    def pct(a: int, b: int) -> str:
        return f"{a} / {b} ({100 * a / max(b, 1):.1f}%)"

    t = Table("check", "value", "gate")
    queried = q("SELECT COUNT(*) FROM works WHERE s2_status IN ('ok','not_found','failed')")
    t.add_row("S2 match rate", pct(q("SELECT COUNT(*) FROM works WHERE s2_status='ok'"), queried), "expect ≥ 80%")
    t.add_row("S2 failed (rerun enrich)", str(q("SELECT COUNT(*) FROM works WHERE s2_status='failed'")), "0")
    t.add_row("preprints with arXiv id", pct(q("SELECT COUNT(*) FROM works WHERE type='preprint' AND arxiv_id IS NOT NULL"),
                                             q("SELECT COUNT(*) FROM works WHERE type='preprint'")), "≥ 90%")
    t.add_row("senior authors CSRankings-verified",
              pct(q("SELECT COUNT(*) FROM work_universities WHERE professor_verified=1"),
                  q("SELECT COUNT(*) FROM work_universities")), "≥ 60%")
    t.add_row("canonical works / all", pct(q("SELECT COUNT(*) FROM works WHERE canonical_id=work_id"),
                                          q("SELECT COUNT(*) FROM works")), "")
    t.add_row("abstract still missing", str(q("SELECT COUNT(*) FROM works WHERE abstract IS NULL")), "flagged")
    t.add_row("arXiv-only works with real S2 venue",
              str(q("SELECT COUNT(*) FROM works WHERE venue_type='repository' AND venue_s2 IS NOT NULL")), "")
    t.add_row("papers with GitHub link", str(q("SELECT COUNT(*) FROM works WHERE has_code=1")), "")
    console.print(t)


def _load_scoring_inputs():
    import numpy as np

    from .embed.prefilter import emb_dir, load_embeddings

    ids, abs_d, title_d = load_embeddings()
    z = np.load(emb_dir() / "protos.npz")
    return ids, abs_d, title_d, [str(c) for c in z["clusters"]], z["protos"], z["owner"].tolist()


def _score(conn, tau_override: float | None = None):
    from .config import load_settings
    from .embed.prefilter import meaning_scores, score_all

    cfg = load_settings().prefilter
    ids, abs_d, title_d, clusters, protos, owner = _load_scoring_inputs()
    meaning = meaning_scores(abs_d, protos, owner, len(clusters))
    tau = {c: (tau_override if tau_override is not None else cfg["tau"][c]) for c in clusters}
    return ids, abs_d, title_d, clusters, meaning, score_all(conn, clusters, meaning, ids, tau, cfg["kw_min"])


@app.command()
def prefilter(
    skip_embed: bool = typer.Option(False, "--skip-embed", help="Reuse saved embeddings"),
    skip_index: bool = typer.Option(False, "--skip-index", help="Don't (re)build the Qdrant all_papers collection"),
    tau: float = typer.Option(None, help="Override τ_c for every subdomain (default: config/settings.yaml)"),
) -> None:
    """(M3) bge-m3 embeddings for all canonical papers → keyword + meaning scores → survivors → Qdrant all_papers."""
    import numpy as np

    from .config import load_settings
    from .db import connect, init_schema
    from .embed.prefilter import embed_all, emb_dir, prototype_matrix, write_prefilter

    conn = connect()
    init_schema(conn)
    if not skip_embed:
        from .embed.bge import BGE
        from .ollama_client import OllamaClient

        try:
            OllamaClient().unload()  # GPU stages never overlap (plan §10.3)
        except Exception:  # noqa: BLE001
            pass
        bge = BGE("cuda")
        clusters, protos, owner = prototype_matrix(bge)
        np.savez(emb_dir() / "protos.npz", clusters=np.array(clusters), protos=protos, owner=np.array(owner))
        n = embed_all(conn, bge, log=console.print)
        bge.close()
        console.print(f"{OK} embedded {n} canonical papers")
    ids, abs_d, title_d, clusters, meaning, scores = _score(conn, tau)
    kept = write_prefilter(conn, scores)
    lo, hi = load_settings().prefilter["target_survivors"]
    t = Table("cluster", "kept (best cluster)")
    for c in clusters:
        t.add_row(c, str(sum(1 for s in scores.values() if s["keep"] and s["best"] == c)))
    console.print(t)
    console.print(f"{OK if lo <= kept <= hi else BAD} survivors: {kept} of {len(ids)} (target {lo}–{hi})")
    if not skip_index:
        from .index.vector_store import build_all_papers, open_store

        store = open_store()
        n = build_all_papers(conn, store, ids, abs_d, title_d)
        store.close()
        console.print(f"{OK if n == len(ids) else BAD} all_papers: {n} points (expected {len(ids)})")


@app.command("gold-sample")
def gold_sample(n_per_cluster: int = 30, n_random: int = 45, seed: int = 7) -> None:
    """Write data/gold/prefilter_unlabeled.csv: ~150 papers to hand-label (top hits, borderline, random)."""
    import csv
    import random

    import numpy as np

    from .config import load_settings
    from .db import connect

    conn = connect()
    ids, _, _, clusters, meaning, _ = _score(conn)
    rng = random.Random(seed)
    order = {c: np.argsort(-meaning[:, j]) for j, c in enumerate(clusters)}
    pick: dict[str, str] = {}
    for c in clusters:
        top = [ids[i] for i in order[c][:150]]
        border = [ids[i] for i in order[c][150:800]]
        for w in rng.sample(top, n_per_cluster // 2) + rng.sample(border, n_per_cluster - n_per_cluster // 2):
            pick.setdefault(w, f"{c}-candidate")
    rest = [w for w in ids if w not in pick]
    for w in rng.sample(rest, n_random):
        pick[w] = "random"
    out = load_settings().path("data_dir") / "gold" / "prefilter_unlabeled.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(pick.items())
    rng.shuffle(rows)
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["work_id", "title", "abstract_snippet", "origin", "label"])
        for w, origin in rows:
            r = conn.execute("SELECT title, abstract FROM works WHERE work_id=?", (w,)).fetchone()
            wr.writerow([w, r["title"], (r["abstract"] or "")[:600], origin, ""])
    console.print(f"{OK} wrote {len(rows)} rows to {out}")


@app.command()
def calibrate(gold: str = typer.Option("data/gold/prefilter.csv", help="CSV with work_id,label (cluster id or none)"),
              target_recall: float = 0.90) -> None:
    """(M3) Sweep τ over the gold labels: recall of relevant papers vs. number of survivors."""
    import csv

    import numpy as np

    from .config import ROOT, load_settings
    from .db import connect
    from .embed.prefilter import keep_rule, score_all

    conn = connect()
    ids, _, _, clusters, meaning, _ = _score(conn)
    cfg = load_settings().prefilter
    labels = {r["work_id"]: r["label"].strip() for r in csv.DictReader(open(ROOT / gold, encoding="utf-8")) if r["label"].strip()}
    pos = {w for w, l in labels.items() if l != "none"}
    console.print(f"gold: {len(labels)} labelled, {len(pos)} relevant")
    idx = {w: i for i, w in enumerate(ids)}
    kw_all = score_all(conn, clusters, meaning, ids, {c: 0.0 for c in clusters}, 99)  # keyword scores only
    t = Table("τ", "recall", "survivors", "label→cluster correct")
    best = None
    for tau in np.arange(0.35, 0.76, 0.025):
        kept = {w for w in ids if any(keep_rule(float(meaning[idx[w], j]), kw_all[w]["kw"][c], tau, cfg["kw_min"])
                                       for j, c in enumerate(clusters))}
        rec = len(pos & kept) / max(len(pos), 1)
        n = len(kept)
        t.add_row(f"{tau:.3f}", f"{rec:.2%}", str(n), "")
        if rec >= target_recall:
            best = (float(tau), rec, n)
    console.print(t)
    if best:
        console.print(f"{OK} highest τ with recall ≥ {target_recall:.0%}: τ={best[0]:.3f} (recall {best[1]:.1%}, "
                      f"{best[2]} survivors; target {cfg['target_survivors']})")
    else:
        console.print(f"{BAD} no τ in range reaches recall {target_recall:.0%}")


@app.command()
def extract(limit: int = typer.Option(0, help="Process at most N papers (0 = all)"),
            redo: bool = typer.Option(False, help="Re-extract papers already done with this prompt_version")) -> None:
    """(M4) gemma3 structured extraction on pre-filter survivors. Resumable; safe to Ctrl-C and rerun."""
    from .db import connect, init_schema
    from .extract.extractor import run_extract

    conn = connect()
    init_schema(conn)
    res = run_extract(conn, limit=limit or None, redo=redo, log=console.print)
    console.print(f"{OK if res['failed'] == 0 else BAD} {res}")


@app.command("extract-relabel")
def extract_relabel() -> None:
    """Re-apply the final-label rule (LLM + pre-filter lexicon) to stored extractions; no model calls."""
    from .db import connect
    from .extract.extractor import relabel_all

    console.print(f"{OK} {relabel_all(connect())}")


@app.command("extract-gold-sample")
def extract_gold_sample(n: int = 50, seed: int = 11) -> None:
    """Pick n survivors (stratified by best cluster, abstract present) to hand-label: data/gold/extract_sample.json."""
    import random

    from .config import load_settings
    from .db import connect

    conn = connect()
    rng = random.Random(seed)
    clusters = ["SysML_RL_Caching", "Edge_RAG_Vector_Systems", "Spatio_Temporal_MARL"]
    picked = []
    for i, c in enumerate(clusters):
        ids = [r[0] for r in conn.execute(
            "SELECT p.work_id FROM prefilter p JOIN works w USING(work_id) WHERE p.keep=1 AND p.best_cluster=? "
            "AND w.abstract IS NOT NULL ORDER BY p.work_id", (c,))]
        picked += rng.sample(ids, n // 3 + (1 if i < n % 3 else 0))
    out = load_settings().path("data_dir") / "gold" / "extract_sample.json"
    out.write_text(json.dumps(picked))
    console.print(f"{OK} {len(picked)} work_ids -> {out}")


@app.command("extract-eval")
def extract_eval(gold: str = typer.Option("data/gold/extract.json", help="JSON list of gold records"),
                 show: bool = typer.Option(False, help="Print per-paper errors")) -> None:
    """(M4 gate) Run the extractor on the gold set (nothing stored) and report accuracy / precision / recall."""
    from rapidfuzz import fuzz

    from .config import ROOT
    from .db import connect
    from .extract.extractor import KINDS as ALL_KINDS, extract_one, final_labels
    KINDS = {k: v for k, v in ALL_KINDS.items() if k != 'hardware'}  # gate covers datasets/baselines/metrics
    from .extract.normalize import normalize
    from .ollama_client import OllamaClient

    conn, client = connect(), OllamaClient()
    recs = json.loads((ROOT / gold).read_text(encoding="utf-8"))
    inv_path = ROOT / 'data/gold/extract_invalid.json'
    invalid = set(json.loads(inv_path.read_text())) if inv_path.exists() else set()
    sub_ok = fails = adj_extra_ok = fin_ok = 0
    tp = {k: 0 for k in KINDS}
    npred = {k: 0 for k in KINDS}
    ngold = {k: 0 for k in KINDS}
    for rec in recs:
        w = conn.execute("SELECT title, abstract, tldr FROM works WHERE work_id=?", (rec["work_id"],)).fetchone()
        ex, _, _, err = extract_one(client, w["title"], w["abstract"], w["tldr"])
        if ex is None:
            fails += 1
            console.print(f"[red]FAIL[/red] {rec['work_id']}: {err}")
            for k in KINDS:
                ngold[k] += len(rec.get(k, []))
            continue
        ok = ex.subdomain_cluster == rec["subdomain"]
        sub_ok += ok
        pf = conn.execute("SELECT best_cluster, kw_scores_json FROM prefilter WHERE work_id=?", (rec["work_id"],)).fetchone()
        fin_ok += final_labels(ex, pf["best_cluster"], json.loads(pf["kw_scores_json"]))["subdomain_cluster"] == rec["subdomain"]
        if show and not ok:
            console.print(f"[yellow]subdomain[/yellow] {w['title'][:70]!r}: gold={rec['subdomain']} pred={ex.subdomain_cluster}")
        for k, field in KINDS.items():
            pred = {normalize(k, x) for x in getattr(ex, field)}
            gold_set = {normalize(k, x) for x in rec.get(k, [])}
            def same(a: str, b: str) -> bool:
                return a == b or fuzz.ratio(a, b) >= 85 or (min(len(a), len(b)) >= 4 and (a in b or b in a))

            hit = {p for p in pred if any(same(p, g) for g in gold_set)}
            found = {g for g in gold_set if any(same(p, g) for p in pred)}
            tp[k] += len(hit)
            adj_extra_ok += len({p for p in pred - hit if p not in invalid})
            npred[k] += len(pred)
            ngold[k] += len(gold_set)
            tp_r = len(found)
            if show and (pred - hit or gold_set - found):
                console.print(f"  {k}: extra={sorted(pred - hit)} missed={sorted(gold_set - found)}")
            tp[k + "_r"] = tp.get(k + "_r", 0) + tp_r
    n = len(recs)
    P = sum(tp[k] for k in KINDS) / max(sum(npred.values()), 1)
    R = sum(tp[k + "_r"] for k in KINDS) / max(sum(ngold.values()), 1)
    t = Table("metric", "value", "gate")
    t.add_row("subdomain accuracy (raw LLM)", f"{sub_ok}/{n} ({100 * sub_ok / n:.1f}%)", "")
    t.add_row("subdomain accuracy (final: LLM + pre-filter)", f"{fin_ok}/{n} ({100 * fin_ok / n:.1f}%)", "≥ 85%")
    adj = (sum(tp[k] for k in KINDS) + adj_extra_ok) / max(sum(npred.values()), 1)
    t.add_row("entity precision (adjudicated)", f"{adj:.1%}", "≥ 90%")
    t.add_row("entity precision (strict vs minimal gold)", f"{P:.1%}", "")
    t.add_row("entity recall", f"{R:.1%}", "≥ 70%")
    t.add_row("JSON failures", f"{fails}/{n}", "< 1%")
    for k in KINDS:
        t.add_row(f"  {k} P / R", f"{tp[k] / max(npred[k], 1):.0%} / {tp[k + '_r'] / max(ngold[k], 1):.0%}", "")
    console.print(t)


@app.command("extract-post")
def extract_post(skip_hf: bool = typer.Option(False, "--skip-hf"), skip_specter: bool = typer.Option(False, "--skip-specter")) -> None:
    """(M4) Hugging Face GitHub links + S2 pass 2 (SPECTER2 vectors) for pre-filter survivors. Needs no GPU."""
    from .db import connect, init_schema
    from .extract.post import run_hf, run_specter
    from .http_client import HttpClient

    conn = connect()
    init_schema(conn)
    http = HttpClient(conn)
    try:
        if not skip_hf:
            console.print(f"HF: {run_hf(http, conn, log=console.print)}")
        if not skip_specter:
            console.print(f"SPECTER2: {run_specter(http, conn, log=console.print)}")
    finally:
        http.close()


@app.command()
def index(skip_embed: bool = typer.Option(False, "--skip-embed", help="Reuse saved composite vectors"),
          check_only: bool = typer.Option(False, "--check-only", help="Only run the M5 gate checks")) -> None:
    """(M5) Normalise entities, composite vectors, Qdrant all_papers + relevant_papers, professor profiles, gate checks."""
    import numpy as np

    from .db import connect, init_schema
    from .embed.prefilter import load_embeddings
    from .index import build as B
    from .index.fts import rebuild_fts
    from .index.vector_store import open_store
    from .similarity.professors import build_profiles

    conn = connect()
    init_schema(conn)
    if not check_only:
        console.print(f"entities: {B.renormalize_entities(conn)}")
        console.print(f"FTS rows: {rebuild_fts(conn)}")
        if not skip_embed:
            from .embed.bge import BGE
            from .ollama_client import OllamaClient

            try:
                OllamaClient().unload()
            except Exception:  # noqa: BLE001
                pass
            bge = BGE("cuda")
            ids = B.embed_composites(conn, bge)
            bge.close()
            console.print(f"{OK} composite vectors for {len(ids)} extracted papers")
        store = open_store()
        B.build_relevant(conn, store, log=console.print)
        ids, abs_d, _ = load_embeddings()
        console.print(f"professor profiles: {build_profiles(conn, ids, abs_d)}")
    else:
        store = open_store()
    res = B.check(conn, store)
    store.close()
    t = Table("M5 gate check", "value", "expected")
    for name, (got, want) in res["counts"].items():
        t.add_row(f"{name} points vs SQLite", f"{got} / {want}", "equal")
    for k, v in res["p95_ms"].items():
        t.add_row(f"p95 search latency ({k})", f"{v} ms", "< 300 ms")
    for k, (n, ok) in res["smoke"].items():
        t.add_row(f"filter smoke: {k}", f"{n} results, all match: {ok}", "True")
    console.print(t)


@app.command()
def match(skip_topics: bool = typer.Option(False, "--skip-topics", help="Keep existing micro-topics"),
          no_label: bool = typer.Option(False, "--no-label", help="Keyword labels instead of gemma3 (fast)"),
          skip_pairs: bool = typer.Option(False, "--skip-pairs"),
          k: int = typer.Option(20, "--k", help="Matches kept per paper")) -> None:
    """(M6) Micro-topics (HDBSCAN + gemma3 labels), top-k cross-university `pairs` and `clusters`."""
    import json as _json

    from .db import connect, init_schema
    from .index.vector_store import open_store
    from .match.clusters import build_clusters
    from .match.pairs import build_pairs
    from .similarity import micro_topics as MT
    from .similarity.signals import Corpus
    from .similarity.scorer import default_weights

    conn = connect()
    init_schema(conn)
    corpus = Corpus.load(conn)
    if not skip_topics:
        client = None
        if not no_label:
            from .ollama_client import OllamaClient

            client = OllamaClient()
        console.print(f"micro-topics: {MT.build(conn, corpus, client, log=console.print)}")
        if client is not None:
            client.unload()
        corpus.reload_topics(conn)
    if not skip_pairs:
        from .config import load_settings

        gw = load_settings().path("data_dir") / "gold" / "weights.json"
        weights = _json.loads(gw.read_text()) if gw.exists() else default_weights()
        console.print(f"weights: {'calibrated (data/gold/weights.json)' if gw.exists() else 'defaults'}")
        store = open_store()
        console.print(f"pairs: {build_pairs(conn, corpus, store, weights=weights, k=k, log=console.print)}")
    console.print(f"clusters: {build_clusters(conn, corpus)}")


@app.command("match-calibrate")
def match_calibrate() -> None:
    """(M6) Fit similarity weights on data/gold/pairs.csv (LogReg, keeps defaults if AUC doesn't improve); writes weights.json."""
    import json as _json

    from .config import load_settings
    from .db import connect
    from .match.calibrate import run
    from .similarity.signals import Corpus

    res = run(Corpus.load(connect()))
    g = load_settings().path("data_dir") / "gold"
    (g / "weights.json").write_text(_json.dumps(res["weights"], indent=1))
    (g / "calibration.json").write_text(_json.dumps(res, indent=1))
    t = Table("M6 calibration", "value")
    for k, v in res.items():
        if k not in ("weights", "fitted_weights"):
            t.add_row(k, str(v))
    console.print(t)
    console.print({k: round(v, 3) for k, v in res["weights"].items()})
    console.print(f"Set similarity.tau_close in config/settings.yaml to {res['tau_close']}")


snap_app = typer.Typer(help="Save and switch between named versions of the whole working state (DB, embeddings, vectors, calibration)",
                       no_args_is_help=True)
app.add_typer(snap_app, name="snapshot")


def _snap(fn, *a, **k):
    from .snapshots import SnapshotError

    try:
        return fn(*a, **k)
    except SnapshotError as e:
        console.print(f"{BAD} {e}")
        raise typer.Exit(1)


@snap_app.command("save")
def snap_save(name: str, force: bool = typer.Option(False, "--force", help="Overwrite an existing snapshot")) -> None:
    """Copy the active state into data/snapshots/NAME (the active state is unchanged)."""
    from . import snapshots as S
    from .config import ROOT

    d = _snap(S.save, ROOT, name, force=force, log=console.print)
    console.print(f"{OK} saved {name} -> {d}")


@snap_app.command("list")
def snap_list() -> None:
    """Show the active label and all saved snapshots."""
    from . import snapshots as S
    from .config import ROOT

    t = Table("name", "created", "works", "extracted", "pairs", "universities", "size")
    for r in S.list_snapshots(ROOT):
        t.add_row(r["name"], r.get("created", ""), str(r.get("works", "?")), str(r.get("extracted", "?")), str(r.get("pairs", "?")),
                  str(r.get("universities", "?")), f"{r.get('bytes', 0) / 1e9:.2f} GB")
    console.print(f"active state: [bold]{S.active_label(ROOT)}[/bold]")
    console.print(t)


@snap_app.command("use")
def snap_use(name: str, keep_current_as: str = typer.Option(None, "--keep-current-as", help="Name to park the current state under")) -> None:
    """Switch to snapshot NAME. The current state is parked (not deleted) under its label, so you can switch back."""
    from . import snapshots as S
    from .config import ROOT

    _snap(S.use, ROOT, name, keep_current_as=keep_current_as, log=console.print)


@snap_app.command("delete")
def snap_delete(name: str, yes: bool = typer.Option(False, "--yes")) -> None:
    """Delete a saved snapshot (frees its disk space). Cannot delete the active state."""
    from . import snapshots as S
    from .config import ROOT

    if not yes:
        console.print(f"{BAD} add --yes to delete snapshot {name}")
        raise typer.Exit(1)
    _snap(S.delete, ROOT, name)
    console.print(f"{OK} deleted {name}")


@app.command()
def status() -> None:
    """Row counts per stage and when each stage last ran."""
    from .db import connect, init_schema
    from .pipeline import status_rows

    conn = connect()
    init_schema(conn)
    counts, last = status_rows(conn)
    t = Table("table / metric", "rows")
    for k, v in counts.items():
        t.add_row(k, str(v))
    console.print(t)
    t = Table("stage", "last finished", "runs")
    for r in last:
        t.add_row(*map(str, r))
    console.print(t if last else "no runs recorded yet (run-all logs them)")


@app.command("run-all")
def run_all_cmd(from_: str = typer.Option(None, "--from", help="First stage"), to: str = typer.Option(None, "--to", help="Last stage"),
                skip: list[str] = typer.Option(None, "--skip", help="Stage(s) to skip"),
                quick: bool = typer.Option(False, "--quick", help="Reuse saved artefacts: no network, no re-embedding, no re-labelling"),
                dry_run: bool = typer.Option(False, "--dry-run"),
                fresh: bool = typer.Option(False, "--fresh", help="Start from an empty data/ (keeps http cache, gold labels); needs --yes"),
                yes: bool = typer.Option(False, "--yes")) -> None:
    """(M9) Run init -> harvest -> enrich -> prefilter -> extract -> extract-post -> index -> match -> report, one process per stage."""
    from .config import load_settings
    from .db import connect, init_schema
    from .pipeline import fresh as make_fresh, run_all

    st = load_settings()
    if fresh:
        if not yes:
            console.print(f"{BAD} --fresh moves {st.path('db')} to a .bak file and deletes embeddings/vectors/qdrant_storage; add --yes to confirm")
            raise typer.Exit(1)
        if not dry_run:
            console.print(f"old database kept as {make_fresh(st.path('db'), st.path('data_dir'))}")
    conn = connect()
    init_schema(conn)
    try:
        res = run_all(conn, start=from_, stop=to, skip=list(skip or []), quick=quick, dry_run=dry_run, log=console.print)
    except ValueError as e:
        console.print(f"{BAD} {e}")
        raise typer.Exit(2)
    t = Table("table", "before", "after")
    for k, b, a in res["changed"]:
        t.add_row(k, str(b), str(a))
    console.print(t if res["changed"] else "no table counts changed (rerun reproduced the same counts)")
    console.print(f"{OK if res['ok'] else BAD} stages: " + ", ".join(f"{r['stage']}={r['rc']}" for r in res["stages"]))
    if not res["ok"]:
        raise typer.Exit(1)


def _overrides(uni, exclude_uni, subdomain, from_, to, type_, has_code, dataset, min_unis, k, intent, mode=None) -> dict:
    return {"uni": list(uni or []), "exclude_uni": list(exclude_uni or []), "subdomain": subdomain, "year_from": from_,
            "year_to": to, "types": list(type_ or []), "has_code": True if has_code else None, "datasets": list(dataset or []),
            "min_universities": min_unis, "k": k, "intent": intent, "mode": mode}


@app.command()
def ask(question: str = typer.Argument(..., help="Natural-language question"),
        uni: list[str] = typer.Option(None, "--uni", help="Only these universities (repeatable)"),
        exclude_uni: list[str] = typer.Option(None, "--exclude-uni"),
        subdomain: str = typer.Option(None, "--subdomain", help="SysML_RL_Caching | Edge_RAG_Vector_Systems | Spatio_Temporal_MARL"),
        from_: int = typer.Option(None, "--from", help="Earliest year"), to: int = typer.Option(None, "--to", help="Latest year"),
        type_: list[str] = typer.Option(None, "--type", help="conference-paper | preprint"),
        has_code: bool = typer.Option(False, "--has-code"), dataset: list[str] = typer.Option(None, "--dataset"),
        min_unis: int = typer.Option(None, "--min-unis"), k: int = typer.Option(None, "--k"),
        intent: str = typer.Option(None, "--intent"),
        mode: str = typer.Option(None, "--mode", help="diverse (default) | one_per_uni | overall | one_uni"),
        no_answer: bool = typer.Option(False, "--no-answer", help="Skip the gemma3 synthesis (table only)"),
        show_plan: bool = typer.Option(False, "--show-plan")) -> None:
    """(M7) Ask a question: gemma3 plan -> deterministic retrieval -> table + cross-university summary -> validated synthesis."""
    from .rag.chat import run_question
    from .rag.execute import Engine

    eng = Engine()
    out = run_question(eng, question, overrides=_overrides(uni, exclude_uni, subdomain, from_, to, type_, has_code, dataset,
                                                           min_unis, k, intent, mode), answer=not no_answer, console=console)
    if show_plan:
        console.print_json(json.dumps(out.result.R.plan.model_dump()))


@app.command()
def chat(no_answer: bool = typer.Option(False, "--no-answer", help="Tables only")) -> None:
    """(M7) Interactive REPL; follow-ups refine the previous query. Commands: :plan :why P3 :open P3 :filters :save :reset."""
    from .rag.chat import repl
    from .rag.execute import Engine

    repl(Engine(), console, answer=not no_answer)


@app.command("ask-eval")
def ask_eval(path: str = typer.Option("data/gold/rag_eval.json", help="JSON list of {q, expect:[regex], topic?: bool}"),
             answer: bool = typer.Option(True, "--answer/--no-answer", help="Also run the synthesis + validator"),
             show: bool = typer.Option(False, help="Print the top-8 titles per question")) -> None:
    """(M7 gate) Run the evaluation questions and report top-8 hit rate, university coverage and citation validity."""
    from .rag.evaluate import run_eval

    run_eval(path, answer=answer, show=show, console=Console(width=200))


@app.command()
def report(top: int = typer.Option(10, help="Number of clusters"), no_draft: bool = typer.Option(False, "--no-draft", help="Skip gemma3 paragraphs"),
           out: str = typer.Option("BENCHMARK_CANDIDATES.md")) -> None:
    """(M8) Write BENCHMARK_CANDIDATES.md for the top cross-university clusters."""
    from pathlib import Path

    from .config import ROOT
    from .rag.execute import Engine
    from .report.benchmark_report import build_report

    eng = Engine(llm=not no_draft)
    console.print(build_report(eng, top=top, draft=not no_draft, out_path=ROOT / Path(out), log=console.print))


if __name__ == "__main__":
    app()
