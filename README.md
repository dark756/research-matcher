# research-matcher

Cross-university CS paper store (17 universities, papers from 2025-01-01, three subdomains) with a local RAG engine for
finding close work at other universities. Stack: OpenAlex + Semantic Scholar, bge-m3 embeddings, `gemma3:4b-it-qat` via Ollama.
Full design: [plan.md](plan.md).

## Talk to it

Prerequisites: Ollama running with `gemma3:4b-it-qat`; the project's `.venv`; a built database (`data/papers.db`).

```
cd "C:\Users\gurka\Desktop\research folder\codebase"
.venv\Scripts\python.exe -m research_matcher.cli chat            # interactive
.venv\Scripts\python.exe -m research_matcher.cli ask "your question" [--uni ETH] [--from 2026] [--has-code] [--no-answer]
```

If the venv is already active (e.g. a VS Code terminal showing `(.venv)`), drop the long prefix:

```
research-matcher chat
research-matcher ask "your question" --no-answer
```

or, if that command isn't found, `python -m research_matcher.cli chat`.

Which papers are picked (`--mode`, or `:mode X` in chat; default in `config/settings.yaml` → `rag.select_mode`):
`diverse` (default: spread over universities, max 3 each), `one_per_uni` (best paper of each university), `overall` (plain top-k by
score), `one_uni` (prefer the single best-matching university, fill the rest from others). Saying "one per university" in the
question also switches to `one_per_uni`. The written answer sees up to 12 papers (`rag.context_papers`) in a 16K window (`rag.num_ctx`).

Example questions:
- Find similar work across universities on learned cache eviction for key-value stores.
- ETH vs CMU vs Toronto on KV-cache quantization: who does what, shared baselines?
- Who at NUS/NTU does MARL for ride-hailing dispatch, with code, 2026 only?
- Show papers from other universities similar to W4413360446
- Which professors in Europe work on topics similar to Gustavo Alonso?
- Which datasets and baselines are most used in Edge_RAG_Vector_Systems, and by which unis?

In `chat`, follow-ups refine the last query ("only with code", "drop UCL", "more like P2"). Commands: `:plan` `:why P3` `:open P3`
`:filters` `:save` `:reset` `:help` `:quit`. `--no-answer` gives the table only (faster).

Timing: first question ≈ 30 s (loads the corpus ≈ 8 s and bge-m3 on CPU ≈ 17 s, plus gemma3), later ones ≈ 30–40 s with the
written answer, ≈ 10 s table-only. Close GPU-heavy apps (games) first; gemma3 needs the 6 GB GPU.

## Other commands

| Command | Does |
|---|---|
| `status` | row counts per stage, last run times |
| `report` | writes `BENCHMARK_CANDIDATES.md` (top clusters, claims to verify, 3050 stress-test plan) |
| `ask-eval` | 20-question RAG evaluation (M7 gate) |
| `run-all [--from S] [--to S] [--quick] [--dry-run]` | all stages in order; `--quick` reuses saved artefacts (~15 min) |
| `run-all --fresh --yes` | rebuild from empty data (old DB kept as `.bak`); ≈ 6–7 h, mostly extraction |

## Versions (snapshots)

Keep several versions of the whole working state (DB, embeddings, vectors, calibration, `universities.yaml`, `settings.yaml`,
benchmark report) and switch with one command. Close `chat` first: a running session holds the database and the switch refuses.

```
research-matcher snapshot save v1-17unis        # copy of the active state (≈2 GB, a few seconds)
research-matcher snapshot list                  # active label + saved versions
research-matcher snapshot use v1-17unis --keep-current-as v2-expanded    # switch; the current state is parked, not deleted
research-matcher snapshot use v2-expanded       # switch back (v1 is parked under its own label)
research-matcher snapshot delete NAME --yes     # free disk space
```

Switching is a rename on the same drive (about a second, no extra disk) and is rolled back automatically if anything fails.
Shared across versions, not snapshotted: `data/gold` labels, `data/cache`, `data/notes`, `.env.local`. `data/papers.db.bak-*`
(left by `run-all --fresh`) is not a snapshot and can be deleted.

Individual stages: `init harvest enrich prefilter extract extract-post index match`.

## Setup from scratch

1. Python 3.11+, `python -m venv .venv`, `pip install -e .` plus torch (CUDA), FlagEmbedding, hdbscan deps (scikit-learn), networkx.
2. Install Ollama and `ollama pull gemma3:4b-it-qat`.
3. Put `OPENALEX_API_KEY`, `OPENALEX_EMAIL`, `S2_API_KEY` in `.env.local`.
4. `research-matcher run-all` (first build, ≈ 6–7 h).

Tests: `.venv\Scripts\python.exe -m pytest tests -q`
