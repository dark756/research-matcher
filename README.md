# research-matcher

**A fully local RAG system that harvests 36,569 recent CS papers from 17 research universities, extracts what each one claims, and finds the closest work at other universities so that results can be compared and verified.**

Built to compile a cross-university benchmark of papers, the professors behind them, and their subdomains (cache/RL for systems, edge RAG and vector systems, spatio-temporal MARL). It runs on a single laptop GPU with no cloud inference.

---

## Architecture

```
OpenAlex ──► harvest ──► Semantic Scholar enrich ──► bge-m3 pre-filter (cosine + keywords)
(36.6k papers,            (venue, abstract, tldr,         │  7.9% survive (2,882 papers)
 17 universities)          SPECTER2, citations)           ▼
                                              gemma3:4b-it-qat via Ollama
                                              (JSON-schema extraction: datasets, baselines,
                                               metrics, claims; grounded against the abstract)
                                                          │
CSRankings ──► professor verification ───────────────────►│
                                                          ▼
               SQLite + FTS5 (BM25)  +  NumPy flat vector index (bge-m3 dense, 1024-d)
                                                          │
                      14-signal cross-university scorer + MMR diversification (≤3 per university)
                                                          │
                              ask / chat (gemma3 query plan → retrieval → cited answer)
                              report (benchmark candidates, claims to verify)
```

| Layer | Choice |
|---|---|
| Metadata | OpenAlex (primary), Semantic Scholar (enrichment, 1 req/s global limiter), CSRankings (professor check) |
| Embeddings | bge-m3 via FlagEmbedding (dense 1024-d + sparse lexical) |
| LLM | `gemma3:4b-it-qat` via Ollama. It has no native tool calling, so queries are turned into JSON-schema plans |
| Store | SQLite (FTS5 for BM25) and a NumPy flat vector index. An embedded-Qdrant backend is included but off by default (see results) |
| Answers | Every `[P#]` citation is validated, and university, professor and link fields come from the database, not the model |

Full design: [plan.md](plan.md). Planned multi-version extension: [re_plan.md](re_plan.md).

## Hardware constraints

Designed and tuned for a **laptop RTX 3050 with 6 GB VRAM** (plus CPU RAM), with no cloud GPU.

- gemma3 4B QAT (about 4 GB) and bge-m3 are never resident on the GPU together. Pipeline stages run as separate subprocesses so VRAM is freed between them, and gemma3 is unloaded after each stage.
- At query time bge-m3 runs on CPU so that gemma3 keeps the GPU.
- The answer window is 16K tokens (`rag.num_ctx`) with up to 12 context papers, sized to fit in 6 GB.
- The generated benchmark report includes a drafted stress-test plan per cluster for the same 6 GB budget. These drafts are LLM output and have not been run.

## Results

Measured on this machine (RTX 3050 6 GB Laptop, Windows 11):

| Metric | Result |
|---|---|
| Corpus | 37,279 works harvested, 36,569 canonical after dedupe; 17 universities; papers from 2025-01-01; 18,539 professor profiles |
| Pre-filter | 2,882 survivors (7.9%); recall on a 135-paper gold set was 100%, which is optimistic because the sample was skewed toward top candidates |
| Extraction | 2,882 of 2,882 papers extracted, 0 JSON failures, about 4.8 h at 0.17 papers/s on the 3050 |
| Extraction quality (50-abstract gold set, tuned on it) | subdomain accuracy 88%, entity precision 96%, recall 93% |
| Cross-university output | 9,880 ranked pairs, 59 clusters (43 span 3 or more universities) |
| RAG evaluation (20 questions) | expected paper in the top 8 for **19/20**; **12/12** topic queries return 3 or more universities; **0** invalid citations; 0 unsupported entity, professor or university terms |
| Retrieval latency (36.6k vectors) | p95 4 ms unfiltered, 41 ms filtered, using the NumPy flat index. Local Qdrant filtered p95 was 492 ms, which missed the 300 ms target, so it is not the default |
| Full rebuild | `run-all --fresh` reproduces the deterministic stages exactly; LLM-dependent counts drift by under 1% |
| Tests | 54 pytest tests |

Stage timings from the last full run: harvest 16.3 min, enrich 1.9, prefilter 13.6, extract 288.8, extract-post 10.6, index 1.4, match 9.1, report 2.5.

**Known limits.** Under a strict rubric only 42% of matches above the "close" threshold were judged close on a held-out sample (98% close or related), so treat pairs below the strong threshold as "related". Claims are parsed from abstracts only and must be checked against the papers. Gold labels were produced with Claude assistance and the thresholds were tuned on them.

## How to run it

Prerequisites: Windows or Linux, Python 3.11+, an NVIDIA GPU with 6 GB or more, [Ollama](https://ollama.com), and free API keys for OpenAlex and Semantic Scholar.

```bash
# 1. Install
python -m venv .venv && .venv/Scripts/activate        # Linux/macOS: source .venv/bin/activate
pip install -e . torch FlagEmbedding scikit-learn networkx   # install CUDA torch per pytorch.org
ollama pull gemma3:4b-it-qat

# 2. Configure keys (copy .env.example to .env.local and fill in OPENALEX_API_KEY, OPENALEX_EMAIL, S2_API_KEY)

# 3. Build, then ask (first build is about 6-7 h, mostly extraction; it is resumable)
research-matcher run-all
research-matcher ask "ETH vs CMU vs Toronto on KV-cache quantization: who does what, shared baselines?"
```

The database and embeddings are not stored in this repository (they are several GB), so step 3 builds them from the public APIs.

### Example questions

- Find similar work across universities on learned cache eviction for key-value stores.
- Who at NUS/NTU does MARL for ride-hailing dispatch, with code, 2026 only?
- Which professors in Europe work on topics similar to Gustavo Alonso?
- Which datasets and baselines are most used in Edge_RAG_Vector_Systems, and by which universities?

`research-matcher chat` is interactive: follow-ups refine the last query ("only with code", "drop UCL", "more like P2"). Useful flags: `--uni ETH`, `--from 2026`, `--has-code`, `--no-answer` (table only, faster), `--mode diverse|one_per_uni|overall|one_uni`.

### Other commands

| Command | Does |
|---|---|
| `status` | row counts per stage, last run times |
| `report` | writes `BENCHMARK_CANDIDATES.md` (top clusters, claims to verify, 3050 stress-test plan) |
| `ask-eval` | 20-question RAG evaluation |
| `run-all [--from S] [--to S] [--quick] [--dry-run]` | all stages in order; `--quick` reuses saved artefacts (about 15 min) |
| `snapshot save/list/use/delete` | keep and switch between whole working states |

Individual stages: `init harvest enrich prefilter extract extract-post index match`. Tests: `pytest tests -q`.

## Repository layout

```
src/research_matcher/  harvest/ enrich/ embed/ extract/ index/ match/ similarity/ rag/ report/
config/                universities, subdomains, aliases, settings
tests/                 pytest suite
plan.md  re_plan.md    design reference and next-step plan
BENCHMARK_CANDIDATES.md  example generated report
```
