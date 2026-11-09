base req: i need to compile a list of what papers, who worked on them (professors), what subdomains, so i can merge them with similar papers published from other reach unis and make my own benchmark paper comparing and verifying their findings

using rag system and gemma3 4b-it-qat already downloaded on ollama and FlagEmbedding already pip installed on python 3.11 venv in this folder for bge-m3

---

# Implementation Plan: `research-matcher` — Cross-University CS Paper Store + Local RAG Engine

> **Rev 3 (2026-11-05). Ready for implementation.** ✅ = verified live on this machine or against the OpenAlex API on 2026-11-05.
>
> **Locked decisions**
> | Topic | Decision |
> |---|---|
> | LLM | `gemma3:4b-it-qat` via Ollama (capabilities ✅ `completion, vision`, **no native tools**, so we use JSON-schema "query plans", not tool calling) |
> | Embeddings | bge-m3 via **FlagEmbedding (pip)**, dense 1024-d + sparse lexical |
> | GPU | `torch 2.15.0+cu132` ✅ installed, CUDA works on the RTX 3050 6 GB Laptop |
> | Metadata source | **OpenAlex = primary** (API key in `.env.local` ✅). **Semantic Scholar = enrichment** (`S2_API_KEY` in `.env.local` ✅ working). **Rate limit: 1 request/second, cumulative across ALL S2 endpoints**, enforced by one global limiter (§2.1) |
> | Filters | Published **2025-01-01 onwards**, types **conference-paper + preprint**, domain **Computer Science**, the 17 institutions below |
> | Output | **RAG query engine** (`ask` / `chat`) with multi-signal cross-university similarity, plus `report` for benchmark candidates. **No CSV/XLSX exports.** |

---

## 1. Environment (✅ 2026-11-05)

| Item | State |
|---|---|
| Python | `.venv` 3.11.0 |
| torch | `2.15.0+cu132` (driver 610.78, CUDA UMD 13.3) ✅ fp16 matmul on GPU OK |
| FlagEmbedding | 1.4.2 ✅ imports. bge-m3 weights **not yet downloaded** (~2.3 GB on first `BGEM3FlagModel` load) |
| Ollama | `gemma3:4b-it-qat` (4.0 GB, ctx up to 131,072) ✅ |
| SQLite | 3.38.4 with **FTS5** ✅ (used for BM25 keyword retrieval) |
| scikit-learn | 1.9.1 ✅ `HDBSCAN` (micro-topics) and `LogisticRegression` (weight calibration) available |
| Secrets | `.env.local` ✅ contains `OPENALEX_API_KEY`, `OPENALEX_EMAIL`, `S2_API_KEY`. Load with `load_dotenv(".env.local")`. **Never log keys** (redact `api_key=` in cached URLs; the S2 key goes in the `x-api-key` header, so never write headers to cache or logs) |
| To install (M0) | `pip install pydantic httpx qdrant-client python-dotenv tenacity rapidfuzz pytest prompt_toolkit` |

### 1.1 OpenAlex API budget (✅ read from response headers)
- Daily limit: **10,000 credits** (`X-RateLimit-Limit: 10000`, ≈ $1/day). List/filter request = **1 credit**; any `search` request = **10 credits**. Resets daily (`X-RateLimit-Reset` = seconds).
- Estimated full harvest ≈ **450–1,200 credits** (§4.4), so one day's budget is plenty, and reruns are free thanks to the HTTP cache.
- The HTTP client must read `X-RateLimit-Remaining` on every response, **stop at < 500 remaining**, and honor `Retry-After` on 429.

---

## 2. Data sources (final)

| Source | Used for | Default |
|---|---|---|
| **OpenAlex** (`api.openalex.org`) | Harvest (the only source that can filter by institution): works, abstracts, authors, affiliations + lineage, topics/keywords, references, related works, citations, venue, OA links | **ON**, primary |
| **Semantic Scholar** (`api.semanticscholar.org`) | Enrichment by DOI/arXiv ID: **real venue** for arXiv-only papers (✅ test: arXiv 1706.03762 → "Neural Information Processing Systems", type conference), `tldr`, abstract fallback, citation + influential-citation counts, `externalIds` (ArXiv, DBLP, CorpusId), `openAccessPdf`, **SPECTER2 paper embeddings** (✅ 768-d) | **ON**, at **1 req/s cumulative** (§2.1) |
| CSRankings faculty CSV (static file on GitHub, one download, no API, no key) | Professor verification (✅ 34,749 faculty, all 17 unis covered, 66–274 each) | ON (downloaded once in `init`, cached) |
| Hugging Face Papers API (`/api/papers/{arxiv_id}` → `githubRepo`) | GitHub links for arXiv papers | ON, **only for pre-filter survivors** (~2–5k cached calls); `--no-hf` to disable |
| GitHub search API | Last-resort code lookup | **OFF** by default (`enable_github_search: false`) |
**How the two sources split the work:**
- **Venue fix:** ICLR/ICML/USENIX papers that OpenAlex only knows as arXiv preprints get their real venue from S2 `publicationVenue` (stored as `venue_s2`; the displayed venue prefers S2 when OpenAlex says "arXiv"). The abstract-regex "Accepted at …" hint stays as a fallback.
- **Abstracts:** OpenAlex covers 88.7% (32,330 / 36,435). S2 `abstract` fills gaps; papers still missing one are embedded by title + keywords + topics + tldr and flagged `abstract_missing`.
- **arXiv IDs:** derived from OpenAlex (DOI `10.48550/arXiv.<id>` or the `arxiv.org/abs/<id>` URL), and cross-checked or filled from S2 `externalIds.ArXiv`.

### 2.1 Semantic Scholar usage rules (rate limit: **1 request/second, cumulative across all endpoints**)
- **One global limiter** (`S2RateLimiter`, token bucket: 1 token/s, burst 1) wraps **every** S2 call: batch, search, single-paper, whatever. No S2 calls run in parallel, ever: no thread pools and no async fan-out for S2.
  - To keep the limit when two CLI commands run at once, the limiter also stores `last_request_at` in SQLite (`rate_limits` table, `BEGIN IMMEDIATE` lock) and sleeps until ≥ 1.05 s (5% safety margin) have passed since the last S2 request from any process.
- **Backoff on 429/5xx:** honor `Retry-After` if present. Otherwise exponential with jitter: 2 s → 4 → 8 → 16 → 32 → 60 s cap, max 6 attempts, then mark the batch `s2_failed` and continue (resumable). Each retry also consumes a limiter token.
- **Batch endpoint only (default):** `POST /graph/v1/paper/batch` with **≤ 500 IDs** per request (`DOI:…`, `ARXIV:…`). This is the only way to stay fast at 1 req/s.
  - **Pass 1, all harvested works (~37.3k IDs, ≈ 75 requests ≈ 2–3 min):**
    `fields=externalIds,title,venue,publicationVenue,year,publicationTypes,abstract,tldr,citationCount,influentialCitationCount,openAccessPdf`
  - **Pass 2, pre-filter survivors only (2–5k IDs, ≈ 4–10 requests):** `fields=embedding.specter_v2` (768-d vectors, kept in a separate pass so large responses stay small). Stored in `data/embeddings/specter2.npy` plus SQLite index.
- **Optional, off by default:** `GET /graph/v1/paper/search/match?query=<title>` for works with neither a DOI nor an arXiv ID. It costs 1 request per paper, so only for survivors: `s2_title_match: false`; 1,000 papers ≈ 17 min.
- Responses are cached in `http_cache` (2xx only), so reruns cost zero S2 requests. Papers S2 doesn't know (`null` entries in the batch response) are recorded as `s2_status = not_found` and never re-queried.
- Header: `x-api-key: <S2_API_KEY>`. The key is never written to the cache key, logs or `query_log`.

---

## 3. Target universities (17) ✅ re-verified 2026-11-05

| # | Short | University | Ctry | ROR | OpenAlex ID | CSRankings affiliation (faculty) | Aliases for queries |
|---|---|---|---|---|---|---|---|
| 1 | ETH | ETH Zurich | CH | `05a28rw58` | `I35440088` | ETH Zurich (66) | eth, ethz, eth zürich |
| 2 | EPFL | École Polytechnique Fédérale de Lausanne | CH | `02s376052` | `I5124864` | EPFL (101) | epfl, lausanne |
| 3 | TUM | Technical University of Munich | DE | `02kkvpp62` | `I62916508` | TU Munich (122) | tum, tu munich, tu münchen |
| 4 | OXF | University of Oxford | GB | `052gg0110` | `I40120149` | University of Oxford (113) | oxford |
| 5 | CAM | University of Cambridge | GB | `013meh722` | `I241749` | University of Cambridge (107) | cambridge |
| 6 | IMP | Imperial College London | GB | `041kmwe10` | `I47508984` | Imperial College London (106) | imperial, icl |
| 7 | UCL | University College London | GB | `02jx3x895` | `I45129253` | University College London (125) | ucl |
| 8 | CMU | Carnegie Mellon University | US | `05x2bcf33` | `I74973139` | Carnegie Mellon University (274) | cmu, carnegie mellon |
| 9 | COL | Columbia University | US | `00hj8s172` | `I78577930` | Columbia University (73) | columbia |
| 10 | UCSD | University of California San Diego | US | `0168r3w48` | `I36258959` | Univ. of California - San Diego (197) | ucsd, uc san diego |
| 11 | GT | Georgia Institute of Technology | US | `01zkghx44` | `I130701444` | Georgia Institute of Technology (240) | gt, gatech, georgia tech |
| 12 | UofT | University of Toronto | CA | `03dbr7087` | `I185261750` | University of Toronto (160) | uoft, u of t, toronto |
| 13 | UW | University of Waterloo | CA | `01aff2v68` | `I151746483` | University of Waterloo (159) | waterloo, uwaterloo |
| 14 | UBC | University of British Columbia | CA | `03rmrcq20` | `I141945490` | University of British Columbia (96) | ubc |
| 15 | SFU | Simon Fraser University | CA | `0213rcc28` | `I18014758` | Simon Fraser University (86) | sfu, simon fraser |
| 16 | NUS | National University of Singapore | SG | `01tgyzw49` | `I165932596` | National University of Singapore (117) | nus |
| 17 | NTU | Nanyang Technological University | SG | `02e7b5302` | `I172675005` | Nanyang Technological University (169) | ntu, nanyang |

Countries for query parsing: CH, DE, GB (UK), US, CA, SG. Regions: "Europe" = CH+DE+GB, "North America" = US+CA, "Asia" = SG.

---

## 4. Harvest specification (OpenAlex)

### 4.1 Query A — main set ✅ **36,435 works** (32,330 with abstract)
```
GET https://api.openalex.org/works
  filter=authorships.institutions.lineage:I35440088|I5124864|I62916508|I40120149|I241749|I47508984|I45129253|
                                          I74973139|I78577930|I36258959|I130701444|I185261750|I151746483|
                                          I141945490|I18014758|I165932596|I172675005,
         from_publication_date:2025-01-01,
         type:conference-paper|preprint,
         primary_topic.field.id:fields/17,
         is_retracted:false,is_paratext:false
  per_page=100, cursor=* → meta.next_cursor
  select=<see 4.5>
  api_key=<from .env.local>, mailto=<OPENALEX_EMAIL>
```
Breakdown ✅: preprint 18,409 · conference-paper 18,026 · 2025: 16,375 · 2026 (YTD): 20,060.
By subfield ✅: AI 18,222 · CV 7,139 · Info Systems 3,289 · Theory 2,067 · Networks 1,767 · HW/Arch 1,054 · HCI 934 · Signal Proc 755 · CS Apps 534 · Software 430 · Graphics 244.
(All CS subfields kept, as specified. The pre-filter handles relevance. Optional toggle `exclude_subfields: [1707,1709,1704,1711]` → 27,363.)

### 4.2 Query B — conference proceedings published as journals ✅ **643 works** (toggle `include_proceedings_journals: true`, default ON)
**Why:** OpenAlex types SIGMOD, VLDB, POPL/PLDI/OOPSLA, SIGMETRICS, CoNEXT and FSE papers as `article`, because these conferences publish in journal-style proceedings. Your `conference-paper` filter alone would **silently drop SIGMOD and VLDB**, which are core venues for `SysML_RL_Caching` and `Edge_RAG_Vector_Systems`.
```
filter=authorships.institutions.lineage:<17 IDs>, from_publication_date:2025-01-01, type:article,
       locations.source.id:S4387289859|S4210226185|S4210216081|S4210193547|S4387290834|S4404663975
```
| Source ID | Venue (= conference) | 2025+ works |
|---|---|---|
| `S4387289859` | Proc. ACM on Management of Data (SIGMOD) | 143 |
| `S4210226185` | Proc. VLDB Endowment (VLDB) | 126 |
| `S4210216081` | Proc. ACM on Programming Languages (POPL/PLDI/OOPSLA/ICFP) | 217 |
| `S4210193547` | Proc. ACM Meas. Anal. Comput. Syst. (SIGMETRICS) | 21 |
| `S4387290834` | Proc. ACM on Networking (CoNEXT) | 19 |
| `S4404663975` | Proc. ACM on Software Engineering (FSE) | 116 |
These are stored with `type = "conference-paper"` and `type_raw = "article"`.

### 4.3 Query C — subdomain keyword supplement (outside the CS field) ✅ **≈206 works** (toggle `keyword_supplement: true`)
**Why:** OpenAlex files some CS-method papers under Engineering or Decision Sciences. For `Spatio_Temporal_MARL`, **32 keyword hits are outside CS vs 15 inside**.
```
filter=authorships.institutions.lineage:<17>, from_publication_date:2025-01-01, type:conference-paper|preprint,
       primary_topic.field.id:!fields/17, title_and_abstract.search:<cluster boolean query>
```
| Cluster | Boolean query (v1, extend in M3) | inside CS | outside CS (added) |
|---|---|---|---|
| SysML_RL_Caching | `("reinforcement learning" OR "learned") AND (caching OR "cache replacement" OR "cache eviction" OR "query optimization" OR "query optimizer" OR "cardinality estimation" OR "load balancing" OR autoscaling OR "cluster scheduling" OR "job scheduling")` | 71 | 8 |
| Edge_RAG_Vector_Systems | `"approximate nearest neighbor" OR HNSW OR DiskANN OR "vector database" OR "vector search" OR "product quantization" OR "KV cache" OR "retrieval-augmented generation"` | 810 | 166 |
| Spatio_Temporal_MARL | `("multi-agent reinforcement learning" OR "graph neural network" OR "learned index") AND ("vehicle routing" OR dispatch OR fleet OR "ride-hailing" OR "resource allocation" OR emergency OR spatio-temporal)` | 15 | 32 |
Tagged `harvest_source = keyword_supplement`. They still must pass the pre-filter.
⚠ **Early signal:** strict keywords find only ~50 MARL/routing papers. Expect that cluster to be the thinnest; the meaning-based pre-filter (§7) is what widens it.

### 4.4 Credit estimate
A ≈ 365 pages + B ≈ 7 + C ≈ 3 queries × ≤3 pages × 10 credits + author batches (~150–600 calls for senior authors at target unis) + 1 institution check ≈ **450–1,100 credits** out of the 10,000/day.

### 4.5 `select` fields (top-level only)
`id,doi,ids,title,publication_year,publication_date,type,language,primary_location,locations,authorships,abstract_inverted_index,primary_topic,topics,keywords,referenced_works,related_works,cited_by_count,fwci,citation_normalized_percentile,open_access,best_oa_location,is_retracted`
✅ Sample check: `related_works` present 25/25, `keywords` 25/25, `referenced_works` 19/25 (overall `has_references` = 14,054 / 36,435 = 39%; the similarity engine treats missing references as "unknown", not "dissimilar").

### 4.6 Authors batch
`GET /authors?filter=openalex_id:A1|A2|…` (≤100 per call), `select=id,display_name,orcid,summary_stats,last_known_institutions,works_count`. Only for authors who are senior or lead authors at a target uni.

### 4.7 Other filters available later (✅ tested; add to config without code changes)
Institution type/country/continent · author ID/ORCID · corresponding author/institution · authors/institutions/countries count ·
dates (`from_/to_publication_date`, created/updated) · language · subfield/topic/keyword IDs (with `!` negation) ·
text search (`title.search`, `abstract.search`, `title_and_abstract.search`, `fulltext.search`, `semantic.search`) ·
venue ID/type/h-index/publisher, `indexed_in`, `doi_starts_with` · `cited_by_count`, `fwci`, top 1%/10% · has abstract/DOI/PDF/references,
OA status/license · funders/awards. Invalid now: `type_crossref`, `grants.funder`. Note: `group_by` cannot be combined with `select`.

---

## 5. Data stored per paper

| Group | Fields |
|---|---|
| Identity | `work_id`, `doi`, `arxiv_id` (derived / S2), `s2_paper_id`, `s2_corpus_id`, `dblp_key` (from S2 `externalIds`), `title`, `title_norm`, `year`, `publication_date`, `type`, `type_raw`, `language`, `canonical_id`, `merged_ids[]` |
| Venue | `venue_name` (display: S2 venue if OpenAlex says arXiv), `venue_source_id`, `venue_type` (conference/journal/repository), `venue_s2` + `venue_s2_type` + `venue_s2_aliases` (e.g., NeurIPS/NIPS), `venue_hint` (regex "accepted at X" from abstract) |
| Links | `doi_url`, `openalex_url`, `arxiv_abs_url`, `landing_page_url`, `oa_pdf_url`, `github_urls[]` + `code_source` (abstract_regex / hf) + `code_confidence`, `has_code` |
| Authors (all) | `author_id`, `name`, `orcid`, `position`, `is_corresponding`, institutions with lineage, `raw_affiliation` |
| Professor per target uni | `senior_author` (PI), `lead_author`, `professor_verified` (CSRankings), `homepage`, `scholar_id`, `h_index`, `works_count`, `attribution_confidence` |
| Universities | `target_universities[]`, `sub_institutions[]`, `other_institutions[]` (incl. industry), `countries[]`, `is_multi_target_collab` |
| Content | `abstract` + `abstract_source` (openalex/s2), `abstract_missing`, `tldr` (S2), OpenAlex `primary_topic` / `subfield` / `field`, `topics[]` (with scores), `keywords[]` (with scores) |
| Citation graph | `referenced_works[]`, `related_works[]`, `cited_by_count` (OpenAlex), `s2_citations`, `s2_influential_citations`, `fwci`, `top_10pct`, `citation_velocity` |
| Subdomain | `subdomain_cluster`, `prefilter_scores{}`, `llm_label`, `labels_agree`, `is_relevant`, `confidence`, `needs_review`, `micro_topic_id` + `micro_topic_label` |
| Extracted (LLM, grounded) | `sub_problem_solved`, `core_algorithm_or_architecture`, `baselines[]`, `datasets[]`, `metrics[]` (raw + normalized), `reported_results[]` {metric, value, compared_to, dataset}, `hardware_mentioned[]`, `grounding_dropped` |
| Access | `is_oa`, `oa_status`, `license`, `oa_pdf_url` (OpenAlex, else S2 `openAccessPdf`) |
| Matching | `pairs[]` (score + per-signal breakdown + shared entities), `cluster_ids[]` |
| Vectors | `abstract_dense`, `title_dense`, `composite_dense` (extracted papers), sparse lexical weights (SQLite), `specter2` (S2, survivors) |
| Provenance | `harvest_source` (A/B/C), `harvested_at`, `s2_status` (ok/not_found/failed) + `s2_fetched_at`, `llm_model`, `prompt_version`, `extracted_at`, `raw_json` (zlib), `s2_raw_json` (zlib) |

### 5.1 SQLite schema (`data/papers.db`)
| Table | Purpose / key columns |
|---|---|
| `http_cache` | `key` PK (sha256 of method + **redacted** URL + body; headers never included), `url_redacted`, `status`, `body` (zlib), `fetched_at` |
| `rate_limits` | `service` PK (`s2`, `openalex`), `last_request_at`, `credits_remaining`. Used by the global limiter across processes |
| `institutions` | `openalex_id` PK, `ror`, `name`, `short`, `country`, `csrankings_name`, `aliases_json` |
| `works` | identity, venue, links, content, citation-graph, access and provenance fields above (`referenced_works_json`, `related_works_json`, `topics_json`, `keywords_json`) |
| `works_fts` | FTS5 virtual table over (`title`, `keywords`, `abstract`) for BM25 |
| `authors` | `author_id` PK, `name`, `orcid`, `h_index`, `works_count`, `last_known_insts_json` |
| `authorships` | (`work_id`, `author_id`) PK, `position_idx`, `author_position`, `is_corresponding`, `inst_ids_json`, `raw_affiliation` |
| `work_universities` | (`work_id`, `inst_id`) PK, `senior_author_id`, `lead_author_id`, `attribution_confidence` |
| `faculty` | CSRankings rows: `name`, `name_norm`, `affiliation`, `homepage`, `scholar_id`, `orcid`, `matched_author_id` |
| `code_links` | `work_id`, `url`, `source`, `confidence` |
| `prefilter` | `work_id` PK, `kw_scores_json`, `dense_scores_json`, `best_cluster`, `keep` |
| `extractions` | `work_id` PK, `model`, `prompt_version`, `raw_json`, `is_relevant`, `subdomain_cluster`, `sub_problem`, `core_method`, `confidence`, `needs_review`, `grounding_dropped`, `error` |
| `entities` | (`work_id`, `kind`, `normalized`) PK, `raw`. Kind ∈ dataset / baseline / metric / hardware |
| `reported_results` | `work_id`, `metric`, `value`, `compared_to`, `dataset` |
| `sparse_vecs` | `work_id`, `field` (title/composite), `lexical_json` |
| `micro_topics` | `topic_id` PK, `subdomain`, `label`, `size`, `universities_json`, `centroid` (blob) |
| `pairs` | (`a`, `b`) PK, `score`, `signals_json` (every signal value), `shared_json` |
| `clusters` | `cluster_id` PK, `subdomain`, `members_json`, `universities_json`, `score` |
| `professor_profiles` | (`author_id`, `inst_id`) PK, `centroid` (blob), `subdomain_counts_json`, `n_papers` |
| `query_log` | `ts`, `question`, `plan_json`, `result_ids_json`, `answer`, `validation_json` |
| `runs` | `stage`, `started`, `finished`, `params_json`, `counts_json` |

### 5.2 Professor attribution
1. Per (paper, target uni): `senior_author` = the last-positioned author whose institution lineage contains that uni; `lead_author` = the first-positioned one.
2. CSRankings verification: ORCID match, else `name_norm` (accent-stripped, lowercase, initials handled) rapidfuzz ≥ 90 **and** affiliation = that uni's `csrankings_name`. This sets `professor_verified`.
3. Unverified: h-index ≥ 15 gives `attribution_confidence = medium`, else `low`. Every RAG answer shows ✔ (verified) or ? (unverified).

---

## 6. Subdomains (`config/subdomains.yaml`)
Each entry has an `id`, a 2–3 sentence `definition` (pasted into LLM prompts), 3–5 `prototypes` (embedded for the pre-filter), a `lexicon` (regex terms) and an `openalex_query` (§4.3).

| ID | Scope |
|---|---|
| `SysML_RL_Caching` | RL/learned models for caching (Redis/KV), DB query optimization, microservice load balancing, cloud scheduling |
| `Edge_RAG_Vector_Systems` | ANN indexing (HNSW/DiskANN), hybrid vector-relational queries, KV-cache quantization, low-latency edge/local RAG |
| `Spatio_Temporal_MARL` | MARL, GNNs, learned spatial indexes for vehicle routing, fleet dispatch, emergency resource allocation |
| `none` | everything else (stored + searchable by meaning, just not LLM-extracted) |

Below the subdomains, **micro-topics** are discovered automatically (§8.3). For example, `Edge_RAG_Vector_Systems` might split into "filtered ANN search", "KV-cache quantization", "on-device RAG"…

---

## 7. Pre-filter (Stage 3) — why and how
- **Why:** gemma3 takes ~4–7 s per paper. ~37k papers would be ~2–3 days of GPU time. Most aren't in our 3 subdomains, so a cheap local pass keeps only plausible candidates, targeting **2,000–5,000** (an overnight LLM run).
- **Signal 1 — keywords:** lexicon hits in title + abstract, title hits ×2.
- **Signal 2 — meaning:** bge-m3 embeds title + abstract (all ~37k, ~6–10 min on GPU). The score is the max cosine to each subdomain's prototype descriptions.
- **Keep rule:** `meaning ≥ τ_c` OR (`keywords ≥ 2` AND `meaning ≥ τ_c − 0.05`). All Query C papers are evaluated the same way.
- **Calibration:** hand-label 150 papers (`data/gold/prefilter.csv`). Choose τ_c for ≥ 90% recall within the survivor budget.
- **Nothing is deleted:** rejected papers stay searchable in the RAG via their abstract vectors; they just lack LLM-extracted fields. The abstract and title vectors computed here are reused by the RAG index.

---

## 8. Similarity engine — finding close papers across universities (core of the RAG)

### 8.1 Signals (each normalized to 0–1)
| # | Signal | How computed | Covers |
|---|---|---|---|
| s1 | `dense_abstract` | bge-m3 cosine of title + abstract | overall meaning |
| s2 | `dense_composite` | bge-m3 cosine of "Problem/Method/Baselines/Datasets/Metrics" composite (extracted papers only) | same problem + same method family |
| s3 | `title_dense` | bge-m3 cosine of titles | **name similarity**: semantic |
| s4 | `title_lexical` | rapidfuzz `token_set_ratio` on stopword-stripped titles, plus bge-m3 sparse match on titles | **name similarity**: shared technical words/acronyms (e.g., "DiskANN", "LSM") |
| s5 | `sparse_lexical` | bge-m3 sparse lexical score on abstract/composite | rare-term overlap |
| s6 | `subdomain` | same `subdomain_cluster` = 1 | the subdomain (hard filter in cross-uni mode, see 8.2) |
| s7 | `micro_topic` | same HDBSCAN micro-topic = 1; else cosine of the two micro-topic centroids | fine-grained topic |
| s8 | `openalex_topics` | weighted Jaccard of OpenAlex `topics` (with scores); same `primary_topic` bonus | OpenAlex's own topic classifier, an independent signal |
| s9 | `keywords` | Jaccard of OpenAlex `keywords` | |
| s10 | `datasets` | Jaccard of normalized datasets | benchmark comparability |
| s11 | `baselines` | Jaccard of normalized baselines | benchmark comparability |
| s12 | `metrics` | Jaccard of normalized metrics | benchmark comparability |
| s13 | `biblio_coupling` | \|refs(A) ∩ refs(B)\| / √(\|refs(A)\|·\|refs(B)\|) | same literature lineage |
| s14 | `citation_link` | A cites B, B cites A, or B ∈ A.related_works | direct relation |
| s15 | `specter2` | cosine of S2 SPECTER2 vectors (survivors only) | citation-informed scientific similarity, a second model independent of bge-m3 |

### 8.2 Combined score
```
S(A,B) = Σ w_i · s_i  /  Σ w_i      (sum only over signals AVAILABLE for both papers, e.g., s2/s10–s12 need extraction, s13 needs refs)
```
Default weights (`config/settings.yaml → similarity.weights`):
`dense_abstract .20 · dense_composite .17 · title_dense .07 · title_lexical .04 · sparse_lexical .07 · micro_topic .06 · openalex_topics .06 · keywords .04 · datasets .07 · baselines .07 · metrics .03 · biblio_coupling .05 · citation_link .02 · specter2 .05`
- **Subdomain handling:** in cross-uni matching modes, subdomain is a **hard filter** (same subdomain, or the subdomain the user asked for). In free search it's a soft multiplier (×0.85 when they differ).
- **Cross-university constraint:** a pair is only valid if A and B share **no** target university (Qdrant `must_not universities MatchAny`).
- **Benchmark eligibility** (for `report` and the "comparable" badge): shares ≥1 dataset, or ≥1 baseline, or ≥2 metrics.
- **Calibration (M6):** hand-label ≥150 pairs as `close / related / unrelated` (`data/gold/pairs.csv`). Fit `LogisticRegression` on the signal vectors to learn the weights; keep the defaults if AUC doesn't improve. τ_close is chosen at the best F1.
- **Explainability:** every match stores `signals_json`. The RAG prints a "why matched" line with the top 3 contributing signals and shared items, e.g. `meaning 0.83 · same micro-topic "filtered ANN" · shared: SIFT1M, DiskANN`.

### 8.3 Micro-topics (precomputed in `match`)
- `HDBSCAN(min_cluster_size=5)` on `abstract_dense` (fallback `composite_dense`) within each subdomain, plus one run over `none`-subdomain papers so general topics are also browsable.
- gemma3 labels each micro-topic from its 8 most central titles (JSON: `label` ≤ 6 words, `description` 1 sentence).
- Stored with the universities they span. A topic spanning ≥3 target unis is a **cross-uni topic**.

### 8.4 Candidate generation (fast, high recall) → re-rank → diversify
1. **Union of candidates:** Qdrant `abstract` top-150 ∪ Qdrant `composite` top-100 (if extracted) ∪ SQLite FTS5 BM25 top-100 (title ×3, keywords ×2, abstract ×1) ∪ same micro-topic members ∪ citation neighbors (refs, related_works) that are in our store. Apply payload filters (subdomain, unis, year, type, has_code…) at this step.
2. **Re-rank** by S (8.2).
3. **University-diverse selection:** MMR (λ = 0.75) with a per-university cap (default 3). Stop when `k` results are selected **and** ≥ `min_universities` (default 3, hard minimum 2) are represented. If fewer universities exist above τ, say so explicitly rather than padding with weak matches.

### 8.5 Professor similarity
`professor_profiles.centroid` = recency-weighted mean of `abstract_dense` over the papers where that person is the senior author at a target uni. "Who works on similar things to Prof X at other unis?" ranks other professors by centroid cosine × subdomain-distribution overlap (+ shared micro-topics), excluding Prof X's own uni.

---

## 9. RAG engine (`ask` / `chat`)

### 9.1 What you can ask (all must work — this is the M7 eval set seed)
- "Find similar work across universities on learned cache eviction for key-value stores."
- "Which unis are working on filtered vector search? Group by university with professors."
- "Show papers from other universities similar to W4391234567" / "…similar to P3" (from the previous answer).
- "ETH vs CMU vs Toronto on KV-cache quantization: who does what, shared baselines?"
- "Who at NUS/NTU does MARL for ride-hailing dispatch, with code, 2026 only?"
- "Which professors in Europe work on topics similar to [Prof name]?"
- "Which datasets and baselines are most used in Edge_RAG_Vector_Systems, and by which unis?"
- "List cross-university micro-topics in SysML_RL_Caching that span ≥4 universities."

### 9.2 Step 1 — gemma3 turns the question into a QueryPlan (Ollama `format` = JSON schema, temperature 0)
```python
class QueryPlan(BaseModel):
    intent: Literal["topic_across_unis", "similar_to_paper", "compare_unis", "find_professors",
                    "similar_professors", "entity_stats", "cross_uni_topics", "paper_detail"]
    semantic_text: str                      # rewritten search text (expanded acronyms, no filler)
    subdomain: Literal["SysML_RL_Caching","Edge_RAG_Vector_Systems","Spatio_Temporal_MARL","any"]
    universities_include: list[str] = []    # free text; resolved by alias table (NOT by the LLM)
    universities_exclude: list[str] = []
    countries: list[str] = []
    min_universities: int = 3
    per_university_limit: int = 3
    year_from: int | None = None
    year_to: int | None = None
    types: list[Literal["conference-paper","preprint"]] = []
    has_code: bool | None = None
    datasets: list[str] = []
    baselines: list[str] = []
    metrics: list[str] = []
    professor: str | None = None
    paper_ref: str | None = None            # W-id, DOI, arXiv id, title fragment, or "P3"
    k: int = 12
```
- The system prompt contains the subdomain definitions, the university short names, and 6 few-shot examples (question → plan).
- **Deterministic post-processing:** university and country names go through the alias table (unknown → warning); entities go through `aliases.yaml`; `paper_ref` is resolved via ID regex, then the chat's previous results, then FTS title match. If `subdomain == any` and the query is clearly about one, the subdomain is inferred from the prototype cosine of `semantic_text` (≥ τ_c).
- CLI flags override the plan: `--uni`, `--exclude-uni`, `--subdomain`, `--from`, `--to`, `--type`, `--has-code`, `--dataset`, `--min-unis`, `--k`, `--intent`.

### 9.3 Step 2 — deterministic execution (Python, no LLM)
| Intent | Execution |
|---|---|
| `topic_across_unis` | embed `semantic_text` (bge-m3 on **CPU**) → §8.4 with query-vs-paper signals (s1, s3–s5, s6–s9, entity overlap with plan entities) → diverse selection → group by university |
| `similar_to_paper` | resolve the paper → read precomputed `pairs` + live §8.4 using the paper as the query → exclude its own universities |
| `compare_unis` | same as the topic mode, restricted to `universities_include`, per-uni cap = k / n_unis, plus a shared-entity table across the unis |
| `find_professors` | topic retrieval → aggregate by (senior_author, uni) → rank by Σ S, show their top papers |
| `similar_professors` | §8.5 |
| `entity_stats` | pure SQL over `entities` × `work_universities` |
| `cross_uni_topics` | SQL over `micro_topics` (span ≥ N unis) |
| `paper_detail` | SQL |

### 9.4 Step 3 — output
1. **Deterministic table** (Python + `rich`, values straight from SQLite):
   `[P#] | University | Professor (✔/?) | Title | Year | Type/Venue | Subdomain › micro-topic | Datasets | Baselines | Paper link | GitHub | why matched`, grouped under a header per university.
2. **Cross-uni summary block** (deterministic): universities covered, shared datasets/baselines/metrics across groups, and which pairs are benchmark-eligible.
3. **gemma3 synthesis**: context = the plan plus the top ≤8 papers as compact JSON (title, uni, professor, abstract trimmed to ~180 words, extracted fields) ≈ 3–5k tokens.
   Options: `temperature 0.2, top_p 0.9, repeat_penalty 1.1, num_ctx 8192, num_predict 768, seed 42`.
   Prompt rules: compare approaches *across universities*; cite `[P#]` for every claim; name the shared datasets/baselines; say "not in the data" when unsure; no outside knowledge about results.
4. **Validator:** every `[P#]` exists; every dataset/baseline/professor/university named appears in the context. Otherwise the sentence is marked ⚠. Everything is logged to `query_log`.

### 9.5 `chat` REPL
Keeps the last plan and results, so follow-ups refine them ("only with code", "drop UCL", "more like P2", "compare P1 and P4", "now 2026 only").
Commands: `:plan` (show the parsed plan), `:why P3` (full signal breakdown), `:open P3` (links), `:filters`, `:save` (append to `notes/YYYY-MM-DD.md`), `:reset`.

### 9.6 Runtime
`ask`/`chat`: gemma3 on GPU; bge-m3 on **CPU** (`devices="cpu"`, query ≈ <1 s). Latency target: plan 2–3 s + retrieval < 1 s + answer 10–20 s.

---

## 10. Pipeline stages (CLI)

| Cmd | Does | GPU | Est. time |
|---|---|---|---|
| `init` | verify 17 ROR→OpenAlex, Ollama model, CUDA, `.env.local` key (print credits remaining), create schema, download + load CSRankings | — | 1 min |
| `harvest` | Queries A + B + C, rebuild abstracts, authorships, `work_universities`, author batches, FTS index | — | 15–30 min |
| `enrich` | derive arXiv IDs → **S2 pass 1** (batch ≤500, 1 req/s global, ≈75 requests) → dedupe (DOI → arXiv ID → S2 paperId → title fuzzy ≥95 + first-author surname; prefer conference-paper) → venue (S2 > hint) → abstract/tldr fill → CSRankings verification → abstract-regex GitHub links | — | 5–8 min |
| `prefilter` | bge-m3 abstract + title vectors (dense + sparse) for all; keyword + prototype scores; Qdrant `all_papers` | bge-m3 | 10–15 min |
| `extract` | gemma3 structured extraction on survivors (schema §10.1; abstract + S2 tldr as input); then HF code lookup and **S2 pass 2** (SPECTER2, ≈4–10 requests) on survivors | gemma3 | 3–8 h (resumable) |
| `index` | normalize entities, composite vectors, Qdrant `relevant_papers`, `professor_profiles` | bge-m3 | 5 min |
| `match` | micro-topics (HDBSCAN + gemma labels), precompute `pairs` (top-20 cross-uni per paper) and `clusters` | CPU + gemma3 | 20–40 min |
| `ask` / `chat` | RAG (§9) | gemma3 (+CPU bge-m3) | interactive |
| `report` | `BENCHMARK_CANDIDATES.md` for top clusters (claims to verify, 3050 stress-test plan) | gemma3 | ~30 min |
| `status` / `run-all` | counts per stage / stages in order with model unloads | | |

### 10.1 Extraction schema (Stage `extract`)
```python
class ReportedResult(BaseModel):
    metric: str; value: str; compared_to: str | None; dataset: str | None
class PaperExtraction(BaseModel):
    sub_problem_solved: str
    core_algorithm_or_architecture: str
    baseline_algorithms_compared: list[str]      # ≤8, only if named in text
    datasets_or_benchmarks_used: list[str]       # ≤8, only if named in text
    evaluation_metrics: list[str]                # ≤8
    reported_results: list[ReportedResult]       # ≤5
    hardware_mentioned: list[str]
    subdomain_cluster: Literal["SysML_RL_Caching","Edge_RAG_Vector_Systems","Spatio_Temporal_MARL","none"]
    is_relevant_cs_systems_or_ml: bool
    confidence: Literal["high","medium","low"]
```
`POST /api/chat`, `format=schema`, options `temperature 0.0, seed 42, num_ctx 4096, num_predict 640`, `keep_alive "30m"`.
Grounding check: drop list items that don't fuzzy-match (partial_ratio ≥ 85) the title or abstract. Retry once with `num_predict 1024` on invalid JSON.
Final subdomain = the LLM label if it agrees with the pre-filter's best cluster or confidence is high; otherwise `needs_review`.

### 10.2 Qdrant (embedded, `QdrantClient(path="./data/qdrant_storage")`, single process)
- `all_papers` (~37k): named vectors `abstract`, `title` (1024, cosine). Payload: `work_id, title, year, type, venue, target_universities, countries, senior_pis, subdomain (best guess), micro_topic_id, has_code, cited_by_count, extracted`.
- `relevant_papers` (~1.5–4k): `abstract`, `title`, `composite`. Payload: the full §5 set.
- Payload indexes: `subdomain_cluster`, `target_universities`, `year`, `type`, `has_code`, `micro_topic_id`.
- Local mode warns above ~20k points. The M5 gate is p95 search < 300 ms on `all_papers`; otherwise swap in a NumPy flat index behind the same `VectorStore` interface.

### 10.3 Ollama settings (Windows user env via `setx`, then restart the Ollama tray app)
`OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`, `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_MAX_LOADED_MODELS=1`.
GPU stages never overlap: before a bge-m3 GPU stage, unload gemma3 (`POST /api/generate {"model":"gemma3:4b-it-qat","keep_alive":0}`); after it, `del model; torch.cuda.empty_cache()`.

---

## 11. Config files (to create in M0)

`config/settings.yaml` (sketch):
```yaml
models: {llm: "gemma3:4b-it-qat", embed: "BAAI/bge-m3", ollama_url: "http://localhost:11434"}
harvest:
  from_publication_date: "2025-01-01"
  types: [conference-paper, preprint]
  field: "fields/17"
  exclude_subfields: []                # e.g. [1707,1709,1704,1711]
  include_proceedings_journals: true   # Query B
  proceedings_sources: [S4387289859, S4210226185, S4210216081, S4210193547, S4387290834, S4404663975]
  keyword_supplement: true             # Query C
  per_page: 100
  min_credits_remaining: 500
enrich: {hf_code_lookup: true, github_search: false}
semantic_scholar:
  enabled: true
  base_url: "https://api.semanticscholar.org/graph/v1"
  rate_limit_rps: 1.0            # CUMULATIVE across ALL S2 endpoints (key-level limit)
  min_interval_s: 1.05           # 5% safety margin
  batch_size: 500                # max IDs per /paper/batch
  backoff: {base_s: 2, cap_s: 60, max_attempts: 6, jitter: true}
  pass1_fields: "externalIds,title,venue,publicationVenue,year,publicationTypes,abstract,tldr,citationCount,influentialCitationCount,openAccessPdf"
  pass2_specter: true            # embedding.specter_v2 for pre-filter survivors
  title_match_fallback: false    # /paper/search/match, 1 req per paper
http:
  openalex_max_rps: 10
prefilter: {tau: {SysML_RL_Caching: 0.55, Edge_RAG_Vector_Systems: 0.55, Spatio_Temporal_MARL: 0.55}, kw_min: 2, target_survivors: [2000, 5000]}
extract: {num_ctx: 4096, num_predict: 640, temperature: 0.0, seed: 42, prompt_version: "x1"}
similarity:
  weights: {dense_abstract: .20, dense_composite: .17, title_dense: .07, title_lexical: .04, sparse_lexical: .07,
            micro_topic: .06, openalex_topics: .06, keywords: .04, datasets: .07, baselines: .07, metrics: .03,
            biblio_coupling: .05, citation_link: .02, specter2: .05}
  tau_close: 0.65
  subdomain_mismatch_penalty: 0.85
  mmr_lambda: 0.75
  per_university_limit: 3
  min_universities: 3
rag: {num_ctx: 8192, num_predict: 768, temperature: 0.2, top_p: 0.9, repeat_penalty: 1.1, context_papers: 8, embed_device: cpu}
```
Plus `config/universities.yaml` (§3), `config/subdomains.yaml` (§6), `config/aliases.yaml` (entity normalization).
(The τ values are placeholders; M3/M6 calibration replaces them.)

---

## 12. Project layout
```
codebase/
├─ plan.md   pyproject.toml   .env.local (exists, gitignored)   .env.example
├─ config/   settings.yaml  universities.yaml  subdomains.yaml  aliases.yaml
├─ src/research_matcher/
│  ├─ cli.py            # typer app: init harvest enrich prefilter extract index match ask chat report status run-all
│  ├─ config.py         # pydantic models for yaml + env
│  ├─ db.py             # schema, migrations, helpers
│  ├─ http_client.py    # httpx + sqlite cache + tenacity + OpenAlex credit guard + key redaction
│  ├─ rate_limit.py     # S2RateLimiter: global 1 req/s token bucket + cross-process SQLite lock
│  ├─ ollama_client.py  # chat(format=schema), unload(), health()
│  ├─ harvest/   openalex.py  abstracts.py  authors.py  csrankings.py
│  ├─ enrich/    dedupe.py  ids.py  venue_hints.py  professors.py  code_links.py  semantic_scholar.py (batch pass 1 + SPECTER2 pass 2)
│  ├─ embed/     bge.py (GPU/CPU, dense+sparse)  prefilter.py
│  ├─ extract/   schema.py  prompts.py  extractor.py  grounding.py  normalize.py
│  ├─ index/     vector_store.py (Qdrant impl + NumPy fallback)  fts.py
│  ├─ similarity/ signals.py  scorer.py  diversify.py  micro_topics.py  professors.py
│  ├─ match/     pairs.py  clusters.py
│  ├─ rag/       query_plan.py  resolve.py  execute.py  render.py  answer.py  validate.py  chat.py
│  └─ report/    benchmark_report.py
├─ tests/        test_abstracts.py test_ids.py test_dedupe.py test_professors.py test_normalize.py test_rate_limit.py
│                test_signals.py test_diversify.py test_query_plan_resolve.py test_validate.py
└─ data/         papers.db  qdrant_storage/  gold/  notes/  cache/
```

---

## 13. Implementation order (milestones & gates)

- [x] **M0a** GPU torch (`2.15.0+cu132`) ✅
- [x] **M0b-key** OpenAlex API key in `.env.local` ✅ (credits header confirmed)
- [x] **M0c-key** Semantic Scholar API key in `.env.local` ✅ (batch call 200 OK; venue + SPECTER2 fields confirmed). Limit **1 req/s cumulative**
- [x] **M0** ✅ gate passed 2026-11-05 (17/17, gemma3, CUDA, credits 9,177, CSRankings 2,311 faculty). deps install; bge-m3 download; Ollama env vars; scaffold §12; `config/*.yaml`; `db.py`; `http_client.py`; `ollama_client.py`; `init`.
  **Gate:** `init` shows 17/17 ✅, gemma3 ✅, CUDA ✅, credits remaining, CSRankings loaded.
- [x] **M1** ✅ 2026-11-05: 37,279 works (A 36,432 / B 642 / C 205), 0 without target uni, 88.9% abstracts, 667 credits. `harvest` (A + B + C). **Gate:** stored ≈ 36,435 + 643 + ~206 (±1%, minus overlaps); abstract-rebuild tests pass; every work has ≥1 target uni; credits used < 1,500.
- [~] **M2** done with 2 gate misses (2026-11-05): S2 match 92.1% ✅, limiter tests ✅, dedupe spot-check ✅, 0 failed; **arXiv ID 89.4% (gate 90%: rest are SSRN/bioRxiv/Zenodo, no arXiv ID exists)**; **professor-verified 30.7% (gate 60%: CSRankings lacks many senior authors incl. well-known CS faculty; non-CS authors)**. Downstream uses `canonical_id = work_id`. `enrich` (incl. S2 pass 1). **Gate:** dedupe spot-check 30; ≥ 60% of senior authors CSRankings-verified; arXiv ID for ≥ 90% of preprints; S2 match rate reported (expect ≥ 80%); **unit test proves the limiter never issues 2 S2 requests within 1 s** (including under retries and two concurrent processes); zero 429s in a normal run.
- [x] **M3** ✅ 2026-11-05: τ=0.58 → 2,882 survivors; `all_papers` 36,568 points. Gold = 135 papers (30 relevant) labelled by Claude, sample skewed to top candidates → recall 100% is optimistic; recall cliff at τ>0.60 (77% @0.625). Re-check at M4/M6. `prefilter` + gold labels + τ calibration. **Gate:** recall ≥ 90%; survivors 2–5k; `all_papers` populated.
- [x] **M4** ✅ 2026-11-06: 2,882/2,882 survivors extracted, 0 JSON failures. Gold (50, Claude-labelled, tuned-on → optimistic): subdomain 88% final (80% raw LLM), entity precision 96% adjudicated (strict vs minimal gold 63%), recall 93%. Output: Edge 274 / MARL 145 / SysML 74 / none 2,389; 365 of 493 relevant are benchmark-eligible; 613 flagged needs_review; SPECTER2 2,375 vectors; HF code links 184. Two LLM calls per paper (entities + few-shot classifier); final label = LLM demoted to none without a lexicon hit (no flipping). Known weak spots: baselines (only 84 of 493 relevant papers name any), MARL set contains generic-MARL papers. `extract` (prompt tuned on a 50-abstract gold set first, then the overnight run). **Gate:** subdomain accuracy ≥ 85%, entity precision ≥ 90%, recall ≥ 70%, JSON failures < 1%.
- [x] **M5** ✅ 2026-11-06: all_papers 36,568 = SQLite, relevant_papers 2,882 = extractions, 4/4 payload-filter smoke tests, 18,539 professor profiles. **Backend swapped Qdrant → NumPy flat index** (`index.backend` in settings): local Qdrant filtered p95 was 492 ms (>300), NumPy is 4.6 ms unfiltered / 35 ms filtered. Qdrant code and its collection are kept (`index.backend: qdrant`). Backend-neutral filter spec in `index/vector_store.py`; BM25 helper in `index/fts.py`. `index`. **Gate:** point counts match SQLite; payload-filter smoke tests; p95 < 300 ms.
- [~] **M6** ⚠ partial 2026-11-06 (see result below) similarity + `match` + micro-topics + weight calibration on 150 pairs. **Gate:** ≥ 70% of the top-20 cross-uni matches judged "close"; micro-topic labels sensible on spot-check.
  - **M6 result (2026-11-06) — code done, gate only partly met:**
    - **Built:** `similarity/{signals,scorer,diversify,micro_topics}.py`, `match/{pairs,clusters,calibrate}.py`, CLI `match` and `match-calibrate`, 7 new tests (40 pass). 14 signals (s6 `subdomain` is the hard filter); candidates = dense top-150 ∪ composite top-100 ∪ BM25 top-100 ∪ micro-topic members ∪ citation neighbours; MMR λ=0.75, ≤3 per university, ≥3 universities or an explicit "only N" note.
    - **Micro-topics:** sklearn `HDBSCAN` (leaf, PCA-32; noise in the 3 subdomains soft-assigned at cos ≥ 0.70). 793 topics (SysML 4, Edge 8, MARL 9, none 772); gemma3 labelled 171 (all subdomain topics + the 150 largest `none` topics, rest keyword fallback). Spot-check: labels are sensible ("KV Cache Compression Techniques", "Graph-Based ANN Search", "Multi-Agent Pathfinding with RL"); some MARL topics are generic-GNN (inherited from M4).
    - **Pairs / clusters:** 9,860 `pairs` (top-20 cross-uni for each of the 493 papers in the 3 subdomains; others are matched live by `ask`). 59 `clusters` from pairs ≥ τ_strong 0.60 (recursive Louvain, ≤12 members), 43 span ≥3 universities; they read as coherent (e.g. KV-cache serving, KV routing in disaggregated serving, KV eviction).
    - **Calibration** on 150 hand-labelled pairs (`data/gold/pairs.csv`; Claude-labelled, 29 close / 89 related / 32 unrelated; strict rubric: *close* = same specific problem + technique family, comparable in a benchmark): 5-fold CV AUC for "close" **0.67 default weights → 0.77 fitted**, so fitted weights are used (`data/gold/weights.json`; title_lexical .24, sparse .15, micro_topic .14, title_dense .10, dense_abstract .10, dense_composite .09, specter2 .08, citation_link .07, datasets .03, biblio .02, rest 0). Best-F1 τ_close = 0.54 (F1 0.62).
    - **Gate (≥70% of top-20 matches "close"): NOT met under the strict rubric.** On 150 calibration pairs 24% of the top-20 slots are close (89% close-or-related); on a **fresh held-out sample of 50 matches above τ_close** (`data/gold/pairs_heldout.csv`) 42% are close, 98% close-or-related. Only the stricter tier τ_strong 0.60 reached 70% (7/10, tiny n) and is used for clusters. Cause: each subdomain has only 74–274 papers, so a top-20 reaches far down; closeness of a rank-15 match is genuinely rarer. Treat `pairs` below τ_strong as "related", not "close".
    - **Caveats:** labels are mine and the sample is small (29 positives), so weights/τ are noisy; micro_topic weight depends on this HDBSCAN run; `none`/unextracted papers have no composite/entity signals.
- [x] **M7** RAG `ask`/`chat`. **Gate (20-question eval):** ≥ 80% return ≥1 expected paper in the top-8; ≥ 90% of topic queries return ≥ 3 universities when they exist; 0 invalid `[P#]`; all uni/professor/link fields come from the DB. ✅ 2026-11-06 (with caveats below)
    - **Built:** `src/research_matcher/rag/` — `query_plan.py` (gemma3 → QueryPlan, 9 few-shot, follow-ups get the previous plan), `resolve.py` (alias table for universities/countries/regions, paper_ref via P#/W-id/DOI/arXiv/FTS, professor fuzzy match, intent safety nets), `execute.py` (all 8 intents, query-vs-paper signals, filters shared between vector store and Python), `render.py` (grouped table + cross-uni summary), `answer.py` (gemma3 synthesis over ≤8 papers), `validate.py` (⚠ marks), `chat.py` (REPL: `:plan :why :open :filters :save :reset`), `evaluate.py`. CLI: `ask` (all override flags, `--no-answer`, `--show-plan`), `chat`, `ask-eval`. `match_paper` gained `extra_flt`/`keep`. 47 tests pass.
    - **Gate result** (`data/gold/rag_eval.json`, 20 questions, run with synthesis): top-8 hit **19/20 (95%)**; ≥3 universities **12/12 topic queries**; invalid `[P#]` **0**; unsupported entity/professor/university terms **0**; display fields come from SQLite via `load_details`.
    - **Post-M7 change 2026-11-07:** synthesis window 8192 → 16384 tokens and context papers 8 → 12 (`rag.num_ctx`, `rag.context_papers`; planner window `rag.plan_num_ctx` 8192). New selection toggle `--mode` / `:mode` / `rag.select_mode`: `diverse` (default), `one_per_uni`, `overall`, `one_uni` (`similarity/diversify.py: select_by_mode`). The M7 gate numbers below were measured with the old 8K/8-paper settings and the default mode.
    - **Caveats:** (1) the "expected paper" check is a regex over title+abstract that I wrote, not a human-labelled paper list. (2) The one miss (Q13, RL for query optimisation) is a corpus gap (1 matching paper), not a retrieval bug. (3) Query-mode thresholds `rag.tau_query 0.40` / `floor_query 0.33` are set by eye from score distributions (relevant 0.4–0.7, nonsense ≈0.25–0.3). (4) Latency is above the plan target: ≈30–40 s per question with synthesis (plan ≈3 s, retrieval ≈3 s after warm-up, answer ≈25 s on this machine). (5) Validator is lexical (known dataset/baseline/professor/university names), so it can't catch wrong claims that use no known name.
- [x] **M8** `report` → `BENCHMARK_CANDIDATES.md`. ✅ 2026-11-06
    - `research-matcher report [--top 10] [--no-draft] [--out FILE]`; code in `src/research_matcher/report/benchmark_report.py`. Takes the `clusters` table, ranks by number of benchmark-comparable cross-university pairs → universities → link score, writes 10 sections. Each has: member table (uni, professor ✔/?, title, year, datasets, baselines, DOI/arXiv/code links), shared datasets/baselines/metrics, "claims to verify" (verbatim rows from `reported_results`, parsed from abstracts), and two gemma3-drafted parts (proposed benchmark, RTX 3050 6 GB stress-test plan) cited with `[P#]` and checked by the `ask` validator (⚠ on failure).
    - Result: 10 clusters, 4 sentences flagged ⚠ in the drafts (e.g. model names like Llama not in the cluster data).
    - **Caveats:** most clusters share few entities (e.g. cluster 6: only `longbench` across 3 of 7 papers), since extraction reads abstracts only; "claims" are raw abstract fragments (some are vague, e.g. "= achieves"); stress-test plans are LLM drafts, not tested; two clusters can share a micro-topic label (cluster id is shown).
- [x] **M9** `run-all` from empty `data/` (cache kept) reproduces the results. ✅ 2026-11-07 (near-identical, not bit-exact)
    - **Full `run-all --fresh --yes` result** (old DB kept as `data/papers.db.bak-*`, counts before in `data/gold/snapshot_before_fresh.json`): all 9 stages exit 0; extraction 2,882/2,882 ok (≈4.8 h at 0.17 papers/s), M5 checks pass (36,569/36,569 and 2,882/2,882 points; p95 4 ms / 41 ms). Counts before → after: works 37,279 → 37,279; prefilter survivors 2,882 → 2,882; extractions 2,882 → 2,882; professor profiles 18,539 → 18,539; clusters 59 → 59; canonical works 36,568 → 36,569; work_universities 40,952 → 40,950; entities 5,981 → 6,001; reported_results 2,859 → 2,848; micro-topics 793 → 795; pairs 9,860 → 9,880. Subdomain split now 276 / 142 / 76 (494 vs 493 relevant before).
    - **Reading:** deterministic stages reproduce exactly (works, survivors, profiles, clusters); the small drift (<1%) sits where gemma3 output or a dedupe tie enters (entities, claims, micro-topic labels/count, one canonical paper). The ranking and gates of earlier milestones were not re-run on the new data; `data/gold` calibrations (pairs, weights) were fitted to the old run.
    - (Earlier partial status below kept for history.)
    - **Built:** `src/research_matcher/pipeline.py` + CLI. `run-all [--from S] [--to S] [--skip S] [--quick] [--dry-run] [--fresh --yes]` runs init → harvest → enrich → prefilter → extract → extract-post → index → match → report, one subprocess per stage (VRAM freed), unloads gemma3 after each, logs to `runs`, and prints a before/after table-count diff. `--fresh --yes` moves `papers.db` to `papers.db.bak-<ts>` (never deletes it), keeps only `http_cache` + `rate_limits` in the new DB, removes embeddings/vectors/qdrant_storage; gold labels and config are kept. `status` shows counts per stage and last run times. 50 tests pass.
    - **Verified:** `run-all --quick` (no network, no re-embedding, no re-labelling) ran all 9 stages with exit 0. Pairs are bit-identical across reruns. It exposed a real bug: clusters changed between runs (58/59/62/63) because Louvain depended on string-hash order; fixed by running it on sorted integer ids → 59 identical clusters over 3 reruns.
    - **NOT verified:** a true `--fresh` run from empty data. It would redo harvest (≈15–30 min), enrich, prefilter embeddings, **extract (3–8 h)**, index, match with gemma3 labels, report. gemma3 runs at temperature 0 / seed 42 so it should be near-identical but is not guaranteed bit-exact; micro-topic labels are LLM text. `--fresh` also drops extractions (they are not in the HTTP cache), so a fresh run re-pays the full extraction time. Run `run-all --fresh --yes` when the machine is free, then compare the diff table.
- [ ] Optional: M10 full-text PDF claims for top clusters (S2/OpenAlex OA PDF URLs) · M11 Gradio web UI on `rag/` · S2 `/paper/search/match` title fallback.

---

## 14. Risks & mitigations
| Risk | Mitigation |
|---|---|
| OpenAlex credits (10k/day) | Cache, credit guard, `select`, few search calls; harvest ≈ 1k credits |
| SIGMOD/VLDB typed as `article` ✅ observed | Query B allow-list |
| MARL/routing cluster thin (~50 strict keyword hits) ✅ observed | Query C + meaning-based pre-filter; report honestly when < 3 unis exist |
| arXiv-only venue labels | S2 `publicationVenue` (✅ verified) + abstract venue hints |
| S2 1 req/s cumulative limit → 429s / throttling | Single global limiter (all endpoints, cross-process), batch endpoint only (≤500 IDs), exponential backoff with `Retry-After`, cache, resumable `s2_failed` batches. Whole S2 workload ≈ 80–90 requests ≈ 2–3 min |
| References missing for 61% | Similarity renormalizes over available signals |
| Gemma3 4B hallucination | JSON schema, grounding check, deterministic tables, validator, gold-set gates |
| Professor mislabels | CSRankings + confidence flag in every answer |
| VRAM contention | Strict stage order; bge-m3 on CPU at query time |
| API key leakage | Keys only in `.env.local`; OpenAlex key redacted in cache keys and logs; S2 key only in a header, never persisted |
