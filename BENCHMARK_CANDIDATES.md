# Benchmark candidates

Generated 2026-11-07 from 10 cross-university clusters (clusters = papers linked at score ≥ 0.6; ranked by number of benchmark-comparable cross-university pairs, then universities, then score). Papers, professors, links, datasets, baselines and claims come straight from the database; claims are parsed from abstracts and must be checked against the papers before use. ✔ = professor verified against CSRankings, ? = not verified.

## 1. KV Cache Compression & Management (cluster 4)

*Cluster 4 · Edge_RAG_Vector_Systems · 6 papers · 5 universities (COL, ETH, NTU, NUS, UofT) · mean link score 0.66 · 1/14 cross-university pairs are benchmark-comparable*

| Ref | University | Professor | Title | Year | Datasets | Baselines | Links |
|---|---|---|---|---|---|---|---|
| P1 | NUS | Zikang Zhou? (NUS) | RateQuant: Optimal Mixed-Precision KV Cache Quantization via Rate-Distortion Theory | 2026 | — | — | [doi](https://doi.org/10.48550/arxiv.2605.06675) [arXiv](https://arxiv.org/abs/2605.06675) |
| P2 | ETH | Yawei Li? (ETH) | RDKV: Rate-Distortion Bit Allocation for Joint Eviction and Quantization of the KV Cache | 2026 | infinitebench, longbench, ruler | — | [doi](https://doi.org/10.48550/arxiv.2605.08317) [arXiv](https://arxiv.org/abs/2605.08317) |
| P3 | UofT | Sijie Han? (UofT) | Hurwitz Quaternion Multiplicative Quantization for KV Cache Compression | 2026 | coqa, gsm8k, truthfulqa | kivi-4 | [doi](https://doi.org/10.48550/arxiv.2605.27646) [arXiv](https://arxiv.org/abs/2605.27646) |
| P4 | NTU | Weichen Liu✔ (NTU) | SlimKV: Joint Token-Feature KV Cache Compression with Reconstruction-Free Beacon Attention | 2026 | longbench, needle-in-a-haystack | — | [doi](https://doi.org/10.18653/2026.emnlp-main.983) [arXiv](https://arxiv.org/abs/2610.02953) |
| P5 | NUS | Xiaobin Hu? (NUS) | Last But Not Least: Boundary Attention CalibratiON for Multimodal KV Cache Compression | 2026 | — | — | [doi](https://doi.org/10.18653/2026.emnlp-main.331) [arXiv](https://arxiv.org/abs/2606.14782) |
| P6 | COL | Yang, Jiayi? (COL) | PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs | 2025 | 7b-70b long-cot llms | — | [doi](https://doi.org/10.48550/arxiv.2505.18610) [arXiv](https://arxiv.org/abs/2505.18610) |

**Shared across papers**
- datasets: longbench (P2, P4)

**Claims to verify** (reported in the abstracts, taken verbatim from the extraction; unverified)
- P4: score = 16×/32× compression vs uncompressed model on LongBench
- P4: score = 4×/8× compression vs uncompressed model on LongBench
- P4: score = over 96% vs uncompressed model on LongBench
- P4: attention speedup = 7.34× vs uncompressed model on Needle-in-a-Haystack
- P4: end-to-end decoding speedup = 3.38× vs uncompressed model on Needle-in-a-Haystack
- P6: reasoning benchmark performance = 8%
- P6: throughput = 2.73-5.18x
- P1: KIVI perplexity = 49.3

> Drafted by gemma3 from the fields above; sentences marked ⚠ failed the citation / name check.

### Proposed Benchmark
*   The common task is KV cache compression for large language models, aiming to reduce memory usage during inference. [P1], [P2], [P3], [P4], and [P5] all address this problem. [P6] focuses on CoT LLMs specifically.
*   Several papers share the use of rate-distortion theory as a guiding principle for quantization, particularly in P1 and P2. [P1] uses reverse waterfilling from rate-distortion theory, while [P2] employs rate-distortion bit allocation.
*   A fair comparison would involve evaluating each method on the same datasets (e.g., longbench, infinitebench) using comparable metrics like perplexity (PPL), cache retention, and decode speedup. [P2], [P3], and [P4] all use these datasets. [P1] uses kivi perplexity. [P5] focuses on decoding latency and KV-cache compression.
*   The benchmark should assess the trade-off between compression ratio (memory reduction) and performance degradation (e.g., accuracy, speed). [P2], [P3], and [P4] provide metrics for this trade-off.
### Stress-test Plan for an RTX 3050 Laptop (6 GB VRAM)
*   Papers with methods likely to be feasible on the hardware include P3 and potentially P4, given their simpler quantization techniques. [P3] uses Hurwitz Quaternion Multiplicative Quantization which is less computationally intensive than some other approaches. [P4]’s low-rank-aware training might also be manageable. [P1], [P2], and [P5] are likely too complex due to the rate-distortion methods or boundary attention calibration.
*   To reduce memory usage, we should shrink the dataset size significantly – perhaps using a subset of longbench or coqa. [P3] would benefit from this most.
*   We could also experiment with smaller batch sizes (e.g., reducing batch size to 1 or 2) and potentially reduce model size if possible (though this is less controllable). [P3] might be the best candidate for these adjustments.
*   Key numbers to compare would include PPL, zero-shot accuracy (for P3), and decoding latency (for P5). [P3] provides PPL and zero-shot accuracy; [P5] focuses on decoding latency.
We should prioritize metrics that are sensitive to memory constraints. ⚠['sensitive' not in context]

## 2. KV Cache Compression & Management (cluster 9)

*Cluster 9 · Edge_RAG_Vector_Systems · 6 papers · 5 universities (CAM, CMU, GT, NUS, TUM) · mean link score 0.64 · 1/14 cross-university pairs are benchmark-comparable*

| Ref | University | Professor | Title | Year | Datasets | Baselines | Links |
|---|---|---|---|---|---|---|---|
| P1 | CMU | Yidi Miao? (CMU) | CONF-KV: Confidence-Aware KV Cache Eviction with Mixed-Precision Storage for Long-Horizon LLM | 2026 | needle-in-a-haystack, visualwebarena | — | [doi](https://doi.org/10.48550/arxiv.2605.24786) [arXiv](https://arxiv.org/abs/2605.24786) |
| P2 | CMU | Virginia Smith✔ (CMU) | EpiKV: Epiphany-Aware KV Cache Eviction Without the Attention Matrix | 2026 | — | — | [doi](https://doi.org/10.48550/arxiv.2606.26472) [arXiv](https://arxiv.org/abs/2606.26472) |
| P3 | TUM | Peng Xie? (TUM) | Error Certificates for KV-Cache Eviction via Randomized Design | 2026 | longbench | — | [doi](https://doi.org/10.48550/arxiv.2607.21475) [arXiv](https://arxiv.org/abs/2607.21475) |
| P4 | GT | Tianfang Xie? (GT) | Beyond Mean Attention: Diversity-Aware, Layer-Wise Scoring for KV Cache Eviction | 2026 | english longbench | — | [doi](https://doi.org/10.48550/arxiv.2609.30738) [arXiv](https://arxiv.org/abs/2609.30738) |
| P5 | CAM | Young D. Kwon? (CAM) | Monte Carlo Estimation for KV Cache Eviction | 2026 | andpro, longbench, ruler | — | [doi](https://doi.org/10.48550/arxiv.2610.07643) [arXiv](https://arxiv.org/abs/2610.07643) |
| P6 | NUS | Ye Dong? (NUS) | MPCache: MPC-Friendly KV Cache Eviction for Efficient Private LLM Inference | 2025 | — | — | [doi](https://doi.org/10.52202/085713-3251) [arXiv](https://arxiv.org/abs/2501.06807) |

**Shared across papers**
- datasets: longbench (P3, P5)

**Claims to verify** (reported in the abstracts, taken verbatim from the extraction; unverified)
- P5: Qwen2.5-14B average = 45.49 vs 48.24
- P5: Mistral-7B average = 45.20 vs 51.05
- P5: RULER margin over AnDPro = 11.65 vs 128K contexts
- P5: LORE-KV time = 1.46-2.77x vs AnDPro
- P6: decoding latency = 1.8 ~ 2.01x
- P6: communication reduction = 3.39 ~ 8.37x
- P1: perplexity = 1.5--2.1
- P1: retrieval accuracy = 91.4%

> Drafted by gemma3 from the fields above; sentences marked ⚠ failed the citation / name check.

### Proposed benchmark
*   The papers address the challenge of KV cache eviction in long-horizon LLM inference and reasoning models [P1, P2, P3, P4, P5].
*   Several papers share the use of datasets like `needle-in-a-haystack`, `visualwebarena`, `longbench`, `english longbench`, `andpro`, and `ruler` [P1, P3, P4, P5].
*   A fair comparison would focus on evaluating metrics such as `peak memory`, `perplexity`, `retrieval accuracy`, `accuracy`, `auc`, `confidence`, `recall`, `macro`, and `ruler margin` [P1, P2, P3, P4, P5].
*   The primary goal is to compare the effectiveness of different KV eviction strategies in reducing GPU memory consumption and improving inference performance [P1, P2, P3, P4, P5].
### Stress-test plan for an RTX 3050 Laptop (6 GB VRAM)
*   Papers with methods likely feasible on a 6GB RTX 3050 include P3 ([Error Certificates for KV-Cache Eviction via Randomized Design]) and potentially P5 ([Monte Carlo Estimation for KV Cache Eviction]) due to their relatively simple algorithms. [P3, P5]
*   To reduce memory usage, consider shrinking the dataset size (e.g., using smaller subsets of `longbench`) and reducing batch sizes. [P3, P5]
*   Focus on comparing metrics like `auc`, `confidence`, `recall` (for P3) and `ruler margin` (for P5), as these are less computationally intensive than metrics like `peak memory` or `perplexity`. [P3, P5]  The `accuracy` metric from P2 is likely too demanding.

## 3. KV Cache Compression & Management (cluster 3)

*Cluster 3 · Edge_RAG_Vector_Systems · 7 papers · 5 universities (CMU, GT, IMP, NTU, NUS) · mean link score 0.63 · 1/19 cross-university pairs are benchmark-comparable*

| Ref | University | Professor | Title | Year | Datasets | Baselines | Links |
|---|---|---|---|---|---|---|---|
| P1 | GT | Brady Steele? (GT) | On the Limits of Learned Importance Scoring for KV Cache Compression | 2026 | — | — | [doi](https://doi.org/10.48550/arxiv.2601.14279) [arXiv](https://arxiv.org/abs/2601.14279) |
| P2 | NUS | Xuanyao Fong? (NUS) | LOKI: An LLM Accelerator with Optimized KV Cache and In-Memory Computing Cores | 2026 | — | nvidia a100 gpu | [doi](https://doi.org/10.1109/iscas66217.2026.11562728) |
| P3 | CMU | Shaorui Pi? (CMU) | CacheRoute: KV-Cache-Aware Routing for LLM Backend Services on Kubernetes | 2026 | lmsys-chat-1m dataset | — | [doi](https://doi.org/10.1109/eitce70137.2026.11634378) |
| P4 | GT | Ahmed Saeed✔ (GT) | ARK: Avoiding Routing Collisions for KV Cache Transfer in Disaggregated LLM Inference | 2026 | — | — | [doi](https://doi.org/10.1145/3789240.3828750) [arXiv](https://arxiv.org/abs/2610.41160) |
| P5 | NTU | Dayu Wang? (NTU) | AnchorKV: Training-Free KV Cache Compression via Joint Coverage Diversity and Semantic Importance Scoring | 2026 | longbench | — | [doi](https://doi.org/10.18653/2026.emnlp-main.2428) |
| P6 | IMP | Hongxiang Fan? (IMP) | VEDA: Efficient LLM Generation Through Voting-based KV Cache Eviction and Dataflow-flexible Accelerator | 2025 | — | — | [doi](https://doi.org/10.1109/dac63849.2025.11132861) [arXiv](https://arxiv.org/abs/2507.00797) |
| P7 | CMU | Ziqi Liu? (CMU) | Accurate KV Cache Eviction via Anchor Direction Projection for Efficient LLM Inference | 2025 | longbench | — | [doi](https://doi.org/10.52202/085713-0350) |

**Shared across papers**
- datasets: longbench (P5, P7)
- metrics: throughput (P2, P3)

**Claims to verify** (reported in the abstracts, taken verbatim from the extraction; unverified)
- P6: hardware complexity = O(1)
- P6: hardware complexity = O(N)
- P2: energy efficiency = 155.8×
- P2: throughput = 1.87×
- P7: cache accuracy = 96.07%
- P7: KV cache budget = 3.44%
- P7: KV cache budget reduction = 46.0%
- P3: p95 TTFT = 11–43%

> Drafted by gemma3 from the fields above; sentences marked ⚠ failed the citation / name check.

### Proposed benchmark
*   The papers investigate different approaches to KV cache compression and routing, all aiming to improve LLM inference efficiency. [P1, P2, P3, P4, P5, P7]
*   Several papers share the use of metrics like "kv-cache hit rates" and "throughput". [P3, P5]
*   A fair comparison would focus on evaluating the impact of each method on KV cache utilization and inference speed. [P1, P2, P3, P4, P5, P7]
*   The datasets used are primarily focused on long-context LLMs, with "lmsys-chat-1m dataset" and "longbench" being prominent examples. ⚠['lmsys chat 1m' not in context] [P3, P5, P7]
### Stress-test plan for an RTX 3050 Laptop (6 GB VRAM)
*   Papers likely to be feasible: P1, P4, and potentially P7 due to their focus on simpler techniques like importance scoring or routing. [P1, P4, P7]
*   Shrink dataset size significantly – consider using a subset of "longbench" or creating a smaller synthetic dataset. [P5, P7]
*   Reduce batch size to the minimum possible while still allowing for reasonable throughput.
Experiment with batch sizes as low as 1 or 2. [P1, P4, P7]
*   Model size should be kept minimal – consider using a smaller LLM (e.g., one with fewer parameters) if available. [P1, P4, P7]
*   Compare metrics like "mean ttft" (P4), "cache accuracy" (P7), and potentially simplified versions of "kv-cache hit rates" or "throughput" if feasible to compute given the limited VRAM. [P4, P7]

## 4. KV Cache Compression & Management (cluster 0)

*Cluster 0 · Edge_RAG_Vector_Systems · 3 papers · 4 universities (NUS, SFU, UBC, UW) · mean link score 0.64 · 1/3 cross-university pairs are benchmark-comparable*

| Ref | University | Professor | Title | Year | Datasets | Baselines | Links |
|---|---|---|---|---|---|---|---|
| P1 | SFU/UBC | Yuzhen Mao? (UBC); Ke Li✔ (SFU) | IceCache: Memory-efficient KV-cache Management for Long-Sequence LLMs | 2026 | longbench | — | [doi](https://doi.org/10.48550/arxiv.2604.10539) [arXiv](https://arxiv.org/abs/2604.10539) [code](https://github.com/yuzhenmao/IceCache) |
| P2 | NUS | Hui Chen? (NUS) | PrefixKV: Adaptive Prefix KV Cache is What Vision Instruction-Following Models Need for Efficient Generation | 2025 | — | — | [doi](https://doi.org/10.52202/085713-3159) [arXiv](https://arxiv.org/abs/2412.03409) [code](https://github.com/THU-MIG/PrefixKV) |
| P3 | UW | Falahati, Ali? (UW) | SentenceKV: Efficient LLM Inference via Sentence-Level Semantic KV Caching | 2025 | longbench, needle-in-a-haystack, pg-19 | — | [doi](https://doi.org/10.48550/arxiv.2504.00970) [arXiv](https://arxiv.org/abs/2504.00970) [code](https://github.com/zzbright1998/SentenceKV) |

**Shared across papers**
- datasets: longbench (P1, P3)

**Claims to verify** (reported in the abstracts, taken verbatim from the extraction; unverified)
- P2: state-of-the-art performance = achieves
- P1: accuracy = 99%
- P1: KV cache token budget = 25%

> Drafted by gemma3 from the fields above; sentences marked ⚠ failed the citation / name check.

### Proposed benchmark
*   The papers share the task of improving LLM inference efficiency, specifically addressing memory and computational bottlenecks related to KV caches [P1, P2, P3].
*   All three papers utilize the `longbench` dataset as a common evaluation platform [P1, P3]. [P2] does not specify a dataset.
*   A fair comparison would focus on measuring inference latency and memory usage under similar hardware conditions [P1, P3].
The generation quality metric in [P2] is relevant but less directly comparable due to the different problem addressed. [P2] focuses on vision-language models.
*   Comparing the effectiveness of their respective caching strategies (semantic token clustering vs.
adaptive layer-wise retention vs.
sentence-level semantic KV caching) will be key [P1, P2, P3].
### Stress-test plan for an RTX 3050 Laptop (6 GB VRAM)
*   **P1:** The `PagedAttention` method is likely feasible to rerun on a 6GB GPU, particularly with reduced batch sizes and potentially smaller model sizes [P1].
*   **P2:** The adaptive layer-wise KV retention recipe might be challenging due to the complexity of the approach.
Consider shrinking the dataset size or reducing the number of layers processed in each iteration [P2].
*   **P3:** Sentence-level semantic KV caching could also be feasible with reduced batch sizes and smaller model sizes, focusing on a subset of the `longbench` dataset [P3].
*   Comparing inference latency (seconds per token) and memory usage (GB) at different batch sizes (e.g., 1, 2, 4) will provide valuable insights into feasibility [P1, P3].
Model size should be kept minimal to fit within the VRAM limit.

## 5. Vector Search Acceleration Techniques (cluster 19)

*Cluster 19 · Edge_RAG_Vector_Systems · 3 papers · 4 universities (EPFL, ETH, GT, UW) · mean link score 0.63 · 1/3 cross-university pairs are benchmark-comparable*

| Ref | University | Professor | Title | Year | Datasets | Baselines | Links |
|---|---|---|---|---|---|---|---|
| P1 | GT | Kexin Rong✔ (GT) | HONEYBEE: Efficient Role-based Access Control for Vector Databases via Dynamic Partitioning | 2026 | — | — | [doi](https://doi.org/10.1145/3786625) [arXiv](https://arxiv.org/abs/2505.01538) |
| P2 | EPFL/ETH | Anıl Eren Göçer? (ETH); Anastasia Ailamaki✔ (EPFL) | QVCache: A Query-Aware Vector Cache | 2026 | — | — | [doi](https://doi.org/10.48550/arxiv.2602.02057) [arXiv](https://arxiv.org/abs/2602.02057) |
| P3 | UW | Ihab F. Ilyas✔ (UW) | Quake: Adaptive Indexing for Vector Search | 2025 | wikipedia | diskann, hnsw, scann, svs | [doi](https://doi.org/10.48550/arxiv.2506.03437) [arXiv](https://arxiv.org/abs/2506.03437) |

**Shared across papers**
- metrics: query latency (P1, P2, P3); recall (P1, P2)

**Claims to verify** (reported in the abstracts, taken verbatim from the extraction; unverified)
- P1: query latency = 13.5X lower
- P1: memory usage = 1.24X increase
- P1: additional memory consumption = 90.4% reduction
- P3: query latency = 1.5-38x
- P3: update latency = 4.5-126x
- P2: query latency reduction = up to 40-1000x
- P2: recall = comparable to the underlying ANN backend

> Drafted by gemma3 from the fields above; sentences marked ⚠ failed the citation / name check.

### Proposed benchmark
*   The papers aim to improve vector search performance. [P1], [P2], and [P3] all focus on this task.
*   A fair comparison requires evaluating query latency and recall across different datasets. [P1], [P2], and [P3] report these metrics. [P3] specifically includes update latency as a metric. [P2] also reports cache-hit latency.
*   The common dataset is "wikipedia" which is used by [P3].
The other papers do not specify datasets, so a comparison would need to use a shared dataset or multiple ones. [P1] and [P2] have no specified datasets.
*   A fair comparison should consider the baselines used by each paper (e.g., diskann, hnsw, scann, svs for [P3]) to understand how their methods compare against established techniques. [P3] uses these as baselines.
### Stress-test plan for an RTX 3050 Laptop (6 GB VRAM)
*   [P2]'s method of "semantic query repetition" appears feasible to rerun on a 6GB GPU, given its relatively simple implementation and lack of hardware requirements specified in the data [P2].
*   To assess feasibility, consider shrinking the dataset size.
Reducing the dataset to a smaller subset (e.g., 10% or 25%) would be a good starting point for [P2] [P2].
*   Reduce batch size significantly – perhaps down to 1 or 2 – to minimize memory usage. [P2]
*   Compare cache-hit latency and query latency, focusing on the smallest dataset and lowest batch size. [P2]
*   [P1]’s “dynamic partitioning” method likely requires more computational resources than are available on a laptop GPU, so it should be excluded from this stress test [P1].
Similarly, [P3]'s multi-level partitioning scheme is unlikely to be feasible given the hardware constraints. [P3]

## 6. Multi-Agent LLM KV Caching (cluster 10)

*Cluster 10 · Edge_RAG_Vector_Systems · 4 papers · 3 universities (COL, IMP, UCSD) · mean link score 0.65 · 1/4 cross-university pairs are benchmark-comparable*

| Ref | University | Professor | Title | Year | Datasets | Baselines | Links |
|---|---|---|---|---|---|---|---|
| P1 | UCSD | Tajana Rosing✔ (UCSD) | AgentKVShift: Efficient KV Cache Reuse for Agentic Memory Systems | 2026 | — | — | [doi](https://doi.org/10.48550/arxiv.2607.21604) [arXiv](https://arxiv.org/abs/2607.21604) |
| P2 | COL/IMP | Yiren Zhao? (IMP); Bowen Yang? (COL) | AgentKV: Phase-Aware KV Eviction for Agentic LLMs | 2026 | — | r-kv, sglang, tri-attention | [doi](https://doi.org/10.18653/2026.emnlp-main.2514) [arXiv](https://arxiv.org/abs/2609.14872) |
| P3 | COL | Suhas Kumar Morisetty? (COL) | WAKMan: Workload Aware KVCache Manager | 2026 | 20 mixes | lru, s3fifo | [doi](https://doi.org/10.48550/arxiv.2610.36355) [arXiv](https://arxiv.org/abs/2610.36355) |
| P4 | UCSD | Yufei Ding✔ (UCSD) | KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows | 2025 | — | sglang | [doi](https://doi.org/10.52202/085713-4208) [arXiv](https://arxiv.org/abs/2507.07400) |

**Shared across papers**
- baselines: sglang (P2, P4)

**Claims to verify** (reported in the abstracts, taken verbatim from the extraction; unverified)
- P4: speedup = 1.83x
- P4: speedup = 2.19x
- P1: recompute ratio = 10-30%
- P1: recompute ratio = 45-55%
- P1: prefill speedups = 2-3.5x
- P1: recompute ratio = 5x lower
- P1: F1 = over 2x
- P2: task score = 5.5 points

> Drafted by gemma3 from the fields above; sentences marked ⚠ failed the citation / name check.

### Proposed benchmark
*   The papers share the task of optimizing KV caches in agentic LLM systems [P1, P2, P3].
*   Several papers utilize metrics like `f1` and `output-token throughput` to evaluate their methods [P1, P2].
*   A fair comparison would involve evaluating each method's performance on the same datasets (though none are explicitly listed) using the specified baselines [P2, P3]. [P2] uses r-kv, sglang and tri-attention as baselines. [P3] compares against lru and s3fifo.
*   The primary goal is to assess which method offers the most significant improvement in serving efficiency for agentic LLMs [P1, P2, P3].
### Stress-test plan for an RTX 3050 Laptop (6 GB VRAM)
*   **Feasible Papers:**  `P3`'s workload-aware KV cache manager is likely the most feasible to rerun due to its simpler method and reliance on metrics like `cache hits` [P3].
`P4`’s workflow-aware KV cache management might also be possible with significant shrinking, but depends on implementation details. [P4]
*   **Shrinkable Parameters:** Dataset size should be drastically reduced (e.g., to a small subset of the "20 mixes" dataset) and batch sizes significantly lowered [P3].
Model size would need to be minimized – potentially using smaller models or quantized versions [P3, P4].
*   **Numbers to Compare:** Focus on comparing `cache hits` (if possible for P3), `optimality gap` (if available), and any reported speedup metrics [P3].
If feasible, compare the performance of a reduced model size with the original. [P3]

## 7. KV Cache Compression & Management (cluster 2)

*Cluster 2 · Edge_RAG_Vector_Systems · 8 papers · 9 universities (CAM, ETH, GT, IMP, NUS, UCL, UCSD, UW, UofT) · mean link score 0.62 · 0/26 cross-university pairs are benchmark-comparable*

| Ref | University | Professor | Title | Year | Datasets | Baselines | Links |
|---|---|---|---|---|---|---|---|
| P1 | CAM/NUS | Dianbo Liu? (NUS); Qingyu Shi? (CAM) | VQKV: High-Fidelity and High-Ratio Cache Compression via Vector-Quantization | 2026 | longbench | — | [doi](https://doi.org/10.48550/arxiv.2603.16435) [arXiv](https://arxiv.org/abs/2603.16435) |
| P2 | GT | Rajeev Patwari? (GT) | KV Pareto: Systems-Level Optimization of KV Cache and Model Compression for Long Context Inference | 2026 | — | — | [doi](https://doi.org/10.18653/v1/2026.eacl-industry.9) [arXiv](https://arxiv.org/abs/2512.01953) |
| P3 | UCSD | Mingu Kang✔ (UCSD) | STAR-KV: Low-Rank KV Cache Compression via Soft Thresholding for Adaptive Rank Control | 2026 | llms | — | [doi](https://doi.org/10.48550/arxiv.2606.08382) [arXiv](https://arxiv.org/abs/2606.08382) [code](https://github.com/PriyanshBhatnagar/STAR-KV) |
| P4 | UCL | Quan V. Dang? (UCL) | DynaCalKV: Key-Value Cache Compression via Head Grouping and Adaptive Rank Allocation | 2026 | — | — | [doi](https://doi.org/10.48550/arxiv.2607.24331) [arXiv](https://arxiv.org/abs/2607.24331) |
| P5 | GT/UW/UofT | Xiang Li? (GT); Wenjuan Guo? (UW); Yue Ning? (UofT) | Reasoning-Aware Error-Bounded KV-Cache Compression and Sparse Attention for Long-Context LLMs | 2026 | 1,000 long-context traces, 4k to 32k tokens | — | [doi](https://doi.org/10.21203/rs.3.rs-10952127/v1) |
| P6 | UCSD | Farinaz Koushanfar✔ (UCSD) | Key, Value, Compress: A Systematic Exploration of KV Cache Compression Techniques | 2025 | — | — | [doi](https://doi.org/10.1109/cicc63670.2025.10983416) [arXiv](https://arxiv.org/abs/2503.11816) |
| P7 | IMP | Che Liu? (IMP) | MEDA: Dynamic KV Cache Allocation for Efficient Multimodal Long-Context Inference | 2025 | — | — | [doi](https://doi.org/10.18653/v1/2025.naacl-long.125) |
| P8 | ETH | Yekun Chai? (ETH) | EvolKV: Evolutionary KV Cache Compression for LLM Inference | 2025 | gsm8k | — | [doi](https://doi.org/10.18653/v1/2025.findings-emnlp.88) [arXiv](https://arxiv.org/abs/2509.08315) |

**Shared across papers**
- metrics: performance (P1, P6, P8)

**Claims to verify** (reported in the abstracts, taken verbatim from the extraction; unverified)
- P8: performance = 7 percentage points
- P8: memory efficiency = 1.5%
- P1: compression ratio = 82.8%
- P1: reconstruction fidelity = 98.6%
- P1: generation length = 4.3x longer
- P3: KV cache compression = 75%
- P3: KV cache reduction = 20x
- P3: speedup for the attention module = 6.9x

> Drafted by gemma3 from the fields above; sentences marked ⚠ failed the citation / name check.

### Proposed benchmark
*   The common task is to compress the Key-Value (KV) cache in Large Language Models (LLMs) to improve inference performance and reduce memory usage. [P1, P2, P3, P4, P5, P6]
*   Several papers share the use of the `longbench` dataset for evaluating compression ratios and performance. [P1, P3]
*   A fair comparison would assess the compression ratio achieved by each method alongside its impact on overall model performance (e.g., generation throughput or accuracy). [P1, P3, P5]
*   The metrics to compare include compression ratio, performance, reconstruction fidelity, end-to-end generation throughput, KV cache reduction, speedup for the attention module, and inference latency. [P1, P3, P5, P6]
### Stress-test plan for an RTX 3050 Laptop (6 GB VRAM)
*   Papers with methods likely to be feasible on a 6GB GPU include P3 (STAR-KV) due to its low-rank compression and P8 (EvolKV) using evolutionary search, though the latter might require significant parameter tuning. [P3, P8]
*   To make the models runnable, consider shrinking the dataset size – potentially using a subset of `llms` or `gsm8k`. [P3, P8]
*   Reduce batch sizes to minimize VRAM usage.
A small batch size (e.g., 1) might be necessary. [P3, P8]
*   Compare the inference latency and memory efficiency achieved with reduced model sizes and smaller datasets. [P3, P8]

## 8. KV Cache Compression & Management (cluster 8)

*Cluster 8 · Edge_RAG_Vector_Systems · 12 papers · 7 universities (CMU, COL, ETH, GT, NTU, NUS, TUM) · mean link score 0.64 · 0/56 cross-university pairs are benchmark-comparable*

| Ref | University | Professor | Title | Year | Datasets | Baselines | Links |
|---|---|---|---|---|---|---|---|
| P1 | COL | Yanxuan Yu? (COL) | CXL-SpecKV: A Disaggregated FPGA Speculative KV-Cache for Datacenter LLM Serving | 2026 | — | — | [doi](https://doi.org/10.1145/3748173.3779188) [arXiv](https://arxiv.org/abs/2512.11920) |
| P2 | TUM | Ulf Schlichtmann? (TUM) | KV Packet: Recomputation-Free Context-Independent KV Caching for LLMs | 2026 | — | — | [doi](https://doi.org/10.48550/arxiv.2604.13226) [arXiv](https://arxiv.org/abs/2604.13226) [code](https://github.com/ChuangtaoChen-TUM/KVPacket) |
| P3 | NTU | Chengzhi Lu? (NTU) | High Throughput and Low Latency LLM Serving via Adaptive KV Caching | 2026 | — | — | [doi](https://doi.org/10.1145/3767295.3803570) |
| P4 | NUS | Qilong Feng? (NUS) | CacheFlow: Efficient LLM Serving via Automated 3D-Parallel KV Cache Restoration | 2026 | — | — | [doi](https://doi.org/10.48550/arxiv.2604.25080) [arXiv](https://arxiv.org/abs/2604.25080) |
| P5 | ETH | Gustavo Alonso✔ (ETH) | ObjectCache: Layerwise Object-Storage Retrieval for KV Cache Reuse | 2026 | ceph rgw, daos, nixl | — | [doi](https://doi.org/10.48550/arxiv.2605.22850) [arXiv](https://arxiv.org/abs/2605.22850) |
| P6 | CMU/COL | Shuo Wang? (CMU); Zhizhen Zhu? (COL) | SLO-Aware Disaggregated LLM Serving Over Heterogeneous Edge-Cloud Clusters | 2026 | lmsys-chat-1m, longbench | distserve, mooncake | [doi](https://doi.org/10.1109/aiotc71250.2026.11689386) |
| P7 | GT | Tushar Krishna✔ (GT) | DynamoServe: A Distributed Tiered Memory System for Multi-tenant LLM Serving | 2026 | — | — | [doi](https://doi.org/10.1145/3789240.3829347) |
| P8 | COL | Yanxuan Yu? (COL) | To Keep or Not to Keep: Learning KV Cache Retention in Disaggregated LLM Serving Systems | 2026 | — | — | [doi](https://doi.org/10.1145/3793230.3837769) [arXiv](https://arxiv.org/abs/2610.28637) |
| P9 | NUS | Yuxin Cao? (NUS) | Preserving Provenance in Shared KV Caches for LLM Serving | 2026 | sglang 0.5.20, vllm | sglang, vllm | [doi](https://doi.org/10.48550/arxiv.2609.38706) [arXiv](https://arxiv.org/abs/2609.38706) |
| P10 | GT | Shimeng Yu✔ (GT) | An Optical KV Cache Network for Disaggregated LLM Serving | 2026 | — | — | [doi](https://doi.org/10.48550/arxiv.2610.18444) [arXiv](https://arxiv.org/abs/2610.18444) |
| P11 | NUS | Yangjie Zhou? (NUS) | eLLM: Elastic Memory Management Framework for Efficient LLM Serving | 2025 | — | — | [doi](https://doi.org/10.48550/arxiv.2506.15155) [arXiv](https://arxiv.org/abs/2506.15155) |
| P12 | COL | Yanxuan Yu? (COL) | TinyServe: Query-Aware Cache Selection for Efficient LLM Serving | 2025 | — | — | [doi](https://doi.org/10.1145/3746027.3758181) [arXiv](https://arxiv.org/abs/2509.12211) |

**Shared across papers**
- metrics: ttft (P2, P5, P10); latency (P5, P7, P9); accuracy (P1, P9); throughput (P1, P7)

**Claims to verify** (reported in the abstracts, taken verbatim from the extraction; unverified)
- P11: decoding throughput = 2.32x higher
- P11: batch size = 3x larger
- P12: speedup = 3.4×
- P12: memory savings = 2×
- P12: accuracy drop = negligible
- P1: throughput = 3.2×
- P1: memory costs = 2.8×
- P2: TTFT = lower

> Drafted by gemma3 from the fields above; sentences marked ⚠ failed the citation / name check.

### Proposed benchmark
Here's a proposed benchmark based on the provided JSON data:
*   **Common Task:** All papers address challenges related to efficiently serving Large Language Models (LLMs) by optimizing key-value (KV) cache management.
Specifically, they aim to reduce memory costs, latency, and improve throughput during LLM inference.
*   **Datasets/Baselines/Metrics:** Several papers share the use of datasets like `lmsys-chat-1m`, `longbench`, `sglang 0.5.20`, `vllm`.
Common metrics include: `accuracy`, `f1 scores`, `flops`, `ttft`, `end-to-end serving latency`, `restoration latency`, `goodput`, `kv cache transfer volume`, `slo violation rates`, `latency`, `network transfer`, and `accuracy drop`.
Baselines often involve comparing against existing systems like `distserve` and `mooncake`.
*   **Fair Comparison:** A fair comparison would focus on reproducing results using the same datasets, metrics, and hardware configurations.
It’s important to note that many papers lack code availability, making replication difficult.
The use of shared datasets is crucial for establishing a common ground. [P6], [P9]
### Stress-test plan for an RTX 3050 Laptop (6 GB VRAM)
Here's a stress test plan considering the limited resources of an RTX 3050 laptop:
*   **Feasible Papers:**  Papers with simpler methods and minimal hardware requirements are most likely to be runnable.
This includes [P2], [P3], [P8], [P11] and potentially [P12].
*   **Shrinkable Components:** To fit within the 6GB VRAM limit, we should prioritize shrinking:
    *   **Dataset Size:** Reduce the dataset size significantly.
Using smaller subsets of `lmsys-chat-1m` or `longbench` would be a good starting point. [P6]
    *   **Batch Size:** Lowering the batch size dramatically reduces memory consumption. [P11]
    *   **Model Size:** If possible, use smaller LLM models (e.g., smaller versions of Llama 2) to reduce KV cache size. ⚠['llama' not in context] [P9], [P10]
*   **Metrics to Compare:** Given the constraints, focus on comparing:
    *   `latency`: Measure inference time for different configurations. [P2], [P3], [P8]
    *   `ttft`: Track the Time-To-First Token (TTFT) as a measure of initial latency. [P2], [P5]
    *   `accuracy drop`: If possible, evaluate the impact on model accuracy when using reduced configurations. [P12]
Note: Papers like [P4], [P5], [P7], [P10] and [P12] are likely too demanding for the RTX 3050 laptop due to their complex methods or hardware requirements. [P1] and [P9] require significant computational resources and may not be feasible without specialized hardware.

## 9. Graph Retrieval-Augmented Generation (cluster 22)

*Cluster 22 · Edge_RAG_Vector_Systems · 7 papers · 7 universities (CAM, GT, NTU, NUS, OXF, UBC, UofT) · mean link score 0.61 · 0/19 cross-university pairs are benchmark-comparable*

| Ref | University | Professor | Title | Year | Datasets | Baselines | Links |
|---|---|---|---|---|---|---|---|
| P1 | CAM/UBC | Zirun Xu? (UBC); Kuanyan Zhu? (CAM) | AutoPrunedRetriever: Persistent Reasoning Graphs for Efficient and Extensible Retrieval-Augmented Generation | 2026 | — | — | [doi](https://doi.org/10.48550/arxiv.2602.04926) [arXiv](https://arxiv.org/abs/2602.04926) |
| P2 | NTU | Haoyang Zhong? (NTU) | A Unified Framework for Context-Aware and Relation-Aware Graph Retrieval-Augmented Generation | 2026 | — | — | [doi](https://doi.org/10.1145/3774904.3792720) [arXiv](https://arxiv.org/abs/2606.18075) |
| P3 | NUS | Yatao Bian? (NUS) | Question-Adaptive Graph Learning for Multi-hop Retrieval Augmented Generation | 2026 | — | — | [doi](https://doi.org/10.1145/3805712.3809693) [arXiv](https://arxiv.org/abs/2510.11541) [code](https://github.com/Jerry2398/QSGNN) |
| P4 | UofT | Rahul C. Shetty? (UofT) | Enhancing Context-Aware Search with Retrieval-Augmented Generation | 2025 | — | bm25, slm embeddings | [doi](https://doi.org/10.36227/techrxiv.174060240.09460752/v1) |
| P5 | OXF | Philip H.S. Torr✔ (OXF) | Can Knowledge-Graph-based Retrieval Augmented Generation Really Retrieve What You Need? | 2025 | stark benchmark | gpt-4o | [doi](https://doi.org/10.52202/085713-3200) [arXiv](https://arxiv.org/abs/2510.16582) |
| P6 | NTU/NUS | Qika Lin? (NUS); Luu Anh Tuan✔ (NTU) | HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation | 2025 | — | — | [doi](https://doi.org/10.52202/085713-5089) [arXiv](https://arxiv.org/abs/2503.21322) [code](https://github.com/LHRLAB/HyperGraphRAG) |
| P7 | GT | Siqi Miao? (GT) | Weak-to-Strong GraphRAG: Aligning Weak Retrievers with Large Language Models for Graph-based Retrieval Augmented Generation | 2025 | — | — | [doi](https://doi.org/10.48550/arxiv.2506.22518) [arXiv](https://arxiv.org/abs/2506.22518) |

**Shared across papers**
- metrics: accuracy (P1, P2)

**Claims to verify** (reported in the abstracts, taken verbatim from the extraction; unverified)
- P4: relevance score = 0.500
- P4: relevance score = 0.7668
- P4: relevance score = 0.9202
- P5: hit rate = 10%
- P5: recall = 10%
- P1: accuracy = 1.8-10.8 points
- P1: input tokens = 96%
- P1: input tokens = 60-96%

> Drafted by gemma3 from the fields above; sentences marked ⚠ failed the citation / name check.

### Proposed benchmark
*   The common task is retrieval-augmented generation (RAG), specifically focusing on leveraging knowledge graphs for improved performance [P1, P2, P3, P5, P6].
*   Several papers share the use of accuracy as a primary metric [P1, P2, P3, P6].
Some also consider input tokens or retrieval efficiency [P1, P6].
*   A fair comparison would need to evaluate how well each system retrieves relevant information and generates coherent responses based on that retrieved knowledge. [P1, P2, P5]
*   The datasets used are primarily focused on benchmark datasets like the Stark Benchmark [P5], though specific dataset usage varies between papers [P1, P2, P3, P6]. [P5]
### Stress-test plan for an RTX 3050 Laptop (6 GB VRAM)
*   Papers with methods likely to be feasible include P3 and P6 due to their relatively simple graph representation learning frameworks [P3, P6].
*   To reduce memory requirements, consider shrinking the dataset size.
For P3, reducing the number of nodes/edges in the graph would be beneficial [P3].
For P6, smaller hypergraphs could be tested.
*   Reduce batch size to 1 or even smaller if necessary [P3, P6].
*   Compare metrics like "answer accuracy" (P6) and “performance” (P3), focusing on whether the system can still produce reasonable results with limited resources.
The primary focus should be on demonstrating feasibility rather than achieving state-of-the-art performance [P3, P6].

## 10. Retrieval-Augmented Generation Optimization (cluster 23)

*Cluster 23 · Edge_RAG_Vector_Systems · 8 papers · 6 universities (CMU, EPFL, ETH, NTU, NUS, UCSD) · mean link score 0.64 · 0/26 cross-university pairs are benchmark-comparable*

| Ref | University | Professor | Title | Year | Datasets | Baselines | Links |
|---|---|---|---|---|---|---|---|
| P1 | UCSD | Tajana Rosing✔ (UCSD) | HeteroRAGCache: Software-Hardware Co-Design for Efficient RAG Caching using Emerging Memories | 2026 | — | — | [doi](https://doi.org/10.1145/3787109.3815268) |
| P2 | NTU | Chaitanya Dhananjay Jadhav? (NTU) | Joint Optimization of Secure and Energy Efficient Retrieval Augmented Generation for Mobile Edge Computing | 2026 | — | — | [doi](https://doi.org/10.1109/infocom59046.2026.11571647) |
| P3 | EPFL | Martijn de Vos? (EPFL) | Leveraging Approximate Caching for Faster Retrieval-Augmented Generation | 2025 | medrag, mmlu | — | [doi](https://doi.org/10.1145/3721146.3721941) |
| P4 | ETH | Wenqi Jiang? (ETH) | PipeRAG: Fast Retrieval-Augmented Generation via Adaptive Pipeline Parallelism | 2025 | — | — | [doi](https://doi.org/10.1145/3690624.3709194) |
| P5 | NUS | Dan Chen? (NUS) | HeterRAG: Heterogeneous Processing-in-Memory Acceleration for Retrieval-augmented Generation | 2025 | — | — | [doi](https://doi.org/10.1145/3695053.3731089) |
| P6 | ETH | Gustavo Alonso✔ (ETH) | RAGO: Systematic Performance Optimization for Retrieval-Augmented Generation Serving | 2025 | — | — | [doi](https://doi.org/10.1145/3695053.3731093) [arXiv](https://arxiv.org/abs/2503.14649) |
| P7 | UCSD | Hadi Esmaeilzadeh✔ (UCSD) | In-Storage Acceleration of Retrieval Augmented Generation as a Service | 2025 | — | — | [doi](https://doi.org/10.1145/3695053.3731032) |
| P8 | CMU | Yuwei An? (CMU) | HyperRAG: Enhancing Quality-Efficiency Tradeoffs in Retrieval-Augmented Generation with Reranker KV-Cache Reuse | 2025 | — | — | [doi](https://doi.org/10.48550/arxiv.2504.02921) [arXiv](https://arxiv.org/abs/2504.02921) |

**Shared across papers**
- metrics: throughput (P7, P8)

**Claims to verify** (reported in the abstracts, taken verbatim from the extraction; unverified)
- P3: retrieval latency = up to 59%
- P6: QPS = 2ˆincrease
- P6: time-to-first-token latency = 55% reduction
- P7: throughput = 4.3×
- P7: throughput = 1.5×
- P2: attack-success rate = 50%
- P2: SCR = 1.8×

> Drafted by gemma3 from the fields above; sentences marked ⚠ failed the citation / name check.

### Proposed benchmark
*   The common task across these papers is improving the efficiency and/or speed of Retrieval Augmented Generation (RAG). [P2, P3, P4, P5, P6, P7]
*   Several papers share the use of datasets like "medrag" and "mmlu" for evaluation. [P3]
*   A fair comparison would need to evaluate metrics such as retrieval latency, generation latency, attack success rate, qps, throughput, recall, and downstream performance. [P2, P3, P4, P6, P7]
*   The papers’ methods are diverse (co-design, approximate caching, pipeline parallelism, heterogeneous PIM, RAGO), requiring careful consideration of how to meaningfully compare them. [P1, P2, P3, P4, P5, P6]
### Stress-test plan for an RTX 3050 Laptop (6 GB VRAM)
*   Papers with methods likely feasible on the hardware are P3 and potentially P4 due to their focus on caching or pipeline parallelism. [P3, P4]
*   To reduce dataset size, consider using smaller subsets of "medrag" or "mmlu". [P3]
*   Reducing batch size would be crucial for fitting models within the 6GB VRAM limit. [P3, P4]
*   Comparing metrics like retrieval latency and recall (for P3) is feasible, as these are less computationally intensive than generation quality or attack success rate. [P3]
