"""Extraction prompts. Two calls per paper: (1) entity extraction, (2) a short subdomain classification with few-shot
examples (a 4B model decides the label far better in its own focused call). Bump settings.extract.prompt_version
whenever this text changes."""
from __future__ import annotations

BODY_RULES = """You extract structured facts from a computer-science paper's title and abstract.

Rules:
- Always output every field; use an empty list when nothing is named.
- Use ONLY what the text states. Never use outside knowledge.
- Every list item must be a proper NAME exactly as written in the text. Never output generic phrases ("three datasets", "benchmark datasets", "math tasks"), numbers or percentages ("16.4%", "10% error bound"), or model names as datasets/metrics.
- baseline_algorithms_compared: named methods or systems the paper compares against or outperforms.
- datasets_or_benchmarks_used: named datasets, benchmarks or workloads evaluated on (e.g. SIFT1M, TPC-H, MuJoCo, GSM8K). LLMs/models are NOT datasets.
- evaluation_metrics: named measures (e.g. recall@10, latency, hit ratio, QPS, accuracy). Not numbers, not model names.
- reported_results: only numeric claims explicitly in the text; value copied as written (e.g. "2.3x", "17%").
- hardware_mentioned: named hardware only (e.g. A100, Raspberry Pi).
- sub_problem_solved: one sentence. core_algorithm_or_architecture: one short phrase naming the method.
- is_relevant_cs_systems_or_ml: true if this is a computer-science systems or machine-learning paper."""

CLASSIFY_RULES = """You classify a computer-science paper into at most one research subdomain, from its title and abstract.
Choose a subdomain ONLY if the paper's own contribution is about that topic; otherwise answer "none". The words "edge", "RAG", "LLM", "reinforcement learning" or "multi-agent" alone are NOT enough.

Subdomains:
- SysML_RL_Caching: caching or cache eviction/replacement, model/content caching (including AI-model caching at the network edge), query optimization, cardinality estimation, load balancing, autoscaling, or cluster/job scheduling (typically with RL or learned models).
- Edge_RAG_Vector_Systems: vector search, approximate nearest neighbor indexes, vector databases; KV-cache management for LLM inference (eviction, quantization, offloading, compression); or the efficiency, latency or serving systems of RAG, including local/on-device RAG.
- Spatio_Temporal_MARL: multi-agent RL or graph neural networks for vehicle routing, coordination of connected/autonomous vehicles, fleet dispatch, ride-hailing, traffic, or emergency/spatial resource allocation.
- none: everything else, including: attacks on or privacy of RAG; RAG answer quality or misinformation; decentralized, distributed or edge LLM inference and quantization without KV-cache or RAG focus; DNN latency estimation; multi-agent LLM prompting; LLM reasoning or alignment with RL; generic RL or MARL algorithms; robotics; computer vision.

confidence: high if the choice is clear from the text, low if guessing."""

# (title, abstract snippet, label) – paraphrased, NOT taken from the gold set
FEWSHOT = [
    ("Learned Admission and Eviction for CDN Caches", "We use deep reinforcement learning to decide which objects to admit to and evict from content delivery network caches, improving byte hit ratio over LRU.", "SysML_RL_Caching"),
    ("Predicate-Aware Graph Index for Filtered Nearest Neighbor Search", "Filtered approximate nearest neighbor queries combine vector similarity with attribute predicates. We extend HNSW so filters are applied during graph traversal.", "Edge_RAG_Vector_Systems"),
    ("Query-Aware Offloading of the KV Cache for Long-Context Serving", "KV cache memory limits batch size in LLM serving. We offload cold KV entries to CPU memory and prefetch the entries each decode step needs.", "Edge_RAG_Vector_Systems"),
    ("Cooperative Order Dispatching for Ride-Hailing with Multi-Agent RL", "Each driver is an agent; we learn a shared policy for matching drivers and requests across a city to reduce passenger waiting time.", "Spatio_Temporal_MARL"),
    ("Robust Retrieval-Augmented QA Against Noisy Passages", "We train the generator to ignore irrelevant retrieved passages, improving exact-match accuracy on open-domain QA benchmarks.", "none"),
    ("Splitting Transformer Layers Across Volunteer Devices", "We shard a language model across consumer devices connected by slow links and compress the activations sent between them.", "none"),
    ("Credit Assignment in Cooperative Multi-Agent Reinforcement Learning", "We propose a value decomposition method for cooperative MARL and evaluate on StarCraft multi-agent challenge maps.", "none"),
    ("Poisoning the Knowledge Base of Retrieval-Augmented Generation", "We craft malicious documents that, once retrieved, steer the LLM toward attacker-chosen answers.", "none"),
    ("Proactive Placement of Generative Models on Edge Servers with Deep RL", "We cache AI models on base stations and allocate bandwidth jointly using an actor-critic agent to maximise model hit ratio.", "SysML_RL_Caching"),
    ("Neural Architecture Search for Graph Classification", "We search per-layer aggregation operators for GNNs and report accuracy on citation networks.", "none"),
]


def body_system_prompt() -> str:
    return BODY_RULES


def classify_messages(title: str, abstract: str | None) -> list[dict]:
    msgs = [{"role": "system", "content": CLASSIFY_RULES}]
    for t, a, label in FEWSHOT:
        msgs.append({"role": "user", "content": f"Title: {t}\nAbstract: {a}"})
        msgs.append({"role": "assistant", "content": '{"subdomain_cluster": "%s", "confidence": "high"}' % label})
    msgs.append({"role": "user", "content": user_prompt(title, abstract, None)})
    return msgs


def user_prompt(title: str, abstract: str | None, tldr: str | None) -> str:
    body = (abstract or "").strip()
    if len(body) > 2600:
        body = body[:2600] + "…"
    parts = [f"Title: {title}"]
    parts.append(f"Abstract: {body}" if body else "Abstract: (not available)")
    if tldr and not body:
        parts.append(f"TL;DR: {tldr}")
    return "\n".join(parts)
