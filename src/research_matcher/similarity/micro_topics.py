"""Micro-topics: HDBSCAN on abstract vectors within each subdomain (+ one run over `none`), labelled by gemma3 (plan §8.3)."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from typing import Callable

import numpy as np

from .signals import Corpus

SOFT_ASSIGN = 0.70
SUBDOMAINS = ["SysML_RL_Caching", "Edge_RAG_Vector_Systems", "Spatio_Temporal_MARL", "none"]
LABEL_SCHEMA = {"type": "object", "properties": {"label": {"type": "string"}, "description": {"type": "string"}},
                "required": ["label", "description"]}


def cluster_group(vecs: np.ndarray, min_cluster_size: int = 5, pca_dims: int = 32) -> np.ndarray:
    """HDBSCAN on PCA-reduced, L2-normalised vectors (euclidean ~ cosine). Returns labels, -1 = noise."""
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA

    if len(vecs) < min_cluster_size * 2:
        return np.full(len(vecs), -1)
    x = PCA(n_components=min(pca_dims, len(vecs) - 1), random_state=0).fit_transform(vecs)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=2, metric="euclidean",
                   cluster_selection_method="leaf", copy=True).fit_predict(x)


def label_prompt(titles: list[str]) -> list[dict]:
    body = "\n".join(f"- {t}" for t in titles)
    return [{"role": "system", "content": "You name research topics. Given paper titles that belong to one cluster, give a short topic "
                                          "label (at most 6 words, a noun phrase naming the specific technical topic, no generic words "
                                          "like 'research' or 'systems' alone) and a one-sentence description of what these papers share."},
            {"role": "user", "content": f"Titles in the cluster:\n{body}"}]


def fallback_label(corpus: Corpus, members: list[int]) -> str:
    kw = Counter(k for m in members for k in corpus.meta[m]["keywords"])
    return ", ".join(k for k, _ in kw.most_common(3)) or "unlabelled"


def central_titles(corpus: Corpus, members: list[int], n: int = 8) -> list[str]:
    cent = corpus.abs[members].mean(axis=0)
    cent /= np.linalg.norm(cent) or 1.0
    order = np.argsort(-(corpus.abs[members] @ cent))[:n]
    return [corpus.meta[members[i]]["title"] for i in order]


def build(conn: sqlite3.Connection, corpus: Corpus, client=None, *, label_none_max: int = 150,
          log: Callable[[str], None] = print) -> dict:
    """Cluster each subdomain, write micro_topics + works.micro_topic_id/label. `client` None = keyword fallback labels only.
    For `none` only the label_none_max largest topics (those spanning >=3 universities first) get a gemma label."""
    groups: dict[str, list[int]] = {s: [] for s in SUBDOMAINS}
    for i, m in enumerate(corpus.meta):
        groups[m["subdomain"] if m["subdomain"] in groups else "none"].append(i)
    topics: list[dict] = []
    assign: dict[int, int] = {}
    for sub in SUBDOMAINS:
        idx = groups[sub]
        lab = cluster_group(corpus.abs[idx])
        if sub != "none":  # soft-assign HDBSCAN noise to the nearest topic when it is clearly close (cos >= 0.70)
            cs = sorted(set(lab) - {-1})
            if cs:
                cents = np.stack([corpus.abs[[idx[k] for k in np.nonzero(lab == c)[0]]].mean(axis=0) for c in cs])
                cents /= np.linalg.norm(cents, axis=1, keepdims=True)
                for k in np.nonzero(lab == -1)[0]:
                    sims = cents @ corpus.abs[idx[k]]
                    if sims.max() >= SOFT_ASSIGN:
                        lab[k] = cs[int(sims.argmax())]
        log(f"  {sub}: {len(idx)} papers -> {len(set(lab) - {-1})} topics, {int((lab == -1).sum())} noise")
        for c in sorted(set(lab) - {-1}):
            members = [idx[k] for k in np.nonzero(lab == c)[0]]
            tid = len(topics)
            cent = corpus.abs[members].mean(axis=0)
            cent = (cent / (np.linalg.norm(cent) or 1.0)).astype(np.float32)
            unis = Counter(u for m in members for u in corpus.unis[m])
            topics.append({"topic_id": tid, "subdomain": sub, "members": members, "centroid": cent,
                           "universities": sorted(unis), "n_unis": len(unis)})
            for m in members:
                assign[m] = tid
    # labelling order: subdomain topics first, then `none` topics by (spans>=3 unis, size)
    todo = [t for t in topics if t["subdomain"] != "none"]
    none_t = sorted([t for t in topics if t["subdomain"] == "none"], key=lambda t: (-(t["n_unis"] >= 3), -len(t["members"])))
    todo += none_t[:label_none_max]
    done = 0
    for t in topics:
        t["label"], t["description"] = fallback_label(corpus, t["members"]), ""
    if client is not None:
        for t in todo:
            try:
                r = client.chat_json(label_prompt(central_titles(corpus, t["members"])), LABEL_SCHEMA,
                                     {"temperature": 0.1, "seed": 7, "num_ctx": 4096, "num_predict": 160})
                t["label"], t["description"] = r["label"].strip()[:80], r["description"].strip()
                done += 1
            except Exception as e:  # noqa: BLE001 keep fallback label
                log(f"  label failed for topic {t['topic_id']}: {e}")
            if done and done % 25 == 0:
                log(f"  labelled {done}/{len(todo)}")
    if "description" not in {r[1] for r in conn.execute("PRAGMA table_info(micro_topics)")}:
        conn.execute("ALTER TABLE micro_topics ADD COLUMN description TEXT")
    conn.execute("BEGIN")
    conn.execute("DELETE FROM micro_topics")
    conn.execute("UPDATE works SET micro_topic_id=NULL, micro_topic_label=NULL")
    for t in topics:
        conn.execute("INSERT INTO micro_topics VALUES (?,?,?,?,?,?,?)",
                     (t["topic_id"], t["subdomain"], t["label"], len(t["members"]), json.dumps(t["universities"]),
                      t["centroid"].tobytes(), t["description"]))
    conn.executemany("UPDATE works SET micro_topic_id=?, micro_topic_label=? WHERE work_id=?",
                     [(tid, topics[tid]["label"], corpus.ids[i]) for i, tid in assign.items()])
    conn.execute("COMMIT")
    return {"topics": len(topics), "labelled_by_llm": done, "assigned": len(assign),
            "cross_uni_topics": sum(t["n_unis"] >= 3 for t in topics),
            "by_subdomain": {s: sum(t["subdomain"] == s for t in topics) for s in SUBDOMAINS}}
