"""Cross-university clusters of close papers, built from the precomputed `pairs` graph (plan §10 `match`)."""
from __future__ import annotations

import json
import sqlite3

import networkx as nx

from ..config import load_settings
from ..similarity.signals import Corpus


def build_clusters(conn: sqlite3.Connection, corpus: Corpus, *, tau: float | None = None, max_size: int = 12) -> dict:
    """Edges = pairs with score >= tau_strong. Components are split with recursive Louvain when larger than max_size.
    Only clusters spanning >= 2 target universities are kept."""
    tau = tau if tau is not None else load_settings().similarity.get("tau_strong", 0.60)
    g = nx.Graph()
    for a, b, s in conn.execute("SELECT a, b, score FROM pairs WHERE score >= ? ORDER BY a, b", (tau,)):
        g.add_edge(a, b, weight=s)
    # Louvain on integer node ids in sorted order: string-hash randomisation otherwise changes the partition between runs
    order = sorted(g.nodes)
    ix = {w: i for i, w in enumerate(order)}
    h = nx.Graph()
    h.add_nodes_from(range(len(order)))
    h.add_weighted_edges_from((ix[a], ix[b], d["weight"]) for a, b, d in sorted(g.edges(data=True), key=lambda e: (ix[e[0]], ix[e[1]])))
    groups: list[set[str]] = []

    def split(nodes: set[str], depth: int = 0) -> None:
        if len(nodes) <= max_size or depth >= 4:
            groups.append(nodes)
            return
        parts = nx.algorithms.community.louvain_communities(h.subgraph(sorted(ix[n] for n in nodes)), weight="weight", resolution=2.0 + depth, seed=0)
        if len(parts) == 1:
            groups.append(nodes)
            return
        for p in sorted(parts, key=min):
            split({order[i] for i in p}, depth + 1)

    for comp in sorted(nx.connected_components(g), key=lambda q: sorted(q)[0]):
        split(set(comp))
    conn.execute("BEGIN")
    conn.execute("DELETE FROM clusters")
    n = 0
    for members in groups:
        unis = sorted({u for m in members for u in corpus.unis[corpus.pos[m]]})
        if len(members) < 2 or len(unis) < 2:
            continue
        sub = max({corpus.meta[corpus.pos[m]]["subdomain"] for m in members},
                  key=lambda s: sum(corpus.meta[corpus.pos[m]]["subdomain"] == s for m in members))
        sc = [d["weight"] for _, _, d in g.subgraph(members).edges(data=True)]
        conn.execute("INSERT INTO clusters VALUES (?,?,?,?,?)",
                     (n, sub, json.dumps(sorted(members)), json.dumps(unis), sum(sc) / len(sc) if sc else 0.0))
        n += 1
    conn.execute("COMMIT")
    sizes = sorted((len(json.loads(r[0])) for r in conn.execute("SELECT members_json FROM clusters")), reverse=True)
    return {"clusters": n, "largest": sizes[:5], "tau": tau}
