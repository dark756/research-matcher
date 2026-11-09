import sqlite3
from datetime import date

import numpy as np

from research_matcher.db import SCHEMA
from research_matcher.index.fts import bm25_search, fts_query
from research_matcher.index.vector_store import NumpyStore, build_collection
from research_matcher.similarity.professors import recency_weight


def test_fts_query_is_safe():
    q = fts_query("Learned cache eviction (DiskANN) for the KV-cache: AND NOT *")
    assert '"Learned"' in q and '"DiskANN"' in q and '"KV-cache"' in q and '"the"' not in q
    assert fts_query("the of") == ""


def test_bm25_prefers_title_and_skips_noncanonical():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    for wid, canon, title, abstract in [("W1", "W1", "Vector search at scale", "x"), ("W2", "W2", "Other", "vector search vector"),
                                         ("W3", "W1", "Vector search at scale", "x")]:
        c.execute("INSERT INTO works(work_id, canonical_id, title, abstract, keywords_json) VALUES (?,?,?,?,'[]')",
                  (wid, canon, title, abstract))
        c.execute("INSERT INTO works_fts(title, keywords, abstract, work_id) VALUES (?,?,?,?)", (title, "", abstract, wid))
    got = [w for w, _ in bm25_search(c, "vector search", 5)]
    assert got[0] == "W1" and "W3" not in got  # title weight 3x beats abstract repeats; duplicate row excluded


def test_numpy_store_filters_and_ranking(tmp_path):
    s = NumpyStore(str(tmp_path))
    ids = ["a", "b", "c", "d"]
    vecs = np.eye(4, dtype=np.float32)
    pay = {"a": {"u": ["ETH", "CMU"], "year": 2025, "code": True}, "b": {"u": ["NUS"], "year": 2026, "code": False},
           "c": {"u": ["ETH"], "year": 2026, "code": True}, "d": {"u": [], "year": 2024, "code": False}}
    build_collection(s, "t", ids, {"v": vecs}, pay)
    q = np.array([0.9, 0.5, 0.4, 0.0], dtype=np.float32)
    assert [h.payload["u"] for h in s.search("t", "v", q, 2)][0] == ["ETH", "CMU"]
    got = s.search("t", "v", q, 10, {"must_not": [{"key": "u", "any": ["ETH"]}]})
    assert {h.id for h in got} == {1, 3}                       # list field: ANY element match, then excluded
    got = s.search("t", "v", q, 10, {"must": [{"key": "year", "gte": 2026}, {"key": "code", "eq": True}]})
    assert [h.id for h in got] == [2]
    assert s.search("t", "v", q, 10, {"must": [{"key": "u", "any": ["MIT"]}]}) == []
    s2 = NumpyStore(str(tmp_path))                              # persisted
    assert s2.count("t") == 4 and s2.search("t", "v", q, 1)[0].id == 0


def test_recency_weight_half_life():
    today = date(2026, 11, 6)
    assert abs(recency_weight("2025-11-06", today) - 0.5) < 0.01
    assert recency_weight("2026-11-06", today) > recency_weight("2025-01-01", today)
    assert recency_weight(None, today) == 0.5
