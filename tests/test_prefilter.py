import numpy as np

from research_matcher.embed.prefilter import keep_rule, keyword_score, meaning_scores
from research_matcher.embed.bge import sparse_score

LEX = ["reinforcement learning", "cach(e|ing)", "query optimi[sz](er|ation)"]


def test_keyword_score_title_counts_double():
    assert keyword_score("Learned caching", "uses reinforcement learning", LEX) == 3
    assert keyword_score("Nothing", "reinforcement learning for caching", LEX) == 2
    assert keyword_score("x", "y", LEX) == 0


def test_keep_rule():
    assert keep_rule(0.60, 0, 0.55, 2)             # meaning alone
    assert keep_rule(0.52, 2, 0.55, 2)             # keywords rescue a near miss
    assert not keep_rule(0.52, 1, 0.55, 2)         # not enough keywords
    assert not keep_rule(0.45, 5, 0.55, 2)         # keywords can't rescue a far miss


def test_meaning_scores_max_over_prototypes():
    protos = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32)
    docs = np.array([[1, 0], [0, 1]], dtype=np.float32)
    m = meaning_scores(docs, protos, owner=[0, 0, 1], n_clusters=2)
    assert m.shape == (2, 2)
    assert m[0, 0] == 1.0 and abs(m[0, 1] - 0.6) < 1e-6 and abs(m[1, 1] - 0.8) < 1e-6


def test_sparse_score():
    assert abs(sparse_score({"a": 0.5, "b": 1.0}, {"b": 0.5, "c": 2.0}) - 0.5) < 1e-9
    assert sparse_score({}, {"a": 1.0}) == 0.0
