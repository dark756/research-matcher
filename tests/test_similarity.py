import numpy as np

from research_matcher.similarity.diversify import select_diverse
from research_matcher.similarity.scorer import benchmark_eligible, combine
from research_matcher.similarity.signals import SIGNALS, jaccard, sparse_norm, title_tokens, weighted_jaccard


def test_jaccard_and_nan_when_empty():
    assert jaccard({"a", "b"}, {"b", "c"}) == 1 / 3
    assert np.isnan(jaccard(set(), {"a"}))          # unavailable, not 0
    assert weighted_jaccard({"t1": 1.0}, {"t1": 0.5, "t2": 0.5}) == 0.5 / 1.5


def test_sparse_norm_bounds():
    a = {"x": 1.0, "y": 2.0}
    assert abs(sparse_norm(a, a, 5.0, 5.0) - 1.0) < 1e-9
    assert np.isnan(sparse_norm({}, a, 0, 5.0))


def test_title_tokens_strips_stopwords():
    assert title_tokens("A Study of the DiskANN Index for Filtered Search") == "study diskann index filtered search"


def test_combine_renormalises_over_available_signals():
    w = {s: 0.0 for s in SIGNALS}
    w["dense_abstract"], w["datasets"] = 0.5, 0.5
    row = np.full((2, len(SIGNALS)), np.nan)
    row[0, SIGNALS.index("dense_abstract")] = 0.8                      # datasets unavailable -> only s1 counts
    row[1, SIGNALS.index("dense_abstract")] = 0.8
    row[1, SIGNALS.index("datasets")] = 0.0                            # datasets available and disjoint -> averaged in
    out = combine(row, w)
    assert abs(out[0] - 0.8) < 1e-9 and abs(out[1] - 0.4) < 1e-9


def test_benchmark_eligibility():
    assert benchmark_eligible({"datasets": ["sift1m"]})
    assert benchmark_eligible({"baselines": ["hnsw"]})
    assert not benchmark_eligible({"metrics": ["recall"]})
    assert benchmark_eligible({"metrics": ["recall", "qps"]})


def _unit(n, d=8, seed=0):
    v = np.random.default_rng(seed).normal(size=(n, d))
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def test_diversify_caps_per_university_and_reaches_min_unis():
    # 6 strong candidates from MIT, then weaker ones from other unis
    unis = [frozenset({"MIT"})] * 6 + [frozenset({"ETH"}), frozenset({"NUS"}), frozenset({"CMU"})]
    scores = np.array([0.9, 0.89, 0.88, 0.87, 0.86, 0.85, 0.7, 0.69, 0.68])
    sel, note = select_diverse(scores, _unit(9), unis, k=5, per_uni=3, min_unis=3, tau=0.6)
    picked = [unis[i] for i in sel]
    assert sum(u == frozenset({"MIT"}) for u in picked) <= 3
    assert len({u for p in picked for u in p}) >= 3 and note is None


def test_selection_modes():
    from research_matcher.similarity.diversify import select_by_mode

    unis = [frozenset({"MIT"}), frozenset({"MIT"}), frozenset({"MIT"}), frozenset({"ETH"}), frozenset({"NUS"}), frozenset({"ETH"})]
    scores = np.array([0.9, 0.85, 0.8, 0.7, 0.6, 0.5])
    kw = dict(k=4, per_uni=3, min_unis=3, tau=0.0)
    assert select_by_mode("overall", scores, _unit(6), unis, **kw)[0] == [0, 1, 2, 3]
    sel, note = select_by_mode("one_per_uni", scores, _unit(6), unis, **kw)
    assert sel == [0, 3, 4] and "only 3 universities" in note          # never two from one university
    sel, note = select_by_mode("one_uni", scores, _unit(6), unis, **kw)
    assert sel[:3] == [0, 1, 2] and sel[3] == 3 and "MIT" in note       # MIT first, then best of the others
    assert select_by_mode("diverse", scores, _unit(6), unis, **kw)[0] == select_diverse(scores, _unit(6), unis, **kw)[0]


def test_diversify_does_not_pad_with_weak_matches():
    unis = [frozenset({"MIT"})] * 4 + [frozenset({"ETH"})]
    scores = np.array([0.9, 0.89, 0.88, 0.87, 0.3])                    # the only other uni is far below tau
    sel, note = select_diverse(scores, _unit(5), unis, k=3, per_uni=3, min_unis=3, tau=0.6)
    assert 4 not in sel and note and "not padding" in note      # k reached from MIT; the weak ETH paper is not pulled in
