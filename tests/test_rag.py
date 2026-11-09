import sqlite3
from types import SimpleNamespace

import numpy as np

from research_matcher.rag.execute import make_filters, query_signals
from research_matcher.rag.query_plan import FEW_SHOT, QueryPlan, build_messages
from research_matcher.rag.resolve import Resolved, resolve_places, scan_unis
from research_matcher.rag.validate import CITE, validate
from research_matcher.similarity.signals import SIGNALS


def test_few_shot_plans_are_valid_and_cover_all_intents():
    intents = {QueryPlan.model_validate_json(p).intent for _, p in FEW_SHOT}
    assert intents == set(QueryPlan.model_fields["intent"].annotation.__args__)
    msgs = build_messages("only with code", {"intent": "topic_across_unis"})
    assert msgs[-1]["content"].startswith("Previous plan:") and msgs[0]["role"] == "system"


def test_resolve_places_aliases_regions_and_unknowns():
    unis, countries, warn = resolve_places(["ETH", "toronto", "NUS/NTU", "Europe", "Atlantis"])
    assert unis == {"ETH", "UofT", "NUS", "NTU"}
    assert countries == {"CH", "DE", "GB"}
    assert len(warn) == 1 and "Atlantis" in warn[0]


def test_scan_unis_finds_names_but_not_common_words():
    assert scan_unis("ETH vs CMU vs Toronto on KV-cache quantization") == {"ETH", "CMU", "UofT"}
    assert scan_unis("what can the other unis tell us about a cam and a col") == set()


def _corpus(rows):
    return SimpleNamespace(meta=[r[0] for r in rows], unis=[frozenset(r[1]) for r in rows])


def test_filters_keep_predicate_matches_the_plan():
    plan = QueryPlan(intent="topic_across_unis", semantic_text="x", year_from=2026, has_code=True, types=["preprint"])
    R = Resolved(plan=plan, include=frozenset({"NUS", "NTU"}), exclude=frozenset({"UCL"}), subdomain="Spatio_Temporal_MARL", subdomain_hard=True)
    keep_with, spec = make_filters(R)
    good = {"year": 2026, "type": "preprint", "has_code": True, "subdomain": "Spatio_Temporal_MARL"}
    c = _corpus([(good, {"NUS"}), ({**good, "year": 2025}, {"NUS"}), ({**good, "has_code": False}, {"NUS"}),
                 (good, {"NUS", "UCL"}), (good, {"ETH"}), ({**good, "subdomain": "none"}, {"NTU"})])
    keep = keep_with(c)
    assert [keep(j) for j in range(6)] == [True, False, False, False, False, False]
    keys = {x["key"] for x in spec["must"]}
    assert {"subdomain", "target_universities", "year", "type", "has_code"} <= keys
    assert spec["must_not"] == [{"key": "target_universities", "any": ["UCL"]}]


def test_citation_regex_accepts_brackets_and_parentheses():
    found = [m.group(1) for m in CITE.finditer("a [P1] b (P2 & P6) c [P3, P4] d (P1 and P2)")]
    assert found == ["P1", "P2 & P6", "P3, P4", "P1 and P2"]


def _db():
    c = sqlite3.connect(":memory:")
    c.executescript("""CREATE TABLE entities(work_id,kind,normalized,raw); CREATE TABLE authors(author_id,name);
                       CREATE TABLE work_universities(work_id,inst_id,senior_author_id);
                       INSERT INTO entities VALUES('W1','datasets','sift1m','x'),('W1','baselines','diskann','x'),('W2','datasets','gist1m','x');
                       INSERT INTO authors VALUES('A1','Jane Roe'),('A2','John Smith');
                       INSERT INTO work_universities VALUES('W1','I1','A1'),('W2','I2','A2');""")
    return c


def test_validator_flags_unknown_ids_and_unsupported_terms():
    ctx = [{"id": "P1", "title": "Fast ANN", "abstract": "We use SIFT1M.", "problem": None, "method": None, "universities": ["ETH"],
            "professors": {"ETH": "Jane Roe"}, "datasets": ["sift1m"], "baselines": []}]
    text = ("ETH uses SIFT1M [P1]. Jane Roe leads it [P1]. CMU uses GIST1M [P2]. John Smith agrees [P1].")
    out, rep = validate(_db(), text, ctx, "ANN question", "ann")
    assert rep["invalid_ids"] == ["P2"]
    assert set(rep["unsupported_terms"]) >= {"gist1m", "john smith"}
    assert "unknown P2" in out and "ETH uses SIFT1M [P1]." in out and "⚠" not in out.split("CMU")[0]


def test_query_signals_shapes_and_nan_for_pair_only_signals():
    n, d = 3, 4
    v = np.eye(d, dtype=np.float32)[:n]
    meta = [{"tt": "cache eviction", "micro": None, "keywords": {"cache"}, "extracted": False, "datasets": set(), "baselines": set(),
             "metrics": set()} for _ in range(n)]
    corpus = SimpleNamespace(abs=v, title=v, comp=v, has_comp=np.array([True, False, False]), meta=meta, sparse_title={}, sparse_abs={},
                             sparse_comp={}, self_title={}, self_abs={}, self_comp={}, topic_cent={})
    q = {"dense": v[0], "sparse": {}, "self": 0.0, "tokens": "cache eviction"}
    S = query_signals(corpus, q, np.array([0, 1, 2]), {})
    col = {s: i for i, s in enumerate(SIGNALS)}
    assert S.shape == (3, len(SIGNALS))
    assert S[0, col["dense_abstract"]] == 1.0 and S[1, col["dense_abstract"]] == 0.0
    assert not np.isnan(S[0, col["dense_composite"]]) and np.isnan(S[1, col["dense_composite"]])
    assert np.isnan(S[:, col["biblio_coupling"]]).all() and np.isnan(S[:, col["specter2"]]).all()
    assert S[0, col["title_lexical"]] == 1.0


def test_synthesize_warns_when_context_is_nearly_full(monkeypatch):
    from research_matcher.rag import answer

    class Client:
        last = {"prompt_eval_count": 11000, "eval_count": 700, "done_reason": "length"}

        def chat(self, *a, **k):
            return "text [P1]"

    res = SimpleNamespace(items=[{"work_id": "W1"}], extra={}, R=SimpleNamespace(plan=QueryPlan(intent="topic_across_unis", semantic_text="x")))
    dets = {"W1": {"title": "t", "unis": ["ETH"], "profs": {}, "year": 2026, "abstract": "a", "sub_problem": None, "method": None,
                   "datasets": [], "baselines": [], "metrics": [], "has_code": False}}
    monkeypatch.setattr(answer, "load_settings", lambda: SimpleNamespace(rag={"num_ctx": 12288, "num_predict": 768, "context_papers": 12,
                                                                               "temperature": 0.2, "top_p": 0.9, "repeat_penalty": 1.1}))
    text, ctx = answer.synthesize(SimpleNamespace(client=Client()), "q", res, dets)
    w = res.extra["answer_warnings"]
    assert len(w) == 2 and "output limit" in w[0] and "start of the prompt may have been dropped" in w[1]
