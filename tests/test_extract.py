from research_matcher.extract.grounding import ground, valid_baseline, valid_dataset, valid_metric
from research_matcher.extract.normalize import normalize
from research_matcher.extract.schema import PaperExtraction, ReportedResult
from research_matcher.extract.extractor import final_labels


def mk(**kw):
    base = dict(baseline_algorithms_compared=[], datasets_or_benchmarks_used=[], evaluation_metrics=[], reported_results=[],
                hardware_mentioned=[], sub_problem_solved="s", core_algorithm_or_architecture="c", subdomain_cluster="none",
                is_relevant_cs_systems_or_ml=True, confidence="high")
    base.update(kw)
    return PaperExtraction(**base)


def test_entity_filters():
    assert valid_dataset("SIFT1M") and valid_dataset("LongBench v2")
    assert not valid_dataset("three benchmark datasets") and not valid_dataset("math tasks") and not valid_dataset("RAG datasets")
    assert valid_metric("recall@10") and valid_metric("latency") and not valid_metric("16.4%") and not valid_metric("10% error bound")
    assert valid_baseline("R-KV") and not valid_baseline("state-of-the-art baselines") and not valid_baseline("strong baseline")


def test_grounding_drops_hallucinations():
    text = "We evaluate DiskANN against HNSW on SIFT1M and report 2.3x higher QPS."
    ex = mk(baseline_algorithms_compared=["HNSW", "ScaNN"], datasets_or_benchmarks_used=["SIFT1M", "GIST1M"],
            evaluation_metrics=["QPS", "recall@10"],
            reported_results=[ReportedResult(metric="QPS", value="2.3x"), ReportedResult(metric="QPS", value="9.9x")])
    ex, dropped = ground(ex, text)
    assert ex.baseline_algorithms_compared == ["HNSW"] and ex.datasets_or_benchmarks_used == ["SIFT1M"]
    assert ex.evaluation_metrics == ["QPS"] and [r.value for r in ex.reported_results] == ["2.3x"]
    assert dropped == 4


def test_alias_normalisation():
    assert normalize("datasets", "SIFT-1M") == "sift1m" and normalize("baselines", "Vamana") == "diskann"
    assert normalize("metrics", "Hit Rate") == "hit ratio" and normalize("datasets", "Foo (Bar)") == "foo bar"


def test_final_labels():
    e = mk(subdomain_cluster="Edge_RAG_Vector_Systems", confidence="low")
    r = final_labels(e, "Edge_RAG_Vector_Systems", {"Edge_RAG_Vector_Systems": 1})
    assert r["subdomain_cluster"] == "Edge_RAG_Vector_Systems" and r["needs_review"] == 0 and r["labels_agree"] == 1
    r = final_labels(mk(subdomain_cluster="Edge_RAG_Vector_Systems", confidence="high"), "SysML_RL_Caching",
                     {"Edge_RAG_Vector_Systems": 1})
    assert r["needs_review"] == 0 and r["labels_agree"] == 0
    r = final_labels(e, "SysML_RL_Caching", {"Edge_RAG_Vector_Systems": 1})
    assert r["needs_review"] == 1


def test_final_label_demotes_never_flips():
    e = mk(subdomain_cluster="Spatio_Temporal_MARL", confidence="high")
    r = final_labels(e, "Spatio_Temporal_MARL", {"Spatio_Temporal_MARL": 0, "SysML_RL_Caching": 5})
    assert r["subdomain_cluster"] == "none" and r["needs_review"] == 1  # no flip to SysML despite many hits
    assert final_labels(mk(subdomain_cluster="none"), "SysML_RL_Caching", {"SysML_RL_Caching": 0})["subdomain_cluster"] == "none"
