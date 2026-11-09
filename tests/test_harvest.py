import sqlite3

from research_matcher.db import SCHEMA
from research_matcher.harvest.openalex import (derive_arxiv_id, filter_a, filter_c, norm_title, parse_work,
                                               store_work)

T = {"I1", "I2"}


def work(wid="https://openalex.org/W1", insts=(("I1",),), **kw):
    auth = [{"author": {"id": f"https://openalex.org/A{i}", "display_name": f"Au{i}"}, "author_position": "middle",
             "institutions": [{"id": f"https://openalex.org/{x[0]}",
                               "lineage": [f"https://openalex.org/{y}" for y in x]} for x in ins]}
            for i, ins in enumerate(insts)]
    w = {"id": wid, "doi": "https://doi.org/10.48550/arXiv.2501.01234", "title": "A Study of DiskANN: Éclair!",
         "publication_year": 2025, "type": "preprint", "authorships": auth,
         "abstract_inverted_index": {"hi": [0], "there": [1]}, "keywords": [{"display_name": "ann", "score": .5}],
         "topics": [], "referenced_works": [], "related_works": [], "primary_location": {"source": {
             "display_name": "arXiv", "type": "repository", "id": "https://openalex.org/S1"}}}
    w.update(kw)
    return w


def conn():
    c = sqlite3.connect(":memory:", isolation_level=None)
    c.executescript(SCHEMA)
    return c


def test_arxiv_from_doi_and_url():
    assert derive_arxiv_id({"doi": "https://doi.org/10.48550/arXiv.2501.01234v2"}) == "2501.01234"
    w = {"primary_location": {"landing_page_url": "https://arxiv.org/abs/2403.00001v3"}}
    assert derive_arxiv_id(w) == "2403.00001"
    assert derive_arxiv_id({"doi": "https://doi.org/10.1/x"}) is None


def test_norm_title():
    assert norm_title("A Study: Éclair!") == "a study eclair"


def test_lead_senior_and_lineage():
    # author0 at a sub-institution of I1, author1 elsewhere, author2 at I1 directly, author3 at I2
    w = work(insts=[[("I9", "I1")], [("I7",)], [("I1",)], [("I2",)]])
    p = parse_work(w, "A", T)
    wu = {u["inst_id"]: u for u in p["work_unis"]}
    assert wu["I1"]["lead_author_id"] == "A0" and wu["I1"]["senior_author_id"] == "A2"
    assert wu["I2"]["lead_author_id"] == "A3" == wu["I2"]["senior_author_id"]
    assert p["row"]["abstract"] == "hi there" and p["row"]["arxiv_id"] == "2501.01234"


def test_author_without_openalex_id_is_kept():
    w = work(insts=[[("I1",)]])
    w["authorships"][0]["author"] = {"id": None, "display_name": "Anon"}
    p = parse_work(w, "A", T)
    assert p["work_unis"][0]["lead_author_id"] == "X:W1:0"


def test_no_target_uni_and_type_override():
    assert parse_work(work(insts=[[("I7",)]]), "A", T)["work_unis"] == []
    p = parse_work(work(type="article"), "B", T, type_override="conference-paper")
    assert p["row"]["type"] == "conference-paper" and p["row"]["type_raw"] == "article"


def test_store_priority_and_fts():
    c = conn()
    a = parse_work(work(), "A", T)
    assert store_work(c, a)
    assert not store_work(c, parse_work(work(), "C", T))  # lower priority doesn't overwrite
    assert c.execute("SELECT harvest_source FROM works").fetchone()[0] == "A"
    assert c.execute("SELECT COUNT(*) FROM works_fts WHERE works_fts MATCH 'diskann'").fetchone()[0] == 1
    assert store_work(c, parse_work(work(), "C", T), refresh=True)  # refresh keeps the better source label
    assert c.execute("SELECT harvest_source FROM works").fetchone()[0] == "A"
    assert store_work(c, a, refresh=True)
    assert c.execute("SELECT COUNT(*) FROM works_fts").fetchone()[0] == 1  # no duplicate FTS rows
    assert c.execute("SELECT COUNT(*) FROM authorships").fetchone()[0] == 1


def test_filters_match_plan():
    cfg = {"from_publication_date": "2025-01-01", "types": ["conference-paper", "preprint"], "field": "fields/17"}
    f = filter_a(cfg, ["I1", "I2"])
    assert "authorships.institutions.lineage:I1|I2" in f and "type:conference-paper|preprint" in f
    assert "primary_topic.field.id:!fields/17" in filter_c(cfg, ["I1"], '"a" OR b')
