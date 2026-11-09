import sqlite3

import httpx
import pytest

from research_matcher import http_client as hc
from research_matcher.db import SCHEMA, connect, init_schema
from research_matcher.enrich.code_links import find_github
from research_matcher.enrich.dedupe import UnionFind, run_dedupe
from research_matcher.enrich.professors import FacultyIndex, _initial_match
from research_matcher.enrich.semantic_scholar import apply_s2, is_real_venue, s2_id
from research_matcher.enrich.venue_hints import venue_hint
from research_matcher.rate_limit import S2RateLimiter


def mem():
    c = sqlite3.connect(":memory:", isolation_level=None)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c


def add(c, wid, title, typ="preprint", doi=None, arxiv=None, venue=None, vtype="repository", abstract="x", first="Ann Lee"):
    c.execute("INSERT INTO works(work_id,title,title_norm,year,type,doi,arxiv_id,venue_name,venue_type,abstract,"
              "publication_date) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
              (wid, title, title.lower(), 2025, typ, doi, arxiv, venue, vtype, abstract, "2025-03-01"))
    c.execute("INSERT OR IGNORE INTO authors(author_id,name) VALUES (?,?)", ("A" + first, first))
    c.execute("INSERT INTO authorships(work_id,author_id,position_idx) VALUES (?,?,0)", (wid, "A" + first))
    c.execute("INSERT INTO work_universities(work_id,inst_id) VALUES (?,?)", (wid, "I1"))


def test_s2_id_and_venue():
    assert s2_id("10.1/x", "2501.1") == "ARXIV:2501.1" and s2_id("10.1/x", None) == "DOI:10.1/x"
    assert s2_id(None, None) is None
    assert not is_real_venue("arXiv.org") and not is_real_venue(None) and is_real_venue("ICLR")


def test_apply_s2_fills_venue_and_abstract():
    c = mem()
    add(c, "W1", "t", venue="arXiv", abstract=None)
    c.execute("UPDATE works SET abstract=NULL WHERE work_id='W1'")
    c.execute("INSERT INTO works_fts(title,keywords,abstract,work_id) VALUES ('t','','','W1')")
    apply_s2(c, "W1", {"paperId": "p", "externalIds": {"ArXiv": "2501.9", "CorpusId": 5}, "abstract": "Hello",
                       "publicationVenue": {"name": "Neural Information Processing Systems", "type": "conference"},
                       "tldr": {"text": "short"}, "citationCount": 3})
    r = c.execute("SELECT * FROM works").fetchone()
    assert r["venue_name"] == "Neural Information Processing Systems" and r["abstract"] == "Hello"
    assert r["abstract_source"] == "s2" and r["arxiv_id"] == "2501.9" and r["tldr"] == "short"


def test_dedupe_prefers_conference_and_merges():
    c = mem()
    add(c, "W1", "learning cache eviction with deep rl", doi="10.1/a", arxiv="2501.1")
    add(c, "W2", "learning cache eviction with deep rl", typ="conference-paper", arxiv="2501.1",
        venue="NSDI", vtype="conference")
    add(c, "W3", "learning cache eviction with deep rls", first="Ann Lee")  # fuzzy title, same surname
    add(c, "W4", "something else entirely about graphs", first="Bob Roe")
    stats = run_dedupe(c)
    canon = {r["work_id"]: r["canonical_id"] for r in c.execute("SELECT work_id, canonical_id FROM works")}
    assert canon["W1"] == canon["W2"] == canon["W3"] == "W2" and canon["W4"] == "W4"
    assert stats["duplicates"] == 2


def test_union_find():
    u = UnionFind()
    u.union("a", "b")
    u.union("b", "c")
    assert u.find("a") == u.find("c") != u.find("d")


def test_initial_match_and_faculty_index():
    assert _initial_match("j smith", "john smith") and not _initial_match("k smith", "john smith")
    c = mem()
    c.execute("INSERT INTO faculty(name,name_norm,affiliation,orcid) VALUES ('Zoë Müller','zoe muller','ETH Zurich','')")
    c.execute("INSERT INTO faculty(name,name_norm,affiliation,orcid) VALUES ('Wei Wang 0001','wei wang','CMU','')")
    ix = FacultyIndex(c)
    assert ix.match("ETH Zurich", "Zoe Müller", None) is not None
    assert ix.match("ETH Zurich", "Z. Muller", None) is not None
    assert ix.match("ETH Zurich", "Wei Wang", None) is None  # right name, wrong university


def test_github_and_hint():
    assert find_github("Code at https://github.com/foo/bar-baz.") == ["https://github.com/foo/bar-baz"]
    assert find_github("see github.com/a/b and github.com/a/b.git") == ["https://github.com/a/b"]
    assert venue_hint("We show X. Accepted at ICLR 2026.") == "ICLR 2026"
    assert venue_hint("nothing here") is None


# --- gate: limiter never issues 2 S2 requests within 1 s, including under retries ---
class FakeClient:
    def __init__(self, statuses, log):
        self.statuses, self.log = list(statuses), log

    def post(self, url, json=None, headers=None):
        self.log.append(self.now())
        code = self.statuses.pop(0)
        return httpx.Response(code, json=[{"paperId": "p"}] if code == 200 else {}, headers={"Retry-After": "0"})


def test_limiter_spacing_holds_under_429_retries(tmp_path, monkeypatch):
    conn = connect(tmp_path / "t.db")
    init_schema(conn)
    now = [1000.0]
    monkeypatch.setattr(hc.time, "sleep", lambda s: now.__setitem__(0, now[0] + s))
    http = hc.HttpClient(conn)
    http.s2_limiter = S2RateLimiter(conn, 1.05, clock=lambda: now[0],
                                    sleep=lambda s: now.__setitem__(0, now[0] + s))
    sent: list[float] = []
    fc = FakeClient([429, 429, 503, 200], sent)
    fc.now = lambda: now[0]
    http.client = fc
    assert http.s2_post("/paper/batch", {"fields": "x"}, {"ids": ["a"]}) == [{"paperId": "p"}]
    assert len(sent) == 4 and all(b - a >= 1.05 - 1e-9 for a, b in zip(sent, sent[1:]))
    # cached 2xx: no further request
    http.s2_post("/paper/batch", {"fields": "x"}, {"ids": ["a"]})
    assert len(sent) == 4
