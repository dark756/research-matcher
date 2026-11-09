import sqlite3

import pytest

from research_matcher.pipeline import NAMES, diff, fresh, select_stages


def test_stage_selection_and_validation():
    assert select_stages(None, None, []) == NAMES
    assert select_stages("index", "report", ["match"]) == ["index", "report"]
    with pytest.raises(ValueError):
        select_stages("nope", None, [])
    with pytest.raises(ValueError):
        select_stages("report", "index", [])


def test_diff_reports_only_changes():
    assert diff({"a": 1, "b": 2}, {"a": 1, "b": 3, "c": 0}) == [("b", 2, 3), ("c", -1, 0)]


def test_fresh_keeps_http_cache_and_backs_up_old_db(tmp_path):
    db = tmp_path / "papers.db"
    c = sqlite3.connect(db)
    c.executescript("CREATE TABLE http_cache(key PRIMARY KEY, body); INSERT INTO http_cache VALUES('k','v');"
                    "CREATE TABLE works(work_id); INSERT INTO works VALUES('W1');")
    c.commit()
    c.close()
    (tmp_path / "embeddings").mkdir()
    (tmp_path / "embeddings" / "x.npy").write_text("x")
    bak = fresh(db, tmp_path)
    assert bak.exists() and not (tmp_path / "embeddings").exists()
    n = sqlite3.connect(db)
    assert n.execute("SELECT COUNT(*) FROM http_cache").fetchone()[0] == 1
    assert n.execute("SELECT name FROM sqlite_master WHERE name='works'").fetchone() is None
    assert sqlite3.connect(bak).execute("SELECT COUNT(*) FROM works").fetchone()[0] == 1
