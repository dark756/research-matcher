import sqlite3

import pytest

from research_matcher import snapshots as S


def _project(tmp_path, tag):
    (tmp_path / "data" / "embeddings").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "gold").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config").mkdir(exist_ok=True)
    db = tmp_path / "data" / "papers.db"
    db.unlink(missing_ok=True)
    c = sqlite3.connect(db)
    c.executescript("CREATE TABLE works(work_id); CREATE TABLE extractions(work_id, error); CREATE TABLE pairs(a);"
                    "CREATE TABLE institutions(x);")
    c.executemany("INSERT INTO works VALUES (?)", [(f"W{i}",) for i in range(3 if tag == "v1" else 5)])
    c.commit()
    c.close()
    (tmp_path / "data" / "embeddings" / "a.npy").write_text(tag)
    (tmp_path / "config" / "universities.yaml").write_text(tag)
    (tmp_path / "data" / "gold" / "pairs.csv").write_text("shared labels")


def _works(tmp_path):
    c = sqlite3.connect(tmp_path / "data" / "papers.db")
    n = c.execute("SELECT COUNT(*) FROM works").fetchone()[0]
    c.close()                                                            # an open handle blocks the switch on Windows
    return n


def test_save_then_switch_back_and_forth(tmp_path):
    _project(tmp_path, "v1")
    S.save(tmp_path, "v1-17unis", log=lambda *_: None)
    assert (tmp_path / "data" / "papers.db").exists()                    # save leaves the active state alone
    _project(tmp_path, "v2")                                             # work continues: the expanded version
    assert _works(tmp_path) == 5
    S.use(tmp_path, "v1-17unis", keep_current_as="v2-expanded", log=lambda *_: None)
    assert _works(tmp_path) == 3 and (tmp_path / "config" / "universities.yaml").read_text() == "v1"
    assert S.active_label(tmp_path) == "v1-17unis"
    names = {r["name"]: r for r in S.list_snapshots(tmp_path)}
    assert set(names) == {"v2-expanded"} and names["v2-expanded"]["works"] == 5
    S.use(tmp_path, "v2-expanded", log=lambda *_: None)                   # back again; v1 gets parked under its own label
    assert _works(tmp_path) == 5 and (tmp_path / "data" / "embeddings" / "a.npy").read_text() == "v2"
    assert {r["name"] for r in S.list_snapshots(tmp_path)} == {"v1-17unis"}
    assert (tmp_path / "data" / "gold" / "pairs.csv").read_text() == "shared labels"     # shared files untouched


def test_guards(tmp_path):
    _project(tmp_path, "v1")
    S.save(tmp_path, "a", log=lambda *_: None)
    with pytest.raises(S.SnapshotError):
        S.save(tmp_path, "a", log=lambda *_: None)                       # no silent overwrite
    with pytest.raises(S.SnapshotError):
        S.save(tmp_path, "bad name!", log=lambda *_: None)
    with pytest.raises(S.SnapshotError):
        S.use(tmp_path, "missing")
    S.save(tmp_path, "current", log=lambda *_: None)
    with pytest.raises(S.SnapshotError):                                 # parking would clobber snapshots/current
        S.use(tmp_path, "a")
    S.use(tmp_path, "a", keep_current_as="b", log=lambda *_: None)
    assert _works(tmp_path) == 3
