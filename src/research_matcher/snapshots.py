"""Named snapshots of the whole working state, with instant switching between them.

A snapshot is a folder under data/snapshots/<name>/ holding the version-specific files (DB, embeddings, vector store, calibration,
the two config files that depend on which universities are loaded, the benchmark report) plus a manifest.json.
 - `save NAME`  copies the active state into snapshots/NAME (the active state keeps running).
 - `use NAME`   parks the active state under its own label and moves NAME into place. Moves are renames on the same drive, so
                switching takes about a second and needs no extra disk; nothing is deleted.
data/.active holds the label of the active state (default "current"). Shared, not snapshotted: data/gold labels, data/cache,
data/notes, the Qdrant folder (unused with the numpy backend), .env.local."""
from __future__ import annotations

import json
import re
import shutil
import sqlite3
import time
from pathlib import Path

# paths relative to the project root that make up "a version"
ITEMS = ["data/papers.db", "data/embeddings", "data/vectors", "data/gold/weights.json", "data/gold/calibration.json",
         "config/universities.yaml", "config/settings.yaml", "BENCHMARK_CANDIDATES.md"]
DB_SIDECARS = ["data/papers.db-wal", "data/papers.db-shm"]
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,60}$")


class SnapshotError(Exception):
    pass


def snap_dir(root: Path) -> Path:
    return root / "data" / "snapshots"


def active_label(root: Path) -> str:
    p = root / "data" / ".active"
    return p.read_text().strip() if p.exists() else "current"


def _check_name(name: str) -> None:
    if not NAME_RE.match(name):
        raise SnapshotError(f"bad name {name!r}: use letters, digits, . _ - (max 61 chars)")


def _size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def _checkpoint(root: Path) -> None:
    """Fold the WAL into the DB file so the DB is one self-contained file."""
    db = root / "data" / "papers.db"
    if db.exists():
        c = sqlite3.connect(db)
        c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        c.close()


def _counts(db: Path) -> dict:
    try:
        c = sqlite3.connect(db)
        out = {"works": c.execute("SELECT COUNT(*) FROM works").fetchone()[0],
               "extracted": c.execute("SELECT COUNT(*) FROM extractions WHERE error IS NULL").fetchone()[0],
               "pairs": c.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]}
        try:
            out["universities"] = c.execute("SELECT COUNT(*) FROM institutions").fetchone()[0]
        except sqlite3.Error:
            pass
        c.close()
        return out
    except sqlite3.Error:
        return {}


def list_snapshots(root: Path) -> list[dict]:
    rows = []
    d = snap_dir(root)
    for p in sorted(d.iterdir()) if d.exists() else []:
        m = p / "manifest.json"
        if m.is_file():
            rows.append({"name": p.name, **json.loads(m.read_text())})
    return rows


def save(root: Path, name: str, *, force: bool = False, log=print) -> Path:
    """Copy the active state into snapshots/<name>."""
    _check_name(name)
    dest = snap_dir(root) / name
    if dest.exists():
        if not force:
            raise SnapshotError(f"snapshot {name!r} already exists (use --force to overwrite)")
        shutil.rmtree(dest)
    _checkpoint(root)
    items = [i for i in ITEMS if (root / i).exists()]
    need = sum(_size(root / i) for i in items)
    free = shutil.disk_usage(root).free
    if free < need * 1.1:
        raise SnapshotError(f"not enough disk space: need {need / 1e9:.1f} GB, free {free / 1e9:.1f} GB")
    tmp = snap_dir(root) / f".{name}.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    for i in items:
        src, dst = root / i, tmp / i
        dst.parent.mkdir(parents=True, exist_ok=True)
        log(f"  copy {i} ({_size(src) / 1e6:.0f} MB)")
        shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)
    manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "from_label": active_label(root), "items": items,
                "bytes": need, **_counts(root / "data" / "papers.db")}
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1))
    tmp.rename(dest)
    return dest


def _move(src: Path, dst: Path, done: list) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    done.append((src, dst))


def use(root: Path, name: str, *, keep_current_as: str | None = None, log=print) -> None:
    """Make snapshot `name` the active state. The current state is parked as snapshots/<label> (label = data/.active, or
    --keep-current-as). Everything is rolled back if any move fails (for example a running process holds the DB open)."""
    _check_name(name)
    src = snap_dir(root) / name
    if not (src / "manifest.json").is_file():
        raise SnapshotError(f"no snapshot named {name!r} (see `snapshot list`)")
    cur = keep_current_as or active_label(root)
    _check_name(cur)
    if cur == name:
        raise SnapshotError(f"{name!r} is already the active label")
    park = snap_dir(root) / cur
    if park.exists():
        raise SnapshotError(f"snapshots/{cur} already exists; pass --keep-current-as NEW_NAME to park the current state under another name")
    _checkpoint(root)
    for s in DB_SIDECARS:
        if (root / s).exists():
            raise SnapshotError("database has -wal/-shm files: another research-matcher process is probably running; stop it first")
    items = [i for i in ITEMS if (src / i).exists() or (root / i).exists()]
    done: list = []
    tmp_park = snap_dir(root) / f".{cur}.parking"
    try:
        shutil.rmtree(tmp_park, ignore_errors=True)
        parked = []
        for i in items:                                   # 1) current -> parking area
            if (root / i).exists():
                _move(root / i, tmp_park / i, done)
                parked.append(i)
        for i in items:                                   # 2) snapshot -> active
            if (src / i).exists():
                _move(src / i, root / i, done)
        m = json.loads((src / "manifest.json").read_text())
        mp = tmp_park / "manifest.json"
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "from_label": cur, "items": parked,
                                  "bytes": sum(_size(tmp_park / i) for i in parked), **_counts(tmp_park / "data" / "papers.db")}, indent=1))
        tmp_park.rename(park)
        shutil.rmtree(src)                                # now holds only the manifest
        (root / "data" / ".active").write_text(name)
        log(f"active: {name} (previous state parked as {cur!r}; {m.get('works', '?')} works)")
    except Exception as e:  # noqa: BLE001
        for s, d in reversed(done):                       # undo
            try:
                d.rename(s)
            except OSError:
                pass
        shutil.rmtree(tmp_park, ignore_errors=True)
        raise SnapshotError(f"switch failed and was rolled back: {type(e).__name__}: {e}") from e


def delete(root: Path, name: str) -> None:
    _check_name(name)
    p = snap_dir(root) / name
    if not (p / "manifest.json").is_file():
        raise SnapshotError(f"no snapshot named {name!r}")
    shutil.rmtree(p)
