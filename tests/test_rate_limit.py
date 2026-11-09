import multiprocessing as mp
import time

from research_matcher.db import connect, init_schema
from research_matcher.http_client import cache_key, redact_url
from research_matcher.rate_limit import S2RateLimiter


def test_redaction_strips_api_key():
    u = "https://api.openalex.org/works?filter=a:b&api_key=SECRET&per_page=1"
    assert "SECRET" not in redact_url(u)
    assert cache_key("GET", u) == cache_key("GET", u.replace("SECRET", "OTHER"))


def test_limiter_spacing_fake_clock(tmp_path):
    conn = connect(tmp_path / "t.db")
    init_schema(conn)
    now = [1000.0]
    lim = S2RateLimiter(conn, 1.05, clock=lambda: now[0], sleep=lambda s: now.__setitem__(0, now[0] + s))
    times = [lim.acquire() for _ in range(5)]
    assert all(b - a >= 1.05 - 1e-9 for a, b in zip(times, times[1:]))


def _worker(db, q):
    conn = connect(db)
    lim = S2RateLimiter(conn, 1.05)
    out = []
    for _ in range(3):
        lim.acquire()
        out.append(time.time())
    q.put(out)


def test_limiter_two_processes(tmp_path):
    db = tmp_path / "t.db"
    init_schema(connect(db))
    q = mp.Queue()
    ps = [mp.Process(target=_worker, args=(db, q)) for _ in range(2)]
    [p.start() for p in ps]
    res = sorted(t for _ in ps for t in q.get(timeout=30))
    [p.join() for p in ps]
    assert len(res) == 6
    assert all(b - a >= 1.0 for a, b in zip(res, res[1:]))  # 1.05 reserved; allow clock jitter
