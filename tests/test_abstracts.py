from research_matcher.harvest.abstracts import rebuild_abstract


def test_rebuild_orders_by_position():
    inv = {"world": [1], "Hello": [0], "again": [2, 4], "hello": [3]}
    assert rebuild_abstract(inv) == "Hello world again hello again"


def test_repeated_words():
    assert rebuild_abstract({"a": [0, 2], "b": [1]}) == "a b a"


def test_missing_or_empty():
    assert rebuild_abstract(None) is None
    assert rebuild_abstract({}) is None
