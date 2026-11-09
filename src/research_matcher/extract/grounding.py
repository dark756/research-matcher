"""Grounding check: drop extracted items that don't appear (fuzzily) in the title/abstract (plan §10.1)."""
from __future__ import annotations

import re

from rapidfuzz import fuzz

from .schema import PaperExtraction, ReportedResult

THRESH = 85
GENERIC_FIRST = {"a", "an", "the", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "several",
                 "multiple", "various", "many", "some", "several", "public", "standard", "popular", "real-world", "in-domain",
                 "out-of-domain", "complex", "diverse", "common", "existing", "synthetic", "benchmark", "our", "new"}


def valid_dataset(item: str) -> bool:
    """Datasets must be names: carry a capital or digit as written, and not be a generic descriptor."""
    t = item.strip()
    if not re.search(r"[A-Z0-9]", t):
        return False
    first = re.split(r"[\s]", t.lower(), maxsplit=1)[0]
    if re.search(r"(?i)(llama|mistral|qwen|gemma|gpt-?\d|claude|gemini|grok|smollm|phi-?\d|resnet|densenet|inception|bert|vit-)", t):
        return False  # models are not datasets
    if re.search(r"(?i)(?<![a-z])(systems?|networks|scheduling|problems?)$", t):
        return False
    return first not in GENERIC_FIRST and not re.search(r"(?i)(?<![a-z])(datasets|benchmarks)$", t)


def valid_metric(item: str) -> bool:
    t = item.strip()
    if re.search(r"(?i)(estimator|q-functions?|threshold|improvements?|tasks?|control|safety)$", t) or t.lower().startswith("shortest"):
        return False
    if "%" in t or re.fullmatch(r"[\d.,x×\s]+", t):
        return False
    return not (re.search(r"\d", t) and "@" not in t and not re.search(r"(?i)top-?\d|f1|auc|ndcg|mrr|bleu|rouge", t))


def valid_baseline(item: str) -> bool:
    if not re.search(r"[A-Z0-9]", item):
        return False
    return not re.search(r"(?i)(?<![a-z])(baselines?|state-of-the-art|sota|feedback|existing (methods|approaches))(?![a-z])", item)


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[\s\-_]+", " ", s.lower()).split())


def grounded(item: str, text_norm: str) -> bool:
    it = _norm(item)
    if len(it) < 2:
        return False
    if it in text_norm:
        return True
    return len(it) >= 4 and fuzz.partial_ratio(it, text_norm) >= THRESH


def ground(ex: PaperExtraction, text: str) -> tuple[PaperExtraction, int]:
    """Return (cleaned extraction, number of dropped items)."""
    t = _norm(text)
    dropped = 0

    def keep(items: list[str]) -> list[str]:
        nonlocal dropped
        out = []
        for i in items:
            if grounded(i, t):
                out.append(i.strip())
            else:
                dropped += 1
        return list(dict.fromkeys(out))

    def only(items: list[str], ok) -> list[str]:
        nonlocal dropped
        good = [i for i in items if ok(i)]
        dropped += len(items) - len(good)
        return good

    ex.baseline_algorithms_compared = keep(only(ex.baseline_algorithms_compared, valid_baseline))
    ex.datasets_or_benchmarks_used = keep(only(ex.datasets_or_benchmarks_used, valid_dataset))
    ex.evaluation_metrics = keep(only(ex.evaluation_metrics, valid_metric))
    ex.hardware_mentioned = keep(ex.hardware_mentioned)
    results: list[ReportedResult] = []
    for r in ex.reported_results:
        digits = re.sub(r"[^\d.]", "", r.value).strip(".")
        in_text = (bool(digits) and digits in re.sub(r"[^\d.]", "", t.replace(" ", ""))) or r.value.lower() in t
        if in_text:
            if r.compared_to and not grounded(r.compared_to, t):
                r.compared_to = None
            if r.dataset and not grounded(r.dataset, t):
                r.dataset = None
            results.append(r)
        else:
            dropped += 1
    ex.reported_results = results
    return ex, dropped
