"""Step 1 of the RAG engine: gemma3 turns a question into a QueryPlan (plan §9.2). Entities such as universities are
only *extracted* here; they are resolved deterministically in resolve.py."""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from ..config import load_settings, load_subdomains, load_universities

Intent = Literal["topic_across_unis", "similar_to_paper", "compare_unis", "find_professors", "similar_professors",
                 "entity_stats", "cross_uni_topics", "paper_detail"]
Sub = Literal["SysML_RL_Caching", "Edge_RAG_Vector_Systems", "Spatio_Temporal_MARL", "any"]


class QueryPlan(BaseModel):
    intent: Intent
    semantic_text: str                       # rewritten search text (expanded acronyms, no filler)
    subdomain: Sub = "any"
    universities_include: list[str] = Field(default_factory=list)   # free text; resolved by the alias table, NOT the LLM
    universities_exclude: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    min_universities: int = 3
    per_university_limit: int = 3
    year_from: int | None = None
    year_to: int | None = None
    types: list[Literal["conference-paper", "preprint"]] = Field(default_factory=list)
    has_code: bool | None = None
    datasets: list[str] = Field(default_factory=list)
    baselines: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    professor: str | None = None
    paper_ref: str | None = None             # W-id, DOI, arXiv id, title fragment, or "P3"
    k: int = 12


def _ex(question: str, **plan) -> tuple[str, str]:
    plan.setdefault("semantic_text", question)
    return question, QueryPlan(**plan).model_dump_json()


FEW_SHOT = [
    _ex("Find similar work across universities on learned cache eviction for key-value stores.",
        intent="topic_across_unis", semantic_text="learned cache eviction policy for key-value stores",
        subdomain="SysML_RL_Caching"),
    _ex("Which unis are working on filtered vector search? Group by university with professors.",
        intent="topic_across_unis", semantic_text="filtered vector search, hybrid vector-relational queries, approximate nearest neighbor with attribute filters",
        subdomain="Edge_RAG_Vector_Systems", k=15),
    _ex("Show papers from other universities similar to P3",
        intent="similar_to_paper", semantic_text="", paper_ref="P3"),
    _ex("ETH vs CMU vs Toronto on KV-cache quantization: who does what, shared baselines?",
        intent="compare_unis", semantic_text="KV-cache quantization for LLM inference", subdomain="Edge_RAG_Vector_Systems",
        universities_include=["ETH", "CMU", "Toronto"]),
    _ex("Who at NUS/NTU does MARL for ride-hailing dispatch, with code, 2026 only?",
        intent="find_professors", semantic_text="multi-agent reinforcement learning for ride-hailing dispatch",
        subdomain="Spatio_Temporal_MARL", universities_include=["NUS", "NTU"], has_code=True, year_from=2026, year_to=2026),
    _ex("Which professors in Europe work on topics similar to Jane Doe?",
        intent="similar_professors", semantic_text="", professor="Jane Doe", countries=["Europe"]),
    _ex("Which datasets and baselines are most used in Edge_RAG_Vector_Systems, and by which unis?",
        intent="entity_stats", semantic_text="datasets and baselines", subdomain="Edge_RAG_Vector_Systems"),
    _ex("List cross-university micro-topics in SysML_RL_Caching that span at least 4 universities.",
        intent="cross_uni_topics", semantic_text="micro-topics", subdomain="SysML_RL_Caching", min_universities=4),
    _ex("Tell me about W4391234567",
        intent="paper_detail", semantic_text="", paper_ref="W4391234567"),
]


def system_prompt() -> str:
    subs = "\n".join(f"- {s['id']}: {' '.join(str(s['definition']).split())}" for s in load_subdomains() if s["id"] != "none")
    unis = ", ".join(f"{u.short} ({u.name})" for u in load_universities())
    return f"""You convert a researcher's question about a store of CS papers into a JSON query plan. Output JSON only.

Intents:
- topic_across_unis: find papers on a topic across universities.
- similar_to_paper: papers from OTHER universities similar to one given paper (paper_ref = W-id, DOI, arXiv id, title fragment, or "P3" for result 3 of the previous answer).
- compare_unis: compare what two or more named universities do on a topic (universities_include lists them).
- find_professors: which professors / who works on a topic.
- similar_professors: professors similar to a NAMED professor (professor = that name).
- entity_stats: which datasets / baselines / metrics are used most, and by which universities.
- cross_uni_topics: list fine-grained topics that span several universities (min_universities = the number asked).
- paper_detail: facts about one paper.

Subdomains (use "any" unless the question clearly names or is about exactly one):
{subs}

Universities in the store: {unis}.
Rules: copy university / country / professor names exactly as the user wrote them (do NOT expand or guess). semantic_text is a short, self-contained search phrase with acronyms expanded and no filler words (empty for similar_to_paper, similar_professors and paper_detail). Only set year_from/year_to, has_code, types, datasets, baselines, metrics when the question asks. 'with code' means has_code true. '2026 only' means year_from and year_to 2026. k is how many papers to show (default 12). If a previous plan is given and the new message is a follow-up, change only what the follow-up asks for (e.g. 'only with code', 'drop UCL', 'more like P2', 'now 2026 only') and keep the rest."""


def build_messages(question: str, prev_plan: dict | None = None) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": system_prompt()}]
    for q, p in FEW_SHOT:
        msgs += [{"role": "user", "content": f"Question: {q}"}, {"role": "assistant", "content": p}]
    prev = f"Previous plan: {json.dumps(prev_plan)}\n" if prev_plan else ""
    msgs.append({"role": "user", "content": f"{prev}Question: {question}"})
    return msgs


def make_plan(client, question: str, prev_plan: dict | None = None) -> tuple[QueryPlan, str | None]:
    """Returns (plan, warning). Falls back to a plain topic search if the model's JSON is unusable."""
    opts = {"temperature": 0.0, "seed": 42, "num_ctx": int(load_settings().rag.get("plan_num_ctx", 8192)), "num_predict": 400}
    raw, err = "", None
    for _ in range(2):
        try:
            raw = client.chat(build_messages(question, prev_plan), schema=QueryPlan.model_json_schema(), options=opts)
            return QueryPlan.model_validate_json(raw), None
        except (ValidationError, json.JSONDecodeError) as e:
            err = f"{type(e).__name__}: {str(e)[:120]}"
        except Exception as e:  # noqa: BLE001  (Ollama down etc.)
            err = f"{type(e).__name__}: {str(e)[:120]}"
            break
    return QueryPlan(intent="topic_across_unis", semantic_text=question), f"planner failed ({err}); using the question as a plain topic search"
