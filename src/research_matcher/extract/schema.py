"""Extraction schema (plan §10.1). Two LLM calls fill one PaperExtraction: ExtractionBody (entities) + Classification."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Subdomain = Literal["SysML_RL_Caching", "Edge_RAG_Vector_Systems", "Spatio_Temporal_MARL", "none"]


class ReportedResult(BaseModel):
    metric: str
    value: str
    compared_to: str | None = None
    dataset: str | None = None


class ExtractionBody(BaseModel):
    sub_problem_solved: str
    core_algorithm_or_architecture: str
    # No defaults: every list is REQUIRED in the JSON schema so the model can't skip them (it did, and recall collapsed).
    baseline_algorithms_compared: list[str] = Field(max_length=8)
    datasets_or_benchmarks_used: list[str] = Field(max_length=8)
    evaluation_metrics: list[str] = Field(max_length=8)
    reported_results: list[ReportedResult] = Field(max_length=5)
    hardware_mentioned: list[str]
    is_relevant_cs_systems_or_ml: bool


class Classification(BaseModel):
    subdomain_cluster: Subdomain
    confidence: Literal["high", "medium", "low"]


class PaperExtraction(ExtractionBody, Classification):
    pass


def body_schema() -> dict:
    return ExtractionBody.model_json_schema()


def classify_schema() -> dict:
    return Classification.model_json_schema()
