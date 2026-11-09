"""Config loading: YAML files in config/ plus secrets from .env.local."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"


class University(BaseModel):
    short: str
    name: str
    country: str
    ror: str
    openalex_id: str
    csrankings_name: str
    aliases: list[str] = []


class Secrets(BaseModel):
    openalex_api_key: str = ""
    openalex_email: str = ""
    s2_api_key: str = ""

    def __repr__(self) -> str:  # never print keys
        return "Secrets(<redacted>)"

    __str__ = __repr__


class Settings(BaseModel):
    models: dict[str, str]
    paths: dict[str, str]
    harvest: dict[str, Any]
    enrich: dict[str, Any]
    csrankings_url: str
    semantic_scholar: dict[str, Any]
    http: dict[str, Any]
    prefilter: dict[str, Any]
    extract: dict[str, Any]
    similarity: dict[str, Any]
    rag: dict[str, Any]
    index: dict[str, Any] = {}

    def path(self, key: str) -> Path:
        p = Path(self.paths[key])
        return p if p.is_absolute() else ROOT / p


def _yaml(name: str) -> Any:
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache
def load_settings() -> Settings:
    return Settings(**_yaml("settings.yaml"))


@lru_cache
def load_universities() -> list[University]:
    return [University(**u) for u in _yaml("universities.yaml")["universities"]]


@lru_cache
def load_geo() -> dict[str, Any]:
    """country_aliases ({CODE: [words]}) and regions ({name: [CODE...]}) from universities.yaml."""
    y = _yaml("universities.yaml")
    return {"country_aliases": y.get("country_aliases", {}), "regions": y.get("regions", {})}


@lru_cache
def load_subdomains() -> list[dict[str, Any]]:
    return _yaml("subdomains.yaml")["subdomains"]


@lru_cache
def load_aliases() -> dict[str, Any]:
    return _yaml("aliases.yaml")


@lru_cache
def load_secrets() -> Secrets:
    load_dotenv(ROOT / ".env.local")
    return Secrets(
        openalex_api_key=os.getenv("OPENALEX_API_KEY", ""),
        openalex_email=os.getenv("OPENALEX_EMAIL", ""),
        s2_api_key=os.getenv("S2_API_KEY", ""),
    )
