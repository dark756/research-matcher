"""SQLite schema (plan §5.1) and helpers."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .config import load_settings, load_universities

SCHEMA = """
CREATE TABLE IF NOT EXISTS http_cache (
  key TEXT PRIMARY KEY, url_redacted TEXT, status INTEGER, body BLOB, fetched_at REAL);
CREATE TABLE IF NOT EXISTS rate_limits (
  service TEXT PRIMARY KEY, last_request_at REAL, credits_remaining INTEGER);
CREATE TABLE IF NOT EXISTS institutions (
  openalex_id TEXT PRIMARY KEY, ror TEXT, name TEXT, short TEXT, country TEXT,
  csrankings_name TEXT, aliases_json TEXT);
CREATE TABLE IF NOT EXISTS works (
  work_id TEXT PRIMARY KEY, doi TEXT, arxiv_id TEXT, s2_paper_id TEXT, s2_corpus_id TEXT, dblp_key TEXT,
  title TEXT, title_norm TEXT, year INTEGER, publication_date TEXT, type TEXT, type_raw TEXT, language TEXT,
  canonical_id TEXT, merged_ids_json TEXT,
  venue_name TEXT, venue_source_id TEXT, venue_type TEXT, venue_s2 TEXT, venue_s2_type TEXT,
  venue_s2_aliases TEXT, venue_hint TEXT,
  doi_url TEXT, openalex_url TEXT, arxiv_abs_url TEXT, landing_page_url TEXT, oa_pdf_url TEXT,
  has_code INTEGER DEFAULT 0, code_source TEXT, code_confidence REAL,
  abstract TEXT, abstract_source TEXT, abstract_missing INTEGER DEFAULT 0, tldr TEXT,
  primary_topic TEXT, subfield TEXT, field TEXT,
  topics_json TEXT, keywords_json TEXT, referenced_works_json TEXT, related_works_json TEXT,
  cited_by_count INTEGER, s2_citations INTEGER, s2_influential_citations INTEGER, fwci REAL,
  top_10pct INTEGER, citation_velocity REAL,
  is_oa INTEGER, oa_status TEXT, license TEXT,
  subdomain_cluster TEXT, llm_label TEXT, labels_agree INTEGER, is_relevant INTEGER,
  confidence TEXT, needs_review INTEGER DEFAULT 0, micro_topic_id INTEGER, micro_topic_label TEXT,
  harvest_source TEXT, harvested_at REAL, s2_status TEXT, s2_fetched_at REAL,
  llm_model TEXT, prompt_version TEXT, extracted_at REAL,
  raw_json BLOB, s2_raw_json BLOB);
CREATE INDEX IF NOT EXISTS idx_works_doi ON works(doi);
CREATE INDEX IF NOT EXISTS idx_works_arxiv ON works(arxiv_id);
CREATE INDEX IF NOT EXISTS idx_works_year ON works(year);
CREATE VIRTUAL TABLE IF NOT EXISTS works_fts USING fts5(title, keywords, abstract, work_id UNINDEXED);
CREATE TABLE IF NOT EXISTS authors (
  author_id TEXT PRIMARY KEY, name TEXT, orcid TEXT, h_index INTEGER, works_count INTEGER,
  last_known_insts_json TEXT);
CREATE TABLE IF NOT EXISTS authorships (
  work_id TEXT, author_id TEXT, position_idx INTEGER, author_position TEXT, is_corresponding INTEGER,
  inst_ids_json TEXT, raw_affiliation TEXT, PRIMARY KEY (work_id, author_id));
CREATE TABLE IF NOT EXISTS work_universities (
  work_id TEXT, inst_id TEXT, senior_author_id TEXT, lead_author_id TEXT, attribution_confidence TEXT,
  professor_verified INTEGER DEFAULT 0, PRIMARY KEY (work_id, inst_id));
CREATE TABLE IF NOT EXISTS faculty (
  name TEXT, name_norm TEXT, affiliation TEXT, homepage TEXT, scholar_id TEXT, orcid TEXT,
  matched_author_id TEXT);
CREATE INDEX IF NOT EXISTS idx_faculty_norm ON faculty(name_norm);
CREATE TABLE IF NOT EXISTS code_links (work_id TEXT, url TEXT, source TEXT, confidence REAL);
CREATE TABLE IF NOT EXISTS prefilter (
  work_id TEXT PRIMARY KEY, kw_scores_json TEXT, dense_scores_json TEXT, best_cluster TEXT, keep INTEGER);
CREATE TABLE IF NOT EXISTS extractions (
  work_id TEXT PRIMARY KEY, model TEXT, prompt_version TEXT, raw_json TEXT, is_relevant INTEGER,
  subdomain_cluster TEXT, sub_problem TEXT, core_method TEXT, confidence TEXT, needs_review INTEGER,
  grounding_dropped INTEGER, error TEXT);
CREATE TABLE IF NOT EXISTS entities (
  work_id TEXT, kind TEXT, normalized TEXT, raw TEXT, PRIMARY KEY (work_id, kind, normalized));
CREATE TABLE IF NOT EXISTS reported_results (
  work_id TEXT, metric TEXT, value TEXT, compared_to TEXT, dataset TEXT);
CREATE TABLE IF NOT EXISTS sparse_vecs (work_id TEXT, field TEXT, lexical_json TEXT,
  PRIMARY KEY (work_id, field));
CREATE TABLE IF NOT EXISTS micro_topics (
  topic_id INTEGER PRIMARY KEY, subdomain TEXT, label TEXT, size INTEGER, universities_json TEXT,
  centroid BLOB, description TEXT);
CREATE TABLE IF NOT EXISTS pairs (
  a TEXT, b TEXT, score REAL, signals_json TEXT, shared_json TEXT, PRIMARY KEY (a, b));
CREATE TABLE IF NOT EXISTS clusters (
  cluster_id INTEGER PRIMARY KEY, subdomain TEXT, members_json TEXT, universities_json TEXT, score REAL);
CREATE TABLE IF NOT EXISTS professor_profiles (
  author_id TEXT, inst_id TEXT, centroid BLOB, subdomain_counts_json TEXT, n_papers INTEGER,
  PRIMARY KEY (author_id, inst_id));
CREATE TABLE IF NOT EXISTS query_log (
  ts REAL, question TEXT, plan_json TEXT, result_ids_json TEXT, answer TEXT, validation_json TEXT);
CREATE TABLE IF NOT EXISTS runs (
  stage TEXT, started REAL, finished REAL, params_json TEXT, counts_json TEXT);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or load_settings().path("db")
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)  # autocommit; explicit BEGIN where needed
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def seed_institutions(conn: sqlite3.Connection) -> int:
    import json

    for u in load_universities():
        conn.execute(
            "INSERT OR REPLACE INTO institutions VALUES (?,?,?,?,?,?,?)",
            (u.openalex_id, u.ror, u.name, u.short, u.country, u.csrankings_name, json.dumps(u.aliases)),
        )
    return len(load_universities())
