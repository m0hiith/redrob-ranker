#!/usr/bin/env python3
"""
Intelligent Candidate Ranking Engine
Redrob Hackathon — Intelligent Candidate Discovery & Ranking Challenge

A recruiter-grade ranking pipeline. It does NOT keyword match. Instead it runs
seven stages:

    1. JD Parser ............. weighted must-have / good-to-have / negative signals
    2. Feature Engineering ... per-candidate numeric features + concept coverage
    3. Semantic Search ....... sentence-transformer cosine fit (lexical fallback)
    4. Career Pattern ........ recruiter-style trajectory & negative-signal analysis
    5. Behavioral Signals .... Redrob platform engagement & reliability
    6. Weighted Ranker ....... final_score = 0.40 sem + 0.30 career + 0.15 behav
                                            + 0.10 location + 0.05 availability
    7. Honeypot Filter ....... fake-candidate detection → forced to the bottom

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
    python rank.py --candidates ./sample_candidates.json --out ./submission.csv

Optional real embeddings (auto-detected; falls back to lexical TF-IDF if absent):
    pip install sentence-transformers
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import math
import os
import re
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─── Stage 1: Job Description (weighted requirements) ─────────────────────────
# Extracted directly from the JD. Edit this block to target a different role.
JOB_DESCRIPTION: Dict = {
    "title": "Senior AI Engineer — Founding Team (search, ranking, retrieval)",
    "responsibilities": [
        "Own the intelligence layer: ranking, retrieval and matching systems",
        "Build and improve large-scale retrieval and ranking systems",
        "Design embedding pipelines and serve them from a vector database",
        "Own offline evaluation with NDCG, MRR and MAP and ship to production",
        "Run production ML services and iterate on relevance",
    ],
    # Must-have requirements (high weight). Each maps to a concept bucket below.
    "must_have": [
        "Embeddings",
        "Retrieval Systems",
        "Vector Databases",
        "Ranking/Search Systems",
        "Python",
        "Evaluation Metrics (NDCG, MRR, MAP)",
        "Product Company Experience",
        "Production ML",
    ],
    # Good-to-have (lighter weight).
    "good_to_have": [
        "LLM Fine-tuning",
        "Learning to Rank",
        "HR Tech",
        "Distributed Systems",
        "Open Source",
    ],
    # Negative signals (recruiter red flags — reduce the career score).
    "negative_signals": [
        "Pure Research",
        "Consulting-only career",
        "CV/speech/robotics without NLP/IR",
        "LangChain-only experience",
        "No production deployments",
        "Title-chasing job-hop pattern",
    ],
    # Where the role lives. The JD is hybrid in Pune/Noida with relocation
    # welcome from Tier-1 Indian cities — NOT a remote role.
    # YoE fit (JD: 5-9 years, ideal 6-8) is encoded in _yoe_score below.
    "location": {"country": "India", "cities": ["Pune", "Noida"], "work_mode": "hybrid"},
}

# ─── Stage 6: Final score weights (sum to 1.0) ───────────────────────────────
WEIGHTS = {
    "semantic":     0.40,   # raised career → reduced semantic by 0.05
    "career":       0.30,   # boosted: now includes YoE modifier; stronger trajectory signal
    "behavioral":   0.15,
    "location":     0.10,
    "availability": 0.05,
}

logger = logging.getLogger(__name__)

# "Today" for recency/availability math. NEVER date.today(): the Stage-3 sandbox
# re-runs ranking on a different day, and a drifting reference date would change
# behavioral scores → different ranking → reproduction failure. The effective
# reference date is derived from the dataset itself (max last_active_date) via
# derive_reference_date(), with this pinned constant as the empty-pool fallback.
DEFAULT_REFERENCE_DATE = date(2026, 6, 1)


def derive_reference_date(candidates: List[Dict]) -> date:
    """
    Deterministic "today": the max parsable last_active_date in the pool.

    Same dataset → same reference date → bit-identical output, no matter when
    or where the pipeline is re-run.
    """
    best: Optional[date] = None
    for c in candidates:
        raw = (c.get("redrob_signals") or {}).get("last_active_date")
        if not raw or not isinstance(raw, str):
            continue
        try:
            d = date.fromisoformat(raw)
        except ValueError:
            continue
        if best is None or d > best:
            best = d
    return best or DEFAULT_REFERENCE_DATE

# ─── Concept ontology ────────────────────────────────────────────────────────
# Maps each must-have / good-to-have requirement to the skill/keyword surface
# forms that satisfy it. This drives concept *coverage* (used by career analysis
# and reasoning) — the headline relevance number still comes from embeddings.
CONCEPTS: Dict[str, set] = {
    "Embeddings": {
        "embeddings", "sentence transformers", "sentence-transformers",
        "word2vec", "glove", "text embeddings", "bge", "minilm",
        # Encoder architectures
        "bi encoder", "bi-encoder", "biencoder", "dual encoder",
        "cross encoder", "cross-encoder", "crossencoder",
        # Late-interaction & passage-retrieval models
        "colbert", "dpr", "e5", "gte",
    },
    "Retrieval Systems": {
        "information retrieval", "retrieval", "rag",
        "retrieval augmented generation", "bm25", "semantic search",
        "haystack", "dense retrieval", "elasticsearch", "opensearch",
        # Search engines
        "solr", "vespa",
        # Retrieval paradigms
        "hybrid search", "sparse retrieval",
        "dense passage retrieval", "dpr", "colbert",
        "two-stage retrieval",
    },
    "Vector Databases": {
        "vector search", "vector database", "faiss", "pinecone",
        "weaviate", "qdrant", "milvus", "pgvector", "chroma",
        # Also used as vector/ANN backends
        "vespa",
        # ANN indexing
        "hnsw", "ann", "approximate nearest neighbor",
        "approximate nearest neighbours",
    },
    "Ranking/Search Systems": {
        "ranking", "search", "recommendation systems", "recommender",
        "learning to rank", "learning-to-rank", "ltr", "xgboost",
        "lightgbm", "search engineer", "relevance",
        # Reranking
        "reranking", "re-ranking", "reranker", "re-ranker",
        "cross encoder", "cross-encoder",
        # Multi-stage pipeline terminology
        "first stage retrieval", "second stage ranking",
    },
    "Python": {"python"},
    "Evaluation Metrics (NDCG, MRR, MAP)": {
        "ndcg", "mrr", "map", "evaluation metrics", "offline evaluation",
        "ab test", "a/b test", "a/b testing", "ab testing",
        "precision@k", "recall@k", "online evaluation",
    },
    "Production ML": {
        "mlops", "mlflow", "kubeflow", "bentoml", "production", "deployment",
        "serving", "model serving", "weights & biases", "wandb",
    },
    "Product Company Experience": {
        # System types that signal real product-ML work
        "ranking system", "recommendation system", "recommendation engine",
        "retrieval system", "search infrastructure", "search platform",
        "search ranking", "consumer product", "marketplace",
        # Scale / production signals
        "production ml", "real-time serving", "at scale",
        "shipped to production", "feature launch",
        # Experimentation (product companies run A/B tests; consulting firms rarely do)
        "a/b testing", "experimentation platform", "online metrics",
    },
    # ── Good-to-have ──
    "LLM Fine-tuning": {
        "fine-tuning llms", "fine-tuning", "peft", "lora", "qlora", "sft", "rlhf",
    },
    "Learning to Rank": {"learning to rank", "learning-to-rank", "ltr"},
    "HR Tech": {"hr tech", "hrtech", "recruiting", "ats", "talent"},
    "Distributed Systems": {
        "distributed systems", "spark", "kafka", "ray", "flink", "kubernetes",
    },
    "Open Source": {"open source", "open-source", "oss", "github"},
}

# Industries that count as genuine *product* company experience.
PRODUCT_INDUSTRIES = {
    "ai/ml", "software", "food delivery", "e-commerce", "ecommerce",
    "fintech", "transportation", "technology", "saas", "internet",
    "product", "consumer internet",
}
# Consulting / services shops — the "consulting-only career" negative signal.
SERVICES_COMPANIES = {
    "wipro", "infosys", "tcs", "tata consultancy", "cognizant", "accenture",
    "mindtree", "capgemini", "hcl", "tech mahindra", "deloitte", "ibm services",
}
SERVICES_INDUSTRIES = {"it services", "consulting", "staffing", "outsourcing"}

# Description phrases that suggest real product-ML work even at services firms.
# Used by _product_company_signal for partial-credit when industry tag is missing/wrong.
_PRODUCT_DESC_SIGNALS = {
    "ranking system", "recommendation system", "recommendation engine",
    "retrieval system", "search infrastructure", "search ranking",
    "consumer product", "marketplace product",
    "real-time serving", "at scale", "shipped to production",
    "a/b testing", "online experiment", "feature launch",
    "production ml system", "production serving",
}

# Out-of-domain specialist skill groups (JD: "primary expertise in CV /
# speech / robotics without significant NLP/IR exposure" is a reject).
CV_SKILLS = {
    "opencv", "yolo", "cnn", "object detection", "image classification",
    "image segmentation", "gans", "computer vision",
}
SPEECH_SKILLS = {
    "asr", "tts", "speech recognition", "speech synthesis", "wav2vec",
    "kaldi", "whisper", "speaker diarization", "audio processing",
}
ROBOTICS_SKILLS = {
    "slam", "ros", "robot operating system", "robotics", "motion planning",
    "path planning", "lidar", "control systems", "autonomous navigation",
}
OOD_SKILL_GROUPS = {
    "cv-only": CV_SKILLS,
    "speech-only": SPEECH_SKILLS,
    "robotics-only": ROBOTICS_SKILLS,
}

# Title-seniority ladder for the title-chaser (job-hopper) signal.
_SENIORITY_RANKS = (
    ("intern", 0), ("trainee", 0),
    ("junior", 1), ("associate", 1),
    ("principal", 5), ("staff", 4), ("lead", 4),
    ("senior", 3),  # checked after staff/principal so "senior staff" ranks 4+
    ("director", 6), ("head", 6), ("vp", 7), ("vice president", 7), ("chief", 7),
)
_DEFAULT_SENIORITY = 2  # plain IC title


def _title_seniority(title: str) -> int:
    best = None
    for marker, level in _SENIORITY_RANKS:
        if re.search(rf"(?<![a-z0-9]){marker}(?![a-z0-9])", title):
            best = level if best is None else max(best, level)
    return _DEFAULT_SENIORITY if best is None else best

# The four pillars that define a full-stack search engineer.
# Covering all four separates specialists from tool-mentioners.
_SEARCH_PILLARS = (
    "Retrieval Systems",
    "Ranking/Search Systems",
    "Evaluation Metrics (NDCG, MRR, MAP)",
    "Production ML",
)

# Job titles that count as directly relevant to the role (beyond the
# Ranking/Search concept surfaces, which are also accepted as title words).
RELEVANT_TITLE_SURFACES = CONCEPTS["Ranking/Search Systems"] | {
    "ml engineer", "machine learning engineer", "ai engineer",
    "applied ml", "nlp engineer", "research engineer", "data scientist",
    "mlops engineer", "search engineer", "recommendation systems engineer",
    "relevance engineer", "ranking engineer", "retrieval engineer",
    "applied scientist", "search platform engineer", "vector search engineer",
}


def _surface_pattern(surfaces) -> "re.Pattern":
    """
    Compile a word-boundary alternation for a set of surface forms.

    Plain `surface in text` substring tests false-positive at scale:
    "search" hits "research", "map" hits "roadmap", "ann" hits "planning",
    "rag" hits "storage". Custom lookaround boundaries (not \\b) are used so
    surfaces containing non-word chars ("a/b test", "c++") still anchor on
    their alphanumeric edges. Longest-first ordering makes multi-word
    surfaces win over their own substrings.
    """
    alts = sorted((re.escape(s) for s in surfaces), key=len, reverse=True)
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(alts) + r")(?![a-z0-9])")


# Compiled once at module load — reused across all 100K candidates.
CONCEPT_PATTERNS: Dict[str, "re.Pattern"] = {
    name: _surface_pattern(surfaces) for name, surfaces in CONCEPTS.items()
}
_RELEVANT_TITLE_PATTERN = _surface_pattern(RELEVANT_TITLE_SURFACES)
_SERVICES_COMPANY_PATTERN = _surface_pattern(SERVICES_COMPANIES)
_PRODUCT_DESC_PATTERN = _surface_pattern(_PRODUCT_DESC_SIGNALS)
_RESEARCH_TITLE_PATTERN = _surface_pattern({"research", "researcher"})
_LANGCHAIN_PATTERN = _surface_pattern({"langchain"})

PROFICIENCY = {"beginner": 0.25, "intermediate": 0.50, "advanced": 0.80, "expert": 1.00}
_VALID_PROFICIENCIES = frozenset(PROFICIENCY)

# A listed skill counts at full coverage only with usage evidence behind it.
TRUSTED_ENDORSEMENTS = 5
TRUSTED_DURATION_MONTHS = 12

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "at",
    "by", "from", "as", "is", "are", "was", "were", "be", "this", "that", "it",
    "team", "worked", "built", "using", "used", "work",
}


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _yoe_score(yoe: float) -> float:
    """
    Smooth seniority-fit score in [0, 1] for a Search/Ranking ML IC role.

    Sweet spot: 5–9 years (score ≥ 0.90).  Peak at 7 years (score = 1.00).
    Formula: piecewise linear between key breakpoints — no hard cliffs.

        0y → 0.20  (floor; some signal from other dimensions)
        3y → 0.60  (junior but capable)
        5y → 0.90  (entering ideal band)
        7y → 1.00  (peak: enough production depth, not overqualified)
        9y → 0.90  (exiting ideal band)
       14y → 0.70  (senior: valuable but risk of overqualified/overpriced)
      15y+ → 0.70  (floor for very experienced candidates)

    Used as a multiplicative modifier inside score_career:
        career_score *= (0.80 + 0.20 * _yoe_score(yoe))
    This bounds the YoE effect to ±20% of the raw career score, preserving
    the primacy of actual job-trajectory signals.
    """
    yoe = float(yoe or 0)
    if yoe <= 0:
        return 0.20
    if yoe <= 3:
        return 0.20 + 0.40 * (yoe / 3.0)
    if yoe < 5:
        # strictly < 5 so yoe=5 falls into the next branch and returns exactly 0.90
        return 0.60 + 0.30 * ((yoe - 3) / 2.0)
    if yoe <= 7:
        return 0.90 + 0.10 * ((yoe - 5) / 2.0)
    if yoe <= 9:
        return 1.00 - 0.10 * ((yoe - 7) / 2.0)
    if yoe <= 14:
        return 0.90 - 0.20 * ((yoe - 9) / 5.0)
    return 0.70


# ─── Stage 3: Semantic Search ─────────────────────────────────────────────────

DEFAULT_MODEL_DIR = str(Path(__file__).parent / "models" / "bge-small-en-v1.5")

# Lexical TF-IDF is for tests/smoke runs only — it does not scale to the full
# pool (per-doc bigram vectors are multi-GB at 100K) and produces a different
# ranking than the submitted semantic one.
LEXICAL_MAX_CANDIDATES = 20_000


class Embedder:
    """
    Produces semantic similarity between the JD and each candidate.

    Loads a sentence-transformer strictly from a LOCAL directory (vendored by
    download_model.py as a documented pre-computation step) — the ranking step
    never touches the network, per the submission spec.

    A TF-IDF lexical fallback exists for tests and small smoke runs only and
    must be opted into explicitly (`allow_lexical_fallback=True`). Silent mode
    degradation is forbidden: it would make the submitted ranking
    irreproducible in the offline evaluation sandbox.
    """

    # BGE models require an asymmetric prefix on the query side only.
    _BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        model_dir: Optional[str] = None,
        allow_lexical_fallback: bool = False,
    ) -> None:
        self.mode = "lexical"
        self._model = None
        self._model_id: Optional[str] = None
        self._query_prefix: str = ""
        # None  → semantic mode loaded successfully.
        # str   → human-readable reason the local model could not be loaded.
        self.load_error: Optional[str] = None

        # Hard guarantee: no network during ranking, even if a cache is stale.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        model_path = Path(model_dir or DEFAULT_MODEL_DIR)
        if not (model_path / "config.json").exists():
            self.load_error = (
                f"local model not found at {model_path} "
                f"(run `python download_model.py` once, with network, to vendor it)"
            )
        else:
            try:
                from sentence_transformers import SentenceTransformer  # noqa: WPS433
                self._model = SentenceTransformer(str(model_path), device="cpu")
                self._model_id = model_path.name
                self._query_prefix = (
                    self._BGE_QUERY_PREFIX if "bge" in model_path.name.lower() else ""
                )
                self.mode = "semantic"
                logger.info("Embedder: loaded %s from %s", model_path.name, model_path)
            except ImportError as exc:
                self.load_error = f"sentence-transformers not installed: {exc}"
            except Exception as exc:
                self.load_error = f"{model_path}: {type(exc).__name__}: {exc}"

        if self.mode == "lexical":
            if not allow_lexical_fallback:
                raise RuntimeError(
                    "Semantic model unavailable and lexical fallback not enabled.\n"
                    f"  Reason: {self.load_error}\n"
                    "  Fix:    python download_model.py   (pre-computation, needs network once)\n"
                    "  Or:     pass --allow-lexical-fallback (tests/smoke runs ONLY — "
                    "produces a different, non-submittable ranking)."
                )
            logger.warning(
                "Embedder: TF-IDF lexical mode (explicitly enabled) — %s", self.load_error
            )

    def similarities(self, jd_doc: str, cand_docs: List[str]) -> List[float]:
        if not cand_docs:
            return []
        if self.mode == "lexical" and len(cand_docs) > LEXICAL_MAX_CANDIDATES:
            raise RuntimeError(
                f"Lexical fallback is fenced to ≤{LEXICAL_MAX_CANDIDATES} candidates "
                f"(got {len(cand_docs)}). It exists for tests/smoke runs only — "
                "vendor the model with `python download_model.py` for full-pool runs."
            )
        raw = (
            self._semantic(jd_doc, cand_docs)
            if self.mode == "semantic"
            else self._lexical(jd_doc, cand_docs)
        )
        return [_clamp(s) for s in raw]

    def _semantic(self, jd_doc: str, cand_docs: List[str]) -> List[float]:
        import numpy as np  # bundled with sentence-transformers

        query = self._query_prefix + jd_doc
        jd_emb = np.asarray(
            self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        )
        cand_emb = np.asarray(
            self._model.encode(cand_docs, normalize_embeddings=True, show_progress_bar=False)
        )
        sims = cand_emb @ jd_emb[0]
        return [_clamp(float(s)) for s in sims]

    def _lexical(self, jd_doc: str, cand_docs: List[str]) -> List[float]:
        docs = [jd_doc] + cand_docs
        tokenised = [_tokenize(d) for d in docs]

        # IDF across the corpus.
        df: Counter = Counter()
        for toks in tokenised:
            for term in set(toks):
                df[term] += 1
        n = len(docs)
        idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}

        vectors = [_tfidf_vector(toks, idf) for toks in tokenised]
        jd_vec = vectors[0]
        return [_cosine(jd_vec, v) for v in vectors[1:]]


def _tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9][a-z0-9+#.]*", text.lower())
    words = [w for w in words if w not in STOPWORDS and len(w) > 1]
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return words + bigrams


def _tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf = Counter(tokens)
    total = sum(tf.values()) or 1
    return {t: (c / total) * idf.get(t, 1.0) for t, c in tf.items()}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return _clamp(dot / (na * nb)) if na and nb else 0.0


def jd_document(jd: Dict) -> str:
    """
    Combine title + requirements + responsibilities into one JD text.

    Each requirement is expanded into its concrete surface forms (e.g. "Vector
    Databases" → "faiss pinecone weaviate qdrant milvus") so that similarity is
    measured against the real tools candidates list, not abstract category names.
    Must-haves are repeated to weight them more heavily.
    """
    parts = [jd["title"]]
    parts += jd["responsibilities"]
    for req in jd["must_have"]:
        surfaces = " ".join(CONCEPTS.get(req, set()))
        parts += [f"{req} {surfaces}".strip()] * 2
    for req in jd["good_to_have"]:
        surfaces = " ".join(CONCEPTS.get(req, set()))
        parts.append(f"{req} {surfaces}".strip())
    return ". ".join(parts)


def candidate_document(c: Dict) -> str:
    """Combine resume text, skills, projects, experience and job titles."""
    p = c.get("profile", {})
    parts = [
        p.get("headline", ""),
        p.get("summary", ""),
        p.get("current_title", ""),
    ]
    parts += [s.get("name", "") for s in c.get("skills", [])]
    for job in c.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    for edu in c.get("education", []):
        parts.append(edu.get("field_of_study", ""))
    return ". ".join(part for part in parts if part)


# ─── Stage 2: Feature Engineering ─────────────────────────────────────────────

def candidate_text_blob(c: Dict) -> str:
    """Lowercased haystack of everything a candidate has written/claimed."""
    return candidate_document(c).lower()


def concept_coverage(c: Dict, jd: Dict) -> Dict[str, float]:
    """
    For each must/good requirement, 0–1 coverage from the candidate's surface.

    "Product Company Experience" is special-cased from industry/company data.
    """
    blob = candidate_text_blob(c)
    skills_list = c.get("skills", [])
    skill_names = {s.get("name", "").lower() for s in skills_list}
    # Trust signals per skill (schema doc: "duration_months + endorsements are
    # the trust signal against keyword-stuffers").
    skill_endorsements = {
        s.get("name", "").lower(): (s.get("endorsements") or 0)
        for s in skills_list
    }
    skill_durations = {
        s.get("name", "").lower(): (s.get("duration_months") or 0)
        for s in skills_list
    }
    coverage: Dict[str, float] = {}

    requirements = jd["must_have"] + jd["good_to_have"]
    for req in requirements:
        if req == "Product Company Experience":
            coverage[req] = _product_company_signal(c)
            continue
        surfaces = CONCEPTS.get(req, set())
        hit = 0.0
        for surface in surfaces:
            if surface in skill_names:
                # Explicit skill = strong, but only at full credit when backed
                # by usage evidence. A bare listing with zero endorsements AND
                # zero duration is exactly what keyword-stuffers produce.
                trusted = (
                    skill_endorsements.get(surface, 0) >= TRUSTED_ENDORSEMENTS
                    or skill_durations.get(surface, 0) >= TRUSTED_DURATION_MONTHS
                )
                hit = max(hit, 1.0 if trusted else 0.8)
        if hit < 0.6 and CONCEPT_PATTERNS[req].search(blob):
            hit = max(hit, 0.6)              # mentioned in text = moderate
        coverage[req] = hit
    return coverage


def _product_company_signal(c: Dict) -> float:
    history = c.get("career_history", []) or []
    if not history:
        ind = c.get("profile", {}).get("current_industry", "").lower()
        return 0.8 if ind in PRODUCT_INDUSTRIES else 0.2
    best = 0.0
    for job in history:
        ind = (job.get("industry", "") or "").lower()
        comp = (job.get("company", "") or "").lower()
        desc = (job.get("description", "") or "").lower()
        if ind in PRODUCT_INDUSTRIES and not _SERVICES_COMPANY_PATTERN.search(comp):
            best = max(best, 1.0)
        elif ind in PRODUCT_INDUSTRIES:
            best = max(best, 0.6)
        elif _PRODUCT_DESC_PATTERN.search(desc):
            # Services/consulting firm but description shows product-ML work style
            # (e.g., shipped ranking system for a client's product). Partial credit.
            best = max(best, 0.4)
    return best


def engineer_features(c: Dict, jd: Dict, coverage: Dict[str, float]) -> Dict:
    """Stage 2 numeric features, surfaced for transparency and reasoning."""
    p = c.get("profile", {})
    must = jd["must_have"]
    must_covered = sum(1 for r in must if coverage.get(r, 0) >= 0.6)
    n_pillars = sum(1 for p_ in _SEARCH_PILLARS if coverage.get(p_, 0) >= 0.6)
    return {
        "years_exp": p.get("years_of_experience", 0),
        "must_have_covered": must_covered,
        "must_have_total": len(must),
        "covered_concepts": [r for r in must if coverage.get(r, 0) >= 0.6],
        "n_search_pillars": n_pillars,
    }


def _search_depth_bonus(coverage: Dict[str, float]) -> float:
    """
    Reward breadth across the four search-engineering pillars.
    Each pillar covered at ≥ 0.6 adds 0.02 to the career score (max +0.08).
    Separates 'mentioned FAISS once' from a genuine full-stack search engineer.
    """
    n = sum(1 for p in _SEARCH_PILLARS if coverage.get(p, 0) >= 0.6)
    return n / 4.0 * 0.08


# ─── Stage 4: Career Pattern Analysis (think like a recruiter) ────────────────

def score_career(c: Dict, coverage: Dict[str, float]) -> Tuple[float, List[str]]:
    """
    Rewards a real production trajectory in relevant roles at product companies.
    Applies negative-signal penalties (consulting-only, pure research, CV-only,
    LangChain-only, no production). Returns (score, list_of_flags).
    """
    history = c.get("career_history", []) or []
    flags: List[str] = []

    if not history:
        return 0.15, flags

    weighted_sum = 0.0
    weight_total = 0.0
    for i, job in enumerate(history):
        w = 1.0 if job.get("is_current") else max(0.30, 1.0 - i * 0.20)
        title = (job.get("title", "") or "").lower()
        desc = (job.get("description", "") or "").lower()
        industry = (job.get("industry", "") or "").lower()
        company = (job.get("company", "") or "").lower()

        title_s = 0.85 if _RELEVANT_TITLE_PATTERN.search(title) else 0.25
        concept_hits = sum(
            1 for name in
            ("Retrieval Systems", "Vector Databases",
             "Ranking/Search Systems", "Embeddings", "Production ML")
            if CONCEPT_PATTERNS[name].search(desc)
        )
        desc_s = _clamp(concept_hits / 3.0)
        prod_s = 1.0 if (industry in PRODUCT_INDUSTRIES and
                         not _SERVICES_COMPANY_PATTERN.search(company)) else 0.35

        job_score = title_s * 0.40 + desc_s * 0.35 + prod_s * 0.25

        # Duration weight: a 2-year deep dive outweighs a 3-month gig.
        # Unknown/zero duration treated as 12 months (neutral-ish).
        dur_months = (job.get("duration_months", 0) or 0)
        duration_weight = min((dur_months if dur_months > 0 else 12) / 24.0, 1.0)
        combined_w = w * duration_weight
        weighted_sum += job_score * combined_w
        weight_total += combined_w

    score = _clamp(weighted_sum / max(weight_total, 1.0))

    # Trajectory bonus: currently in a relevant, production role.
    cur = history[0]
    if cur.get("is_current") and coverage.get("Production ML", 0) >= 0.6:
        score = _clamp(score + 0.05)

    # Depth bonus: reward covering all four search-engineering pillars.
    score = _clamp(score + _search_depth_bonus(coverage))

    # YoE modifier: scales career ±20% based on seniority fit.
    # Applied before negative signals so that trajectory penalties compound
    # naturally on top of the seniority-adjusted base.
    yoe = float((c.get("profile") or {}).get("years_of_experience", 0) or 0)
    score = _clamp(score * (0.80 + 0.20 * _yoe_score(yoe)))

    score, flags = _apply_negative_signals(c, coverage, history, score)
    return score, flags


def _apply_negative_signals(
    c: Dict, coverage: Dict[str, float],
    history: List[Dict], score: float,
) -> Tuple[float, List[str]]:
    flags: List[str] = []
    skill_names = {s.get("name", "").lower() for s in c.get("skills", [])}

    # Consulting-only career: every role at a services shop.
    def is_services(job: Dict) -> bool:
        ind = (job.get("industry", "") or "").lower()
        comp = (job.get("company", "") or "").lower()
        return ind in SERVICES_INDUSTRIES or bool(_SERVICES_COMPANY_PATTERN.search(comp))

    if history and all(is_services(j) for j in history):
        score *= 0.55
        flags.append("consulting-only")

    # Pure research: research titles with no production/MLOps footprint.
    titles = " ".join((j.get("title", "") or "").lower() for j in history)
    if _RESEARCH_TITLE_PATTERN.search(titles) and coverage.get("Production ML", 0) < 0.6:
        score *= 0.70
        flags.append("pure-research")

    # Out-of-domain specialist: CV/speech/robotics-heavy skills with no
    # retrieval/ranking depth (JD explicit reject). One penalty is enough.
    core_hits = (
        coverage.get("Retrieval Systems", 0)
        + coverage.get("Vector Databases", 0)
        + coverage.get("Ranking/Search Systems", 0)
    )
    for ood_flag, group in OOD_SKILL_GROUPS.items():
        if len(skill_names & group) >= 2 and core_hits < 0.6:
            score *= 0.60
            flags.append(ood_flag)
            break

    # Title-chaser: job-hops every ~1.5y up the seniority ladder (JD explicit
    # DO-NOT-WANT: "want 3+ year commitment"). Three+ consecutive completed
    # stints under 20 months, climbing titles toward the present.
    non_current = [j for j in history if not j.get("is_current")]
    streak = []
    for job in non_current:
        dur = job.get("duration_months", 0) or 0
        if 0 < dur < 20:
            streak.append(job)
        else:
            break
    if len(streak) >= 3:
        # history is most-recent-first; reverse to chronological order.
        levels = [
            _title_seniority((j.get("title") or "").lower())
            for j in reversed(streak)
        ]
        non_decreasing = all(b >= a for a, b in zip(levels, levels[1:]))
        if non_decreasing and levels[-1] > levels[0]:
            score *= 0.75
            flags.append("title-chaser")

    # LangChain-only: rides LangChain/RAG buzzwords without real retrieval depth.
    has_langchain = (
        "langchain" in skill_names
        or bool(_LANGCHAIN_PATTERN.search(candidate_text_blob(c)))
    )
    if has_langchain and coverage.get("Vector Databases", 0) < 0.6 and core_hits < 0.6:
        score *= 0.65
        flags.append("langchain-only")

    # No production deployments anywhere.
    if coverage.get("Production ML", 0) < 0.6 and not any(
        CONCEPT_PATTERNS["Production ML"].search((j.get("description", "") or "").lower())
        for j in history
    ):
        score *= 0.80
        flags.append("no-production")

    return _clamp(score), flags


# ─── Stage 5: Behavioral Signals ──────────────────────────────────────────────

def score_behavioral(c: Dict, reference_date: date = DEFAULT_REFERENCE_DATE) -> float:
    """
    Recruiter-facing reliability & engagement. A strong skill match means little
    if the candidate is inactive or never responds.

        recruiter_response_rate     0.28
        last_active recency         0.18
        interview_completion_rate   0.16
        github_activity_score       0.12
        profile_completeness        0.10
        saved_by_recruiters_30d     0.08
        offer_acceptance_rate       0.08
    """
    s = c.get("redrob_signals", {})

    response = s.get("recruiter_response_rate", 0) or 0
    interview = s.get("interview_completion_rate", 0) or 0

    gh_raw = s.get("github_activity_score", -1)
    github = _clamp(gh_raw / 100.0) if gh_raw is not None and gh_raw >= 0 else 0.20

    complete = (s.get("profile_completeness_score", 0) or 0) / 100.0

    recency = 0.50
    last = s.get("last_active_date", "")
    if last:
        try:
            days = (reference_date - date.fromisoformat(last)).days
            recency = _clamp(1.0 - days / 365)
        except ValueError:
            pass

    saved = _clamp((s.get("saved_by_recruiters_30d", 0) or 0) / 15.0)

    offer_raw = s.get("offer_acceptance_rate", -1)
    offer = offer_raw if offer_raw is not None and offer_raw >= 0 else 0.50

    raw = (
        response   * 0.28 +
        recency    * 0.18 +
        interview  * 0.16 +
        github     * 0.12 +
        complete   * 0.10 +
        saved      * 0.08 +
        offer      * 0.08
    )
    return _clamp(raw)


# ─── Location & Availability ──────────────────────────────────────────────────

# JD: "Hyderabad, Mumbai, Delhi NCR welcome" + relocation from Tier-1 cities.
_WELCOME_CITIES = {
    "hyderabad", "mumbai", "navi mumbai", "delhi", "new delhi", "delhi ncr",
    "gurgaon", "gurugram", "ghaziabad", "faridabad", "bengaluru", "bangalore",
}


def score_location(c: Dict, jd: Dict) -> float:
    """
    City-tier fit for a hybrid Pune/Noida role.

        JD cities (Pune/Noida)           1.00
        welcome metros (Hyd/Mum/NCR/Blr) 0.85
        elsewhere in India               0.65
        abroad, willing to relocate      0.45
        abroad                           0.20

    Work-mode preference adjusts ±20%: the JD is hybrid, so a hard "remote"
    preference is a real mismatch while hybrid/flexible/onsite all fit.
    """
    p = c.get("profile", {})
    s = c.get("redrob_signals", {})
    target = jd.get("location", {})

    country = (p.get("country", "") or "").lower()
    loc_text = (p.get("location", "") or "").lower()
    jd_cities = {city.lower() for city in target.get("cities", [])}

    if country == (target.get("country", "") or "").lower():
        if any(city in loc_text for city in jd_cities):
            base = 1.00
        elif any(city in loc_text for city in _WELCOME_CITIES):
            base = 0.85
        else:
            base = 0.65
    elif s.get("willing_to_relocate"):
        base = 0.45
    else:
        base = 0.20

    mode = (s.get("preferred_work_mode", "") or "").lower()
    if mode in {"hybrid", "flexible", "onsite"}:
        mode_s = 1.0
    elif mode == "remote":
        mode_s = 0.5   # JD is hybrid — pure remote is a mismatch
    else:
        mode_s = 0.7   # unstated
    return _clamp(base * 0.80 + mode_s * 0.20)


def score_availability(c: Dict) -> float:
    s = c.get("redrob_signals", {})
    score = 0.0
    if s.get("open_to_work_flag"):
        score += 0.50

    notice = s.get("notice_period_days", 90)
    if notice is None:
        notice = 90
    if notice <= 15:
        score += 0.40
    elif notice <= 30:
        score += 0.33
    elif notice <= 60:
        score += 0.22
    elif notice <= 90:
        score += 0.12

    if (s.get("applications_submitted_30d", 0) or 0) > 0:
        score += 0.10
    return _clamp(score)


# ─── Stage 7: Honeypot Detection ──────────────────────────────────────────────

def _parse_date(raw: object) -> Optional[date]:
    """Tolerant ISO date parse: 'YYYY-MM-DD' or 'YYYY-MM'; anything else → None."""
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()
    for candidate in (text, f"{text}-01"):
        try:
            return date.fromisoformat(candidate)
        except ValueError:
            continue
    return None


# Tolerances for the date-consistency checks. Honeypot flags are a hard gate
# (>10% in top 100 = DQ), so precision matters more than recall: generous
# grace windows keep legitimate-but-sloppy profiles out of the net.
_DURATION_MISMATCH_GRACE_MONTHS = 6.0
_FUTURE_DATE_GRACE_DAYS = 60
_ENDED_CURRENT_ROLE_GRACE_DAYS = 90
_OVERLAP_SOFT_LIMIT_MONTHS = 24.0
_DAYS_PER_MONTH = 30.44


def _date_consistency_flags(
    c: Dict, reference_date: date,
) -> Tuple[int, int, List[str]]:
    """
    The spec's example honeypots are date-impossible profiles. Checks the
    internal consistency of career_history and platform dates.
    Returns (hard_count, soft_count, reasons).
    """
    history = c.get("career_history", []) or []
    signals = c.get("redrob_signals", {}) or {}
    hard = 0
    soft = 0
    reasons: List[str] = []

    intervals: List[Tuple[date, date]] = []
    for job in history:
        if not isinstance(job, dict):
            continue
        start = _parse_date(job.get("start_date"))
        end = _parse_date(job.get("end_date"))
        is_current = bool(job.get("is_current"))
        dur = job.get("duration_months")

        if start and end and start > end:
            hard += 1
            reasons.append(f"role ends ({end}) before it starts ({start})")

        future_cutoff = reference_date.toordinal() + _FUTURE_DATE_GRACE_DAYS
        for d, label in ((start, "starts"), (end, "ends")):
            if d and d.toordinal() > future_cutoff:
                hard += 1
                reasons.append(f"role {label} in the future ({d})")
                break

        if (
            is_current and end
            and (reference_date - end).days > _ENDED_CURRENT_ROLE_GRACE_DAYS
        ):
            hard += 1
            reasons.append(f"marked current but ended {end}")

        span_end = end or (reference_date if is_current else None)
        if (
            start and span_end and span_end >= start
            and isinstance(dur, (int, float)) and dur > 0
        ):
            span_months = (span_end - start).days / _DAYS_PER_MONTH
            # Two-sided check only when both dates are explicit. For open
            # current roles the span runs to the reference date, so a smaller
            # duration_months is just data lag — only the impossible direction
            # (claimed tenure exceeding the available time) is fabrication.
            mismatch = (
                abs(dur - span_months) if end else (dur - span_months)
            )
            if mismatch > _DURATION_MISMATCH_GRACE_MONTHS:
                hard += 1
                reasons.append(
                    f"duration_months={dur:.0f} contradicts dates "
                    f"({start}–{span_end} ≈ {span_months:.0f}mo)"
                )

        if start and span_end and span_end > start:
            intervals.append((start, span_end))

    # Soft: heavy overlap between supposedly full-time roles.
    overlap_days = 0
    intervals.sort()
    for (s1, e1), (s2, e2) in zip(intervals, intervals[1:]):
        overlap_days += max(0, (min(e1, e2) - s2).days)
    if overlap_days / _DAYS_PER_MONTH > _OVERLAP_SOFT_LIMIT_MONTHS:
        soft += 1
        reasons.append(
            f"roles overlap by {overlap_days / _DAYS_PER_MONTH:.0f} months"
        )

    # Soft: platform activity precedes the account itself.
    signup = _parse_date(signals.get("signup_date"))
    last_active = _parse_date(signals.get("last_active_date"))
    if signup and last_active and last_active < signup:
        soft += 1
        reasons.append(f"last active ({last_active}) before signup ({signup})")

    return hard, soft, reasons


def detect_honeypot(
    c: Dict, reference_date: date = DEFAULT_REFERENCE_DATE,
) -> Tuple[bool, List[str]]:
    """
    Catches organizer-inserted fakes: impossible timelines, fabricated tenure,
    expert claims with no usage. Hard flags force rejection; soft flags accumulate.
    Returns (is_honeypot, reasons).
    """
    p = c.get("profile", {})
    yoe = p.get("years_of_experience", 0) or 0
    skills = c.get("skills", [])
    history = c.get("career_history", []) or []

    hard, soft, reasons = _date_consistency_flags(c, reference_date)

    # Hard: used a skill longer than the entire career (+2yr grace).
    max_skill_years = max(
        (sk.get("duration_months", 0) or 0) / 12.0 for sk in skills
    ) if skills else 0.0
    if max_skill_years > yoe + 2.0:
        hard += 1
        reasons.append(f"skill tenure {max_skill_years:.1f}y > exp {yoe:.1f}y")

    # Hard: a single role longer than the whole stated career (+1yr grace).
    for job in history:
        if (job.get("duration_months", 0) or 0) / 12.0 > yoe + 1.0:
            hard += 1
            reasons.append("single role exceeds total experience")
            break

    # Expert skills with no real tenure. Many experts where *every* one has zero
    # usage is fabrication (hard); a partial pattern is merely suspicious (soft).
    # Gap closed: previously 9 experts / 8 zero-usage triggered neither branch
    # because the hard check required ALL zero and the soft check required >10.
    expert = [sk for sk in skills if sk.get("proficiency") == "expert"]
    expert_no_use = [sk for sk in expert if (sk.get("duration_months", 0) or 0) == 0]
    if len(expert) >= 8 and len(expert_no_use) >= len(expert) - 1:
        # All, or all-but-one, expert skills with zero usage → hard fabrication signal.
        hard += 1
        reasons.append(f"{len(expert_no_use)}/{len(expert)} expert skills with zero usage")
    elif len(expert) >= 6 and len(expert_no_use) >= 5:
        soft += 1
        reasons.append("many expert skills with zero usage")

    # Soft: expert claim contradicted by a failing assessment.
    assess = c.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}
    for sk in expert:
        score = assess.get(sk.get("name"))
        if score is not None and score < 30:
            soft += 1
            reasons.append(f"expert {sk.get('name')} but assessment {score:.0f}")
            break

    # Soft: claimed YoE far exceeds summed real tenure.
    summed_years = sum((j.get("duration_months", 0) or 0) for j in history) / 12.0
    if history and yoe > summed_years + 4.0 and summed_years > 0:
        soft += 1
        reasons.append(f"claims {yoe:.1f}y but history sums to {summed_years:.1f}y")

    is_honeypot = hard >= 1 or soft >= 3
    return is_honeypot, reasons


# ─── Reasoning ────────────────────────────────────────────────────────────────

def build_reasoning(
    c: Dict, features: Dict, semantic: float, behavioral: float,
    career_flags: List[str], honeypot_reasons: List[str], is_honeypot: bool,
) -> str:
    p = c.get("profile", {})
    s = c.get("redrob_signals", {})
    history = c.get("career_history", []) or []

    if is_honeypot:
        detail = "; ".join(honeypot_reasons) or "fabricated profile"
        return f"HONEYPOT — {detail}."

    title = p.get("current_title", "Unknown")
    yoe = p.get("years_of_experience", 0) or 0
    rr = s.get("recruiter_response_rate", 0) or 0
    company = (history[0].get("company", "") if history else "") or ""
    location = p.get("city", "") or p.get("location", "") or p.get("country", "") or ""

    # Skill evidence: list concepts actually covered in candidate's profile.
    covered_concepts = features["covered_concepts"]
    skill_phrase = ", ".join(covered_concepts[:3]) if covered_concepts else "no core skill matches"

    # Build a single readable sentence specific to this candidate.
    at_company = f" at {company}" if company else ""
    loc_phrase = f"; {location}-based" if location else ""

    parts = [f"{title}{at_company} with {yoe:.1f}y exp; covers {skill_phrase}"]

    if rr >= 0.70:
        parts.append(f"highly responsive ({rr:.0%})")
    elif rr >= 0.40:
        parts.append(f"moderate responsiveness ({rr:.0%})")
    else:
        parts.append(f"low response rate ({rr:.0%})")

    gh = s.get("github_activity_score", -1)
    if gh is not None and gh >= 60:
        parts.append("active GitHub")

    n_pillars = features.get("n_search_pillars", 0)
    if n_pillars == 4:
        parts.append("full-stack search depth (4/4 pillars)")
    elif n_pillars == 3:
        parts.append("strong search depth (3/4 pillars)")

    if career_flags:
        flag_map = {
            "consulting-only": "consulting-only background",
            "pure-research": "research-only, limited production evidence",
            "cv-only": "Computer Vision focus — limited retrieval/ranking depth",
            "langchain-only": "LangChain usage without deeper retrieval stack",
            "no-production": "no production deployment signals",
        }
        concerns = "; ".join(flag_map.get(f, f) for f in career_flags)
        parts.append(f"concern: {concerns}")

    return ("; ".join(parts) + loc_phrase + ".").replace(";;", ";")  # guard dupes


# ─── Main Pipeline ────────────────────────────────────────────────────────────

# Stage-A → Stage-B handoff size. The CSV needs the top 100; re-ranking the
# top 2,000 prescreened candidates (20× margin) keeps every plausible
# top-100 contender while cutting CPU embedding work from 100K docs
# (~30-55 min, busts the 5-min budget) to ~2K docs (well under a minute).
PRESCREEN_FINALISTS = 2000


def _score_components(
    c: Dict, jd: Dict, reference_date: date,
) -> Tuple[Dict[str, float], float, List[str], float, float, float]:
    """Shared non-semantic scoring used by both stages — identical by construction."""
    coverage = concept_coverage(c, jd)
    career, career_flags = score_career(c, coverage)
    behavioral = score_behavioral(c, reference_date)
    location = score_location(c, jd)
    availability = score_availability(c)
    return coverage, career, career_flags, behavioral, location, availability


def _coverage_proxy(coverage: Dict[str, float], jd: Dict) -> float:
    """
    Stage-A stand-in for the semantic score: must-haves count double.
    Only used to pick finalists; the final blend uses true embeddings.
    """
    must, good = jd["must_have"], jd["good_to_have"]
    total = 2 * sum(coverage.get(r, 0) for r in must) + sum(coverage.get(r, 0) for r in good)
    return total / (2 * len(must) + len(good))


def prescreen(
    candidates: List[Dict], jd: Dict, reference_date: date,
    finalists: int = PRESCREEN_FINALISTS,
) -> List[Dict]:
    """
    Stage A: rank the full pool on cheap features only (concept coverage,
    career, behavioral, location, availability) and return the finalists for
    semantic re-ranking. Deterministic tie-break by candidate_id.
    """
    if len(candidates) <= finalists:
        return candidates
    ranked = []
    for c in candidates:
        coverage, career, _, behavioral, location, availability = (
            _score_components(c, jd, reference_date)
        )
        score = (
            _coverage_proxy(coverage, jd) * WEIGHTS["semantic"] +
            career       * WEIGHTS["career"] +
            behavioral   * WEIGHTS["behavioral"] +
            location     * WEIGHTS["location"] +
            availability * WEIGHTS["availability"]
        )
        ranked.append((-score, c.get("candidate_id", ""), c))
    ranked.sort(key=lambda x: (x[0], x[1]))
    return [c for _, _, c in ranked[:finalists]]


def rank_candidates(
    candidates: List[Dict], jd: Dict, embedder: Embedder,
    reference_date: Optional[date] = None,
    finalists: int = PRESCREEN_FINALISTS,
    show_timings: bool = False,
) -> List[Dict]:
    if reference_date is None:
        reference_date = derive_reference_date(candidates)

    # Stage A: cheap full-pool screen (no embeddings).
    t0 = time.perf_counter()
    pool = prescreen(candidates, jd, reference_date, finalists)
    t_screen = time.perf_counter() - t0

    # Stage B: true semantic similarity for finalists only.
    t0 = time.perf_counter()
    jd_doc = jd_document(jd)
    cand_docs = [candidate_document(c) for c in pool]
    semantic_scores = embedder.similarities(jd_doc, cand_docs)
    t_embed = time.perf_counter() - t0

    t0 = time.perf_counter()
    scored = []
    for c, semantic in zip(pool, semantic_scores):
        coverage, career, career_flags, behavioral, location, availability = (
            _score_components(c, jd, reference_date)
        )
        features = engineer_features(c, jd, coverage)
        is_honeypot, hp_reasons = detect_honeypot(c, reference_date)

        final = (
            semantic     * WEIGHTS["semantic"] +
            career       * WEIGHTS["career"] +
            behavioral   * WEIGHTS["behavioral"] +
            location     * WEIGHTS["location"] +
            availability * WEIGHTS["availability"]
        )

        # Suppress honeypot scores to 0.0.  Keeping the raw score would mislead
        # NDCG/MRR evaluation (a well-crafted fake could report 0.85 at rank 90).
        # Zero is still within the [0,1] CSV spec and, combined with the
        # "HONEYPOT — " reasoning prefix, gives downstream consumers two
        # independent signals without breaking the 4-column format.
        if is_honeypot:
            final = 0.0

        scored.append({
            "candidate_id": c["candidate_id"],
            "score": round(final, 4),
            "reasoning": build_reasoning(
                c, features, semantic, behavioral,
                career_flags, hp_reasons, is_honeypot,
            ),
            "_honeypot": is_honeypot,
            "_breakdown": {
                "semantic":     round(semantic,     4),
                "career":       round(career,       4),
                "behavioral":   round(behavioral,   4),
                "location":     round(location,     4),
                "availability": round(availability, 4),
                "final":        round(final,        4),
            },
        })

    hp_count = sum(1 for r in scored if r["_honeypot"])
    if hp_count:
        logger.warning(
            "Honeypot filter: %d / %d candidate(s) flagged and score-suppressed",
            hp_count, len(scored),
        )

    # Honeypots always sort below legitimate candidates; then by score desc,
    # then candidate_id asc for a stable, reproducible tie-break.
    scored.sort(key=lambda x: (x["_honeypot"], -x["score"], x["candidate_id"]))

    results = []
    for rank, row in enumerate(scored[:100], 1):
        results.append({
            "candidate_id": row["candidate_id"],
            "rank": rank,
            "score": row["score"],
            "reasoning": row["reasoning"],
            "_breakdown": row.get("_breakdown", {}),
        })
    t_blend = time.perf_counter() - t0

    if show_timings:
        print(
            f"  ✓ Stage timings  : screen {t_screen:.1f}s ({len(candidates)} cands) | "
            f"embed {t_embed:.1f}s ({len(pool)} finalists) | blend {t_blend:.1f}s"
        )
    return results


# ─── Stage 0: Schema Validation ───────────────────────────────────────────────

def validate_candidate(c: object) -> Tuple[bool, List[str]]:
    """
    Validate a candidate record before scoring. Returns (is_valid, issues).

    is_valid=False  → structural failure; record must be rejected.
    is_valid=True   → record may be scored; non-empty issues are warnings only.

    Fatal (reject):
      * not a dict
      * candidate_id missing or not a non-empty string
      * profile / skills / career_history / redrob_signals present but wrong type

    Warning (score with caveats):
      * required top-level key absent (downstream .get() defaults kick in)
      * years_of_experience negative or non-numeric
      * skill proficiency not in {beginner, intermediate, advanced, expert}
      * skill / job duration_months negative
      * rate fields (response rate, etc.) outside [0, 1]
      * profile_completeness_score outside [0, 100]
    """
    if not isinstance(c, dict):
        return False, [f"record is not a dict (got {type(c).__name__})"]

    issues: List[str] = []
    fatal = False

    # ── candidate_id ──────────────────────────────────────────────────────────
    cid = c.get("candidate_id")
    if cid is None:
        issues.append("missing candidate_id")
        fatal = True
    elif not isinstance(cid, str) or not cid.strip():
        issues.append(f"candidate_id must be a non-empty string, got {cid!r}")
        fatal = True

    # ── profile ───────────────────────────────────────────────────────────────
    profile = c.get("profile")
    if profile is None:
        issues.append("missing profile (defaults to {})")
    elif not isinstance(profile, dict):
        issues.append(f"profile must be a dict, got {type(profile).__name__}")
        fatal = True
    else:
        yoe = profile.get("years_of_experience")
        if yoe is not None and (not isinstance(yoe, (int, float)) or yoe < 0):
            issues.append(f"profile.years_of_experience invalid: {yoe!r}")

    # ── skills ────────────────────────────────────────────────────────────────
    skills = c.get("skills")
    if skills is None:
        issues.append("missing skills (defaults to [])")
    elif not isinstance(skills, list):
        issues.append(f"skills must be a list, got {type(skills).__name__}")
        fatal = True
    else:
        for i, sk in enumerate(skills):
            if not isinstance(sk, dict):
                issues.append(f"skills[{i}] is not a dict")
                continue
            name = sk.get("name")
            if not name or not isinstance(name, str):
                issues.append(f"skills[{i}].name missing or not a string")
            prof = sk.get("proficiency")
            if prof is not None and prof not in _VALID_PROFICIENCIES:
                issues.append(f"skills[{i}].proficiency unrecognised: {prof!r}")
            dur = sk.get("duration_months")
            if dur is not None and (not isinstance(dur, (int, float)) or dur < 0):
                issues.append(f"skills[{i}].duration_months invalid: {dur!r}")

    # ── career_history ────────────────────────────────────────────────────────
    career = c.get("career_history")
    if career is None:
        issues.append("missing career_history (defaults to [])")
    elif not isinstance(career, list):
        issues.append(f"career_history must be a list, got {type(career).__name__}")
        fatal = True
    else:
        for i, job in enumerate(career):
            if not isinstance(job, dict):
                issues.append(f"career_history[{i}] is not a dict")
                continue
            dur = job.get("duration_months")
            if dur is not None and (not isinstance(dur, (int, float)) or dur < 0):
                issues.append(f"career_history[{i}].duration_months invalid: {dur!r}")

    # ── redrob_signals ────────────────────────────────────────────────────────
    signals = c.get("redrob_signals")
    if signals is None:
        issues.append("missing redrob_signals (defaults to {})")
    elif not isinstance(signals, dict):
        issues.append(f"redrob_signals must be a dict, got {type(signals).__name__}")
        fatal = True
    else:
        for rate_key in (
            "recruiter_response_rate",
            "interview_completion_rate",
            "offer_acceptance_rate",
        ):
            v = signals.get(rate_key)
            # offer_acceptance_rate uses -1 as a "no offer history" sentinel (signals doc).
            allows_sentinel = rate_key == "offer_acceptance_rate"
            if (
                v is not None
                and isinstance(v, (int, float))
                and not (0.0 <= v <= 1.0)
                and not (allows_sentinel and v == -1)
            ):
                issues.append(f"redrob_signals.{rate_key} out of range [0, 1]: {v!r}")
        completeness = signals.get("profile_completeness_score")
        if (
            completeness is not None
            and isinstance(completeness, (int, float))
            and not (0 <= completeness <= 100)
        ):
            issues.append(
                f"redrob_signals.profile_completeness_score out of range [0, 100]: {completeness!r}"
            )

    return not fatal, issues


def load_candidates(path: str) -> List[Dict]:
    p = Path(path)
    # Supports .json (array), .jsonl, and gzip-compressed variants (.jsonl.gz / .json.gz).
    is_gz = p.suffix.lower() == ".gz"
    fmt = (p.suffixes[-2].lower() if is_gz and len(p.suffixes) >= 2 else p.suffix.lower())
    opener = (lambda: gzip.open(p, "rt", encoding="utf-8")) if is_gz \
        else (lambda: open(p, "r", encoding="utf-8"))

    if fmt == ".jsonl":
        raw: List = []
        with opener() as f:
            for ln_no, ln in enumerate(f, 1):
                if not ln.strip():
                    continue
                try:
                    raw.append(json.loads(ln))
                except json.JSONDecodeError as exc:
                    logger.error("Line %d: JSON parse error — %s", ln_no, exc)
    else:
        with opener() as f:
            data = json.load(f)
        raw = data if isinstance(data, list) else [data]

    valid: List[Dict] = []
    rejected = 0
    for i, record in enumerate(raw):
        cid = (
            record.get("candidate_id", f"<record {i}>")
            if isinstance(record, dict)
            else f"<record {i}>"
        )
        is_valid, issues = validate_candidate(record)
        for msg in issues:
            if is_valid:
                logger.warning("Candidate %s: %s", cid, msg)
            else:
                logger.error("Candidate %s: %s", cid, msg)
        if is_valid:
            valid.append(record)
        else:
            rejected += 1
            logger.error("Candidate %s REJECTED — structural validation failed", cid)

    if rejected:
        logger.warning(
            "%d / %d candidate record(s) rejected; %d will be scored",
            rejected, len(raw), len(valid),
        )
    return valid


def write_csv(results: List[Dict], out_path: str) -> None:
    _fields = ["candidate_id", "rank", "score", "reasoning"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in _fields} for r in results)


def _print_diagnostics(results: List[Dict], n: int = 20) -> None:
    """Print a score-component breakdown table for the top-n ranked candidates."""
    header = (
        f"{'Rank':>4}  {'ID':<16}  {'Sem':>5}  {'Career':>6}  "
        f"{'Behav':>5}  {'Loc':>5}  {'Avail':>5}  {'Final':>6}  Honeypot"
    )
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for r in results[:n]:
        b = r.get("_breakdown", {})
        hp = "HP" if r["reasoning"].startswith("HONEYPOT") else ""
        print(
            f"  #{r['rank']:>3}  {r['candidate_id']:<16}  "
            f"{b.get('semantic',0):>5.3f}  {b.get('career',0):>6.3f}  "
            f"{b.get('behavioral',0):>5.3f}  {b.get('location',0):>5.3f}  "
            f"{b.get('availability',0):>5.3f}  {b.get('final',0):>6.4f}  {hp}"
        )
    print("─" * len(header))


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    ap.add_argument("--candidates", default="candidates.jsonl",
                    help=".json array, .jsonl, or gzip-compressed variant")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                    help="Local sentence-transformer directory (vendored by download_model.py)")
    ap.add_argument("--allow-lexical-fallback", action="store_true",
                    help="Tests/smoke ONLY: permit TF-IDF fallback when the local model "
                         "is missing. Produces a different, non-submittable ranking.")
    ap.add_argument("--reference-date", default=None,
                    help="Override the deterministic 'today' (YYYY-MM-DD). Default: "
                         "max last_active_date in the pool — same data, same output.")
    ap.add_argument("--finalists", type=int, default=PRESCREEN_FINALISTS,
                    help="Stage-A prescreen size handed to semantic re-ranking")
    ap.add_argument("--diagnostics", action="store_true",
                    help="Print score-component breakdown table for top-20 candidates")
    args = ap.parse_args()

    print(f"Loading: {args.candidates}")
    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates)} candidates")

    reference_date = (
        date.fromisoformat(args.reference_date) if args.reference_date
        else derive_reference_date(candidates)
    )
    print(f"  ✓ Reference date : {reference_date} (deterministic)")

    print("Initialising embedder …")
    embedder = Embedder(
        model_dir=args.model_dir,
        allow_lexical_fallback=args.allow_lexical_fallback,
    )
    mode = embedder.mode          # "semantic" or "lexical"
    model = embedder._model_id or "TF-IDF lexical fallback"
    if mode == "semantic":
        print("  ✓ Embedding mode : semantic (offline, local model)")
        print(f"  ✓ Model          : {model}")
    else:
        print("  ✗ Embedding mode : lexical (TF-IDF fallback — tests/smoke only)")
        print(f"  ✗ Fallback reason: {embedder.load_error}")
    weights_summary = " + ".join(
        f"{v:.2f}×{k[:3]}" for k, v in WEIGHTS.items()
    )
    print(f"  ✓ Score formula  : {weights_summary}")

    print("Ranking …")
    results = rank_candidates(
        candidates, JOB_DESCRIPTION, embedder, reference_date,
        finalists=args.finalists, show_timings=True,
    )

    hp_count = sum(1 for r in results if r["reasoning"].startswith("HONEYPOT"))
    print(f"Writing {len(results)} rows → {args.out}  ({hp_count} honeypot(s) flagged, score=0.0)")
    write_csv(results, args.out)

    if args.diagnostics:
        _print_diagnostics(results, n=min(20, len(results)))
    else:
        print("\nTop 10:")
        for r in results[:10]:
            print(f"  #{r['rank']:>3}  {r['candidate_id']}  {r['score']:.4f}  {r['reasoning']}")


if __name__ == "__main__":
    main()
