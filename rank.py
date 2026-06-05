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
    6. Weighted Ranker ....... final_score = 0.45 sem + 0.25 career + 0.15 behav
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
import json
import math
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

# ─── Stage 1: Job Description (weighted requirements) ─────────────────────────
# Extracted directly from the JD. Edit this block to target a different role.
JOB_DESCRIPTION: Dict = {
    "title": "Search / Ranking Machine Learning Engineer",
    "responsibilities": [
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
        "Computer Vision only",
        "LangChain-only experience",
        "No production deployments",
    ],
    "experience_years_min": 2,
    "experience_years_ideal": 6,
    # Where the role lives. Used by the location score.
    "location": {"country": "India", "remote_ok": True},
}

# ─── Stage 6: Final score weights (sum to 1.0) ───────────────────────────────
WEIGHTS = {
    "semantic":     0.45,
    "career":       0.25,
    "behavioral":   0.15,
    "location":     0.10,
    "availability": 0.05,
}

REFERENCE_DATE = date(2026, 6, 5)

# ─── Concept ontology ────────────────────────────────────────────────────────
# Maps each must-have / good-to-have requirement to the skill/keyword surface
# forms that satisfy it. This drives concept *coverage* (used by career analysis
# and reasoning) — the headline relevance number still comes from embeddings.
CONCEPTS: Dict[str, set] = {
    "Embeddings": {
        "embeddings", "sentence transformers", "sentence-transformers",
        "word2vec", "glove", "text embeddings", "bge", "minilm",
    },
    "Retrieval Systems": {
        "information retrieval", "retrieval", "rag",
        "retrieval augmented generation", "bm25", "semantic search",
        "haystack", "dense retrieval", "elasticsearch", "opensearch",
    },
    "Vector Databases": {
        "vector search", "vector database", "faiss", "pinecone",
        "weaviate", "qdrant", "milvus", "pgvector", "chroma",
    },
    "Ranking/Search Systems": {
        "ranking", "search", "recommendation systems", "recommender",
        "learning to rank", "learning-to-rank", "ltr", "xgboost",
        "lightgbm", "search engineer", "relevance",
    },
    "Python": {"python"},
    "Evaluation Metrics (NDCG, MRR, MAP)": {
        "ndcg", "mrr", "map", "evaluation metrics", "offline evaluation",
        "ab test", "a/b test", "precision@k", "recall@k",
    },
    "Production ML": {
        "mlops", "mlflow", "kubeflow", "bentoml", "production", "deployment",
        "serving", "model serving", "weights & biases", "wandb",
    },
    "Product Company Experience": set(),  # derived from industry, not skills
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

# Skill surfaces that are purely Computer Vision (the "CV only" negative signal).
CV_SKILLS = {
    "opencv", "yolo", "cnn", "object detection", "image classification",
    "image segmentation", "gans", "computer vision",
}

PROFICIENCY = {"beginner": 0.25, "intermediate": 0.50, "advanced": 0.80, "expert": 1.00}

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "at",
    "by", "from", "as", "is", "are", "was", "were", "be", "this", "that", "it",
    "team", "worked", "built", "using", "used", "work",
}


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ─── Stage 3: Semantic Search ─────────────────────────────────────────────────

class Embedder:
    """
    Produces semantic similarity between the JD and each candidate.

    Primary path: sentence-transformers/all-MiniLM-L6-v2 (true semantic).
    Fallback path: deterministic TF-IDF cosine over the corpus (pure stdlib),
    so the pipeline runs offline with zero dependencies and auto-upgrades the
    moment `sentence-transformers` is installed.
    """

    def __init__(self) -> None:
        self.mode = "lexical"
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer  # noqa: WPS433
            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.mode = "semantic"
        except Exception:  # missing dep, no model cache, no network — degrade.
            self._model = None

    def similarities(self, jd_doc: str, cand_docs: List[str]) -> List[float]:
        if not cand_docs:
            return []
        raw = (
            self._semantic(jd_doc, cand_docs)
            if self.mode == "semantic"
            else self._lexical(jd_doc, cand_docs)
        )
        # Normalize by the cohort max so relevance occupies a usable range and
        # the 0.45 weight has real ranking influence. Absolute cosine magnitude
        # is irrelevant for ordering within a single JD's candidate pool.
        top = max(raw)
        if top <= 1e-9:
            return raw
        return [_clamp(s / top) for s in raw]

    def _semantic(self, jd_doc: str, cand_docs: List[str]) -> List[float]:
        import numpy as np  # bundled with sentence-transformers

        emb = self._model.encode(
            [jd_doc] + cand_docs, normalize_embeddings=True, show_progress_bar=False
        )
        emb = np.asarray(emb)
        sims = emb[1:] @ emb[0]
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
    skill_names = {s.get("name", "").lower() for s in c.get("skills", [])}
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
                hit = max(hit, 1.0)          # explicit skill = strong
            elif surface in blob:
                hit = max(hit, 0.6)          # mentioned in text = moderate
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
        if ind in PRODUCT_INDUSTRIES and not any(s in comp for s in SERVICES_COMPANIES):
            best = max(best, 1.0)
        elif ind in PRODUCT_INDUSTRIES:
            best = max(best, 0.6)
    return best


def engineer_features(c: Dict, jd: Dict, coverage: Dict[str, float]) -> Dict:
    """Stage 2 numeric features, surfaced for transparency and reasoning."""
    p = c.get("profile", {})
    must = jd["must_have"]
    must_covered = sum(1 for r in must if coverage.get(r, 0) >= 0.6)
    return {
        "years_exp": p.get("years_of_experience", 0),
        "must_have_covered": must_covered,
        "must_have_total": len(must),
        "covered_concepts": [r for r in must if coverage.get(r, 0) >= 0.6],
    }


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

    relevant_titles = CONCEPTS["Ranking/Search Systems"] | {
        "ml engineer", "machine learning engineer", "ai engineer",
        "applied ml", "nlp engineer", "research engineer", "data scientist",
        "mlops engineer", "search engineer", "recommendation systems engineer",
    }

    weighted_sum = 0.0
    weight_total = 0.0
    for i, job in enumerate(history):
        w = 1.0 if job.get("is_current") else max(0.30, 1.0 - i * 0.20)
        title = (job.get("title", "") or "").lower()
        desc = (job.get("description", "") or "").lower()
        industry = (job.get("industry", "") or "").lower()
        company = (job.get("company", "") or "").lower()

        title_s = 0.85 if any(t in title for t in relevant_titles) else 0.25
        concept_hits = sum(
            1 for surfaces in
            (CONCEPTS["Retrieval Systems"], CONCEPTS["Vector Databases"],
             CONCEPTS["Ranking/Search Systems"], CONCEPTS["Embeddings"],
             CONCEPTS["Production ML"])
            if any(s in desc for s in surfaces)
        )
        desc_s = _clamp(concept_hits / 3.0)
        prod_s = 1.0 if (industry in PRODUCT_INDUSTRIES and
                         not any(s in company for s in SERVICES_COMPANIES)) else 0.35

        job_score = title_s * 0.40 + desc_s * 0.35 + prod_s * 0.25
        weighted_sum += job_score * w
        weight_total += w

    score = _clamp(weighted_sum / max(weight_total, 1.0))

    # Trajectory bonus: currently in a relevant, production role.
    cur = history[0]
    if cur.get("is_current") and coverage.get("Production ML", 0) >= 0.6:
        score = _clamp(score + 0.05)

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
        return ind in SERVICES_INDUSTRIES or any(s in comp for s in SERVICES_COMPANIES)

    if history and all(is_services(j) for j in history):
        score *= 0.55
        flags.append("consulting-only")

    # Pure research: research titles with no production/MLOps footprint.
    titles = " ".join((j.get("title", "") or "").lower() for j in history)
    if "research" in titles and coverage.get("Production ML", 0) < 0.6:
        score *= 0.70
        flags.append("pure-research")

    # Computer-vision only: CV-heavy skills, no retrieval/ranking depth.
    cv_hits = len(skill_names & CV_SKILLS)
    core_hits = (
        coverage.get("Retrieval Systems", 0)
        + coverage.get("Vector Databases", 0)
        + coverage.get("Ranking/Search Systems", 0)
    )
    if cv_hits >= 2 and core_hits < 0.6:
        score *= 0.60
        flags.append("cv-only")

    # LangChain-only: rides LangChain/RAG buzzwords without real retrieval depth.
    has_langchain = "langchain" in skill_names or "langchain" in candidate_text_blob(c)
    if has_langchain and coverage.get("Vector Databases", 0) < 0.6 and core_hits < 0.6:
        score *= 0.65
        flags.append("langchain-only")

    # No production deployments anywhere.
    if coverage.get("Production ML", 0) < 0.6 and not any(
        any(s in (j.get("description", "") or "").lower()
            for s in CONCEPTS["Production ML"])
        for j in history
    ):
        score *= 0.80
        flags.append("no-production")

    return _clamp(score), flags


# ─── Stage 5: Behavioral Signals ──────────────────────────────────────────────

def score_behavioral(c: Dict) -> float:
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
            days = (REFERENCE_DATE - date.fromisoformat(last)).days
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

def score_location(c: Dict, jd: Dict) -> float:
    p = c.get("profile", {})
    s = c.get("redrob_signals", {})
    target = jd.get("location", {})

    score = 0.40  # baseline
    if (p.get("country", "") or "").lower() == (target.get("country", "") or "").lower():
        score += 0.35
    elif s.get("willing_to_relocate"):
        score += 0.25

    mode = (s.get("preferred_work_mode", "") or "").lower()
    if target.get("remote_ok") and mode in {"flexible", "remote", "hybrid"}:
        score += 0.25
    elif mode == "flexible":
        score += 0.15
    return _clamp(score)


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

def detect_honeypot(c: Dict) -> Tuple[bool, List[str]]:
    """
    Catches organizer-inserted fakes: impossible timelines, fabricated tenure,
    expert claims with no usage. Hard flags force rejection; soft flags accumulate.
    Returns (is_honeypot, reasons).
    """
    p = c.get("profile", {})
    yoe = p.get("years_of_experience", 0) or 0
    skills = c.get("skills", [])
    history = c.get("career_history", []) or []

    hard = 0
    soft = 0
    reasons: List[str] = []

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
    expert = [sk for sk in skills if sk.get("proficiency") == "expert"]
    expert_no_use = [sk for sk in expert if (sk.get("duration_months", 0) or 0) == 0]
    if len(expert) >= 8 and len(expert_no_use) == len(expert):
        hard += 1
        reasons.append(f"{len(expert)} expert skills, all with zero usage")
    elif len(expert) > 10 and len(expert_no_use) >= 5:
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
    title = p.get("current_title", "Unknown")
    yoe = p.get("years_of_experience", 0) or 0
    rr = s.get("recruiter_response_rate", 0) or 0
    covered = features["must_have_covered"]
    total = features["must_have_total"]
    top = ", ".join(features["covered_concepts"][:3]) or "none"

    if is_honeypot:
        return f"⚠ HONEYPOT — {('; '.join(honeypot_reasons)) or 'fabricated profile'}."

    base = (
        f"{title}, {yoe:.1f}y | semantic fit {semantic:.0%} | "
        f"{covered}/{total} must-haves ({top}) | response {rr:.0%}, "
        f"engagement {behavioral:.0%}"
    )
    if career_flags:
        base += f" | flags: {', '.join(career_flags)}"
    return base + "."


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def rank_candidates(candidates: List[Dict], jd: Dict, embedder: Embedder) -> List[Dict]:
    jd_doc = jd_document(jd)
    cand_docs = [candidate_document(c) for c in candidates]
    semantic_scores = embedder.similarities(jd_doc, cand_docs)

    scored = []
    for c, semantic in zip(candidates, semantic_scores):
        coverage = concept_coverage(c, jd)
        features = engineer_features(c, jd, coverage)

        career, career_flags = score_career(c, coverage)
        behavioral = score_behavioral(c)
        location = score_location(c, jd)
        availability = score_availability(c)
        is_honeypot, hp_reasons = detect_honeypot(c)

        final = (
            semantic     * WEIGHTS["semantic"] +
            career       * WEIGHTS["career"] +
            behavioral   * WEIGHTS["behavioral"] +
            location     * WEIGHTS["location"] +
            availability * WEIGHTS["availability"]
        )

        scored.append({
            "candidate_id": c["candidate_id"],
            "score": round(final, 4),
            "reasoning": build_reasoning(
                c, features, semantic, behavioral,
                career_flags, hp_reasons, is_honeypot,
            ),
            "_honeypot": is_honeypot,
        })

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
        })
    return results


def load_candidates(path: str) -> List[Dict]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".jsonl":
        return [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    data = json.loads(text)
    return data if isinstance(data, list) else [data]


def write_csv(results: List[Dict], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["candidate_id", "rank", "score", "reasoning"]
        )
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    ap = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    ap.add_argument("--candidates", default="candidates.jsonl")
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    print(f"Loading: {args.candidates}")
    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates)} candidates")

    print("Initialising embedder …")
    embedder = Embedder()
    print(f"  → mode: {embedder.mode}"
          + ("" if embedder.mode == "semantic"
             else "  (install sentence-transformers for true semantic search)"))

    print("Ranking …")
    results = rank_candidates(candidates, JOB_DESCRIPTION, embedder)

    print(f"Writing {len(results)} rows → {args.out}")
    write_csv(results, args.out)

    print("\nTop 10:")
    for r in results[:10]:
        print(f"  #{r['rank']:>3}  {r['candidate_id']}  {r['score']:.4f}  {r['reasoning']}")


if __name__ == "__main__":
    main()
