"""
Unit tests for the candidate ranking pipeline.

Run:  pytest test_rank.py -v
These tests use the lexical fallback embedder, so they need no model download.
"""

import math

import pytest

import rank


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _signals(**overrides):
    base = {
        "recruiter_response_rate": 0.5,
        "interview_completion_rate": 0.5,
        "github_activity_score": 30,
        "profile_completeness_score": 80,
        "last_active_date": "2026-05-01",
        "saved_by_recruiters_30d": 5,
        "offer_acceptance_rate": 0.5,
        "open_to_work_flag": True,
        "notice_period_days": 60,
        "applications_submitted_30d": 1,
        "skill_assessment_scores": {},
    }
    base.update(overrides)
    return base


def _strong_candidate():
    return {
        "candidate_id": "C_STRONG",
        "profile": {
            "current_title": "Search Engineer", "years_of_experience": 6.0,
            "country": "India",
            "summary": "Built retrieval and ranking systems with embeddings.",
            "headline": "Search / Ranking ML Engineer",
        },
        "career_history": [{
            "company": "Swiggy", "title": "Recommendation Systems Engineer",
            "duration_months": 36, "is_current": True, "industry": "Food Delivery",
            "company_size": "5001-10000",
            "description": "Information Retrieval, FAISS, Pinecone, Embeddings, MLOps, ranking.",
        }],
        "education": [{"tier": "tier_2", "field_of_study": "Computer Science",
                       "start_year": 2014, "end_year": 2018}],
        "skills": [
            {"name": "Embeddings", "proficiency": "expert", "duration_months": 48, "endorsements": 30},
            {"name": "FAISS", "proficiency": "advanced", "duration_months": 36, "endorsements": 20},
            {"name": "Information Retrieval", "proficiency": "expert", "duration_months": 60, "endorsements": 15},
            {"name": "Python", "proficiency": "expert", "duration_months": 72, "endorsements": 40},
        ],
        "redrob_signals": _signals(),
    }


# ─── Configuration invariants ────────────────────────────────────────────────

def test_weights_sum_to_one():
    assert math.isclose(sum(rank.WEIGHTS.values()), 1.0, abs_tol=1e-9)


def test_weights_match_spec():
    assert rank.WEIGHTS["semantic"] == 0.45
    assert rank.WEIGHTS["career"] == 0.25
    assert rank.WEIGHTS["behavioral"] == 0.15
    assert rank.WEIGHTS["location"] == 0.10
    assert rank.WEIGHTS["availability"] == 0.05


# ─── Stage 7: Honeypot detection ──────────────────────────────────────────────

def test_honeypot_skill_tenure_exceeds_experience():
    cand = _strong_candidate()
    cand["profile"]["years_of_experience"] = 2.0
    cand["skills"][0]["duration_months"] = 120  # 10y of a skill on a 2y career
    is_hp, reasons = rank.detect_honeypot(cand)
    assert is_hp is True
    assert any("tenure" in r for r in reasons)


def test_honeypot_many_expert_skills_zero_usage():
    cand = _strong_candidate()
    cand["skills"] = [
        {"name": f"S{i}", "proficiency": "expert", "duration_months": 0, "endorsements": 0}
        for i in range(12)
    ]
    is_hp, reasons = rank.detect_honeypot(cand)
    assert is_hp is True


def test_legit_candidate_is_not_a_honeypot():
    is_hp, _ = rank.detect_honeypot(_strong_candidate())
    assert is_hp is False


def test_honeypot_sorted_below_legitimate():
    fake = _strong_candidate()
    fake["candidate_id"] = "C_FAKE"
    fake["profile"]["years_of_experience"] = 1.0
    fake["skills"][0]["duration_months"] = 120  # triggers hard flag
    fake["redrob_signals"] = _signals(recruiter_response_rate=1.0,
                                      profile_completeness_score=100)
    results = rank.rank_candidates(
        [fake, _strong_candidate()], rank.JOB_DESCRIPTION, rank.Embedder()
    )
    # The legit candidate must outrank the honeypot despite its inflated signals.
    assert results[0]["candidate_id"] == "C_STRONG"
    assert results[-1]["candidate_id"] == "C_FAKE"
    assert "HONEYPOT" in results[-1]["reasoning"]


# ─── Stage 4: Career negative signals ─────────────────────────────────────────

def test_consulting_only_penalised():
    cand = _strong_candidate()
    cand["career_history"] = [{
        "company": "Infosys", "title": "Consultant", "duration_months": 60,
        "is_current": True, "industry": "IT Services", "description": "Delivery work.",
    }]
    coverage = rank.concept_coverage(cand, rank.JOB_DESCRIPTION)
    _, flags = rank.score_career(cand, coverage)
    assert "consulting-only" in flags


def test_strong_candidate_has_no_negative_flags():
    cand = _strong_candidate()
    coverage = rank.concept_coverage(cand, rank.JOB_DESCRIPTION)
    score, flags = rank.score_career(cand, coverage)
    assert flags == []
    assert score > 0.6


# ─── Stage 3: Semantic search ─────────────────────────────────────────────────

def test_semantic_scores_normalised_to_unit_max():
    emb = rank.Embedder()
    sims = emb.similarities("ranking retrieval embeddings", ["ranking retrieval", "cooking food"])
    assert max(sims) == pytest.approx(1.0)
    assert all(0.0 <= s <= 1.0 for s in sims)


def test_relevant_candidate_scores_higher_semantically():
    emb = rank.Embedder()
    jd = rank.jd_document(rank.JOB_DESCRIPTION)
    relevant = rank.candidate_document(_strong_candidate())
    irrelevant = rank.candidate_document({
        "profile": {"headline": "Chef", "summary": "I cook food.", "current_title": "Chef"},
        "skills": [], "career_history": [], "education": [],
    })
    sims = emb.similarities(jd, [relevant, irrelevant])
    assert sims[0] > sims[1]


# ─── Sub-score ranges ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("fn", [
    lambda c: rank.score_behavioral(c),
    lambda c: rank.score_location(c, rank.JOB_DESCRIPTION),
    lambda c: rank.score_availability(c),
])
def test_subscores_in_unit_range(fn):
    val = fn(_strong_candidate())
    assert 0.0 <= val <= 1.0


def test_pipeline_output_shape():
    results = rank.rank_candidates(
        [_strong_candidate()], rank.JOB_DESCRIPTION, rank.Embedder()
    )
    assert results[0].keys() == {"candidate_id", "rank", "score", "reasoning"}
    assert results[0]["rank"] == 1
