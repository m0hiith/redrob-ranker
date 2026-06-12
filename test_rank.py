"""
Unit tests for the candidate ranking pipeline.

Run:  pytest test_rank.py -v
These tests use the lexical fallback embedder, so they need no model download.
"""

import json
import math
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

import rank


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _lexical_embedder() -> rank.Embedder:
    """Deterministic TF-IDF embedder for tests — no model files needed."""
    return rank.Embedder(model_dir="/nonexistent-model-dir", allow_lexical_fallback=True)

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
    # Career raised to 0.30, semantic lowered to 0.40: richer career signal (now
    # includes YoE modifier) justifies the shift; improves discrimination between
    # strong trajectories and consulting/irrelevant backgrounds.
    assert rank.WEIGHTS["semantic"] == 0.40
    assert rank.WEIGHTS["career"] == 0.30
    assert rank.WEIGHTS["behavioral"] == 0.15
    assert rank.WEIGHTS["location"] == 0.10
    assert rank.WEIGHTS["availability"] == 0.05


# ─── _yoe_score: seniority-fit curve ─────────────────────────────────────────

def test_yoe_score_floor_at_zero():
    assert rank._yoe_score(0) == pytest.approx(0.20)


def test_yoe_score_floor_negative():
    assert rank._yoe_score(-5) == pytest.approx(0.20)


def test_yoe_score_peak_at_seven():
    assert rank._yoe_score(7) == pytest.approx(1.00)


def test_yoe_score_ideal_band_five_to_nine():
    for yoe in (5.0, 6.0, 7.0, 8.0, 9.0):
        s = rank._yoe_score(yoe)
        assert s >= 0.90 - 1e-9, f"yoe={yoe} should be ≥ 0.90, got {s}"


def test_yoe_score_decreasing_below_five():
    assert rank._yoe_score(3) < rank._yoe_score(5)
    assert rank._yoe_score(1) < rank._yoe_score(3)


def test_yoe_score_decreasing_above_nine():
    assert rank._yoe_score(12) < rank._yoe_score(9)
    assert rank._yoe_score(15) <= rank._yoe_score(12)


def test_yoe_score_floor_at_very_senior():
    assert rank._yoe_score(20) == pytest.approx(0.70)


def test_yoe_score_output_in_unit_range():
    for yoe in (0, 1, 2, 3, 5, 7, 9, 12, 15, 20):
        s = rank._yoe_score(yoe)
        assert 0.0 <= s <= 1.0, f"yoe={yoe} produced out-of-range score {s}"


def test_yoe_modifier_penalises_junior_career():
    """A 1-year candidate's career score should be lower than an identical 7-year candidate."""
    junior = _strong_candidate()
    junior["candidate_id"] = "C_JUNIOR"
    junior["profile"]["years_of_experience"] = 1.0

    senior = _strong_candidate()
    senior["profile"]["years_of_experience"] = 7.0

    cov_j = rank.concept_coverage(junior, rank.JOB_DESCRIPTION)
    cov_s = rank.concept_coverage(senior, rank.JOB_DESCRIPTION)

    score_junior, _ = rank.score_career(junior, cov_j)
    score_senior, _ = rank.score_career(senior, cov_s)

    assert score_senior > score_junior, (
        f"7y candidate ({score_senior:.3f}) should outscore 1y ({score_junior:.3f})"
    )


def test_yoe_modifier_does_not_penalise_ideal_range():
    """A candidate at 7 years should not lose score relative to a 6-year equivalent."""
    c_6 = _strong_candidate()
    c_6["candidate_id"] = "C_6Y"
    c_6["profile"]["years_of_experience"] = 6.0

    c_7 = _strong_candidate()
    c_7["profile"]["years_of_experience"] = 7.0

    cov_6 = rank.concept_coverage(c_6, rank.JOB_DESCRIPTION)
    cov_7 = rank.concept_coverage(c_7, rank.JOB_DESCRIPTION)

    score_6, _ = rank.score_career(c_6, cov_6)
    score_7, _ = rank.score_career(c_7, cov_7)

    # Both should be very close; 7y is at peak but both are in the ideal band
    assert abs(score_7 - score_6) < 0.05, (
        f"6y and 7y scores should be close, got {score_6:.3f} vs {score_7:.3f}"
    )


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
        [fake, _strong_candidate()], rank.JOB_DESCRIPTION, _lexical_embedder()
    )
    # The legit candidate must outrank the honeypot despite its inflated signals.
    assert results[0]["candidate_id"] == "C_STRONG"
    assert results[-1]["candidate_id"] == "C_FAKE"
    assert "HONEYPOT" in results[-1]["reasoning"]
    # Score must be suppressed so evaluation metrics (NDCG/MRR) are not inflated.
    assert results[-1]["score"] == 0.0


def test_honeypot_score_suppressed_to_zero_single_candidate():
    """A honeypot run through the full pipeline must always emit score=0.0."""
    fake = _strong_candidate()
    fake["candidate_id"] = "C_FAKE_SOLO"
    fake["profile"]["years_of_experience"] = 1.0
    fake["skills"][0]["duration_months"] = 120  # hard flag: skill tenure > yoe
    results = rank.rank_candidates([fake], rank.JOB_DESCRIPTION, _lexical_embedder())
    assert results[0]["score"] == 0.0
    assert results[0]["reasoning"].startswith("HONEYPOT")


def test_honeypot_all_experts_but_one_zero_usage():
    """9 experts / 8 zero-usage was previously a detection gap — now a hard flag."""
    cand = _strong_candidate()
    cand["profile"]["years_of_experience"] = 5.0
    cand["skills"] = [
        {"name": f"S{i}", "proficiency": "expert",
         "duration_months": 0 if i < 8 else 6, "endorsements": 0}
        for i in range(9)  # 9 experts, 8 with zero usage
    ]
    is_hp, reasons = rank.detect_honeypot(cand)
    assert is_hp is True
    assert any("expert" in r for r in reasons)


def test_legitimate_candidate_with_many_expert_skills_not_flagged():
    """8 expert skills with real usage months must not trigger the honeypot."""
    cand = _strong_candidate()
    cand["profile"]["years_of_experience"] = 8.0
    cand["skills"] = [
        {"name": f"S{i}", "proficiency": "expert",
         "duration_months": 24, "endorsements": 10}
        for i in range(8)
    ]
    is_hp, _ = rank.detect_honeypot(cand)
    assert is_hp is False


# ─── P8: date-consistency honeypot checks ─────────────────────────────────────

def _dated_job(**overrides):
    job = {
        "company": "Acme", "title": "ML Engineer", "industry": "Software",
        "start_date": "2022-01-01", "end_date": "2024-01-01",
        "duration_months": 24, "is_current": False,
        "description": "Built ranking systems.",
    }
    job.update(overrides)
    return job


def test_honeypot_role_ends_before_it_starts():
    cand = _strong_candidate()
    cand["career_history"] = [_dated_job(start_date="2024-01-01", end_date="2022-01-01")]
    is_hp, reasons = rank.detect_honeypot(cand)
    assert is_hp is True
    assert any("before it starts" in r for r in reasons)


def test_honeypot_future_dated_role():
    cand = _strong_candidate()
    cand["career_history"] = [_dated_job(start_date="2031-01-01", end_date=None,
                                         is_current=True, duration_months=1)]
    is_hp, reasons = rank.detect_honeypot(cand)
    assert is_hp is True
    assert any("future" in r for r in reasons)


def test_honeypot_duration_contradicts_dates():
    # 2 calendar years of dates but claims 8 years of tenure in the role.
    cand = _strong_candidate()
    cand["career_history"] = [_dated_job(duration_months=96)]
    is_hp, reasons = rank.detect_honeypot(cand)
    assert is_hp is True
    assert any("contradicts dates" in r for r in reasons)


def test_honeypot_current_role_with_past_end_date():
    cand = _strong_candidate()
    cand["career_history"] = [_dated_job(is_current=True, end_date="2024-01-01")]
    is_hp, reasons = rank.detect_honeypot(cand)
    assert is_hp is True
    assert any("marked current" in r for r in reasons)


def test_stale_duration_on_current_role_is_not_flagged():
    """Open role with duration smaller than the span = data lag, not fabrication."""
    cand = _strong_candidate()
    cand["career_history"] = [_dated_job(start_date="2020-01-01", end_date=None,
                                         is_current=True, duration_months=36)]
    is_hp, reasons = rank.detect_honeypot(cand)
    assert is_hp is False, reasons


def test_overlapping_roles_soft_flag_only():
    """Two heavily overlapping roles → soft flag, not an instant honeypot."""
    from datetime import date
    cand = _strong_candidate()
    cand["profile"]["years_of_experience"] = 8.0
    cand["career_history"] = [
        _dated_job(start_date="2018-01-01", end_date="2022-01-01", duration_months=48),
        _dated_job(company="Other", start_date="2018-06-01", end_date="2022-06-01",
                   duration_months=48),
    ]
    hard, soft, reasons = rank._date_consistency_flags(cand, date(2026, 6, 1))
    assert hard == 0
    assert soft >= 1
    assert any("overlap" in r for r in reasons)
    is_hp, _ = rank.detect_honeypot(cand)
    assert is_hp is False  # one soft flag alone must not condemn


def test_last_active_before_signup_soft_flag():
    from datetime import date
    cand = _strong_candidate()
    cand["redrob_signals"]["signup_date"] = "2025-06-01"
    cand["redrob_signals"]["last_active_date"] = "2024-01-01"
    hard, soft, reasons = rank._date_consistency_flags(cand, date(2026, 6, 1))
    assert hard == 0
    assert soft >= 1
    assert any("signup" in r for r in reasons)


def test_undated_history_triggers_no_date_flags():
    """The strong fixture has no start/end dates — date checks must all skip."""
    from datetime import date
    hard, soft, reasons = rank._date_consistency_flags(_strong_candidate(), date(2026, 6, 1))
    assert (hard, soft, reasons) == (0, 0, [])


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

def test_semantic_scores_in_unit_range():
    emb = _lexical_embedder()
    sims = emb.similarities("ranking retrieval embeddings", ["ranking retrieval", "cooking food"])
    assert all(0.0 <= s <= 1.0 for s in sims)
    assert sims[0] > sims[1]  # relevant doc scores above irrelevant


def test_relevant_candidate_scores_higher_semantically():
    emb = _lexical_embedder()
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
        [_strong_candidate()], rank.JOB_DESCRIPTION, _lexical_embedder()
    )
    assert {"candidate_id", "rank", "score", "reasoning"}.issubset(results[0].keys())
    assert results[0]["rank"] == 1
    # _breakdown carries the component scores used by --diagnostics
    bd = results[0]["_breakdown"]
    assert set(bd.keys()) == {"semantic", "career", "behavioral", "location", "availability", "final"}
    assert all(0.0 <= v <= 1.0 for v in bd.values())


# ─── T2-B: Search specialist depth signal ─────────────────────────────────────

def test_depth_bonus_full_coverage():
    coverage = {p: 1.0 for p in rank._SEARCH_PILLARS}
    assert rank._search_depth_bonus(coverage) == pytest.approx(0.08)


def test_depth_bonus_partial_coverage():
    coverage = {p: 1.0 for p in rank._SEARCH_PILLARS}
    coverage["Production ML"] = 0.0  # 3 of 4 pillars
    assert rank._search_depth_bonus(coverage) == pytest.approx(0.06)


def test_depth_bonus_zero_coverage():
    assert rank._search_depth_bonus({}) == pytest.approx(0.0)


def test_depth_bonus_threshold_boundary():
    # 0.59 is below threshold, 0.60 is at threshold
    below = {p: 0.59 for p in rank._SEARCH_PILLARS}
    at = {p: 0.60 for p in rank._SEARCH_PILLARS}
    assert rank._search_depth_bonus(below) == pytest.approx(0.0)
    assert rank._search_depth_bonus(at) == pytest.approx(0.08)


# ─── P6: reasoning — varied, JD-linked, rank-consistent, grounded ─────────────

def _reasoning_for(cand, rank_pos=1):
    cov = rank.concept_coverage(cand, rank.JOB_DESCRIPTION)
    features = rank.engineer_features(cand, rank.JOB_DESCRIPTION, cov)
    return rank.build_reasoning(cand, features, rank_pos, [], [], False)


def test_reasonings_vary_across_candidates():
    """20 near-identical profiles must not produce identical template strings."""
    texts = set()
    for i in range(20):
        cand = _strong_candidate()
        cand["candidate_id"] = f"CAND_{i:07d}"
        texts.add(_reasoning_for(cand))
    assert len(texts) >= 3   # hash-seeded phrasing variants engage


def test_reasoning_is_deterministic():
    cand = _strong_candidate()
    assert _reasoning_for(cand) == _reasoning_for(cand)


def test_reasoning_references_jd():
    assert "JD" in _reasoning_for(_strong_candidate())


def test_reasoning_claims_are_grounded_in_coverage():
    cand = _strong_candidate()
    cov = rank.concept_coverage(cand, rank.JOB_DESCRIPTION)
    features = rank.engineer_features(cand, rank.JOB_DESCRIPTION, cov)
    text = rank.build_reasoning(cand, features, 1, [], [], False)
    for concept in features["covered_concepts"][:3]:
        assert concept in text  # everything named in the text exists on the profile


def test_reasoning_tone_matches_rank_band():
    cand = _strong_candidate()
    top = _reasoning_for(cand, rank_pos=5)
    tail = _reasoning_for(cand, rank_pos=95)
    assert top != tail
    assert any(m in tail for m in ("Ranked lower", "Held back", "Tail-of-list"))


def test_reasoning_voices_honest_concern():
    cand = _strong_candidate()
    cand["redrob_signals"]["recruiter_response_rate"] = 0.1
    cov = rank.concept_coverage(cand, rank.JOB_DESCRIPTION)
    features = rank.engineer_features(cand, rank.JOB_DESCRIPTION, cov)
    text = rank.build_reasoning(cand, features, 3, ["consulting-only"], [], False)
    assert "Watch-item" in text
    assert "consulting" in text


def test_honeypot_reasoning_keeps_prefix():
    text = rank.build_reasoning(_strong_candidate(), {}, 1, [], ["impossible dates"], True)
    assert text.startswith("HONEYPOT — impossible dates")


# ─── P9: JD negative signals + city-tier location ─────────────────────────────

def test_title_chaser_flagged():
    """3 short climbing stints behind the current role → title-chaser penalty."""
    cand = _strong_candidate()
    cand["career_history"] = [
        {"company": "Now Inc", "title": "Principal Engineer", "duration_months": 6,
         "is_current": True, "industry": "Software", "description": "Ranking work."},
        {"company": "C3", "title": "Staff Engineer", "duration_months": 16,
         "is_current": False, "industry": "Software", "description": "Backend."},
        {"company": "C2", "title": "Senior Engineer", "duration_months": 18,
         "is_current": False, "industry": "Software", "description": "Backend."},
        {"company": "C1", "title": "Software Engineer", "duration_months": 15,
         "is_current": False, "industry": "Software", "description": "Backend."},
    ]
    coverage = rank.concept_coverage(cand, rank.JOB_DESCRIPTION)
    _, flags = rank.score_career(cand, coverage)
    assert "title-chaser" in flags


def test_long_tenure_not_flagged_as_title_chaser():
    cand = _strong_candidate()  # one 36-month current role
    coverage = rank.concept_coverage(cand, rank.JOB_DESCRIPTION)
    _, flags = rank.score_career(cand, coverage)
    assert "title-chaser" not in flags


@pytest.mark.parametrize("skills,expected_flag", [
    ([{"name": "Kaldi"}, {"name": "ASR"}], "speech-only"),
    ([{"name": "SLAM"}, {"name": "ROS"}], "robotics-only"),
    ([{"name": "OpenCV"}, {"name": "YOLO"}], "cv-only"),
])
def test_out_of_domain_specialists_flagged(skills, expected_flag):
    cand = _strong_candidate()
    cand["profile"]["summary"] = "Engineer."
    cand["profile"]["headline"] = "Engineer"
    cand["profile"]["current_title"] = "Engineer"
    cand["career_history"] = [{
        "company": "Acme", "title": "Engineer", "duration_months": 48,
        "is_current": True, "industry": "Software",
        "description": "Built specialised perception/audio/robotics pipelines.",
    }]
    cand["skills"] = [
        {**sk, "proficiency": "expert", "duration_months": 36, "endorsements": 10}
        for sk in skills
    ]
    coverage = rank.concept_coverage(cand, rank.JOB_DESCRIPTION)
    _, flags = rank.score_career(cand, coverage)
    assert expected_flag in flags


def test_speech_specialist_with_retrieval_depth_not_flagged():
    """The JD escape hatch: domain skills WITH NLP/IR depth are fine."""
    cand = _strong_candidate()
    cand["skills"] += [
        {"name": "Kaldi", "proficiency": "expert", "duration_months": 36, "endorsements": 10},
        {"name": "ASR", "proficiency": "expert", "duration_months": 36, "endorsements": 10},
    ]
    coverage = rank.concept_coverage(cand, rank.JOB_DESCRIPTION)
    _, flags = rank.score_career(cand, coverage)
    assert not any(f.endswith("-only") for f in flags)


def _located(city, country="India", mode="hybrid", relocate=False):
    c = _strong_candidate()
    c["profile"]["location"] = city
    c["profile"]["country"] = country
    c["redrob_signals"]["preferred_work_mode"] = mode
    c["redrob_signals"]["willing_to_relocate"] = relocate
    return c


def test_location_city_tiers_ordered():
    jd = rank.JOB_DESCRIPTION
    pune = rank.score_location(_located("Pune, Maharashtra"), jd)
    hyd = rank.score_location(_located("Hyderabad, Telangana"), jd)
    jaipur = rank.score_location(_located("Jaipur, Rajasthan"), jd)
    abroad_reloc = rank.score_location(_located("Toronto", "Canada", relocate=True), jd)
    abroad = rank.score_location(_located("Toronto", "Canada"), jd)
    assert pune > hyd > jaipur > abroad_reloc > abroad


def test_remote_preference_scores_below_hybrid():
    jd = rank.JOB_DESCRIPTION
    hybrid = rank.score_location(_located("Pune", mode="hybrid"), jd)
    remote = rank.score_location(_located("Pune", mode="remote"), jd)
    assert hybrid > remote


def test_jd_block_matches_real_jd():
    assert "Senior AI Engineer" in rank.JOB_DESCRIPTION["title"]
    assert "experience_years_min" not in rank.JOB_DESCRIPTION  # dead config removed
    assert rank.JOB_DESCRIPTION["location"]["cities"] == ["Pune", "Noida"]


# ─── P11: endorsement/duration trust modifier ─────────────────────────────────

def test_bare_skill_listing_gets_stuffer_discount():
    """A skill with zero endorsements AND zero duration covers at 0.8, not 1.0."""
    stuffer = _strong_candidate()
    stuffer["skills"] = [{"name": "FAISS", "proficiency": "expert",
                          "duration_months": 0, "endorsements": 0}]
    stuffer["profile"]["summary"] = ""
    stuffer["profile"]["headline"] = ""
    stuffer["career_history"] = []
    cov = rank.concept_coverage(stuffer, rank.JOB_DESCRIPTION)
    assert cov["Vector Databases"] == pytest.approx(0.8)


@pytest.mark.parametrize("evidence", [
    {"endorsements": 5, "duration_months": 0},
    {"endorsements": 0, "duration_months": 12},
])
def test_evidenced_skill_gets_full_coverage(evidence):
    cand = _strong_candidate()
    cand["skills"] = [{"name": "FAISS", "proficiency": "expert", **evidence}]
    cand["profile"]["summary"] = ""
    cand["profile"]["headline"] = ""
    cand["career_history"] = []
    cov = rank.concept_coverage(cand, rank.JOB_DESCRIPTION)
    assert cov["Vector Databases"] == pytest.approx(1.0)


# ─── P3: two-stage ranking (prescreen → semantic re-rank) ────────────────────

def _junk_candidate(i: int):
    return {
        "candidate_id": f"C_JUNK_{i:04d}",
        "profile": {"current_title": "Accountant", "years_of_experience": 4.0,
                    "country": "India", "summary": "Bookkeeping and audits.",
                    "headline": "Accountant"},
        "career_history": [{"company": "Ledger LLP", "title": "Accountant",
                            "duration_months": 48, "is_current": True,
                            "industry": "Accounting",
                            "description": "Managed accounts and tax filings."}],
        "education": [], "skills": [{"name": "Excel", "proficiency": "advanced",
                                     "duration_months": 48, "endorsements": 3}],
        "redrob_signals": _signals(),
    }


def test_prescreen_keeps_strong_candidate():
    from datetime import date
    pool = [_junk_candidate(i) for i in range(50)] + [_strong_candidate()]
    finalists = rank.prescreen(pool, rank.JOB_DESCRIPTION, date(2026, 6, 1), finalists=5)
    assert len(finalists) == 5
    assert any(c["candidate_id"] == "C_STRONG" for c in finalists)


def test_prescreen_passthrough_when_pool_small():
    from datetime import date
    pool = [_strong_candidate()]
    assert rank.prescreen(pool, rank.JOB_DESCRIPTION, date(2026, 6, 1), finalists=2000) == pool


def test_prescreen_is_deterministic():
    from datetime import date
    pool = [_junk_candidate(i) for i in range(30)]
    a = rank.prescreen(pool, rank.JOB_DESCRIPTION, date(2026, 6, 1), finalists=10)
    b = rank.prescreen(list(reversed(pool)), rank.JOB_DESCRIPTION, date(2026, 6, 1), finalists=10)
    assert [c["candidate_id"] for c in a] == [c["candidate_id"] for c in b]


def test_two_stage_pipeline_ranks_strong_first():
    pool = [_junk_candidate(i) for i in range(40)] + [_strong_candidate()]
    results = rank.rank_candidates(
        pool, rank.JOB_DESCRIPTION, _lexical_embedder(), finalists=10,
    )
    assert results[0]["candidate_id"] == "C_STRONG"
    # Only finalists are scored/output (10 < 100 cap here).
    assert len(results) == 10
    ranks = [r["rank"] for r in results]
    assert ranks == list(range(1, 11))
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


# ─── P5: deterministic reference date ─────────────────────────────────────────

def test_reference_date_derived_from_pool_max_last_active():
    from datetime import date
    pool = [
        {"redrob_signals": {"last_active_date": "2026-03-10"}},
        {"redrob_signals": {"last_active_date": "2026-05-22"}},
        {"redrob_signals": {"last_active_date": "garbage"}},
        {"redrob_signals": {}},
    ]
    assert rank.derive_reference_date(pool) == date(2026, 5, 22)


def test_reference_date_falls_back_to_pinned_constant():
    assert rank.derive_reference_date([]) == rank.DEFAULT_REFERENCE_DATE
    assert rank.derive_reference_date([{"redrob_signals": {}}]) == rank.DEFAULT_REFERENCE_DATE


def test_behavioral_score_is_reference_date_dependent_but_deterministic():
    from datetime import date
    c = _strong_candidate()  # last_active 2026-05-01
    near = rank.score_behavioral(c, date(2026, 5, 2))
    far = rank.score_behavioral(c, date(2026, 12, 1))
    assert near > far                      # staleness must cost score
    assert near == rank.score_behavioral(c, date(2026, 5, 2))  # deterministic


# ─── P4: word-boundary surface matching (no substring false positives) ───────

def test_research_title_gets_no_search_credit():
    """'Market Research Analyst' must not match the 'search' title surface."""
    assert rank._RELEVANT_TITLE_PATTERN.search("market research analyst") is None
    assert rank._RELEVANT_TITLE_PATTERN.search("search engineer") is not None
    assert rank._RELEVANT_TITLE_PATTERN.search("research engineer") is not None  # explicit surface


@pytest.mark.parametrize("text,concept", [
    ("delivered the product roadmap on time", "Evaluation Metrics (NDCG, MRR, MAP)"),
    ("sprint planning and backlog grooming", "Vector Databases"),       # "ann"
    ("managed cloud storage infrastructure", "Retrieval Systems"),      # "rag"
    ("collaborated across teams", "Open Source"),                       # "oss"
    ("standardised report formats", "HR Tech"),                         # "ats"
])
def test_no_substring_false_positive(text, concept):
    assert rank.CONCEPT_PATTERNS[concept].search(text) is None, (
        f"{concept!r} pattern must not fire on {text!r}"
    )


@pytest.mark.parametrize("text,concept", [
    ("built semantic search with embeddings", "Retrieval Systems"),
    ("evaluated with ndcg, mrr and map", "Evaluation Metrics (NDCG, MRR, MAP)"),
    ("indexed vectors in faiss and qdrant", "Vector Databases"),
    ("hnsw ann index tuning", "Vector Databases"),
    ("re-ranking with a cross-encoder", "Ranking/Search Systems"),
])
def test_real_surfaces_still_match(text, concept):
    assert rank.CONCEPT_PATTERNS[concept].search(text) is not None, (
        f"{concept!r} pattern should fire on {text!r}"
    )


def test_research_analyst_career_scores_below_search_engineer():
    """End-to-end: the substring bug used to give research analysts title credit."""
    analyst = _strong_candidate()
    analyst["candidate_id"] = "C_ANALYST"
    analyst["profile"]["current_title"] = "Market Research Analyst"
    analyst["career_history"] = [{
        "company": "Acme Corp", "title": "Market Research Analyst",
        "duration_months": 36, "is_current": True, "industry": "Consumer Goods",
        "description": "Conducted market research surveys and consumer analysis.",
    }]
    analyst["skills"] = []

    engineer = _strong_candidate()

    cov_a = rank.concept_coverage(analyst, rank.JOB_DESCRIPTION)
    cov_e = rank.concept_coverage(engineer, rank.JOB_DESCRIPTION)
    score_a, _ = rank.score_career(analyst, cov_a)
    score_e, _ = rank.score_career(engineer, cov_e)
    assert score_e > score_a + 0.2


# ─── Embedder: strict offline loading + opt-in fallback ──────────────────────

def test_embedder_raises_when_model_missing_and_no_fallback(tmp_path):
    """Missing local model + no opt-in → hard error, never a silent mode change."""
    with pytest.raises(RuntimeError, match="download_model.py"):
        rank.Embedder(model_dir=str(tmp_path / "nope"))


def test_embedder_lexical_mode_requires_explicit_opt_in(tmp_path):
    emb = rank.Embedder(model_dir=str(tmp_path / "nope"), allow_lexical_fallback=True)
    assert emb.mode == "lexical"
    assert emb._model is None
    assert emb._model_id is None
    assert emb.load_error is not None
    assert "not found" in emb.load_error


def test_embedder_lexical_mode_when_package_missing(tmp_path):
    """Model dir exists but the package is absent → fallback only when opted in."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    with patch.dict(sys.modules, {"sentence_transformers": None}):
        emb = rank.Embedder(model_dir=str(model_dir), allow_lexical_fallback=True)
    assert emb.mode == "lexical"
    assert "sentence-transformers" in emb.load_error.lower()


def test_embedder_lexical_mode_when_model_load_fails(tmp_path):
    """Package present but SentenceTransformer() raises — opt-in fallback engages."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    mock_st = MagicMock()
    mock_st.SentenceTransformer.side_effect = OSError("simulated corrupt model")
    with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
        emb = rank.Embedder(model_dir=str(model_dir), allow_lexical_fallback=True)
    assert emb.mode == "lexical"
    assert "OSError" in emb.load_error
    assert "simulated corrupt model" in emb.load_error


def test_embedder_semantic_mode_clears_load_error(tmp_path):
    """Successful local model load → load_error is None and mode is 'semantic'."""
    model_dir = tmp_path / "bge-test"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    mock_model = MagicMock()
    mock_st = MagicMock()
    mock_st.SentenceTransformer.return_value = mock_model
    with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
        emb = rank.Embedder(model_dir=str(model_dir))
    assert emb.mode == "semantic"
    assert emb.load_error is None
    assert emb._model is mock_model
    # BGE-named model dirs must apply the asymmetric query prefix.
    assert emb._query_prefix.startswith("Represent this sentence")
    # Loading must be pinned to the local path, never a hub model ID.
    assert mock_st.SentenceTransformer.call_args[0][0] == str(model_dir)


def test_embedder_sets_offline_env_guards(tmp_path):
    rank.Embedder(model_dir=str(tmp_path / "nope"), allow_lexical_fallback=True)
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def test_lexical_fallback_fenced_above_max_candidates():
    """The lexical path must refuse full-pool-scale input (P13 fence)."""
    emb = _lexical_embedder()
    docs = ["doc"] * (rank.LEXICAL_MAX_CANDIDATES + 1)
    with pytest.raises(RuntimeError, match="fenced"):
        emb.similarities("jd", docs)


def test_embedder_mode_attribute_is_always_set():
    emb = _lexical_embedder()
    assert emb.mode in {"lexical", "semantic"}


# ─── Stage 0: validate_candidate ────────────────────────────────────────────

def test_validate_valid_record_passes():
    is_valid, issues = rank.validate_candidate(_strong_candidate())
    assert is_valid
    assert issues == []


def test_validate_non_dict_rejected():
    is_valid, issues = rank.validate_candidate("not a dict")
    assert not is_valid
    assert issues


def test_validate_missing_candidate_id_rejected():
    c = _strong_candidate()
    del c["candidate_id"]
    is_valid, issues = rank.validate_candidate(c)
    assert not is_valid
    assert any("candidate_id" in m for m in issues)


def test_validate_empty_candidate_id_rejected():
    c = _strong_candidate()
    c["candidate_id"] = "   "
    is_valid, issues = rank.validate_candidate(c)
    assert not is_valid
    assert any("candidate_id" in m for m in issues)


def test_validate_non_string_candidate_id_rejected():
    c = _strong_candidate()
    c["candidate_id"] = 12345
    is_valid, issues = rank.validate_candidate(c)
    assert not is_valid
    assert any("candidate_id" in m for m in issues)


def test_validate_profile_wrong_type_rejected():
    c = _strong_candidate()
    c["profile"] = "not a dict"
    is_valid, issues = rank.validate_candidate(c)
    assert not is_valid
    assert any("profile" in m for m in issues)


def test_validate_skills_wrong_type_rejected():
    c = _strong_candidate()
    c["skills"] = {"name": "Python"}   # dict instead of list
    is_valid, issues = rank.validate_candidate(c)
    assert not is_valid
    assert any("skills" in m for m in issues)


def test_validate_career_history_wrong_type_rejected():
    c = _strong_candidate()
    c["career_history"] = "5 years at Google"
    is_valid, issues = rank.validate_candidate(c)
    assert not is_valid
    assert any("career_history" in m for m in issues)


def test_validate_redrob_signals_wrong_type_rejected():
    c = _strong_candidate()
    c["redrob_signals"] = [1, 2, 3]
    is_valid, issues = rank.validate_candidate(c)
    assert not is_valid
    assert any("redrob_signals" in m for m in issues)


def test_validate_missing_profile_warns_not_rejected():
    c = _strong_candidate()
    del c["profile"]
    is_valid, issues = rank.validate_candidate(c)
    assert is_valid
    assert any("profile" in m for m in issues)


def test_validate_missing_skills_warns_not_rejected():
    c = _strong_candidate()
    del c["skills"]
    is_valid, issues = rank.validate_candidate(c)
    assert is_valid
    assert any("skills" in m for m in issues)


def test_validate_missing_career_history_warns_not_rejected():
    c = _strong_candidate()
    del c["career_history"]
    is_valid, issues = rank.validate_candidate(c)
    assert is_valid
    assert any("career_history" in m for m in issues)


def test_validate_missing_redrob_signals_warns_not_rejected():
    c = _strong_candidate()
    del c["redrob_signals"]
    is_valid, issues = rank.validate_candidate(c)
    assert is_valid
    assert any("redrob_signals" in m for m in issues)


def test_validate_negative_years_of_experience_warns():
    c = _strong_candidate()
    c["profile"]["years_of_experience"] = -1
    is_valid, issues = rank.validate_candidate(c)
    assert is_valid
    assert any("years_of_experience" in m for m in issues)


def test_validate_unknown_skill_proficiency_warns():
    c = _strong_candidate()
    c["skills"][0]["proficiency"] = "ninja"
    is_valid, issues = rank.validate_candidate(c)
    assert is_valid
    assert any("proficiency" in m for m in issues)


def test_validate_rate_field_out_of_range_warns():
    c = _strong_candidate()
    c["redrob_signals"]["recruiter_response_rate"] = 1.5
    is_valid, issues = rank.validate_candidate(c)
    assert is_valid
    assert any("recruiter_response_rate" in m for m in issues)


def test_validate_negative_skill_duration_warns():
    c = _strong_candidate()
    c["skills"][0]["duration_months"] = -10
    is_valid, issues = rank.validate_candidate(c)
    assert is_valid
    assert any("duration_months" in m for m in issues)


def test_validate_negative_job_duration_warns():
    c = _strong_candidate()
    c["career_history"][0]["duration_months"] = -5
    is_valid, issues = rank.validate_candidate(c)
    assert is_valid
    assert any("duration_months" in m for m in issues)


def test_validate_profile_completeness_out_of_range_warns():
    c = _strong_candidate()
    c["redrob_signals"]["profile_completeness_score"] = 150
    is_valid, issues = rank.validate_candidate(c)
    assert is_valid
    assert any("profile_completeness_score" in m for m in issues)


# ─── Stage 0: load_candidates validation integration ─────────────────────────

def _write_jsonl(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _write_json(records, path):
    with open(path, "w") as f:
        json.dump(records, f)


def test_load_candidates_valid_jsonl_all_returned(tmp_path):
    p = tmp_path / "cands.jsonl"
    _write_jsonl([_strong_candidate()], p)
    result = rank.load_candidates(str(p))
    assert len(result) == 1
    assert result[0]["candidate_id"] == "C_STRONG"


def test_load_candidates_invalid_record_excluded(tmp_path):
    valid = _strong_candidate()
    no_id = {k: v for k, v in _strong_candidate().items() if k != "candidate_id"}
    p = tmp_path / "cands.jsonl"
    _write_jsonl([valid, no_id], p)
    result = rank.load_candidates(str(p))
    assert len(result) == 1
    assert result[0]["candidate_id"] == "C_STRONG"


def test_load_candidates_all_invalid_returns_empty(tmp_path):
    p = tmp_path / "cands.jsonl"
    _write_jsonl([{"no_id": True}, {"also_bad": True}], p)
    result = rank.load_candidates(str(p))
    assert result == []


def test_load_candidates_json_format_invalid_excluded(tmp_path):
    valid = _strong_candidate()
    bad_skills = _strong_candidate()
    bad_skills["candidate_id"] = "C_BAD"
    bad_skills["skills"] = "not a list"
    p = tmp_path / "cands.json"
    _write_json([valid, bad_skills], p)
    result = rank.load_candidates(str(p))
    assert len(result) == 1
    assert result[0]["candidate_id"] == "C_STRONG"


def test_load_candidates_malformed_jsonl_line_skipped(tmp_path):
    p = tmp_path / "cands.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps(_strong_candidate()) + "\n")
        f.write("{this is not valid json}\n")
    result = rank.load_candidates(str(p))
    assert len(result) == 1
    assert result[0]["candidate_id"] == "C_STRONG"


def test_load_candidates_warning_record_still_scored(tmp_path):
    # Missing redrob_signals is a warning, not a rejection — record should be scored.
    c = _strong_candidate()
    del c["redrob_signals"]
    p = tmp_path / "cands.jsonl"
    _write_jsonl([c], p)
    result = rank.load_candidates(str(p))
    assert len(result) == 1


# ─── Original full-stack depth test (unchanged) ──────────────────────────────

def test_full_stack_candidate_scores_higher_than_shallow():
    """4-pillar candidate should outscore an otherwise identical 0-pillar candidate."""
    full_stack = _strong_candidate()
    # full_stack already covers retrieval + ranking via description/skills

    shallow = _strong_candidate()
    shallow["candidate_id"] = "C_SHALLOW"
    shallow["career_history"][0]["description"] = "Worked on data pipelines."
    shallow["skills"] = [
        {"name": "Python", "proficiency": "expert", "duration_months": 60, "endorsements": 10}
    ]

    cov_full = rank.concept_coverage(full_stack, rank.JOB_DESCRIPTION)
    cov_shallow = rank.concept_coverage(shallow, rank.JOB_DESCRIPTION)

    score_full, _ = rank.score_career(full_stack, cov_full)
    score_shallow, _ = rank.score_career(shallow, cov_shallow)

    assert score_full > score_shallow
