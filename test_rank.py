"""
Unit tests for the candidate ranking pipeline.

Run:  pytest test_rank.py -v
These tests use the lexical fallback embedder, so they need no model download.
"""

import json
import math
import sys
from unittest.mock import MagicMock, patch

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
        [fake, _strong_candidate()], rank.JOB_DESCRIPTION, rank.Embedder()
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
    results = rank.rank_candidates([fake], rank.JOB_DESCRIPTION, rank.Embedder())
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
    emb = rank.Embedder()
    sims = emb.similarities("ranking retrieval embeddings", ["ranking retrieval", "cooking food"])
    assert all(0.0 <= s <= 1.0 for s in sims)
    assert sims[0] > sims[1]  # relevant doc scores above irrelevant


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


# ─── Embedder: mode detection and load_error ──────────────────────────────────

def test_embedder_lexical_mode_when_package_missing():
    """Setting sys.modules entry to None simulates the package not being installed."""
    with patch.dict(sys.modules, {"sentence_transformers": None}):
        emb = rank.Embedder()
    assert emb.mode == "lexical"
    assert emb._model is None
    assert emb._model_id is None
    assert emb.load_error is not None
    assert "sentence-transformers" in emb.load_error.lower()


def test_embedder_lexical_mode_when_model_load_fails():
    """Package present but SentenceTransformer() raises — falls back to lexical."""
    mock_st = MagicMock()
    mock_st.SentenceTransformer.side_effect = OSError("simulated download failure")
    with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
        emb = rank.Embedder()
    assert emb.mode == "lexical"
    assert emb.load_error is not None
    # Error message must contain the model ID and exception details.
    assert "OSError" in emb.load_error
    assert "simulated download failure" in emb.load_error


def test_embedder_semantic_mode_clears_load_error():
    """Successful model load → load_error is None and mode is 'semantic'."""
    mock_model = MagicMock()
    mock_st = MagicMock()
    mock_st.SentenceTransformer.return_value = mock_model
    with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
        emb = rank.Embedder()
    assert emb.mode == "semantic"
    assert emb.load_error is None
    assert emb._model is mock_model


def test_embedder_import_error_does_not_retry_second_model():
    """ImportError short-circuits the loop; SentenceTransformer is only constructed once."""
    mock_st = MagicMock()
    # Every SentenceTransformer(model_id) call raises ImportError.
    mock_st.SentenceTransformer.side_effect = ImportError("no module named 'sentence_transformers'")
    with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
        emb = rank.Embedder()
    assert emb.mode == "lexical"
    # ImportError must break the loop — constructor called at most once.
    assert mock_st.SentenceTransformer.call_count <= 1


def test_embedder_mode_attribute_is_always_set():
    """mode is always 'lexical' or 'semantic' — never absent or None."""
    emb = rank.Embedder()
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
