"""
End-to-end archetype ordering harness — a hand-labeled proxy for the hidden
ground truth.

The submission is scored against a hidden ground truth we cannot see (80% of the
score is NDCG@10 + NDCG@50). The unit tests in test_rank.py check each signal in
isolation with the lexical fallback embedder. This file does the complementary
thing: it builds a tiny pool containing one instance of each archetype the job
description explicitly names, runs the *real* two-stage pipeline (bge-small
semantic re-rank, offline CPU), and asserts the relative ordering the JD demands.

The single most important assertion is the JD's own thesis:

    a buzzword-free "Tier-5" candidate who *built* a recommendation/ranking system
    must outrank a keyword-stuffer whose skills list is full of AI terms.

If that fails, the ranker has fallen for the trap the organizers built on purpose.

Run:  HF_HUB_OFFLINE=1 pytest test_archetypes.py -v
Needs the vendored model in ./models (python download_model.py).
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from datetime import date

import pytest

import rank

REFERENCE_DATE = date(2026, 6, 1)


# ─── Shared "neutral" context ────────────────────────────────────────────────
# Every archetype gets the same strong behavioral / location / availability
# context so that differences in the final ranking come from the relevance axis
# (semantic + career), not from confounds. The two exceptions (HONEYPOT and
# STALE_STRONG) deliberately break one dimension to test that dimension.

def _signals(**overrides):
    base = {
        "recruiter_response_rate": 0.75,
        "interview_completion_rate": 0.80,
        "github_activity_score": 45,
        "profile_completeness_score": 90,
        "last_active_date": "2026-05-20",
        "signup_date": "2023-01-01",
        "saved_by_recruiters_30d": 6,
        "offer_acceptance_rate": 0.6,
        "open_to_work_flag": True,
        "willing_to_relocate": True,
        "preferred_work_mode": "hybrid",
        "notice_period_days": 30,
        "applications_submitted_30d": 2,
        "skill_assessment_scores": {},
    }
    base.update(overrides)
    return base


def _pune(profile):
    profile.setdefault("location", "Pune")
    profile.setdefault("country", "India")
    return profile


def _skill(name, prof="advanced", dur=36, end=15):
    return {"name": name, "proficiency": prof, "duration_months": dur, "endorsements": end}


# ─── Archetypes ──────────────────────────────────────────────────────────────

def _perfect():
    """6-8y, product company, shipped end-to-end retrieval/ranking. The bullseye."""
    return {
        "candidate_id": "CAND_0000001",
        "profile": _pune({
            "current_title": "Senior Machine Learning Engineer",
            "current_industry": "Internet",
            "years_of_experience": 7.0,
            "headline": "Search & ranking engineer",
            "summary": "Owns retrieval and ranking systems in production.",
        }),
        "career_history": [{
            "company": "Flipkart", "title": "Senior Machine Learning Engineer",
            "duration_months": 48, "is_current": True, "industry": "Internet",
            "company_size": "5001-10000",
            "description": ("Owned the product search ranking stack: embeddings-based "
                            "retrieval with FAISS, hybrid search, and a learning-to-rank "
                            "re-ranker. Ran offline NDCG/MAP evaluation and online A/B tests; "
                            "managed embedding drift and index refresh in production."),
        }],
        "education": [{"tier": "tier_1", "field_of_study": "Computer Science",
                       "start_year": 2013, "end_year": 2017}],
        "skills": [
            _skill("Embeddings", "expert", 48, 30), _skill("FAISS", "advanced", 40, 18),
            _skill("Information Retrieval", "expert", 60, 22),
            _skill("Learning to Rank", "advanced", 36, 12),
            _skill("Python", "expert", 84, 40),
        ],
        "redrob_signals": _signals(),
    }


def _tier5_plain():
    """
    The JD's central example: NO AI buzzwords in the skills list, but the career
    history describes building a recommendation/ranking system at a product
    company. Must surface near the top.
    """
    return {
        "candidate_id": "CAND_0000002",
        "profile": _pune({
            "current_title": "Senior Software Engineer",
            "current_industry": "Internet",
            "years_of_experience": 7.0,
            "headline": "Backend engineer, personalization",
            "summary": "Builds large-scale personalization and matching systems.",
        }),
        "career_history": [{
            "company": "Myntra", "title": "Senior Software Engineer",
            "duration_months": 50, "is_current": True, "industry": "Internet",
            "company_size": "1001-5000",
            "description": ("Built and owned the product recommendation engine serving "
                            "40M users. Designed the candidate retrieval and ranking "
                            "pipeline, trained the matching models, and measured relevance "
                            "with offline NDCG and online A/B experiments to lift engagement."),
        }],
        "education": [{"tier": "tier_2", "field_of_study": "Computer Science",
                       "start_year": 2013, "end_year": 2017}],
        # Deliberately generic — none of the AI keyword surfaces appear here.
        "skills": [
            _skill("Python", "expert", 84, 35), _skill("Java", "advanced", 60, 20),
            _skill("SQL", "advanced", 72, 18), _skill("Distributed Systems", "advanced", 48, 14),
        ],
        "redrob_signals": _signals(),
    }


def _keyword_stuffer():
    """
    Every AI keyword as an 'expert' skill with zero usage evidence, but the title
    is Marketing Manager and the work is marketing. The explicit trap.
    """
    stuffed = [
        _skill(n, "expert", 0, 0) for n in (
            "Embeddings", "RAG", "Pinecone", "Vector Databases", "Retrieval",
            "Ranking", "LLM", "LangChain", "NDCG", "Semantic Search",
        )
    ]
    return {
        "candidate_id": "CAND_0000003",
        "profile": _pune({
            "current_title": "Marketing Manager",
            "current_industry": "Internet",
            "years_of_experience": 4.0,
            "headline": "AI-driven marketing leader",
            "summary": "Marketing manager passionate about AI, RAG and LLMs.",
        }),
        "career_history": [{
            "company": "Zomato", "title": "Marketing Manager",
            "duration_months": 36, "is_current": True, "industry": "Internet",
            "company_size": "5001-10000",
            "description": ("Ran brand and performance marketing campaigns, managed the "
                            "social calendar and influencer budget, and grew follower count."),
        }],
        "education": [{"tier": "tier_3", "field_of_study": "Marketing",
                       "start_year": 2016, "end_year": 2020}],
        "skills": stuffed,
        "redrob_signals": _signals(),
    }


def _consulting_only():
    """Entire career at services/consulting firms. JD explicit do-not-want."""
    return {
        "candidate_id": "CAND_0000004",
        "profile": _pune({
            "current_title": "Technology Analyst",
            "current_industry": "IT Services",
            "years_of_experience": 8.0,
            "headline": "IT consultant",
            "summary": "Delivers client projects across data and ML.",
        }),
        "career_history": [
            {"company": "Infosys", "title": "Technology Analyst", "duration_months": 60,
             "is_current": True, "industry": "IT Services", "company_size": "10001+",
             "description": "Delivered data pipelines and ML proofs of concept for clients."},
            {"company": "TCS", "title": "Systems Engineer", "duration_months": 42,
             "is_current": False, "industry": "IT Services", "company_size": "10001+",
             "description": "Maintained client reporting systems and ETL jobs."},
        ],
        "education": [{"tier": "tier_3", "field_of_study": "Information Technology",
                       "start_year": 2012, "end_year": 2016}],
        "skills": [_skill("Python", "advanced", 60, 12), _skill("Machine Learning", "intermediate", 24, 6)],
        "redrob_signals": _signals(),
    }


def _pure_research():
    """Academic research, publications, no production deployment. Do-not-want."""
    return {
        "candidate_id": "CAND_0000005",
        "profile": _pune({
            "current_title": "Research Scientist",
            "current_industry": "Research",
            "years_of_experience": 6.0,
            "headline": "NLP researcher",
            "summary": "Publishes on neural retrieval; academic research only.",
        }),
        "career_history": [{
            "company": "IIT Research Lab", "title": "Research Scientist",
            "duration_months": 48, "is_current": True, "industry": "Research",
            "company_size": "201-500",
            "description": ("Published papers on neural information retrieval and ranking "
                            "models; ran offline experiments. No production deployment."),
        }],
        "education": [{"tier": "tier_1", "field_of_study": "Computer Science",
                       "start_year": 2012, "end_year": 2018}],
        "skills": [_skill("Information Retrieval", "expert", 60, 20), _skill("PyTorch", "expert", 48, 15)],
        "redrob_signals": _signals(),
    }


def _cv_only():
    """Computer vision / robotics, no NLP/IR. Do-not-want."""
    return {
        "candidate_id": "CAND_0000006",
        "profile": _pune({
            "current_title": "Computer Vision Engineer",
            "current_industry": "Internet",
            "years_of_experience": 7.0,
            "headline": "Perception engineer",
            "summary": "Builds object detection and SLAM systems.",
        }),
        "career_history": [{
            "company": "Ola Electric", "title": "Computer Vision Engineer",
            "duration_months": 48, "is_current": True, "industry": "Internet",
            "company_size": "1001-5000",
            "description": ("Built object detection, image segmentation, and SLAM for "
                            "autonomous navigation. Deployed perception models to vehicles."),
        }],
        "education": [{"tier": "tier_2", "field_of_study": "Robotics",
                       "start_year": 2012, "end_year": 2016}],
        "skills": [_skill("Computer Vision", "expert", 60, 20), _skill("SLAM", "advanced", 36, 10),
                   _skill("Object Detection", "expert", 48, 14)],
        "redrob_signals": _signals(),
    }


def _job_hopper():
    """Rising titles every ~15 months chasing seniority. Do-not-want."""
    return {
        "candidate_id": "CAND_0000007",
        "profile": _pune({
            "current_title": "Principal Engineer",
            "current_industry": "Internet",
            "years_of_experience": 6.0,
            "headline": "Fast-rising engineer",
            "summary": "ML engineer with retrieval experience.",
        }),
        "career_history": [
            {"company": "StartupD", "title": "Principal Engineer", "duration_months": 14,
             "is_current": True, "industry": "Internet", "company_size": "51-200",
             "description": "Retrieval and ranking work."},
            {"company": "StartupC", "title": "Staff Engineer", "duration_months": 15,
             "is_current": False, "industry": "Internet", "company_size": "51-200",
             "description": "Search relevance."},
            {"company": "StartupB", "title": "Senior Engineer", "duration_months": 16,
             "is_current": False, "industry": "Internet", "company_size": "11-50",
             "description": "Recommendation features."},
            {"company": "StartupA", "title": "Engineer", "duration_months": 15,
             "is_current": False, "industry": "Internet", "company_size": "11-50",
             "description": "Backend and ML."},
        ],
        "education": [{"tier": "tier_2", "field_of_study": "Computer Science",
                       "start_year": 2014, "end_year": 2018}],
        "skills": [_skill("Embeddings", "advanced", 24, 10), _skill("Python", "expert", 60, 20)],
        "redrob_signals": _signals(),
    }


def _honeypot():
    """Internally impossible: a role that ends before it starts + zero-usage experts."""
    return {
        "candidate_id": "CAND_0000008",
        "profile": _pune({
            "current_title": "Senior Machine Learning Engineer",
            "current_industry": "Internet",
            "years_of_experience": 8.0,
            "headline": "Senior ML engineer",
            "summary": "Expert across the entire AI stack.",
        }),
        "career_history": [{
            "company": "Flipkart", "title": "Senior Machine Learning Engineer",
            "start_date": "2025-01-01", "end_date": "2020-01-01",  # ends before it starts
            "duration_months": 60, "is_current": False, "industry": "Internet",
            "company_size": "5001-10000",
            "description": "Embeddings, retrieval, ranking, vector databases, production ML.",
        }],
        "education": [{"tier": "tier_1", "field_of_study": "Computer Science",
                       "start_year": 2010, "end_year": 2014}],
        "skills": [_skill(n, "expert", 0, 0) for n in
                   ("Embeddings", "FAISS", "Pinecone", "RAG", "Retrieval",
                    "Ranking", "LLM", "Vector Databases", "NDCG", "MLOps")],
        "redrob_signals": _signals(),
    }


def _stale_strong():
    """Same strength as PERFECT but unreachable: stale + unresponsive + not open."""
    c = _perfect()
    c["candidate_id"] = "CAND_0000009"
    c["redrob_signals"] = _signals(
        last_active_date="2025-05-20",          # ~1 year stale vs reference
        recruiter_response_rate=0.03,
        open_to_work_flag=False,
        applications_submitted_30d=0,
        saved_by_recruiters_30d=0,
    )
    return c


ARCHETYPES = {
    "PERFECT": _perfect, "TIER5_PLAIN": _tier5_plain, "KEYWORD_STUFFER": _keyword_stuffer,
    "CONSULTING_ONLY": _consulting_only, "PURE_RESEARCH": _pure_research, "CV_ONLY": _cv_only,
    "JOB_HOPPER": _job_hopper, "HONEYPOT": _honeypot, "STALE_STRONG": _stale_strong,
}
ID_TO_NAME = {fn()["candidate_id"]: name for name, fn in ARCHETYPES.items()}


# ─── Fixture: run the real pipeline once ─────────────────────────────────────

@pytest.fixture(scope="module")
def ranked():
    """Run the full two-stage pipeline with the real semantic model, once."""
    try:
        embedder = rank.Embedder()  # strict offline load from ./models
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"semantic model unavailable ({exc}); run python download_model.py")
    pool = [fn() for fn in ARCHETYPES.values()]
    results = rank.rank_candidates(pool, rank.JOB_DESCRIPTION, embedder,
                                   reference_date=REFERENCE_DATE)
    rank_of = {ID_TO_NAME[r["candidate_id"]]: r["rank"] for r in results}
    score_of = {ID_TO_NAME[r["candidate_id"]]: r["score"] for r in results}
    return rank_of, score_of, results


# ─── The ground-truth-proxy assertions ───────────────────────────────────────

def test_perfect_fit_ranks_first(ranked):
    rank_of, _, _ = ranked
    assert rank_of["PERFECT"] == 1


def test_tier5_plain_language_beats_keyword_stuffer(ranked):
    """The JD's central thesis — the whole point of the challenge."""
    rank_of, _, _ = ranked
    assert rank_of["TIER5_PLAIN"] < rank_of["KEYWORD_STUFFER"]


def test_tier5_plain_language_surfaces_near_top(ranked):
    rank_of, _, _ = ranked
    assert rank_of["TIER5_PLAIN"] <= 3


def test_negative_archetypes_rank_below_the_two_real_fits(ranked):
    rank_of, _, _ = ranked
    ceiling = max(rank_of["PERFECT"], rank_of["TIER5_PLAIN"])
    for name in ("KEYWORD_STUFFER", "CONSULTING_ONLY", "PURE_RESEARCH",
                 "CV_ONLY", "JOB_HOPPER"):
        assert rank_of[name] > ceiling, f"{name} should rank below the real fits"


def test_honeypot_is_zeroed_and_last(ranked):
    rank_of, score_of, results = ranked
    assert score_of["HONEYPOT"] == 0.0
    assert rank_of["HONEYPOT"] == len(results)


def test_unreachable_strong_candidate_is_down_weighted(ranked):
    """Behavioral availability is real: a stale, unresponsive twin ranks lower."""
    rank_of, _, _ = ranked
    assert rank_of["STALE_STRONG"] > rank_of["PERFECT"]


def test_scores_are_monotonic_non_increasing(ranked):
    _, _, results = ranked
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
