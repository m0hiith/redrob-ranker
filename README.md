# Redrob Candidate Ranker

A recruiter-grade ranking engine for the **Intelligent Candidate Discovery &
Ranking Challenge**. It does **not** keyword match — it scores semantic fit,
reasons about career trajectory like a recruiter, weighs behavioral signals, and
filters out organizer-planted fakes.

## Quick start

```bash
# Runs out of the box on pure stdlib (lexical fallback for semantic search):
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# For TRUE semantic search (recommended), install the model deps first:
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

`--candidates` accepts a `.json` array or a `.jsonl` file. Output is a Top-100
CSV: `candidate_id, rank, score, reasoning`.

## Pipeline (7 stages)

```
Job Description
      │  Stage 1 — JD Parser: weighted must-have / good-to-have / negative signals
      ▼
Embedding Model (all-MiniLM-L6-v2, lexical TF-IDF fallback)
      │  Stage 3 — Semantic Search: cosine(JD, candidate), normalised per cohort
      ▼
Per-candidate Feature Engineering
      │  Stage 2 — concept coverage over the must-have ontology
      ▼
Career Pattern Analysis  ── Stage 4 — recruiter-style trajectory + negative-signal penalties
      ▼
Behavioral Signals       ── Stage 5 — response rate, recency, interviews, GitHub, saves
      ▼
Honeypot Filter          ── Stage 7 — fabricated profiles forced to the bottom
      ▼
Weighted Ranker          ── Stage 6 — final score
      ▼
Top 100 CSV
```

### Stage 6 — final score

```
final_score = 0.45 · semantic
            + 0.25 · career
            + 0.15 · behavioral
            + 0.10 · location
            + 0.05 · availability
```

## Why it beats a keyword matcher

- **Semantic over lexical.** A "Marketing Manager" who lists `Python, OpenAI,
  LangChain, RAG` does *not* outrank a "Senior AI Engineer" who built search
  infrastructure. The JD is expanded into concrete tool surfaces before
  embedding, so similarity is measured against real skills, not buzzwords.
- **Career reasoning.** Negative signals from the JD are enforced:
  consulting-only careers, pure-research profiles, computer-vision-only skill
  sets, LangChain-only experience, and no-production histories are all
  penalised.
- **Behavioral reality.** A 95%-match candidate who's been inactive 8 months
  with a 5% response rate loses to a 90%-match candidate active yesterday with
  an 85% response rate.
- **Honeypot defense.** Impossible timelines (a skill used longer than the whole
  career, a role longer than total experience) and fabricated expertise (many
  "expert" skills with zero usage, expert claims contradicted by failing
  assessments) are detected and pushed below every legitimate candidate, tagged
  `⚠ HONEYPOT` in the reasoning.

## Semantic search modes

| Mode | When | Quality |
|------|------|---------|
| `semantic` | `sentence-transformers` installed | True embeddings (all-MiniLM-L6-v2) |
| `lexical` | fallback (no deps / offline) | TF-IDF cosine over the cohort |

The runner prints which mode it used. Scores are normalised by the cohort max so
the 0.45 semantic weight has real ranking influence in either mode.

## Tests

```bash
pytest test_rank.py -v
```

Covers the weight invariants, honeypot hard/soft flags, career negative signals,
semantic normalisation, and output shape. Runs on the lexical fallback, so no
model download is needed.

## Tuning

Everything lives at the top of [`rank.py`](rank.py):

- `JOB_DESCRIPTION` — retarget the role (must/good/negative requirements).
- `CONCEPTS` — the skill ontology each requirement expands into.
- `WEIGHTS` — the Stage 6 blend.
- `REFERENCE_DATE` — "today" for recency calculations.
