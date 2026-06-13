# Ranking Validation — proxy for the hidden ground truth

The submission is scored against a hidden ground truth (80% of the composite is
NDCG@10 + NDCG@50). We cannot see it, so we validate the ranking two ways that
*don't* need it: a hand-labeled archetype harness and a weight-sensitivity
analysis. Both are reproducible (`pytest test_archetypes.py`,
`python weight_sensitivity.py --candidates <pool>`).

## 1. Archetype ordering (`test_archetypes.py`)

A 9-candidate pool, one per archetype the JD explicitly names, ranked end-to-end
through the *real* two-stage pipeline (bge-small semantic re-rank, offline CPU).
Observed ordering and subscore breakdown:

| rank | archetype        | score  | sem  | career | behav | loc | avail |
|-----:|------------------|-------:|-----:|-------:|------:|----:|------:|
| 1 | PERFECT             | 0.8868 | 0.82 | 1.00 | 0.74 | 1.00 | 0.93 |
| 2 | STALE_STRONG        | 0.7956 | 0.82 | 1.00 | 0.33 | 1.00 | 0.33 |
| 3 | TIER5_PLAIN         | 0.7851 | 0.70 | 0.83 | 0.74 | 1.00 | 0.93 |
| 4 | PURE_RESEARCH       | 0.7355 | 0.72 | 0.64 | 0.74 | 1.00 | 0.93 |
| 5 | JOB_HOPPER          | 0.6467 | 0.75 | 0.30 | 0.74 | 1.00 | 0.93 |
| 6 | CV_ONLY             | 0.5556 | 0.62 | 0.17 | 0.74 | 1.00 | 0.93 |
| 7 | CONSULTING_ONLY     | 0.5308 | 0.62 | 0.08 | 0.74 | 1.00 | 0.93 |
| 8 | KEYWORD_STUFFER     | 0.0000 | 0.66 | 0.31 | — | — | — |
| 9 | HONEYPOT            | 0.0000 | 0.77 | 0.99 | — | — | — |

**What this confirms**

- **The JD's central thesis holds.** `TIER5_PLAIN` — a "Senior Software Engineer"
  with *no AI keywords in the skills list* but a career history describing a
  recommendation/retrieval/ranking system at a product company — ranks **above**
  `KEYWORD_STUFFER` (a Marketing Manager whose skills list is full of AI terms).
  The ranker rewards demonstrated work over keyword density. This is the trap the
  organizers built on purpose, and the pipeline does not fall for it.
- **Honeypots are neutralized.** `HONEYPOT` scores 0.99 on career and would
  otherwise rank near the top; the honeypot detector (here, a role that ends
  before it starts + ten zero-usage "expert" skills) zeroes it to last.
- **Behavioral availability is real.** `STALE_STRONG` is skill-identical to
  `PERFECT` but stale (last active ~1 year ago), 3% recruiter-response, and not
  open to work — it drops from 0.8868 to 0.7956.

**Honest notes (not papered over)**

- `KEYWORD_STUFFER` here is caught by the *honeypot* filter (10 "expert" skills
  with zero duration and zero endorsements is itself an impossible profile), not
  purely by relevance ranking. A subtler stuffer would be down-ranked by the
  career/title path instead; the trust-discount on bare skill listings
  (`concept_coverage`) handles that case.
- In a 9-candidate pool with only two true fits, the *absolute* ranks of the
  negative archetypes (e.g. `PURE_RESEARCH` at #4) are an artifact of the tiny
  pool — there simply aren't other strong fits to fill ranks 4–7. What matters is
  that every negative archetype ranks **below** both true fits, which holds. At
  100K scale the negatives are buried far below the top (see §3).
- `PURE_RESEARCH` keeps a high semantic score (0.72) because its text genuinely
  resembles the JD; the pure-research and no-production career penalties
  (`×0.70`, `×0.80`) are what pull it under the real fits.

## 2. Weight sensitivity (`weight_sensitivity.py`)

The final score blends five subscores (0.40 semantic / 0.30 career / 0.15
behavioral / 0.10 location / 0.05 availability). A good ranking should not hinge
on the exact weights. Holding the prescreen finalist set fixed and re-blending
the full 100K pool under perturbed weights:

| scenario        | top-10 ∩ base | top-50 ∩ base | #1 unchanged |
|-----------------|--------------:|--------------:|:------------:|
| base            | 10/10 | 50/50 | yes |
| more-semantic   |  9/10 | 47/50 | yes |
| more-career     |  9/10 | 48/50 | yes |
| more-behavioral |  8/10 | 46/50 | yes |
| availability-up |  8/10 | 46/50 | yes |
| equal-ish       |  7/10 | 40/50 | yes |

**The #1 candidate is invariant across every scenario**, and the top-10 holds
8–9/10 under realistic perturbations (7/10 even at near-equal weights). The
ranking reflects the candidates, not a fragile weight choice.

## 3. Real-pool top-20 spot check

The actual top-20 on the full 100K pool are all production AI/ML/recsys/search/
NLP engineers at product companies (Zomato, Amazon, CRED, LinkedIn, Paytm, Meta,
Google, Apple, Razorpay, Freshworks, Zoho, Yellow.ai, Sarvam AI), concentrated in
the JD's 6–8-year ideal band. No marketing/consulting-only/pure-research-only
profiles appear in the top ranks, and the self-check reports 0 honeypots in the
top 100.

## Why the weights were not "tuned" further

Without the hidden ground truth, tuning weights to a synthetic pool would risk
overfitting and regressing a result that both the archetype harness and the real
top-20 already show is on-thesis and robust. The blend is left at its
spec-documented values; any future change would be re-validated through these two
harnesses.
