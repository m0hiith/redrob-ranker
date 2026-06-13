# Redrob Candidate Ranker

Ranks the top 100 of 100,000 candidates against the **Senior AI Engineer —
Founding Team** JD for the *Intelligent Candidate Discovery & Ranking
Challenge*. It does **not** keyword match: it screens the full pool on
recruiter-grade features, semantically re-ranks the finalists, and filters
organizer-planted fakes — fully offline, CPU-only, deterministic.

## Reproduce

```bash
pip install -r requirements.txt        # pinned versions

# One-time pre-computation (network, ~2 min): vendor the embedding model.
python download_model.py

# Ranking — offline, CPU-only, < 5 min:
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# Pre-upload gate (format + pool membership + honeypot rate + reasoning variety):
python self_check.py ./submission.csv ./candidates.jsonl
```

`--candidates` accepts `.json`, `.jsonl`, or gzip variants, and content-sniffs
JSONL inside a `.json` file (the official bundle ships exactly that). Output is
`candidate_id,rank,score,reasoning` — ranks 1–100, scores non-increasing,
ties broken by ascending `candidate_id` as the validator requires.

### Docker (Stage-3 style)

```bash
docker build -t redrob-ranker .        # installs deps + vendors model (pre-computation)
docker run --rm --network=none \
  -v /path/to/candidates.json:/data/candidates.json:ro -v "$PWD/out:/out" \
  redrob-ranker python rank.py --candidates /data/candidates.json --out /out/submission.csv
```

### Sandbox demo

`app.py` is a Streamlit app (deployable to Streamlit Cloud / HF Spaces) that
accepts ≤100 candidates and produces the ranked CSV end-to-end:
`streamlit run app.py`.

## Architecture — two-stage ranking

The same shape the JD itself describes (hybrid retrieval, then re-ranking):

```
100,000 candidates
      │  Stage A — full-pool screen (pure Python, no embeddings)
      │    concept coverage over a skill ontology (word-boundary matched,
      │    endorsement/duration-trusted) + career trajectory + behavioral
      │    + location + availability
      ▼
 top 2,000 finalists          ← 20× margin over the 100 output rows
      │  Stage B — semantic re-rank
      │    BAAI/bge-small-en-v1.5 (local, CPU, offline) cosine vs a
      │    surface-form-expanded JD document
      ▼
 final blend → honeypot suppression → top-100 CSV + per-rank reasoning
```

**Why two stages:** embedding all 100K docs on CPU takes tens of minutes and
busts the 5-minute budget; embedding 2,000 takes well under a minute. The
Stage-A proxy (concept coverage) and Stage-B embeddings agree on what they're
measuring, so no plausible top-100 candidate is lost at a 20× margin.

### Final blend (Stage B)

```
final = 0.40·semantic + 0.30·career + 0.15·behavioral + 0.10·location + 0.05·availability
```

- **semantic** — cosine(JD doc, candidate doc), bge-small-en-v1.5. The JD doc
  expands each requirement into concrete tool surfaces (FAISS, Pinecone,
  NDCG…) so similarity is measured against real skills, not category names.
- **career (0.30)** — recruiter-style trajectory: relevant titles, description
  evidence, product-company roles, duration-weighted; ±20% seniority fit
  (peak 7y, ideal band 5–9y per the JD); then the JD's explicit negative
  signals as multiplicative penalties:
  consulting-only ×0.55, CV/speech/robotics-without-IR ×0.60,
  LangChain-only ×0.65, pure-research ×0.70, title-chasing job-hopper ×0.75,
  no-production ×0.80.
- **behavioral (0.15)** — recruiter response rate, activity recency, interview
  completion, GitHub, profile completeness, recruiter saves, offer acceptance.
  A perfect-on-paper candidate who never responds isn't actually hireable.
- **location (0.10)** — city tiers for the hybrid Pune/Noida role: JD cities
  1.0 → welcome metros (Hyd/Mum/NCR/Blr) 0.85 → rest of India 0.65 → abroad
  with relocation 0.45 → abroad 0.20; a hard "remote" preference scores below
  hybrid/flexible.
- **availability (0.05)** — open-to-work flag, notice period (≤30d favored,
  per the JD), recent applications.

### Honeypot defense

`detect_honeypot` cross-checks internal consistency — skill tenure vs total
experience, role length vs career length, zero-usage "expert" stacks, expert
claims contradicted by failing assessments, claimed YoE vs summed tenure, and
date impossibilities (end-before-start, future roles, `duration_months`
contradicting the date span, "current" roles that ended, heavy role overlap,
activity before signup). Hard flags or 3+ soft flags ⇒ score forced to 0.0 and
sorted below every legitimate candidate, reasoning prefixed `HONEYPOT —`.
`self_check.py` reports the detector's rate inside the submitted top 100
(DQ threshold is 10%).

### Anti-keyword-stuffer design

- Word-boundary concept matching (a "Market Research Analyst" gets no credit
  for containing "search"; "roadmap" doesn't hit "MAP").
- A listed skill reaches full coverage only with ≥5 endorsements or ≥12 months
  of use — bare skill-list mentions (the stuffer signature) are discounted.
- Career evidence (titles + role descriptions at product companies) carries
  the largest non-semantic weight, so plain-language strong fits surface even
  without buzzwords.

### Determinism (Stage-3 reproduction)

Same data ⇒ byte-identical CSV, any day, any machine:
- the reference date for recency math is **derived from the dataset**
  (max `last_active_date`, 2026-05-27 for this pool), never `date.today()`;
- cosine similarities are **cached** in `similarity_cache/` on first run,
  keyed by a SHA-256 **fingerprint of the embedding model, the JD text, and the
  ordered candidate documents** — so a changed JD, a swapped model, or an edited
  profile yields a new key and can never read a stale value. Subsequent runs —
  and Stage-3 evaluation — load identical float64 values, eliminating BLAS
  non-determinism from PyTorch's parallel matmul on Apple Silicon. Cache hit:
  embed step 0 s. (Pre-fingerprint caches keyed on candidate IDs alone are still
  read for backward compatibility and migrated on load.) Pass `--no-cache` to
  recompute from the model and bypass the cache entirely;
- the JD's concept surface forms are emitted in **sorted** order, so the
  embedded text — and therefore the scores — are identical regardless of
  `PYTHONHASHSEED`;
- the model loads from a local vendored directory with `HF_HUB_OFFLINE=1` —
  the ranking step cannot touch the network, and a missing model is a hard
  error, never a silent algorithm change;
- ties broken by ascending `candidate_id` at every stage;
- reasoning phrasing is seeded by `crc32(candidate_id)` — varied but
  reproducible.

### Reasoning (Stage-4)

Generated after ranking so tone matches the band: top-10 leads with strengths,
11–50 names the trade-off, 51–100 leads with the gap. Every claim (title,
company, years, concepts, signal values) comes from fields on the profile;
an honest concern is always voiced when one exists; phrasing varies
deterministically per candidate.

## Measured compute (full 100K pool)

| Constraint | Limit | Measured (100,000 candidates, 465 MB) |
|------------|-------|----------------------------------------|
| Ranking wall-clock | ≤ 5 min | **3 min 5 s** (load ~17 s · screen 60 s · embed 107 s · blend 2 s) |
| Peak RAM | ≤ 16 GB | **1.6 GB** max RSS |
| GPU | none | none (CPU-only torch) |
| Network during ranking | none | none (`HF_HUB_OFFLINE=1`, local model) |

On that run: 88 of the 2,000 finalists were honeypot-flagged and suppressed
(the pool contains ~80 by spec — they are built to prescreen well), 0 honeypots
in the submitted top 100, official `validate_submission.py` passes, and all
100 reasoning strings are unique.

Machine: MacBook Air M1, 8 cores, 8 GB RAM, Python 3.13.

## Tests

```bash
pytest test_rank.py -v
```

112 tests cover: weight invariants, the seniority curve, every honeypot flag
(including the date-consistency set, verified for zero false positives on the
official 50-candidate sample), each negative signal, word-boundary matching
regressions, prescreen determinism, strict-offline embedder behavior, the
lexical fence, loader formats, reasoning variety/grounding/rank-tone, and
candidate validation. Tests run on the lexical fallback — no model download
needed. The TF-IDF fallback itself is **tests/smoke only**: it must be enabled
explicitly (`--allow-lexical-fallback`) and refuses pools >20K.

## Known limitations

- Stage-A uses concept coverage as a semantic proxy; a candidate describing
  relevant work in vocabulary entirely outside the ontology *and* with no
  matching skills could be screened out before embedding. The 2,000-finalist
  margin and the ontology's breadth make this unlikely but not impossible.
- Honeypot detection is precision-tuned (generous grace windows); subtle fakes
  with internally consistent dates can slip through to be handled by ordinary
  scoring.
- Company founding dates aren't in the data, so "8 years at a 3-year-old
  company" is only catchable via the date fields actually present.

## Tuning

Everything lives at the top of [`rank.py`](rank.py): `JOB_DESCRIPTION`
(role/must/good/negative), `CONCEPTS` (the skill ontology), `WEIGHTS`
(the Stage-B blend), `PRESCREEN_FINALISTS` (Stage-A handoff size).
