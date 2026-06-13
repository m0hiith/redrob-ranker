# Redrob Behavioral Signals — Coverage Audit

Audited against `redrob_signals_doc.docx` (source) and `rank.py` (implementation).

## Coverage summary

14 of 23 signals feed scoring. 9 are deliberately omitted.

## Signal-by-signal map

| # | Signal | Status | Where in rank.py |
|---|--------|--------|-----------------|
| 1 | `profile_completeness_score` | ✅ used + validated | `score_behavioral` (0.10 weight); validator range-checks [0,100] |
| 2 | `signup_date` | ✅ used (consistency) | Honeypot: flags `last_active < signup` |
| 3 | `last_active_date` | ✅ used heavily | Behavioral recency (0.18); dataset reference date; reasoning staleness |
| 4 | `open_to_work_flag` | ✅ used | `score_availability` (+0.50) |
| 5 | `profile_views_received_30d` | ⬜ omitted | Recruiter-attention circularity — see note below |
| 6 | `applications_submitted_30d` | ✅ used | `score_availability` (+0.10) |
| 7 | `recruiter_response_rate` | ✅ used + validated | `score_behavioral` (top weight 0.28); reasoning; validated |
| 8 | `avg_response_time_hours` | ⬜ omitted | Redundant with #7 — see note below |
| 9 | `skill_assessment_scores` | ✅ used (consistency) | Honeypot: expert claim vs assessment <30 |
| 10 | `connection_count` | ⬜ omitted | Network vanity, gameable, weak discriminator |
| 11 | `endorsements_received` | ⬜ omitted | Per-skill `skills[].endorsements` used instead (finer-grained; aggregate would double-count) |
| 12 | `notice_period_days` | ✅ used | `score_availability` tiered scoring; reasoning |
| 13 | `expected_salary_range_inr_lpa` | ⬜ omitted | JD states no salary band — nothing to match against |
| 14 | `preferred_work_mode` | ✅ used | `score_location` ±20% mode adjustment |
| 15 | `willing_to_relocate` | ✅ used | `score_location` base score |
| 16 | `github_activity_score` | ✅ used | `score_behavioral` (0.12); reasoning; −1 → 0.20 (mild penalty) |
| 17 | `search_appearance_30d` | ⬜ omitted | Recruiter-attention circularity — see note below |
| 18 | `saved_by_recruiters_30d` | ✅ used | `score_behavioral` (0.08, capped at /15) |
| 19 | `interview_completion_rate` | ✅ used + validated | `score_behavioral` (0.16); validated |
| 20 | `offer_acceptance_rate` | ✅ used + validated | `score_behavioral` (0.08); −1 sentinel → 0.50 (neutral); validated |
| 21 | `verified_email` | ⬜ omitted | Near-zero discrimination; trivially set by honeypots |
| 22 | `verified_phone` | ⬜ omitted | Same as above |
| 23 | `linkedin_connected` | ⬜ omitted | Gameable social signal; weak for role-fit inference |

## Rationale for omissions

**Recruiter-attention signals (#5, #17)** — `profile_views_received_30d` and `search_appearance_30d`
are circular: a candidate ranked highly gets more recruiter attention, which would push them higher
next cycle. Omitting them prevents feedback-loop bias in a single-shot ranking.

**`avg_response_time_hours` (#8)** — deliberately redundant with `recruiter_response_rate` (#7),
which already carries 0.28 of the behavioral weight. Rate captures *whether* a candidate responds;
time captures *how fast*. In a single-shot ranking the marginal information is low enough that adding
a second correlated signal would dilute the cleaner rate signal without meaningfully changing ordering.

**`endorsements_received` (#11)** — the aggregate total is omitted, but the per-skill `endorsements`
field (`skills[].endorsements`) *is* used as a keyword-stuffer trust gate: a skill claimed at
advanced/expert with zero duration_months and zero endorsements is discounted. The aggregate
would double-count a signal already applied at the right granularity.

**`expected_salary_range_inr_lpa` (#13)** — the JD contains no explicit salary band, so there is no
target range to score fit against. Scoring this would require inventing a preference not stated in
the spec.

**Verification booleans (#21, #22)** — `verified_email` and `verified_phone` have near-zero
discriminative power across the pool (penetration is very high) and are trivially fabricated for
honeypots.

**Social/network signals (#10, #23)** — `connection_count` and `linkedin_connected` measure platform
engagement and social reach, not ranking/search/ML competence. Both are gameable and weakly
correlated with role fit for a hands-on engineering position.

## Intentional asymmetry in −1 sentinel handling

Two signals use −1 as "no data":

- **`github_activity_score`** −1 → **0.20** (mild negative). For this role (production AI/search
  engineer), no linked GitHub is a weak negative signal — the JD explicitly mentions external
  validation (papers, talks, OSS). A score of 0.20 reflects genuine (if mild) concern, not a neutral
  assumption.
- **`offer_acceptance_rate`** −1 → **0.50** (neutral). No prior offer history is genuinely
  uninformative — there are many legitimate reasons a strong candidate has never received or
  considered an offer. Treating it as neutral is the correct default.

The asymmetry is intentional, not an oversight.

## `saved_by_recruiters_30d` — consistency note

`saved_by_recruiters_30d` (#18) is kept at 0.08 behavioral weight while `profile_views_received_30d`
(#5) and `search_appearance_30d` (#17) are dropped on circularity grounds. The distinction: a *save*
is a deliberate recruiter intent action, meaningfully stronger than a passive view or a search
appearance. It represents an independent quality signal rather than a side-effect of prior ranking.
The weight is capped (score = min(count/15, 1.0)) to prevent outliers from dominating.

## Doc fidelity note

`redrob_signals_doc.docx` contains intro prose and the 23-row table only — no construction or
correlation guidance follows the table. The dossier §6 transcription matches the source exactly,
including the two −1 sentinels for `github_activity_score` and `offer_acceptance_rate`.
