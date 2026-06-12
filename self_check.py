#!/usr/bin/env python3
"""
Pre-submission self-check — run before EVERY upload.

    python self_check.py <submission.csv> <candidates.json|jsonl>

Layers on top of the official validate_submission.py (which only checks
format) the things the organizers check later and the validator does not:

  1. Official format validation (vendored logic: header, 100 rows, rank
     bijection, score monotonic, candidate_id tie-break ascending).
  2. Every candidate_id actually exists in the pool (typos = Stage-1 reject;
     the official validator explicitly does NOT check this).
  3. Honeypot rate inside the top 100 per our own detector
     (>10% in the top 100 = Stage-3 disqualification; warn at >5%).
  4. Reasoning sanity: non-empty, not all identical.

Exit code 0 = safe to submit, 1 = do not submit.
"""

import csv
import re
import sys
from pathlib import Path

import rank

CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")
HONEYPOT_FAIL_RATE = 0.10
HONEYPOT_WARN_RATE = 0.05


def check_format(rows) -> list:
    errors = []
    if len(rows) != 100:
        errors.append(f"expected exactly 100 data rows, found {len(rows)}")

    seen_ids, seen_ranks = set(), set()
    parsed = []
    for i, row in enumerate(rows, 2):
        cid = row.get("candidate_id", "").strip()
        if not CANDIDATE_ID_PATTERN.match(cid):
            errors.append(f"row {i}: candidate_id {cid!r} is not CAND_XXXXXXX")
        if cid in seen_ids:
            errors.append(f"row {i}: duplicate candidate_id {cid}")
        seen_ids.add(cid)
        try:
            rnk = int(row["rank"])
            if str(rnk) != row["rank"].strip():
                raise ValueError
        except (ValueError, KeyError):
            errors.append(f"row {i}: rank {row.get('rank')!r} is not a strict integer")
            continue
        if rnk in seen_ranks:
            errors.append(f"row {i}: duplicate rank {rnk}")
        seen_ranks.add(rnk)
        try:
            score = float(row["score"])
        except (ValueError, KeyError):
            errors.append(f"row {i}: score {row.get('score')!r} is not a float")
            continue
        parsed.append((rnk, score, cid))

    missing = set(range(1, 101)) - seen_ranks
    if missing:
        errors.append(f"missing ranks: {sorted(missing)[:10]}…")

    parsed.sort()
    for (r1, s1, c1), (r2, s2, c2) in zip(parsed, parsed[1:]):
        if s1 < s2:
            errors.append(f"score increases from rank {r1} ({s1}) to {r2} ({s2})")
        if s1 == s2 and c1 > c2:
            errors.append(
                f"tie at ranks {r1}/{r2} not broken by candidate_id ascending "
                f"({c1} > {c2})"
            )
    return errors


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    csv_path, pool_path = sys.argv[1], sys.argv[2]

    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["candidate_id", "rank", "score", "reasoning"]:
            sys.exit(f"FAIL: header is {reader.fieldnames}, must be "
                     "candidate_id,rank,score,reasoning")
        rows = [r for r in reader if any((v or "").strip() for v in r.values())]

    failures = check_format(rows)

    print(f"Loading pool: {pool_path}")
    pool = rank.load_candidates(pool_path)
    by_id = {c.get("candidate_id"): c for c in pool}
    reference_date = rank.derive_reference_date(pool)

    # 2. Pool membership.
    missing_ids = [r["candidate_id"] for r in rows if r["candidate_id"] not in by_id]
    if missing_ids:
        failures.append(f"{len(missing_ids)} id(s) not in pool: {missing_ids[:5]}…")

    # 3. Honeypot rate in the submitted top 100 (our detector as proxy).
    hp = [
        r["candidate_id"] for r in rows
        if r["candidate_id"] in by_id
        and rank.detect_honeypot(by_id[r["candidate_id"]], reference_date)[0]
    ]
    hp_rate = len(hp) / max(len(rows), 1)
    if hp_rate > HONEYPOT_FAIL_RATE:
        failures.append(
            f"honeypot rate {hp_rate:.0%} exceeds the 10% DQ threshold: {hp[:10]}"
        )
    elif hp_rate > HONEYPOT_WARN_RATE:
        print(f"WARN: honeypot rate {hp_rate:.0%} (>{HONEYPOT_WARN_RATE:.0%}); ids: {hp}")
    else:
        print(f"Honeypot rate in top 100: {hp_rate:.0%} ({len(hp)} flagged)")

    # 4. Reasoning sanity.
    reasonings = [r.get("reasoning", "").strip() for r in rows]
    empty = sum(1 for t in reasonings if not t)
    if empty:
        print(f"WARN: {empty} empty reasoning row(s) — strongly discouraged")
    if len(rows) > 1 and len(set(reasonings)) == 1:
        failures.append("all reasoning strings identical (Stage-4 penalty)")
    else:
        print(f"Reasoning variety: {len(set(reasonings))}/{len(reasonings)} unique")

    if failures:
        print(f"\nFAIL — do not submit ({len(failures)} issue(s)):")
        for e in failures:
            print(f"  - {e}")
        sys.exit(1)
    print(f"\nPASS — {Path(csv_path).name} is safe to submit.")


if __name__ == "__main__":
    main()
