#!/usr/bin/env python3
"""
Blend-weight sensitivity analysis.

The final score is a weighted blend of five subscores. A good ranking should be
*robust*: the top picks shouldn't hinge on the exact weights. This script holds
the prescreen finalist set and their subscores fixed (reusing the committed
similarity cache, so no re-encoding), then re-blends under perturbed weight
vectors and reports how much the top-10 / top-50 move.

    python weight_sensitivity.py --candidates <pool.json|jsonl>

Output is a robustness table; it changes nothing and writes nothing.
"""

import argparse
import hashlib
from datetime import date
from pathlib import Path

import numpy as np

import rank


def _blend(sub: dict, w: dict) -> float:
    if sub["honeypot"]:
        return 0.0
    return (
        sub["semantic"] * w["semantic"]
        + sub["career"] * w["career"]
        + sub["behavioral"] * w["behavioral"]
        + sub["location"] * w["location"]
        + sub["availability"] * w["availability"]
    )


def _ranked_ids(subs: dict, w: dict) -> list:
    rows = [
        (-_blend(s, w), s["honeypot"], cid)  # honeypots sort last via score 0 + flag
        for cid, s in subs.items()
    ]
    rows.sort(key=lambda r: (r[1], r[0], r[2]))
    return [cid for _, _, cid in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--finalists", type=int, default=rank.PRESCREEN_FINALISTS)
    args = ap.parse_args()

    candidates = rank.load_candidates(args.candidates)
    reference_date = rank.derive_reference_date(candidates)
    print(f"Pool: {len(candidates):,} | reference_date: {reference_date}")

    finalists = rank.prescreen(candidates, rank.JOB_DESCRIPTION, reference_date, args.finalists)
    finalist_ids = sorted(c.get("candidate_id", "") for c in finalists)
    key = hashlib.sha256("|".join(finalist_ids).encode()).hexdigest()[:20]
    cache_file = rank.SIMILARITY_CACHE_DIR / f"sem_{key}.npy"
    if not cache_file.exists():
        raise SystemExit(
            f"No similarity cache for this finalist set ({cache_file}).\n"
            "Run rank.py on this pool once first so the cache is populated."
        )
    semantic = np.load(str(cache_file)).tolist()

    subs = {}
    for c, sem in zip(finalists, semantic):
        _, career, _, behavioral, location, availability = rank._score_components(
            c, rank.JOB_DESCRIPTION, reference_date
        )
        is_hp, _ = rank.detect_honeypot(c, reference_date)
        subs[c["candidate_id"]] = {
            "semantic": sem, "career": career, "behavioral": behavioral,
            "location": location, "availability": availability, "honeypot": is_hp,
        }

    base = dict(rank.WEIGHTS)
    base_order = _ranked_ids(subs, base)
    base_top10, base_top50 = set(base_order[:10]), set(base_order[:50])

    scenarios = {
        "base":            base,
        "more-semantic":   {"semantic": 0.50, "career": 0.22, "behavioral": 0.13, "location": 0.10, "availability": 0.05},
        "more-career":     {"semantic": 0.30, "career": 0.40, "behavioral": 0.15, "location": 0.10, "availability": 0.05},
        "more-behavioral": {"semantic": 0.35, "career": 0.25, "behavioral": 0.25, "location": 0.10, "availability": 0.05},
        "availability-up": {"semantic": 0.38, "career": 0.28, "behavioral": 0.14, "location": 0.10, "availability": 0.10},
        "equal-ish":       {"semantic": 0.25, "career": 0.25, "behavioral": 0.20, "location": 0.15, "availability": 0.15},
    }

    print(f"\n{'scenario':<16} {'top10∩base':>11} {'top50∩base':>11} {'#1 same':>8}")
    for name, w in scenarios.items():
        assert abs(sum(w.values()) - 1.0) < 1e-9, f"{name} weights must sum to 1"
        order = _ranked_ids(subs, w)
        t10 = len(set(order[:10]) & base_top10)
        t50 = len(set(order[:50]) & base_top50)
        same1 = "yes" if order[0] == base_order[0] else f"no ({order[0]})"
        print(f"{name:<16} {t10:>9}/10 {t50:>9}/50 {same1:>8}")


if __name__ == "__main__":
    main()
