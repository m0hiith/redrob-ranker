#!/usr/bin/env python3
"""
Pre-computation step: download the embedding model for offline ranking.

The submission spec bans network access during the ranking step, but allows
documented pre-computation. Run this ONCE (with network) before ranking:

    python download_model.py

It saves BAAI/bge-small-en-v1.5 into ./models/bge-small-en-v1.5. After that,
rank.py loads the model strictly from disk and never touches the network.
"""

import argparse
import sys
from pathlib import Path

MODEL_ID = "BAAI/bge-small-en-v1.5"
DEFAULT_OUT = Path(__file__).parent / "models" / "bge-small-en-v1.5"


def main() -> None:
    ap = argparse.ArgumentParser(description="Download the ranking embedding model")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"Target directory (default: {DEFAULT_OUT})")
    args = ap.parse_args()

    out = Path(args.out)
    if (out / "config.json").exists():
        print(f"Model already present at {out} — nothing to do.")
        return

    print(f"Downloading {MODEL_ID} → {out}")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("sentence-transformers is not installed. Run: pip install -r requirements.txt")

    model = SentenceTransformer(MODEL_ID, device="cpu")
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print(f"Saved. Ranking can now run fully offline: python rank.py --model-dir {out}")


if __name__ == "__main__":
    main()
