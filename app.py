#!/usr/bin/env python3
"""
Hosted sandbox demo (submission-spec Section 10.5).

Two tabs:
  • Full-pool Top 100 — the real ranking output over all 100K candidates
    (precomputed by `python rank.py` on the official pool), shown instantly.
  • Try your own sample — upload up to 100 records (.json/.jsonl) and run the
    identical pipeline end-to-end on CPU, offline, live.

Local run:
    pip install -r requirements.txt streamlit
    python download_model.py
    streamlit run app.py
"""

import csv
import io
import tempfile
from pathlib import Path

import streamlit as st

import rank

MAX_SANDBOX_CANDIDATES = 100
FULL_POOL_CSV = Path(__file__).parent / "full_pool_top100.csv"
RESULT_COLUMNS = ["candidate_id", "rank", "score", "reasoning"]

st.set_page_config(
    page_title="Redrob Candidate Ranker", page_icon="🎯", layout="wide"
)
st.title("Redrob Candidate Ranker — sandbox")
st.caption(
    "Two-stage ranking: full-pool feature screen → semantic re-rank "
    "(bge-small-en-v1.5, CPU, offline)."
)


@st.cache_resource(show_spinner="Loading embedding model (first boot only)…")
def get_embedder() -> rank.Embedder:
    # First boot on a fresh host: vendor the model (allowed pre-computation),
    # then load it in semantic mode. If torch / the model can't load on this
    # host, degrade to the lexical fallback so the demo still works (clearly
    # flagged in the UI) rather than crashing the whole app.
    try:
        import download_model
        download_model.ensure_model()
        return rank.Embedder()
    except Exception:
        return rank.Embedder(allow_lexical_fallback=True)


@st.cache_data
def load_full_pool_top100() -> list[dict]:
    if not FULL_POOL_CSV.exists():
        return []
    with FULL_POOL_CSV.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def render_results_table(rows: list[dict]) -> None:
    st.dataframe(
        [{c: r[c] for c in RESULT_COLUMNS} for r in rows],
        width="stretch",
        hide_index=True,
    )


full_pool_tab, sample_tab = st.tabs(
    ["🏆 Full-pool Top 100", "🧪 Try your own sample"]
)

with full_pool_tab:
    rows = load_full_pool_top100()
    if not rows:
        st.info(
            "Full-pool results aren't bundled with this deploy. Run "
            "`python rank.py --candidates candidates.json --out submission.csv` "
            "to generate them."
        )
    else:
        st.markdown(
            f"**Top {len(rows)} of the official 100,000-candidate pool** — the "
            "actual `submission.csv`, produced by `python rank.py` in ~3 min on "
            "an 8-core CPU laptop (deterministic, fully offline). This is what "
            "the submission is graded on; the **Try your own sample** tab runs "
            "the identical pipeline live on a small upload."
        )
        render_results_table(rows)
        st.download_button(
            "Download full submission.csv",
            FULL_POOL_CSV.read_text(encoding="utf-8"),
            file_name="submission.csv",
            mime="text/csv",
        )

with sample_tab:
    st.caption(
        f"Upload up to {MAX_SANDBOX_CANDIDATES} candidates as .json or .jsonl."
    )
    uploaded = st.file_uploader("Candidate sample", type=["json", "jsonl"])

    if uploaded is None:
        st.info(
            "Upload a candidate sample to run the ranker. "
            "Try `sample_candidates.json` from the challenge bundle."
        )
    else:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=Path(uploaded.name).suffix, delete=False
        ) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        candidates = rank.load_candidates(tmp_path)
        if not candidates:
            st.error("No valid candidate records found in the upload.")
            st.stop()
        if len(candidates) > MAX_SANDBOX_CANDIDATES:
            st.warning(
                f"Sandbox caps at {MAX_SANDBOX_CANDIDATES} candidates; "
                f"ranking the first {MAX_SANDBOX_CANDIDATES} of {len(candidates)}."
            )
            candidates = candidates[:MAX_SANDBOX_CANDIDATES]

        embedder = get_embedder()
        if embedder.mode == "semantic":
            st.caption(f"Semantic stage: {embedder._model_id} · CPU · offline.")
        else:
            st.warning(
                "Running in lightweight **lexical** mode on this host — the "
                "bge-small-en-v1.5 model isn't loaded here. The full submission "
                "ranks with semantic embeddings (see the repo / reproduce "
                "command); this hosted demo shows the same pipeline with a "
                "TF-IDF semantic stage."
            )

        with st.spinner(f"Ranking {len(candidates)} candidates…"):
            results = rank.rank_candidates(
                candidates, rank.JOB_DESCRIPTION, embedder
            )

        st.success(f"Ranked {len(results)} candidates.")
        render_results_table(results)

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows({k: r[k] for k in RESULT_COLUMNS} for r in results)
        st.download_button(
            "Download ranked CSV", buf.getvalue(),
            file_name="ranked_sample.csv", mime="text/csv",
        )
