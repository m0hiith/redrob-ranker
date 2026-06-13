#!/usr/bin/env python3
"""
Hosted sandbox demo (submission-spec Section 10.5).

Deploy to Streamlit Cloud or HuggingFace Spaces. Accepts a small candidate
sample (≤100 records, .json or .jsonl), runs the full ranking pipeline
end-to-end on CPU, and offers the ranked CSV for download.

Local run:
    pip install -r requirements.txt streamlit
    python download_model.py
    streamlit run app.py
"""

import csv
import io
import json
import tempfile
from pathlib import Path

import streamlit as st

import rank

MAX_SANDBOX_CANDIDATES = 100

st.set_page_config(page_title="Redrob Candidate Ranker", page_icon="🎯")
st.title("Redrob Candidate Ranker — sandbox")
st.caption(
    "Two-stage ranking: full-pool feature screen → semantic re-rank "
    "(bge-small-en-v1.5, CPU, offline). Upload up to "
    f"{MAX_SANDBOX_CANDIDATES} candidates as .json or .jsonl."
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


uploaded = st.file_uploader("Candidate sample", type=["json", "jsonl"])

if uploaded is not None:
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
            "ranks with semantic embeddings (see the repo / reproduce command); "
            "this hosted demo shows the same pipeline with a TF-IDF semantic stage."
        )

    with st.spinner(f"Ranking {len(candidates)} candidates…"):
        results = rank.rank_candidates(
            candidates, rank.JOB_DESCRIPTION, embedder
        )

    st.success(f"Ranked {len(results)} candidates.")
    st.dataframe(
        [
            {"rank": r["rank"], "candidate_id": r["candidate_id"],
             "score": r["score"], "reasoning": r["reasoning"]}
            for r in results
        ],
        width="stretch",
    )

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    writer.writeheader()
    writer.writerows(
        {k: r[k] for k in ("candidate_id", "rank", "score", "reasoning")}
        for r in results
    )
    st.download_button(
        "Download ranked CSV", buf.getvalue(),
        file_name="submission.csv", mime="text/csv",
    )
else:
    st.info("Upload a candidate sample to run the ranker. "
            "Try `sample_candidates.json` from the challenge bundle.")
