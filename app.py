import streamlit as st
import pandas as pd
import json
import tempfile
import os
import traceback

from step1_load_candidates import load_candidates
from step2_embed_and_score import compute_semantic_scores
from step3_numeric_scoring import compute_numeric_scores, compute_final_score
from main import build_reasoning, MIN_SKILL_SCORE, MIN_SEMANTIC_SCORE, TOP_N


st.set_page_config(page_title="AI Candidate Ranker — Sandbox", layout="wide")

st.title("AI Candidate Ranker — Sandbox")
st.caption(
    "Upload a small candidate sample (.jsonl) to run the full ranking pipeline "
    "end-to-end and inspect the output. This is a reproducibility sandbox for "
    "the Redrob AI Hackathon submission — not the full 100K-candidate run."
)

with st.expander("How this works", expanded=False):
    st.markdown(
        """
        1. Upload a `.jsonl` file where each line is one candidate record
           matching `candidate_schema.json`.
        2. The pipeline runs the same three steps as `main.py`:
           filter → semantic embedding → numeric scoring → final ranking.
        3. Results are shown below and available as a CSV download.

        Keep uploads small (under ~200 candidates) — this sandbox runs on
        shared CPU and is meant for spot-checking, not the full dataset.
        """
    )

uploaded_file = st.file_uploader("Candidate file (.jsonl)", type=["jsonl", "json"])

run_button = st.button("Run ranking", type="primary", disabled=uploaded_file is None)

if run_button and uploaded_file is not None:
    raw_bytes = uploaded_file.read()

    # Write to a temp file so load_candidates() can read it the same way it
    # does in main.py — keeps the sandbox and CLI paths identical.
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".jsonl", delete=False
    ) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        with st.status("Running pipeline...", expanded=True) as status:
            st.write("Step 1 — loading and filtering candidates")
            df = load_candidates(tmp_path)
            st.write(f"→ {len(df)} candidates passed hard filters")

            if len(df) == 0:
                status.update(label="No candidates passed the filters", state="error")
                st.warning(
                    "No candidates survived the hard filters (experience, Python, "
                    "ML skill, active-in-365-days, title, honeypot). Try a larger "
                    "or more varied sample."
                )
                st.stop()

            st.write("Step 2 — computing semantic embeddings")
            df, embeddings = compute_semantic_scores(df)
            st.write(f"→ semantic_score range: {df['semantic_score'].min():.4f} – {df['semantic_score'].max():.4f}")

            st.write("Step 3 — computing numeric scores")
            df = compute_numeric_scores(df)
            df = compute_final_score(df)

            df["score"] = df["final_score"].round(4)
            df = df.sort_values(
                ["score", "candidate_id"], ascending=[False, True]
            ).reset_index(drop=True)

            df_filtered = df[
                (df["skill_score"] > MIN_SKILL_SCORE)
                & (df["semantic_score"] > MIN_SEMANTIC_SCORE)
            ].copy()

            n_gated = len(df_filtered)
            st.write(f"→ {n_gated} candidates passed the skill/semantic score gates")

            if n_gated < TOP_N:
                st.info(
                    f"Fewer than {TOP_N} candidates passed the gates "
                    f"({n_gated} found). This is expected for a small sandbox "
                    "sample — the full 100K-candidate run produces a complete "
                    "top 100."
                )
                df_final = df_filtered.copy()
            else:
                df_final = df_filtered.head(TOP_N).copy()

            df_final = df_final.reset_index(drop=True)
            df_final["rank"] = df_final.index + 1
            # Use a list comprehension instead of .apply(axis=1) — with very
            # small DataFrames (e.g. 0 or 1 rows from a sandbox sample),
            # pandas can misinfer whether build_reasoning returns a scalar
            # or a Series and try to expand it into multiple columns,
            # raising "Cannot set a DataFrame with multiple columns to the
            # single column reasoning". A list comprehension has no such
            # ambiguity.
            if len(df_final) > 0:
                df_final["reasoning"] = [
                    build_reasoning(row) for _, row in df_final.iterrows()
                ]
            else:
                df_final["reasoning"] = pd.Series(dtype="object")

            result = df_final[["candidate_id", "rank", "score", "reasoning"]]
            status.update(label="Done", state="complete")

        st.subheader(f"Ranked output ({len(result)} candidates)")
        st.dataframe(result, use_container_width=True, hide_index=True)

        csv_bytes = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="builder.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.code(traceback.format_exc())
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

elif uploaded_file is None:
    st.info("Upload a .jsonl file to get started.")
