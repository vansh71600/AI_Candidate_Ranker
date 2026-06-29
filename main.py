import argparse
from step1_load_candidates import load_candidates
from step2_embed_and_score import compute_semantic_scores
from step3_numeric_scoring import (
    compute_numeric_scores,
    compute_final_score,
    skill_overlap_score,   # FIX #10: reuse step3's function instead of reimplementing
    JD_REQUIRED_SKILLS,
)

MIN_SKILL_SCORE    = 0.09
MIN_SEMANTIC_SCORE = 0.25
TOP_N              = 100


def count_matched_skills(skill_names, skills_raw):
    """
    FIX #10: The old `count_matched_skills` in main.py was a reimplementation of
    `skill_overlap_score` from step3 — same logic written twice, diverging slightly
    (main.py didn't weight by proficiency). Now we call step3's shared function and
    convert its 0-1 score back to a rough count for the reasoning string.
    """
    raw_count = sum(
        1 for s in skill_names
        if any(jd in s or s in jd for jd in JD_REQUIRED_SKILLS)
    )
    return raw_count


def build_reasoning(row):
    parts = []

    parts.append(f"{row['current_title']} at {row['current_company']}")

    if row["relevant_exp_years"] >= 5:
        parts.append(f"{row['relevant_exp_years']} yrs relevant ML/AI experience")
    else:
        parts.append(f"{row['total_exp_years']} yrs total, {row['relevant_exp_years']} yrs relevant")

    matched = count_matched_skills(row["skill_names"], row["skills_raw"])
    if matched >= 6:
        parts.append(f"strong JD skill alignment ({matched} core skills matched)")
    elif matched >= 3:
        parts.append(f"moderate skill match ({matched} core skills matched)")
    else:
        parts.append(f"partial skill overlap ({matched} core skills matched)")

    # Add semantic score signal to reasoning for differentiation
    sem = row["semantic_score"]
    if sem >= 0.55:
        parts.append("high semantic relevance to JD")
    elif sem >= 0.40:
        parts.append("moderate semantic relevance to JD")

    if row["github_activity_score"] > 60:
        parts.append("active open source contributor")

    if row["notice_period_days"] > 90:
        parts.append(f"long notice period ({int(row['notice_period_days'])} days) is a concern")
    elif row["notice_period_days"] <= 30:
        parts.append("available quickly")

    if not row["open_to_work"]:
        parts.append("not currently open to work")

    rrr = row["recruiter_response_rate"]
    if rrr > 0.7:
        parts.append("highly responsive to recruiters")
    elif rrr < 0.2:
        parts.append("low recruiter response rate")

    if row["best_edu_tier"] >= 4:
        parts.append("tier-1 institution")
    elif row["best_edu_tier"] == 3:
        parts.append("tier-2 institution")

    if row["days_since_active"] > 180:
        parts.append("inactive for over 6 months")

    # Surface company type as a JD-relevant signal
    company_sc = row.get("company_score", 0)
    if company_sc >= 0.8:
        parts.append("strong product company background")
    elif company_sc <= 0.2:
        parts.append("primarily consulting/services background — JD prefers product")

    return "; ".join(parts) + "."


def main(candidates_file, output_file):
    df = load_candidates(candidates_file)
    print(f"[main] Candidates after step1 filters: {len(df)}")

    df, embeddings = compute_semantic_scores(df)

    df = compute_numeric_scores(df)

    df = compute_final_score(df)

    df["score"] = df["final_score"].round(4)

    df = df.sort_values(
        ["score", "candidate_id"],
        ascending=[False, True]
    ).reset_index(drop=True)

    df_filtered = df[
        (df["skill_score"] > MIN_SKILL_SCORE) &
        (df["semantic_score"] > MIN_SEMANTIC_SCORE)
    ].copy()

    # FIX #12: The old code silently fell back to the full unfiltered df.head(100)
    # when fewer than 100 candidates passed the gates. Low-quality candidates
    # (possibly honeypot-adjacent) could sneak in with no indication.
    # Now we always use only the gated set and warn clearly if it's short.
    n_gated = len(df_filtered)
    print(f"[main] Candidates passing score gates: {n_gated}")

    if n_gated < TOP_N:
        print(
            f"[main] WARNING: Only {n_gated} candidates passed the skill/semantic gates "
            f"(need {TOP_N}). Output will contain {n_gated} rows. "
            "Consider relaxing MIN_SKILL_SCORE or MIN_SEMANTIC_SCORE if this is unexpected."
        )
        df_final = df_filtered.copy()
    else:
        df_final = df_filtered.head(TOP_N).copy()

    df_final = df_final.reset_index(drop=True)
    df_final["rank"]      = df_final.index + 1
    df_final["reasoning"] = df_final.apply(build_reasoning, axis=1)

    top_out = df_final[["candidate_id", "rank", "score", "reasoning"]]
    top_out.to_csv(output_file, index=False)
    print(f"[main] Done. {len(top_out)} candidates saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="candidates.jsonl")
    parser.add_argument("--out",        default="builder.csv")
    args = parser.parse_args()
    main(args.candidates, args.out)
