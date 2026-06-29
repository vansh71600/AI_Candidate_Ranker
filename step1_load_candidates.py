import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

CANDIDATES_FILE = "candidates.jsonl"

RELEVANT_INDUSTRIES = {
    "technology", "software", "saas", "ai", "machine learning",
    "data", "internet", "e-commerce", "fintech", "edtech",
    "product", "startup", "analytics"
}

PROFICIENCY_WEIGHT = {"expert": 3, "advanced": 2, "intermediate": 1, "beginner": 0}

TIER_RANK = {"tier_1": 4, "tier_2": 3, "tier_3": 2, "tier_4": 1, "unknown": 0}

# FIX #2: Use a rolling 6-year window computed at call time, not at import time.
# The old code did `datetime.today().year - 6` once at module load — meaning a job
# starting 2020-01-01 would be included even if the code ran in Dec 2026 (nearly 7 yrs).
# We now pass `cutoff_date` into build_career_text_trimmed() so it is always fresh.
SIX_YEAR_WINDOW = timedelta(days=6 * 365)

AI_ML_SKILLS = {
    "machine learning", "ml", "nlp", "deep learning", "embeddings",
    "faiss", "vector", "retrieval", "ranking", "recommendation",
    "transformer", "pytorch", "tensorflow", "scikit-learn", "xgboost",
    "data science", "statistics", "neural network", "classification",
    "regression", "clustering", "bert", "llm"
    # "computer vision" intentionally excluded — handled by its own dedicated filter
}

IRRELEVANT_TITLES = {
    "backend engineer", "frontend engineer",
    "full stack", "data engineer", "devops", "android", "ios",
    "analytics engineer"
    # "computer vision" intentionally excluded — handled by its own dedicated filter
}

NLP_ML_SKILLS = {
    "nlp", "ranking", "retrieval", "embeddings", "recommendation"
}

# Dedicated CV filter: skills that qualify a CV engineer as a genuine NLP/retrieval hybrid
CV_NLP_QUALIFYING_SKILLS = {
    "nlp", "natural language processing", "ranking", "retrieval",
    "embeddings", "recommendation", "information retrieval",
    "semantic search", "text classification", "language model"
}


def days_since(date_str):
    if not date_str:
        return 9999
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return (datetime.today() - dt).days
    except ValueError:
        return 9999


def build_career_text_trimmed(career, cutoff_date=None):
    """
    FIX #2: Accept cutoff_date so the 6-year window is always computed from
    the actual run time, not from a stale module-level constant.
    """
    if cutoff_date is None:
        cutoff_date = datetime.today() - SIX_YEAR_WINDOW

    recent_jobs = []
    for j in career:
        start = j.get("start_date", "")
        if not start:
            continue
        try:
            start_dt = datetime.strptime(start[:10], "%Y-%m-%d")
        except Exception:
            continue
        if start_dt >= cutoff_date:
            recent_jobs.append(j)

    recent_jobs = sorted(recent_jobs, key=lambda x: x.get("start_date", ""), reverse=True)

    parts = []
    for j in recent_jobs:
        desc = (j.get("description") or "").strip()
        if desc:
            parts.append(f"{j.get('title', '')} at {j.get('company', '')} : {desc}")

    return " | ".join(parts)


def is_honeypot(career, skills_raw):
    # check 1 — expert proficiency in many skills but 0 months used
    expert_zero = sum(
        1 for s in skills_raw
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    )
    if expert_zero >= 5:
        return True

    # check 2 — duration at a job exceeds plausible company age
    for job in career:
        start = job.get("start_date", "")
        dur   = job.get("duration_months", 0)
        if not start or not dur:
            continue
        try:
            start_year = int(start[:4])
        except Exception:
            continue
        max_possible_months = (datetime.today().year - start_year) * 12
        if dur > max_possible_months + 12:
            return True

    return False


def flatten_profile(c):
    row = {}

    row["candidate_id"] = c.get("candidate_id", "")

    p = c.get("profile", {})
    row["current_title"]    = (p.get("current_title") or "").strip()
    row["current_company"]  = (p.get("current_company") or "").strip()
    row["current_industry"] = (p.get("current_industry") or "").strip()
    row["location"]         = (p.get("location") or "").strip()
    row["country"]          = (p.get("country") or "").strip()

    career = c.get("career_history", [])

    total_months = sum(j.get("duration_months", 0) for j in career)
    row["total_exp_years"] = round(total_months / 12, 1)

    relevant_months = sum(
        j.get("duration_months", 0) for j in career
        if any(kw in j.get("industry", "").lower() for kw in RELEVANT_INDUSTRIES)
    )
    row["relevant_exp_years"] = round(relevant_months / 12, 1)

    # FIX #2: pass cutoff computed fresh at flatten time
    cutoff_date = datetime.today() - SIX_YEAR_WINDOW
    row["career_text"] = build_career_text_trimmed(career, cutoff_date)

    row["career_company_sizes"] = [j.get("company_size", "") for j in career]
    row["career_industries"]    = [j.get("industry", "") for j in career]
    row["career_companies"]     = [j.get("company", "") for j in career]
    row["career_titles"]        = [j.get("title", "") for j in career]

    skills_raw = c.get("skills", [])

    row["skill_names"] = [s.get("name", "").lower() for s in skills_raw]

    # skills_text was a dead field (FIX #7): now used in step2 semantic embedding
    # as supplementary candidate text. Kept but renamed for clarity.
    skill_parts = []
    for s in skills_raw:
        name   = s.get("name", "")
        weight = PROFICIENCY_WEIGHT.get(s.get("proficiency", "beginner"), 0)
        if name and weight > 0:
            skill_parts.extend([name] * weight)
    row["skills_text"] = ", ".join(skill_parts)

    row["skills_raw"] = [
        {
            "name":            s.get("name", "").lower(),
            "proficiency":     s.get("proficiency", "beginner"),
            "endorsements":    s.get("endorsements", 0),
            "duration_months": s.get("duration_months", 0),
        }
        for s in skills_raw
    ]

    edu_list = c.get("education", [])
    row["best_edu_tier"] = max(
        (TIER_RANK.get(e.get("tier", "unknown"), 0) for e in edu_list),
        default=0
    )

    sig = c.get("redrob_signals", {})

    row["open_to_work"]              = sig.get("open_to_work_flag", False)
    row["days_since_active"]         = days_since(sig.get("last_active_date"))
    row["recruiter_response_rate"]   = sig.get("recruiter_response_rate", 0.0)
    row["avg_response_time_hrs"]     = sig.get("avg_response_time_hours", 999)
    row["notice_period_days"]        = sig.get("notice_period_days", 90)
    row["willing_to_relocate"]       = sig.get("willing_to_relocate", False)
    row["preferred_work_mode"]       = sig.get("preferred_work_mode", "")
    row["github_activity_score"]     = sig.get("github_activity_score", -1)
    row["profile_completeness"]      = sig.get("profile_completeness_score", 0)
    row["interview_completion_rate"] = sig.get("interview_completion_rate", 0.0)
    row["offer_acceptance_rate"]     = sig.get("offer_acceptance_rate", -1)
    row["verified_email"]            = sig.get("verified_email", False)
    row["linkedin_connected"]        = sig.get("linkedin_connected", False)
    row["saved_by_recruiters_30d"]   = sig.get("saved_by_recruiters_30d", 0)

    sal = sig.get("expected_salary_range_inr_lpa", {})
    row["salary_min_lpa"] = sal.get("min", 0)
    row["salary_max_lpa"] = sal.get("max", 0)

    row["skill_assessment_scores"] = sig.get("skill_assessment_scores", {})

    row["_is_honeypot"] = is_honeypot(career, skills_raw)

    return row


# Tools that are Python-only or overwhelmingly Python-first.
# A candidate listing any of these almost certainly codes in Python
# even if they haven't explicitly listed it as a skill.
PYTHON_IMPLIES = {
    "pytorch", "tensorflow", "scikit-learn", "keras", "numpy", "pandas",
    "faiss", "hugging face", "transformers", "langchain", "spacy",
    "nltk", "mlflow", "bentoml", "xgboost", "lightgbm", "catboost",
    "pyspark", "fastapi", "flask", "django", "jupyter"
}


def has_python(skill_names):
    # Explicit listing
    if any("python" in s for s in skill_names):
        return True
    # Inferred from Python-only tools — these don't exist outside Python
    return any(any(impl in s for impl in PYTHON_IMPLIES) for s in skill_names)


def has_ml_skill(skill_names):
    return any(any(ml in s for ml in AI_ML_SKILLS) for s in skill_names)


def is_irrelevant_title(current_title, career_titles, skills_raw):
    """
    Filters backend/frontend/devops/etc. titles.
    Escape hatch: one NLP/retrieval skill at intermediate+ saves them (genuine crossover).
    Computer vision is NOT checked here — it has its own dedicated filter below.
    """
    all_titles = career_titles + [current_title]
    titles_text = " ".join(all_titles).lower()
    has_irrelevant = any(t in titles_text for t in IRRELEVANT_TITLES)
    if not has_irrelevant:
        return False
    has_strong_nlp = any(
        any(ml in s["name"] for ml in NLP_ML_SKILLS) and
        s.get("proficiency") in {"intermediate", "advanced", "expert"}
        for s in skills_raw
    )
    return not has_strong_nlp


def is_cv_without_nlp(current_title, career_titles, skills_raw):
    """
    Dedicated Computer Vision filter — completely separate from is_irrelevant_title.
    A CV engineer passes through ONLY if they have 2+ NLP/retrieval skills
    at advanced or expert level, making them a genuine NLP-CV hybrid.
    Anyone with 0 or 1 such skills is a CV specialist, not a fit for this JD.
    """
    all_titles = career_titles + [current_title]
    titles_text = " ".join(all_titles).lower()
    if "computer vision" not in titles_text:
        return False  # not a CV candidate, skip this filter entirely

    qualifying_count = sum(
        1 for s in skills_raw
        if any(skill in s["name"] for skill in CV_NLP_QUALIFYING_SKILLS)
        and s.get("proficiency") in {"advanced", "expert"}
    )
    return qualifying_count < 2  # True = should be filtered out


def load_candidates(filepath):
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {filepath}")

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(flatten_profile(json.loads(line)))
            except Exception:
                continue

    df = pd.DataFrame(rows)
    print(f"[step1] Loaded: {len(df)} candidates total")  # FIX #9

    mask_exp        = (df["total_exp_years"] >= 3) & (df["relevant_exp_years"] >= 2.0)
    mask_active     = df["days_since_active"] < 365
    mask_python     = df["skill_names"].apply(has_python)
    mask_ml         = df["skill_names"].apply(has_ml_skill)
    mask_title      = ~df.apply(
        lambda r: is_irrelevant_title(r["current_title"], r["career_titles"], r["skills_raw"]),
        axis=1
    )
    mask_cv         = ~df.apply(
        lambda r: is_cv_without_nlp(r["current_title"], r["career_titles"], r["skills_raw"]),
        axis=1
    )
    mask_honeypot   = ~df["_is_honeypot"]

    print(f"[step1] After exp filter:       {mask_exp.sum()}")
    print(f"[step1] After active filter:    {(mask_exp & mask_active).sum()}")
    print(f"[step1] After python filter:    {(mask_exp & mask_active & mask_python).sum()}")
    print(f"[step1] After ml filter:        {(mask_exp & mask_active & mask_python & mask_ml).sum()}")
    print(f"[step1] After title filter:     {(mask_exp & mask_active & mask_python & mask_ml & mask_title).sum()}")
    print(f"[step1] After CV filter:        {(mask_exp & mask_active & mask_python & mask_ml & mask_title & mask_cv).sum()}")
    print(f"[step1] After honeypot filter:  {(mask_exp & mask_active & mask_python & mask_ml & mask_title & mask_cv & mask_honeypot).sum()}")

    df_filtered = df[
        mask_exp & mask_active & mask_python & mask_ml & mask_title & mask_cv & mask_honeypot
    ].copy().reset_index(drop=True)

    # FIX #1: Warn about candidates with empty career_text — they will get
    # semantic_score ≈ 0 and silently fail the MIN_SEMANTIC_SCORE gate.
    empty_text = df_filtered["career_text"].str.strip().eq("")
    if empty_text.any():
        print(
            f"[step1] WARNING: {empty_text.sum()} candidates have empty career_text "
            "(no recent job descriptions). They will likely fail the semantic gate."
        )

    print(f"[step1] Final filtered pool: {len(df_filtered)} candidates")
    return df_filtered


if __name__ == "__main__":
    df = load_candidates(CANDIDATES_FILE)
