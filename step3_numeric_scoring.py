import numpy as np
import pandas as pd

JD_REQUIRED_SKILLS = {
    "python", "machine learning", "nlp", "information retrieval",
    "embeddings", "faiss", "vector search", "semantic search",
    "ranking", "recommendation", "deep learning", "pytorch",
    "tensorflow", "scikit-learn", "rest api", "system design",
    "hybrid search", "bm25", "transformer"
}

HIGH_VALUE_INDUSTRIES = {
    "AI/ML", "AI Services", "SaaS", "Fintech", "E-commerce",
    "EdTech", "HealthTech", "HealthTech AI", "Conversational AI",
    "Voice AI", "Food Delivery", "AdTech", "Internet", "Gaming"
}

LOW_VALUE_INDUSTRIES = {
    "IT Services", "Consulting", "Manufacturing",
    "Paper Products", "Conglomerate"
}

PROFICIENCY_SCORE = {"expert": 1.0, "advanced": 0.75, "intermediate": 0.5, "beginner": 0.25}

# FIX #6: Added edu_score (5%) and redistributed weights.
# Previous: semantic 35 + skill 25 + behavioral 20 + exp 10 + company 5 + assessment 5 = 100
# Updated:  semantic 35 + skill 25 + behavioral 18 + exp 10 + company 4 + assessment 4 + edu 4 = 100
W_SEMANTIC    = 0.35
W_SKILL       = 0.25
W_BEHAVIORAL  = 0.18
W_EXPERIENCE  = 0.10
W_COMPANY     = 0.04
W_ASSESSMENT  = 0.04
W_EDUCATION   = 0.04


def skill_overlap_score(skill_names, skills_raw):
    if not skill_names:
        return 0.0

    matched = 0.0
    for s in skills_raw:
        name = s.get("name", "").lower()
        if any(jd_skill in name or name in jd_skill for jd_skill in JD_REQUIRED_SKILLS):
            proficiency_boost = PROFICIENCY_SCORE.get(s.get("proficiency", "beginner"), 0.25)
            matched += proficiency_boost

    return min(matched / len(JD_REQUIRED_SKILLS), 1.0)


def experience_score(relevant_exp_years):
    if relevant_exp_years < 3:   return 0.1
    if relevant_exp_years < 5:   return 0.6
    if relevant_exp_years <= 9:  return 1.0
    return 0.85


def education_score(best_edu_tier):
    """
    FIX #6: best_edu_tier was extracted and stored but never used in scoring.
    TIER_RANK maps: tier_1→4, tier_2→3, tier_3→2, tier_4→1, unknown→0.
    This contributes 4% of the final score, rewarding tier-1/2 institutions.
    """
    mapping = {4: 1.0, 3: 0.75, 2: 0.5, 1: 0.25, 0: 0.1}
    return mapping.get(int(best_edu_tier), 0.1)


def behavioral_score(row):
    """
    FIX #5: open_to_work penalty changed from 0.3× to 0.6×.
    A 0.3× multiplier was far too aggressive — it tanked a strong candidate's
    behavioral score by 70% before any other signal was considered.

    FIX #8: Integrated previously unused signals:
      - profile_completeness: higher completeness → slight boost
      - verified_email: unverified → small penalty
      - offer_acceptance_rate: historically declining all offers → small penalty
      - saved_by_recruiters_30d: market validation signal → small boost
    """
    score = 1.0

    # FIX #5: softened from 0.3 to 0.6
    if not row["open_to_work"]:
        score *= 0.6

    if row["days_since_active"] > 180:
        score *= 0.5
    elif row["days_since_active"] > 90:
        score *= 0.75

    rrr = row["recruiter_response_rate"]
    score *= (0.4 + 0.6 * rrr)

    notice = row["notice_period_days"]
    if notice > 60:
        score *= 0.85

    github = row["github_activity_score"]
    if github > 60:
        score = min(score * 1.15, 1.0)
    elif github == -1:
        score *= 0.95

    icr = row["interview_completion_rate"]
    score *= (0.5 + 0.5 * icr)

    # FIX #8a: profile completeness — reward high completeness
    completeness = row.get("profile_completeness", 0)
    if completeness >= 80:
        score = min(score * 1.05, 1.0)
    elif completeness < 40:
        score *= 0.95

    # FIX #8b: verified_email as a trust signal
    if not row.get("verified_email", False):
        score *= 0.97

    # FIX #8c: offer_acceptance_rate — chronic offer-decliners are risky
    oar = row.get("offer_acceptance_rate", -1)
    if oar != -1 and oar < 0.2:
        score *= 0.90

    # FIX #8d: saved_by_recruiters_30d — market validation
    saved = row.get("saved_by_recruiters_30d", 0)
    if saved >= 5:
        score = min(score * 1.05, 1.0)

    return min(score, 1.0)


def company_type_score(career_industries):
    if not career_industries:
        return 0.5

    score = 0.0
    for industry in career_industries:
        if industry in HIGH_VALUE_INDUSTRIES:
            score += 1.0
        elif industry in LOW_VALUE_INDUSTRIES:
            score += 0.1
        else:
            score += 0.5

    return min(score / len(career_industries), 1.0)


def assessment_score(skill_assessment_scores):
    if not skill_assessment_scores:
        return 0.5

    RELEVANT_ASSESSMENT_SKILLS = {
        "python", "machine learning", "nlp", "deep learning",
        "data science", "algorithms", "system design"
    }

    relevant_scores = [
        v for k, v in skill_assessment_scores.items()
        if any(rs in k.lower() for rs in RELEVANT_ASSESSMENT_SKILLS)
    ]

    if not relevant_scores:
        return 0.5

    return sum(relevant_scores) / (len(relevant_scores) * 100)


def compute_numeric_scores(df):
    df["skill_score"]      = df.apply(lambda r: skill_overlap_score(r["skill_names"], r["skills_raw"]), axis=1)
    df["exp_score"]        = df["relevant_exp_years"].apply(experience_score)
    df["edu_score"]        = df["best_edu_tier"].apply(education_score)  # FIX #6
    df["behavioral_score"] = df.apply(behavioral_score, axis=1)
    df["company_score"]    = df["career_industries"].apply(company_type_score)
    df["assessment_score"] = df["skill_assessment_scores"].apply(assessment_score)

    return df


def compute_final_score(df):
    df["final_score"] = (
        W_SEMANTIC   * df["semantic_score"]   +
        W_SKILL      * df["skill_score"]      +
        W_BEHAVIORAL * df["behavioral_score"] +
        W_EXPERIENCE * df["exp_score"]        +
        W_COMPANY    * df["company_score"]    +
        W_ASSESSMENT * df["assessment_score"] +
        W_EDUCATION  * df["edu_score"]         # FIX #6
    )

    return df
