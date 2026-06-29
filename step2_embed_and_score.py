import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# FIX #11: Do NOT instantiate the model at module level.
# The old code ran `model = SentenceTransformer(...)` at import time, which loaded
# the model into memory even when the module was only imported for testing or
# inspection. The model is now lazy-loaded inside compute_semantic_scores().
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print("[step2] Loading embedding model (paraphrase-MiniLM-L3-v2)...")
        _model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
        print("[step2] Model loaded.")
    return _model


JD_SKILLS = """
Python, embeddings based retrieval, vector databases, FAISS, semantic search,
dense retrieval, hybrid search, BM25, NLP, ranking systems, recommendation systems,
information retrieval, evaluation frameworks, NDCG, MRR, LLM re-ranking,
production ML systems, system design, REST APIs
"""

JD_RESPONSIBILITIES = """
own the intelligence layer for candidate discovery and ranking,
build and improve embedding based retrieval pipelines,
design hybrid search combining dense and sparse retrieval,
implement ranking and matching algorithms,
evaluate ranking quality using NDCG MRR and offline benchmarks,
ship end to end systems to real users at product companies,
work on talent intelligence platform, improve recruiter search experience
"""

JD_REQUIREMENTS = """
5 to 9 years experience in applied machine learning or information retrieval,
experience at product companies not consulting firms,
shipped production systems to real users,
strong Python engineering skills,
experience with vector databases and semantic search,
not pure research background, hands on builder
"""

JD = JD_SKILLS + JD_RESPONSIBILITIES + JD_REQUIREMENTS


def compute_semantic_scores(df):
    model = _get_model()

    # FIX #7: skills_text was a dead field — now appended to career_text for
    # embedding so proficiency-weighted skill terms contribute to semantic score.
    candidate_texts = (
        df["career_text"].fillna("") + " " + df["skills_text"].fillna("")
    ).str.strip().tolist()

    # FIX #1: Warn explicitly if any candidate still has empty combined text.
    empty_count = sum(1 for t in candidate_texts if not t)
    if empty_count:
        print(
            f"[step2] WARNING: {empty_count} candidates have empty embedding text. "
            "Their semantic_score will be near 0 and they will likely be gated out."
        )

    jd_vector = model.encode([JD])[0]

    embeddings = model.encode(
        candidate_texts,
        batch_size=256,
        show_progress_bar=True,
    )

    scores = cosine_similarity(embeddings, jd_vector.reshape(1, -1)).flatten()
    df["semantic_score"] = scores
    print(f"[step2] Semantic scores — min: {scores.min():.4f}, max: {scores.max():.4f}, mean: {scores.mean():.4f}")

    return df, embeddings
