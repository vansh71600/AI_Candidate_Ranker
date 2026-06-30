# AI Candidate Ranker

**Redrob AI Hackathon — Intelligent Candidate Discovery & Ranking Challenge**
**Team:** Builder | **Participant:** Vansh Sharma | **Institute:** IIT Guwahati

---

## Problem Statement

Recruiters go through hundreds of profiles and still often miss the right person — not because the talent isn't there, but because keyword filters can't see what actually matters. This system ranks candidates the way a great recruiter would: by understanding who actually fits the role, not by matching keywords.

---

## Reproduce

```bash
pip install -r requirements.txt
python main.py --candidates ./candidates.jsonl --out ./builder.csv
python validate_submission.py builder.csv
```

Place `candidates.jsonl` in the repo root before running. Output is `builder.csv`.

**Runtime:** under 5 minutes on CPU · **Memory:** under 4 GB · **No GPU · No network calls during ranking**

> **Kaggle users:** If you see a torch circular import error, run this first:
> ```bash
> pip uninstall torchcodec -y --quiet
> pip install sentence-transformers --no-deps --quiet
> pip install "transformers>=4.34.0" "huggingface-hub>=0.19.0" tokenizers tqdm --quiet
> ```

---

## Try it without installing anything

A live sandbox is hosted on Streamlit Cloud — upload a small `.jsonl` sample and get ranked output back:

**https://aicandidateranker-tt5xkj5crbym8rp2p6ff9h.streamlit.app/**

For samples smaller than 100 candidates, the sandbox ranks every candidate that passes the hard filters by score, rather than enforcing the stricter score gates used in the full submission run — this keeps small demo uploads useful even when very few candidates would otherwise clear the production thresholds.

---

## Pipeline

```
candidates.jsonl
      │
      ▼
Step 1 — Load, Parse & Filter                     [step1_load_candidates.py]
      Hard filters:
        • total_exp >= 3 yrs, relevant_exp >= 2 yrs
        • Python required (explicit OR inferred from PyTorch/FAISS/etc.)
        • At least one ML skill required
        • Active within 365 days
        • Irrelevant titles removed: backend/frontend/devops/analytics engineer
          (escape hatch: one NLP/retrieval skill at intermediate+ saves them)
        • Computer Vision filter (separate): needs 2+ advanced NLP/retrieval
          skills to qualify as a CV-NLP hybrid
        • Honeypot detection:
            - expert proficiency in 5+ skills with 0 months used
            - job duration exceeding plausible company founding date
      Rolling 6-year window for career text
      All 23 redrob_signals extracted
      │
      ▼
Step 2 — Semantic Embedding                       [step2_embed_and_score.py]
      career_text + skills_text combined for richer embedding surface
      Model: paraphrase-MiniLM-L3-v2 (local, CPU, no network at ranking time)
      JD enriched with semantic equivalents to bridge intentional keyword gap
      Cosine similarity → semantic_score
      │
      ▼
Step 3 — Numeric Scoring                          [step3_numeric_scoring.py]
      skill_score      (25%) — JD skill overlap weighted by proficiency
      behavioral_score (18%) — activity, response rate, GitHub, completeness,
                               verified email, offer acceptance, recruiter saves
      exp_score        (10%) — relevant domain experience vs JD sweet spot
      company_score     (4%) — product vs consulting via industry labels
      assessment_score  (4%) — verified platform skill test scores
      edu_score         (4%) — institution tier
      │
      ▼
Main — Final Ranking                              [main.py]
      semantic (35%) + all numeric scores combined
      Rounded to 4dp, sorted score desc then candidate_id asc for ties
      Post-scoring gates: skill_score > 0.09, semantic_score > 0.25
      Top 100 with specific per-candidate reasoning
      │
      ▼
builder.csv
```

### Scoring Formula

```
final_score = 0.35 × semantic_score
            + 0.25 × skill_score
            + 0.18 × behavioral_score
            + 0.10 × exp_score
            + 0.04 × company_score
            + 0.04 × assessment_score
            + 0.04 × edu_score
```

---

## Key Design Decisions

**JD enrichment** — The JD intentionally avoids ML buzzwords. We expand it with semantic equivalents so candidates who built retrieval or ranking systems match even without "FAISS" or "RAG" in their profile.

**Rolling 6-year career window** — Uses a live datetime comparison so the window is always exactly 6 years from run time. Older experience adds noise to semantic matching.

**Skills fed into embedding** — Proficiency-weighted skill terms are appended to career text before encoding, giving the embedding a richer signal surface beyond job descriptions alone.

**Separate CV filter** — Computer vision engineers are handled with their own stricter filter requiring 2+ advanced NLP/retrieval skills. This prevents both false positives and false negatives that a single catch-all title list would cause.

**Current title checked** — Both the candidate's current title and full career history are checked against irrelevant-title filters, not just the history.

**Behavioral score uses all platform signals** — `profile_completeness`, `verified_email`, `offer_acceptance_rate`, `saved_by_recruiters_30d`, `applications_submitted_30d` all feed into behavioral scoring alongside the core activity and response signals.

**Education scored** — Institution tier contributes 4% of the final score.

**Honest, JD-connected reasoning** — `build_reasoning()` scans career text for retrieval, ranking, and production deployment signals and names them explicitly. Concerns (long notice period, passive status, low response rate, limited relevant experience) are always surfaced. No two candidates get structurally identical reasoning.

**Honeypot detection** — Expert proficiency with 0 months used across 5+ skills, and job durations that exceed the plausible company age, are both flagged and filtered before scoring.

**Tie-breaking** — Scores rounded to 4dp. Ties broken by `candidate_id` ascending as required by the spec.

---

## File Structure

```
AI_Candidate_Ranker/
├── main.py                    ← entry point; final ranking + reasoning
├── step1_load_candidates.py   ← load, parse, filter, honeypot detection
├── step2_embed_and_score.py   ← JD enrichment, embedding, semantic score
├── step3_numeric_scoring.py   ← skill, behavioral, experience, company, edu scores
├── app.py                     ← Streamlit sandbox app (hosted demo)
├── requirements.txt           ← dependencies
├── runtime.txt                ← pins Python version for the sandbox deployment
├── submission_metadata.yaml   ← hackathon metadata
└── README.md                  ← this file
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `sentence-transformers` | Local embedding model (paraphrase-MiniLM-L3-v2) |
| `scikit-learn` | Cosine similarity |
| `pandas` | Data loading and manipulation |
| `numpy` | Vector operations |
| `torch` | Required by sentence-transformers (not pinned — see requirements.txt) |
| `streamlit` | Sandbox demo app |

---

## Compute Constraints

| Constraint | Limit | This system |
|---|---|---|
| Runtime | ≤ 5 min wall-clock | ~2–3 min on 4-core CPU |
| Memory | ≤ 16 GB RAM | < 4 GB |
| GPU | Not allowed | CPU only |
| Network during ranking | Not allowed | None — model runs locally |

The embedding model is downloaded once on first run and cached by `sentence-transformers`. All subsequent runs are fully offline.

---

## A note on authorship

The pipeline design — the filter logic, the embedding and cosine-similarity approach, the JD enrichment strategy, and the scoring weights — was designed by me. Claude was used as a coding and debugging assistant: writing and cleaning up the implementation, catching edge cases, fixing deployment and environment issues, and helping deploy the Streamlit sandbox.
