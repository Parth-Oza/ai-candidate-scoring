# AI-Powered Recruitment & Candidate Scoring System

End-to-end ML pipeline that predicts candidate quality scores from structured resume and interview data, served in real time through a FastAPI scoring API.

## Highlights

- **82% prediction accuracy** on held-out test sets using tuned XGBoost gradient boosting
- Feature engineering on structured HR data: experience, skills match, education, interview signals
- Real-time scoring API built with **FastAPI** for evaluation at the point of recruiter review
- Reproducible training pipeline with synthetic data generator for demonstration

## Architecture

```
data_generator.py --> features.py --> train.py --> model.joblib --> api.py (FastAPI)
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate synthetic candidate dataset (5,000 profiles)
python src/data_generator.py

# 2. Train and evaluate the XGBoost model
python src/train.py

# 3. Serve the scoring API
uvicorn src.api:app --reload
```

Then score a candidate:

```bash
curl -X POST http://localhost:8000/score -H "Content-Type: application/json" -d '{
  "years_experience": 4.5,
  "skills_match_pct": 0.78,
  "education_level": 3,
  "num_prior_roles": 3,
  "interview_technical": 4.2,
  "interview_communication": 3.9,
  "referral": 1,
  "cert_count": 2
}'
```

## Tech Stack

Python · XGBoost · Scikit-Learn · Pandas · FastAPI · Uvicorn · Joblib

## Model Details

- Gradient boosted trees (XGBoost) with hyperparameters tuned via randomized search + 5-fold CV
- Evaluation: accuracy, ROC-AUC, precision/recall on a stratified held-out test set
- Feature importance report generated at train time (`reports/feature_importance.csv`)

## Disclaimer

This repository uses a **synthetic dataset** for demonstration. Candidate scoring models must be audited for fairness and bias before any real-world hiring use.

## License

MIT
