"""FastAPI service for real-time candidate scoring."""
from __future__ import annotations

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .features import split_xy

MODEL_PATH = "models/candidate_scorer.joblib"

app = FastAPI(title="Candidate Scoring API", version="1.0.0")
_model = None


class Candidate(BaseModel):
    years_experience: float = Field(ge=0, le=50)
    skills_match_pct: float = Field(ge=0, le=1)
    education_level: int = Field(ge=1, le=4)
    num_prior_roles: int = Field(ge=0, le=20)
    interview_technical: float = Field(ge=1, le=5)
    interview_communication: float = Field(ge=1, le=5)
    referral: int = Field(ge=0, le=1)
    cert_count: int = Field(ge=0, le=20)


def get_model():
    global _model
    if _model is None:
        try:
            _model = joblib.load(MODEL_PATH)
        except FileNotFoundError as exc:
            raise HTTPException(503, "Model not trained yet. Run src/train.py") from exc
    return _model


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score(candidate: Candidate):
    model = get_model()
    X, _ = split_xy(pd.DataFrame([candidate.model_dump()]))
    proba = float(model.predict_proba(X)[0, 1])
    return {
        "quality_score": round(proba, 4),
        "recommendation": "advance" if proba >= 0.5 else "review",
    }
