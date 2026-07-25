"""Feature engineering for the candidate scoring model."""
from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = [
    "years_experience",
    "skills_match_pct",
    "education_level",
    "num_prior_roles",
    "interview_technical",
    "interview_communication",
    "referral",
    "cert_count",
    # engineered
    "exp_per_role",
    "interview_avg",
    "senior_flag",
]

TARGET = "strong_hire"


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features. Returns a new DataFrame."""
    out = df.copy()
    out["exp_per_role"] = out["years_experience"] / (out["num_prior_roles"] + 1)
    out["interview_avg"] = (
        out["interview_technical"] + out["interview_communication"]
    ) / 2.0
    out["senior_flag"] = (out["years_experience"] >= 8).astype(int)
    return out


def split_xy(df: pd.DataFrame):
    df = engineer(df)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET] if TARGET in df.columns else None
    return X, y
