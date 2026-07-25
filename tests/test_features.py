import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features import engineer, FEATURE_COLUMNS  # noqa: E402


def sample_df():
    return pd.DataFrame(
        [
            {
                "years_experience": 6.0,
                "skills_match_pct": 0.8,
                "education_level": 3,
                "num_prior_roles": 2,
                "interview_technical": 4.0,
                "interview_communication": 3.5,
                "referral": 1,
                "cert_count": 2,
                "strong_hire": 1,
            }
        ]
    )


def test_engineer_adds_columns():
    out = engineer(sample_df())
    for col in FEATURE_COLUMNS:
        assert col in out.columns


def test_exp_per_role():
    out = engineer(sample_df())
    assert out.loc[0, "exp_per_role"] == 2.0


def test_interview_avg():
    out = engineer(sample_df())
    assert out.loc[0, "interview_avg"] == 3.75
