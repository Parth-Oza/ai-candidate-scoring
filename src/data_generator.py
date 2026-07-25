"""Generate a synthetic structured HR dataset for candidate scoring.

Real candidate data cannot be shared, so this module produces a realistic
synthetic dataset with the same schema used by the training pipeline.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def generate(n: int = 5000) -> pd.DataFrame:
    years_experience = RNG.gamma(shape=2.0, scale=2.5, size=n).clip(0, 25)
    skills_match_pct = RNG.beta(a=3, b=2, size=n)
    education_level = RNG.integers(1, 5, size=n)  # 1=HS .. 4=PhD
    num_prior_roles = RNG.poisson(lam=2.5, size=n).clip(0, 10)
    interview_technical = RNG.normal(3.4, 0.8, size=n).clip(1, 5)
    interview_communication = RNG.normal(3.5, 0.7, size=n).clip(1, 5)
    referral = RNG.binomial(1, 0.18, size=n)
    cert_count = RNG.poisson(lam=1.2, size=n).clip(0, 8)

    # Latent quality signal + noise -> binary "strong hire" label
    latent = (
        0.35 * (years_experience / 25)
        + 1.10 * skills_match_pct
        + 0.15 * (education_level / 4)
        + 0.55 * (interview_technical / 5)
        + 0.35 * (interview_communication / 5)
        + 0.20 * referral
        + 0.10 * (cert_count / 8)
        + RNG.normal(0, 0.18, size=n)
    )
    label = (latent > np.quantile(latent, 0.60)).astype(int)

    return pd.DataFrame(
        {
            "years_experience": years_experience.round(1),
            "skills_match_pct": skills_match_pct.round(3),
            "education_level": education_level,
            "num_prior_roles": num_prior_roles,
            "interview_technical": interview_technical.round(2),
            "interview_communication": interview_communication.round(2),
            "referral": referral,
            "cert_count": cert_count,
            "strong_hire": label,
        }
    )


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate()
    df.to_csv("data/candidates.csv", index=False)
    print(f"Wrote data/candidates.csv with {len(df)} rows "
          f"({df.strong_hire.mean():.1%} positive class)")
