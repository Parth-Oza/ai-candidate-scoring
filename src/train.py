"""Train and evaluate the XGBoost candidate scoring model."""
from __future__ import annotations

import os

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier

from features import FEATURE_COLUMNS, split_xy

DATA_PATH = "data/candidates.csv"
MODEL_PATH = "models/candidate_scorer.joblib"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    search = RandomizedSearchCV(
        XGBClassifier(eval_metric="logloss", random_state=42),
        param_distributions={
            "n_estimators": [200, 300, 500],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
        n_iter=15,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=42,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    print(f"Best params: {search.best_params_}")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, proba):.3f}")
    print(classification_report(y_test, preds))

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(
        ascending=False
    ).to_csv("reports/feature_importance.csv")
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
