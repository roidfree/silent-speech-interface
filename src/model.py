"""Model factory functions."""

from __future__ import annotations


def create_classifier(random_state: int = 42):
    """Create the v1 XGBoost classifier.

    Import is lazy so the rest of the pipeline remains usable without the
    optional training dependency installed.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for training. Install it with "
            "`pip install xgboost` or sync the project dependencies first."
        ) from exc

    return XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=random_state,
    )
