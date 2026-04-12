"""Train and evaluate XGBoost models from grouped feature tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

from . import config
from .data_loader import METADATA_COLUMNS, load_feature_table, load_split_trial_ids
from .model import create_classifier


def _load_split_payload(split_path: str | Path) -> dict[str, list[str]]:
    with Path(split_path).open() as fh:
        return json.load(fh)


def _subset_for_split(
    features_df: pd.DataFrame,
    split_payload: dict[str, list[str]],
    split_name: str,
    *,
    allow_empty: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    trial_ids = set(split_payload.get(split_name, []))
    subset = features_df[features_df["trial_id"].isin(trial_ids)].copy()
    if subset.empty and not allow_empty:
        raise RuntimeError(f"No rows matched split '{split_name}'")
    X = subset.drop(columns=sorted(METADATA_COLUMNS))
    y = subset["word"].copy()
    return X, y


def _evaluate_predictions(
    y_true: pd.Series,
    y_pred: list[str],
    split_name: str,
) -> dict[str, object]:
    labels = sorted(y_true.unique().tolist())
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "split": split_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "classification_report": report_dict,
        "classification_report_text": classification_report(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
        "n_examples": int(len(y_true)),
    }


def train_model(
    features_csv: str | Path = config.FEATURES_METADATA_PATH,
    split_path: str | Path = config.BY_SESSION_SPLIT_PATH,
    save_path: str | Path = config.get_model_path("agagcl"),
    *,
    features_df: pd.DataFrame | None = None,
    split_payload: dict[str, list[str]] | None = None,
) -> dict[str, object]:
    feature_table = features_df.copy() if features_df is not None else load_feature_table(features_csv)
    payload = split_payload or _load_split_payload(split_path)

    X_train, y_train = _subset_for_split(feature_table, payload, "train")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    clf = create_classifier()
    clf.fit(X_train, y_train_encoded)

    metrics: dict[str, dict[str, object]] = {}
    for split_name in ("val", "test"):
        if not payload.get(split_name):
            continue
        X_split, y_split = _subset_for_split(feature_table, payload, split_name, allow_empty=True)
        if y_split.empty:
            continue
        pred_idx = clf.predict(X_split)
        pred_labels = label_encoder.inverse_transform(pred_idx)
        metrics[split_name] = _evaluate_predictions(y_split, list(pred_labels), split_name)

    bundle = {
        "model": clf,
        "feature_columns": list(X_train.columns),
        "label_classes": list(label_encoder.classes_),
        "metrics": metrics,
        "val_accuracy": metrics.get("val", {}).get("accuracy"),
        "val_macro_f1": metrics.get("val", {}).get("macro_f1"),
        "test_accuracy": metrics.get("test", {}).get("accuracy"),
        "test_macro_f1": metrics.get("test", {}).get("macro_f1"),
    }

    save_target = Path(save_path)
    save_target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, save_target)
    bundle["save_path"] = str(save_target)
    return bundle


def main(
    features_csv: str | Path = config.FEATURES_METADATA_PATH,
    split_path: str | Path = config.BY_SESSION_SPLIT_PATH,
    save_path: str | Path = config.get_model_path("agagcl"),
) -> dict[str, object]:
    bundle = train_model(features_csv=features_csv, split_path=split_path, save_path=save_path)
    print(f"Saved model to {bundle['save_path']}")
    for split_name in ("val", "test"):
        metrics = bundle["metrics"].get(split_name)
        if not metrics:
            continue
        print(f"{split_name.title()} accuracy: {metrics['accuracy']:.4f}")
        print(f"{split_name.title()} macro F1: {metrics['macro_f1']:.4f}")
        print(f"{split_name.title()} classification report:")
        print(metrics["classification_report_text"])
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--electrode", default=config.DEFAULT_ELECTRODE)
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--features-csv", default=None)
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--save-path", default=None)
    args = parser.parse_args()

    electrode_name = config.normalize_electrode_name(args.electrode)
    namespace = args.namespace or electrode_name
    features_csv = args.features_csv or str(config.get_features_metadata_path(namespace))
    split_path = args.split_path or str(config.get_by_session_split_path(namespace))
    save_path = args.save_path or str(config.get_model_path(namespace))
    main(features_csv, split_path, save_path)
