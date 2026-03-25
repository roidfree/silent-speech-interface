"""Train the v1 XGBoost model from the feature table and grouped split."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

from . import config
from .data_loader import load_training_data
from .model import create_classifier


def main(
    features_csv: str = str(config.FEATURES_METADATA_PATH),
    split_path: str = str(config.BY_SESSION_SPLIT_PATH),
    save_path: str = "results/models/xgboost.joblib",
) -> None:
    X_train, y_train = load_training_data(features_csv, split_path, "train")
    X_val, y_val = load_training_data(features_csv, split_path, "val")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)

    clf = create_classifier()
    clf.fit(X_train, y_train_encoded)

    val_pred_encoded = clf.predict(X_val)
    val_acc = accuracy_score(y_val_encoded, val_pred_encoded)
    val_macro_f1 = f1_score(y_val_encoded, val_pred_encoded, average="macro")
    val_pred = label_encoder.inverse_transform(val_pred_encoded)
    report = classification_report(y_val, val_pred)

    save_target = Path(save_path)
    save_target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "feature_columns": list(X_train.columns),
            "label_classes": list(label_encoder.classes_),
            "val_accuracy": float(val_acc),
            "val_macro_f1": float(val_macro_f1),
        },
        save_target,
    )
    print(f"Saved model to {save_target}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Validation macro F1: {val_macro_f1:.4f}")
    print("Validation classification report:")
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", default=str(config.FEATURES_METADATA_PATH))
    parser.add_argument("--split-path", default=str(config.BY_SESSION_SPLIT_PATH))
    parser.add_argument("--save-path", default="results/models/xgboost.joblib")
    args = parser.parse_args()
    main(args.features_csv, args.split_path, args.save_path)
