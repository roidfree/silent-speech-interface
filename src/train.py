"""Minimal training script.

Usage:
    python -m src.train --processed-dir data/processed --save-path results/models/rf.joblib
"""
from pathlib import Path
import argparse
import joblib
import numpy as np

from .data_loader import load_dataset
from .preprocessing import extract_features
from .model import create_classifier


def prepare_features(X_raw):
    X = [extract_features(x) for x in X_raw]
    # pad or truncate to same length if needed
    maxlen = max(x.shape[0] for x in X)
    X_mat = np.vstack([np.pad(x, (0, maxlen - x.shape[0]), 'constant') for x in X])
    return X_mat


def main(processed_dir: str, save_path: str):
    X_raw, y = load_dataset(processed_dir)
    if len(X_raw) == 0:
        raise RuntimeError(f"No data found in {processed_dir}; put .mat files under yes/ and no/")
    X = prepare_features(X_raw)
    y = np.asarray(y)

    clf = create_classifier()
    clf.fit(X, y)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, save_path)
    print(f"Saved model to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-dir', type=str, default='data/processed', help='Path to processed data folder')
    parser.add_argument('--save-path', type=str, default='results/models/rf.joblib', help='Model save path')
    args = parser.parse_args()
    main(args.processed_dir, args.save_path)
