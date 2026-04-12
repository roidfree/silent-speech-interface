"""Helpers for loading model-ready feature tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config


METADATA_COLUMNS = {"trial_id", "block_id", "recording_id", "electrode", "word", "session_token"}


def load_feature_table(features_csv: str | Path = config.FEATURES_METADATA_PATH) -> pd.DataFrame:
    """Load the flattened trial-level feature table."""
    return pd.read_csv(features_csv)


def load_split_trial_ids(split_path: str | Path, split_name: str) -> list[str]:
    """Load trial IDs for a named split."""
    with Path(split_path).open() as fh:
        payload = json.load(fh)
    if split_name not in payload:
        raise KeyError(f"Split '{split_name}' not found in {split_path}")
    return list(payload[split_name])


def load_training_data(
    features_csv: str | Path = config.FEATURES_METADATA_PATH,
    split_path: str | Path = config.BY_SESSION_SPLIT_PATH,
    split_name: str = "train",
) -> tuple[pd.DataFrame, pd.Series]:
    """Load X/y for one split from the feature table."""
    features = load_feature_table(features_csv)
    trial_ids = set(load_split_trial_ids(split_path, split_name))
    subset = features[features["trial_id"].isin(trial_ids)].copy()
    if subset.empty:
        raise RuntimeError(f"No rows matched split '{split_name}' in {split_path}")
    X = subset.drop(columns=sorted(METADATA_COLUMNS))
    y = subset["word"].copy()
    return X, y


def load_split_subset(
    features_csv: str | Path,
    split_path: str | Path,
    split_name: str,
) -> pd.DataFrame:
    """Load the full feature subset for a named split."""
    features = load_feature_table(features_csv)
    trial_ids = set(load_split_trial_ids(split_path, split_name))
    return features[features["trial_id"].isin(trial_ids)].copy()
