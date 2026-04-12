"""Project configuration for the silent-speech pipeline."""

from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Raw input and generated output locations.
RAW_DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = PROJECT_ROOT / "processed"
METADATA_DIR = PROJECT_ROOT / "metadata"
SPLITS_DIR = PROJECT_ROOT / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"

# Backwards-compatible legacy defaults for the original Ag/AgCl v1 workflow.
PROCESSED_BLOCKS_DIR = PROCESSED_DIR / "blocks"
PROCESSED_TRIALS_DIR = PROCESSED_DIR / "trials"
PROCESSED_FEATURES_DIR = PROCESSED_DIR / "features"

BLOCKS_METADATA_PATH = METADATA_DIR / "blocks.csv"
TRIALS_METADATA_PATH = METADATA_DIR / "trials.csv"
FEATURES_METADATA_PATH = METADATA_DIR / "features.csv"
QUARANTINE_METADATA_PATH = METADATA_DIR / "quarantine_blocks.csv"

BY_BLOCK_SPLIT_PATH = SPLITS_DIR / "by_block.json"
BY_SESSION_SPLIT_PATH = SPLITS_DIR / "by_session.json"

SUPPORTED_ELECTRODES = ("agagcl", "pedot")
DEFAULT_ELECTRODE = "agagcl"
ELECTRODE_SOURCE_DIRS = {
    "agagcl": RAW_DATA_DIR / "AgAgCl",
    "pedot": RAW_DATA_DIR / "PEDOT5%GLYC2.5" / "raw",
}

# Deterministic group split defaults.
SPLIT_SEED = 42
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# Jayla code truth: filtering and detector parameters.
SAMPLING_RATE = 2000
NOTCH_Q = 30.0
NOTCH_FREQS = [50.0]
BP_LOW = 20.0
BP_HIGH = 450.0
BP_ORDER = 4

WINDOW_MS = 40.0
HOP_MS = 20.0
SMOOTHING_CUTOFF_HZ = 10.0
THRESHOLD_DECAY = 0.99
THRESHOLD_MIN_RATIO = 0.4
MIN_ACTIVE_CH = 2
IGNORE_START_MS = 8500.0
IGNORE_END_MS = 40000.0
MIN_DURATION_MS = 950.0
MERGE_GAP_MS = 1080.0
ONSET_BUFFER_MS = 20.0
OFFSET_BUFFER_MS = 120.0
THRESHOLD_FLOOR = 0.021

# Feature extraction configuration for XGBoost v1.
FEATURE_WINDOW_MS = WINDOW_MS
FEATURE_HOP_MS = HOP_MS
FEATURE_NAMES = ("rms", "mav", "var", "wl", "sd")
FEATURE_AGGREGATIONS = ("mean", "std", "max")

EXPECTED_CHANNELS = tuple(str(i) for i in range(1, 9))
PEDOT_SUSPICIOUS_CHANNEL_IDS = {"28"}
MATCHED_RANDOM_SEED = 42


def normalize_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not slug:
        raise ValueError("Expected a non-empty slug")
    return slug


def normalize_electrode_name(electrode: str) -> str:
    normalized = normalize_slug(electrode)
    aliases = {
        "agagcl": "agagcl",
        "ag_agcl": "agagcl",
        "pedot": "pedot",
        "pedot5_glyc2_5": "pedot",
        "pedot5_glyc2_5_raw": "pedot",
    }
    if normalized not in aliases:
        raise KeyError(f"Unsupported electrode '{electrode}'")
    return aliases[normalized]


def get_default_source_dir(electrode: str) -> Path:
    return ELECTRODE_SOURCE_DIRS[normalize_electrode_name(electrode)]


def get_namespace_root(base_dir: Path, namespace: str) -> Path:
    parts = [normalize_slug(part) for part in namespace.split("/") if part.strip()]
    if not parts:
        raise ValueError("Expected a non-empty namespace")
    root = base_dir
    for part in parts:
        root /= part
    return root


def get_processed_blocks_dir(namespace: str) -> Path:
    return get_namespace_root(PROCESSED_DIR, namespace) / "blocks"


def get_processed_trials_dir(namespace: str) -> Path:
    return get_namespace_root(PROCESSED_DIR, namespace) / "trials"


def get_processed_features_dir(namespace: str) -> Path:
    return get_namespace_root(PROCESSED_DIR, namespace) / "features"


def get_metadata_dir(namespace: str) -> Path:
    return get_namespace_root(METADATA_DIR, namespace)


def get_blocks_metadata_path(namespace: str) -> Path:
    return get_metadata_dir(namespace) / "blocks.csv"


def get_trials_metadata_path(namespace: str) -> Path:
    return get_metadata_dir(namespace) / "trials.csv"


def get_features_metadata_path(namespace: str) -> Path:
    return get_metadata_dir(namespace) / "features.csv"


def get_quarantine_metadata_path(namespace: str) -> Path:
    return get_metadata_dir(namespace) / "quarantine_blocks.csv"


def get_manifest_metadata_path(namespace: str) -> Path:
    return get_metadata_dir(namespace) / "recording_manifest.csv"


def get_recording_summary_path(namespace: str) -> Path:
    return get_metadata_dir(namespace) / "recording_summary.csv"


def get_sampled_trial_ids_path(namespace: str) -> Path:
    return get_metadata_dir(namespace) / "selected_trial_ids.json"


def get_splits_dir(namespace: str) -> Path:
    return get_namespace_root(SPLITS_DIR, namespace)


def get_by_block_split_path(namespace: str) -> Path:
    return get_splits_dir(namespace) / "by_block.json"


def get_by_session_split_path(namespace: str) -> Path:
    return get_splits_dir(namespace) / "by_session.json"


def get_results_dir(namespace: str) -> Path:
    return get_namespace_root(RESULTS_DIR, namespace)


def get_model_dir(namespace: str) -> Path:
    return get_results_dir(namespace) / "models"


def get_model_path(namespace: str, filename: str = "xgboost.joblib") -> Path:
    return get_model_dir(namespace) / filename


def get_report_dir(namespace: str) -> Path:
    return get_results_dir(namespace) / "reports"
