"""Project configuration for the v1 silent-speech pipeline."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Raw input and generated output locations.
RAW_DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = PROJECT_ROOT / "processed"
PROCESSED_BLOCKS_DIR = PROCESSED_DIR / "blocks"
PROCESSED_TRIALS_DIR = PROCESSED_DIR / "trials"
PROCESSED_FEATURES_DIR = PROCESSED_DIR / "features"

METADATA_DIR = PROJECT_ROOT / "metadata"
BLOCKS_METADATA_PATH = METADATA_DIR / "blocks.csv"
TRIALS_METADATA_PATH = METADATA_DIR / "trials.csv"
FEATURES_METADATA_PATH = METADATA_DIR / "features.csv"
QUARANTINE_METADATA_PATH = METADATA_DIR / "quarantine_blocks.csv"

SPLITS_DIR = PROJECT_ROOT / "splits"
BY_BLOCK_SPLIT_PATH = SPLITS_DIR / "by_block.json"
BY_SESSION_SPLIT_PATH = SPLITS_DIR / "by_session.json"

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
