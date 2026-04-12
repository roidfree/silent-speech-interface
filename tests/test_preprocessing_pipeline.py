from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.preprocessing import JaylaDetector, JaylaPreprocessor, fuse_channel_masks  # noqa: E402
from src.preprocessing.pipeline import (  # noqa: E402
    BlockRecord,
    assign_label_balanced_group_splits,
    extract_session_token,
    load_block,
)


def _load_jayla_module(module_name: str, relative_path: str):
    module_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    original_sys_path = list(sys.path)
    sys.path.insert(0, str((ROOT / "SSI-jayla").resolve()))
    sys.path.insert(0, str((ROOT / "SSI-jayla" / "src").resolve()))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_sys_path
    return module


def test_preprocessor_matches_jayla_code():
    jayla_preprocess_module = _load_jayla_module("jayla_preprocess2", "SSI-jayla/src/preprocess2.py")
    raw = np.load(ROOT / "data" / "AgAgCl" / "bye" / "s1" / "bye_CNC_s1_ch1.npy").squeeze().astype(float)

    ours = JaylaPreprocessor().process(raw)
    theirs = jayla_preprocess_module.EMGPreprocessor2().process_signal(raw)["filtered"]

    assert np.allclose(ours, theirs)


def test_detector_matches_jayla_code():
    jayla_detector_module = _load_jayla_module("jayla_detector", "SSI-jayla/src/speech_detection.py")
    raw = np.load(ROOT / "data" / "AgAgCl" / "bye" / "s1" / "bye_CNC_s1_ch1.npy").squeeze().astype(float)

    ours = JaylaDetector().detect(raw, return_metadata=True)
    theirs = jayla_detector_module.SpeechActivityDetector(method="spc").detect(raw, return_metadata=True)

    assert np.allclose(ours["clean_signal"], theirs["clean_signal"])
    assert np.array_equal(ours["labels"], theirs["labels"])
    assert ours["segments"] == theirs["segments"]
    assert np.isclose(ours["final_threshold"], theirs["final_threshold"])


def test_load_block_stacks_eight_channels():
    record = BlockRecord(word="bye", source_dir=ROOT / "data" / "AgAgCl" / "bye" / "s1", session_token="s1")
    block, error = load_block(record)
    assert error is None
    assert block is not None
    assert block.shape == (8, 379000)


def test_load_block_quarantines_missing_or_corrupt_channels():
    record = BlockRecord(word="bye", source_dir=ROOT / "data" / "AgAgCl" / "bye" / "s4", session_token="s4")
    block, error = load_block(record)
    assert block is None
    assert error is not None


def test_fusion_requires_two_channels():
    masks = np.array(
        [
            [False, True, True, False, False],
            [False, False, True, False, False],
            [False, False, False, False, False],
        ]
    )
    fused = fuse_channel_masks(masks, min_active_ch=2)
    assert np.array_equal(fused, np.array([False, False, True, False, False]))


def test_extract_session_token():
    assert extract_session_token("s4") == "s4"
    assert extract_session_token("pain_s4") == "s4"
    assert extract_session_token("yes_s1 (empty cuz recordings are dogshi)") == "s1"


def test_assign_label_balanced_group_splits_preserves_class_coverage():
    rows = []
    label_by_group = {
        "s1": {"bye": 5, "help": 5, "yes": 0},
        "s2": {"bye": 5, "help": 5, "yes": 5},
        "s3": {"bye": 5, "help": 5, "yes": 5},
        "s4": {"bye": 5, "help": 5, "yes": 0},
        "s5": {"bye": 5, "help": 5, "yes": 5},
    }
    for group, counts in label_by_group.items():
        for word, count in counts.items():
            for idx in range(count):
                rows.append({"trial_id": f"{group}_{word}_{idx}", "session_token": group, "word": word})
    trials_df = pd.DataFrame(rows)

    splits = assign_label_balanced_group_splits(trials_df, "session_token", "word", seed=42)
    assert set(splits) == {"train", "val", "test"}
    for split_groups in splits.values():
        split_words = set(trials_df[trials_df["session_token"].isin(split_groups)]["word"].unique())
        assert split_words == {"bye", "help", "yes"}
