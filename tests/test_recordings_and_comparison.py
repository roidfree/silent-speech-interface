from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.compare_electrodes import build_matched_trial_subsets  # noqa: E402
from src.recordings import (  # noqa: E402
    build_recording_manifest,
    derive_stable_channel_subset,
    filter_summary_for_channels,
    summarize_recordings,
)


def test_pedot_manifest_parses_flat_raw_and_flags_suspicious_channels(tmp_path: Path):
    raw_dir = tmp_path / "pedot_raw"
    raw_dir.mkdir()
    for channel in ("2", "3", "28"):
        np.save(raw_dir / f"yes_CE_s5_ch{channel}.npy", np.ones((1, 100), dtype=np.float32))

    manifest = build_recording_manifest("pedot", data_dir=raw_dir)

    assert set(manifest["channel_id"]) == {"2", "3", "28"}
    assert manifest.loc[manifest["channel_id"] == "28", "is_suspicious_channel"].item() is True
    assert manifest.loc[manifest["channel_id"] == "28", "qc_status"].item() == "suspicious_channel"


def test_agagcl_manifest_reads_nested_layout(tmp_path: Path):
    source_dir = tmp_path / "AgAgCl" / "bye" / "s1"
    source_dir.mkdir(parents=True)
    for channel in range(1, 9):
        np.save(source_dir / f"bye_CNC_s1_ch{channel}.npy", np.ones((1, 120), dtype=np.float32))

    manifest = build_recording_manifest("agagcl", data_dir=tmp_path / "AgAgCl")

    assert len(manifest) == 8
    assert manifest["word"].unique().tolist() == ["bye"]
    assert manifest["session_token"].unique().tolist() == ["s1"]
    assert set(manifest["channel_id"]) == {str(channel) for channel in range(1, 9)}


def test_stable_channel_subset_uses_intersection_of_usable_recordings():
    summary_df = pd.DataFrame(
        [
            {"recording_id": "a", "word": "bye", "session_token": "s1", "valid_channels": "2,3,4", "is_usable": True},
            {"recording_id": "b", "word": "bye", "session_token": "s2", "valid_channels": "2,4", "is_usable": True},
            {"recording_id": "c", "word": "bye", "session_token": "s3", "valid_channels": "2,4,5", "is_usable": True},
        ]
    )

    assert derive_stable_channel_subset(summary_df) == ("2", "4")
    filtered = filter_summary_for_channels(summary_df, ("2", "4"))
    assert set(filtered["recording_id"]) == {"a", "b", "c"}


def test_recording_summary_marks_length_mismatch_and_failures():
    manifest = pd.DataFrame(
        [
            {
                "recording_id": "pedot:bye:s1",
                "electrode": "pedot",
                "word": "bye",
                "session_token": "s1",
                "label_token": "CE",
                "channel_id": "2",
                "path": "foo.npy",
                "sample_count": 100,
                "qc_status": "ok",
                "is_suspicious_channel": False,
            },
            {
                "recording_id": "pedot:bye:s1",
                "electrode": "pedot",
                "word": "bye",
                "session_token": "s1",
                "label_token": "CE",
                "channel_id": "3",
                "path": "bar.npy",
                "sample_count": 120,
                "qc_status": "ok",
                "is_suspicious_channel": False,
            },
            {
                "recording_id": "pedot:bye:s2",
                "electrode": "pedot",
                "word": "bye",
                "session_token": "s2",
                "label_token": "CE",
                "channel_id": "2",
                "path": "baz.npy",
                "sample_count": None,
                "qc_status": "load_error:boom",
                "is_suspicious_channel": False,
            },
        ]
    )

    summary = summarize_recordings(manifest)
    by_id = summary.set_index("recording_id")
    assert by_id.loc["pedot:bye:s1", "has_length_mismatch"]
    assert not bool(by_id.loc["pedot:bye:s1", "is_usable"])
    assert "length_mismatch" in by_id.loc["pedot:bye:s1", "qc_status"]
    assert "channel_qc_failures" in by_id.loc["pedot:bye:s2", "qc_status"]


def test_build_matched_trial_subsets_respects_min_budget_per_split():
    ag = pd.DataFrame(
        [
            {"trial_id": "a1", "word": "bye", "session_token": "s1", "f": 1.0},
            {"trial_id": "a2", "word": "bye", "session_token": "s1", "f": 2.0},
            {"trial_id": "a3", "word": "bye", "session_token": "s1", "f": 3.0},
            {"trial_id": "a4", "word": "no", "session_token": "s2", "f": 4.0},
        ]
    )
    pedot = pd.DataFrame(
        [
            {"trial_id": "p1", "word": "bye", "session_token": "s1", "f": 1.0},
            {"trial_id": "p2", "word": "bye", "session_token": "s1", "f": 2.0},
            {"trial_id": "p3", "word": "no", "session_token": "s2", "f": 3.0},
            {"trial_id": "p4", "word": "no", "session_token": "s2", "f": 4.0},
            {"trial_id": "p5", "word": "no", "session_token": "s2", "f": 5.0},
        ]
    )
    split_payloads = {
        "agagcl": {"train": ["a1", "a2", "a3"], "val": ["a4"], "test": []},
        "pedot": {"train": ["p1", "p2"], "val": ["p3", "p4", "p5"], "test": []},
    }

    subset_tables, subset_payloads, budget_df = build_matched_trial_subsets(
        {"agagcl": ag, "pedot": pedot},
        split_payloads,
    )

    assert len(subset_payloads["agagcl"]["train"]) == 2
    assert len(subset_payloads["pedot"]["train"]) == 2
    assert len(subset_payloads["agagcl"]["val"]) == 1
    assert len(subset_payloads["pedot"]["val"]) == 1
    assert set(subset_tables["agagcl"]["trial_id"]) == set(sum(subset_payloads["agagcl"].values(), []))
    assert set(subset_tables["pedot"]["trial_id"]) == set(sum(subset_payloads["pedot"].values(), []))
    assert set(budget_df["trial_budget"]) == {1, 2}
