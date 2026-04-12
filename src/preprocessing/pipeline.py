"""End-to-end conversion pipeline for electrode-aware silent-speech datasets."""

from __future__ import annotations

import argparse
import json
import itertools
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .. import config
from ..recordings import (
    build_recording_manifest,
    derive_stable_channel_subset,
    filter_summary_for_channels,
    load_block_from_manifest_rows,
    summarize_recordings,
)
from .preprocessing import JaylaConfig, JaylaDetector, extract_features, segment_multichannel_block


@dataclass(frozen=True)
class BlockRecord:
    word: str
    source_dir: Path
    session_token: str


def extract_session_token(name: str) -> str:
    match = re.search(r"(s\d+)", name)
    return match.group(1) if match else name


def iter_block_dirs(root: Path) -> list[BlockRecord]:
    records: list[BlockRecord] = []
    for word_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for block_dir in sorted(path for path in word_dir.iterdir() if path.is_dir()):
            records.append(
                BlockRecord(
                    word=word_dir.name,
                    source_dir=block_dir,
                    session_token=extract_session_token(block_dir.name),
                )
            )
    return records


def _channel_number(path: Path) -> str | None:
    match = re.search(r"_ch(\d+)\.npy$", path.name)
    return match.group(1) if match else None


def load_block(record: BlockRecord) -> tuple[np.ndarray | None, str | None]:
    """Legacy loader kept for Ag/AgCl compatibility tests."""
    channels: dict[str, np.ndarray] = {}
    channel_lengths: set[int] = set()

    for path in sorted(record.source_dir.glob("*.npy")):
        channel_id = _channel_number(path)
        if channel_id is None:
            continue
        try:
            signal = np.load(path).squeeze()
        except Exception as exc:
            return None, f"corrupt channel file {path.name}: {exc}"
        channels[channel_id] = np.asarray(signal)
        channel_lengths.add(int(np.asarray(signal).shape[-1]))

    missing = [ch for ch in config.EXPECTED_CHANNELS if ch not in channels]
    if missing:
        return None, f"missing channel(s): {', '.join(missing)}"
    if len(channel_lengths) != 1:
        return None, "channel lengths do not match"

    block = np.stack([channels[ch] for ch in config.EXPECTED_CHANNELS], axis=0).astype(np.float32)
    return block, None


def assign_group_splits(groups: Iterable[str], seed: int) -> dict[str, list[str]]:
    unique_groups = sorted(set(groups))
    rng = random.Random(seed)
    rng.shuffle(unique_groups)

    n_train, n_val, n_test = _resolve_group_split_counts(len(unique_groups))
    train_groups = unique_groups[:n_train]
    val_groups = unique_groups[n_train : n_train + n_val]
    test_groups = unique_groups[n_train + n_val :]

    return {"train": train_groups, "val": val_groups, "test": test_groups}


def _resolve_group_split_counts(n_groups: int) -> tuple[int, int, int]:
    n_train = max(1, round(n_groups * config.TRAIN_RATIO))
    n_val = max(1, round(n_groups * config.VAL_RATIO))
    n_train = min(n_train, n_groups - 2) if n_groups >= 3 else max(1, n_groups - 1)
    n_val = min(n_val, n_groups - n_train - 1) if n_groups - n_train >= 2 else max(0, n_groups - n_train - 1)
    n_test = n_groups - n_train - n_val
    if n_test <= 0 and n_groups > 1:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val = max(0, n_val - 1)

    return n_train, n_val, n_test


def assign_label_balanced_group_splits(
    trials_df: pd.DataFrame,
    group_column: str,
    label_column: str,
    seed: int,
) -> dict[str, list[str]]:
    """Search for a group split that preserves label coverage across splits."""
    groups = sorted(trials_df[group_column].unique().tolist())
    n_train, n_val, n_test = _resolve_group_split_counts(len(groups))
    if len(groups) > 12:
        return assign_group_splits(groups, seed)

    total_trials = len(trials_df)
    overall_label_counts = trials_df[label_column].value_counts().sort_index()
    target_trial_counts = {
        "train": total_trials * config.TRAIN_RATIO,
        "val": total_trials * config.VAL_RATIO,
        "test": total_trials * config.TEST_RATIO,
    }
    group_trial_counts = trials_df.groupby(group_column).size()
    group_label_counts = trials_df.groupby([group_column, label_column]).size().unstack(fill_value=0)
    group_label_counts = group_label_counts.reindex(index=groups, columns=overall_label_counts.index, fill_value=0)

    best_score: tuple[float, ...] | None = None
    best_split: dict[str, list[str]] | None = None

    for train_groups in itertools.combinations(groups, n_train):
        remaining_after_train = [group for group in groups if group not in train_groups]
        for val_groups in itertools.combinations(remaining_after_train, n_val):
            test_groups = tuple(group for group in remaining_after_train if group not in val_groups)
            if len(test_groups) != n_test:
                continue

            split_groups = {
                "train": list(train_groups),
                "val": list(val_groups),
                "test": list(test_groups),
            }
            score_parts: list[float] = []
            total_missing_labels = 0

            for split_name in ("train", "val", "test"):
                selected = split_groups[split_name]
                split_label_counts = group_label_counts.loc[selected].sum(axis=0)
                missing_labels = int((split_label_counts == 0).sum())
                total_missing_labels += missing_labels

                split_trial_count = float(group_trial_counts.loc[selected].sum())
                score_parts.append(abs(split_trial_count - target_trial_counts[split_name]))

                target_label_counts = overall_label_counts * (
                    config.TRAIN_RATIO
                    if split_name == "train"
                    else config.VAL_RATIO if split_name == "val" else config.TEST_RATIO
                )
                score_parts.append(float((split_label_counts - target_label_counts).abs().sum()))

            score = (float(total_missing_labels), *score_parts)
            if best_score is None or score < best_score:
                best_score = score
                best_split = split_groups

    if best_split is None:
        return assign_group_splits(groups, seed)
    return best_split


def materialize_trial_splits(
    trials_df: pd.DataFrame,
    group_column: str,
    seed: int,
    label_column: str | None = None,
) -> dict[str, list[str]]:
    if label_column is None:
        group_splits = assign_group_splits(trials_df[group_column].tolist(), seed)
    else:
        group_splits = assign_label_balanced_group_splits(trials_df, group_column, label_column, seed)
    payload = {"train": [], "val": [], "test": []}
    for split_name, groups in group_splits.items():
        payload[split_name] = sorted(trials_df[trials_df[group_column].isin(groups)]["trial_id"].tolist())
    return payload


def _clean_output_dirs(namespace: str) -> None:
    for directory, prefix in (
        (config.get_processed_blocks_dir(namespace), "block_"),
        (config.get_processed_trials_dir(namespace), "trial_"),
    ):
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.glob(f"{prefix}*.npy"):
            path.unlink()
    config.get_processed_features_dir(namespace).mkdir(parents=True, exist_ok=True)
    config.get_metadata_dir(namespace).mkdir(parents=True, exist_ok=True)
    config.get_splits_dir(namespace).mkdir(parents=True, exist_ok=True)


def _save_pipeline_outputs(
    namespace: str,
    manifest_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    trials_df: pd.DataFrame,
    features_df: pd.DataFrame,
    quarantine_df: pd.DataFrame,
) -> None:
    config.get_metadata_dir(namespace).mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(config.get_manifest_metadata_path(namespace), index=False)
    summary_df.to_csv(config.get_recording_summary_path(namespace), index=False)
    blocks_df.to_csv(config.get_blocks_metadata_path(namespace), index=False)
    trials_df.to_csv(config.get_trials_metadata_path(namespace), index=False)
    features_df.to_csv(config.get_features_metadata_path(namespace), index=False)
    quarantine_df.to_csv(config.get_quarantine_metadata_path(namespace), index=False)


def run_pipeline(
    electrode: str = config.DEFAULT_ELECTRODE,
    data_dir: str | Path | None = None,
    namespace: str | None = None,
    manifest_df: pd.DataFrame | None = None,
    allowed_channels: tuple[str, ...] | list[str] | None = None,
    selected_recording_ids: set[str] | None = None,
) -> dict[str, object]:
    electrode_name = config.normalize_electrode_name(electrode)
    namespace = namespace or electrode_name
    _clean_output_dirs(namespace)

    manifest = manifest_df.copy() if manifest_df is not None else build_recording_manifest(electrode_name, data_dir=data_dir)
    if manifest.empty:
        summary_df = summarize_recordings(manifest)
        _save_pipeline_outputs(
            namespace,
            manifest,
            summary_df,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
        for split_path in (config.get_by_block_split_path(namespace), config.get_by_session_split_path(namespace)):
            split_path.parent.mkdir(parents=True, exist_ok=True)
            with split_path.open("w") as fh:
                json.dump({"train": [], "val": [], "test": []}, fh, indent=2)
        return {
            "electrode": electrode_name,
            "namespace": namespace,
            "allowed_channels": (),
            "processed_blocks": 0,
            "processed_trials": 0,
            "quarantined_blocks": 0,
            "feature_rows": 0,
            "selected_recordings": 0,
        }

    summary_df = summarize_recordings(manifest)
    channel_subset = tuple(str(channel) for channel in (allowed_channels or derive_stable_channel_subset(summary_df)))
    filtered_summary = filter_summary_for_channels(summary_df, channel_subset)
    if selected_recording_ids is not None:
        filtered_summary = filtered_summary[filtered_summary["recording_id"].isin(selected_recording_ids)].reset_index(drop=True)

    detector = JaylaDetector(JaylaConfig())
    blocks_rows: list[dict[str, object]] = []
    trials_rows: list[dict[str, object]] = []
    features_rows: list[dict[str, object]] = []
    quarantine_rows: list[dict[str, object]] = []

    blocks_dir = config.get_processed_blocks_dir(namespace)
    trials_dir = config.get_processed_trials_dir(namespace)
    block_index = 0
    trial_index = 0

    manifest_groups = {recording_id: group for recording_id, group in manifest.groupby("recording_id", sort=True)}

    for _, summary_row in filtered_summary.sort_values(["word", "session_token"]).iterrows():
        recording_id = str(summary_row["recording_id"])
        block_index += 1
        block_id = f"block_{block_index:06d}"
        block, error = load_block_from_manifest_rows(manifest_groups[recording_id], channel_subset)
        if error is not None or block is None:
            quarantine_rows.append(
                {
                    "recording_id": recording_id,
                    "electrode": electrode_name,
                    "word": summary_row["word"],
                    "session_token": summary_row["session_token"],
                    "reason": error or "unknown error",
                }
            )
            continue

        block_file = blocks_dir / f"{block_id}.npy"
        np.save(block_file, block)
        blocks_rows.append(
            {
                "block_id": block_id,
                "recording_id": recording_id,
                "electrode": electrode_name,
                "word": summary_row["word"],
                "session_token": summary_row["session_token"],
                "block_file": str(block_file.relative_to(config.PROJECT_ROOT)),
                "channel_count": int(block.shape[0]),
                "channel_ids": ",".join(channel_subset),
                "length_samples": int(block.shape[1]),
                "sample_rate": config.SAMPLING_RATE,
                "status": "processed",
            }
        )

        segmented = segment_multichannel_block(block, detector=detector)
        for onset, offset in segmented["segments"]:
            onset = int(max(0, onset))
            offset = int(min(block.shape[1], offset))
            if offset <= onset:
                continue

            trial_index += 1
            trial_id = f"trial_{trial_index:06d}"
            trial = block[:, onset:offset]
            trial_file = trials_dir / f"{trial_id}.npy"
            np.save(trial_file, trial)

            trials_rows.append(
                {
                    "trial_id": trial_id,
                    "block_id": block_id,
                    "recording_id": recording_id,
                    "electrode": electrode_name,
                    "word": summary_row["word"],
                    "session_token": summary_row["session_token"],
                    "trial_file": str(trial_file.relative_to(config.PROJECT_ROOT)),
                    "start_sample": onset,
                    "end_sample": offset,
                    "length_samples": int(offset - onset),
                }
            )

            feature_row = {
                "trial_id": trial_id,
                "block_id": block_id,
                "recording_id": recording_id,
                "electrode": electrode_name,
                "word": summary_row["word"],
                "session_token": summary_row["session_token"],
            }
            feature_row.update(extract_features(trial))
            features_rows.append(feature_row)

    blocks_df = pd.DataFrame(blocks_rows)
    trials_df = pd.DataFrame(trials_rows)
    features_df = pd.DataFrame(features_rows)
    quarantine_df = pd.DataFrame(quarantine_rows)

    _save_pipeline_outputs(namespace, manifest, summary_df, blocks_df, trials_df, features_df, quarantine_df)

    if not trials_df.empty:
        by_block = materialize_trial_splits(trials_df, "block_id", seed=config.SPLIT_SEED)
        by_session = materialize_trial_splits(
            trials_df,
            "session_token",
            seed=config.SPLIT_SEED,
            label_column="word",
        )
    else:
        by_block = {"train": [], "val": [], "test": []}
        by_session = {"train": [], "val": [], "test": []}

    with config.get_by_block_split_path(namespace).open("w") as fh:
        json.dump(by_block, fh, indent=2)
    with config.get_by_session_split_path(namespace).open("w") as fh:
        json.dump(by_session, fh, indent=2)

    return {
        "electrode": electrode_name,
        "namespace": namespace,
        "allowed_channels": channel_subset,
        "processed_blocks": len(blocks_df),
        "processed_trials": len(trials_df),
        "quarantined_blocks": len(quarantine_df),
        "feature_rows": len(features_df),
        "selected_recordings": len(filtered_summary),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--electrode", default=config.DEFAULT_ELECTRODE)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--channels", default=None, help="Comma-separated channel ids to force")
    args = parser.parse_args()

    channels = tuple(part.strip() for part in args.channels.split(",") if part.strip()) if args.channels else None
    summary = run_pipeline(
        electrode=args.electrode,
        data_dir=args.data_dir,
        namespace=args.namespace,
        allowed_channels=channels,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
