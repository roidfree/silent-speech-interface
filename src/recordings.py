"""Electrode-aware recording ingestion and manifest helpers."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


AGAGCL_CHANNEL_RE = re.compile(r"_ch(?P<channel>\d+)\.npy$")
PEDOT_RAW_RE = re.compile(
    r"(?P<word>[^_]+)_(?P<label_token>[^_]+)_(?P<session_token>s\d+)_ch(?P<channel>\d+)\.npy$"
)


def _path_relative_to_project(path: Path) -> str:
    try:
        return str(path.relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(path)


def _load_sample_count(path: Path) -> tuple[int | None, str]:
    try:
        signal = np.load(path, allow_pickle=True)
    except Exception as exc:
        return None, f"load_error:{exc}"

    if not isinstance(signal, np.ndarray):
        return None, "invalid_array"
    if signal.dtype.kind not in "fiu":
        return None, f"invalid_dtype:{signal.dtype}"
    if signal.ndim == 1:
        sample_count = int(signal.shape[0])
    elif signal.ndim == 2 and signal.shape[0] == 1:
        sample_count = int(signal.shape[-1])
    else:
        return None, f"invalid_shape:{signal.shape}"
    if sample_count <= 0:
        return None, "empty_signal"
    return sample_count, "ok"


def _iter_agagcl_rows(root: Path, electrode: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for word_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for session_dir in sorted(path for path in word_dir.iterdir() if path.is_dir()):
            session_token = session_dir.name
            recording_id = f"{electrode}:{word_dir.name}:{session_token}"
            for path in sorted(session_dir.glob("*.npy")):
                match = AGAGCL_CHANNEL_RE.search(path.name)
                if match is None:
                    continue
                sample_count, qc_status = _load_sample_count(path)
                rows.append(
                    {
                        "recording_id": recording_id,
                        "electrode": electrode,
                        "word": word_dir.name,
                        "session_token": session_token,
                        "label_token": None,
                        "channel_id": match.group("channel"),
                        "path": _path_relative_to_project(path),
                        "sample_count": sample_count,
                        "qc_status": qc_status,
                        "is_suspicious_channel": False,
                    }
                )
    return rows


def _iter_pedot_rows(root: Path, electrode: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("*.npy")):
        match = PEDOT_RAW_RE.match(path.name)
        if match is None:
            continue
        channel_id = match.group("channel")
        sample_count, sample_qc = _load_sample_count(path)
        is_suspicious_channel = channel_id in config.PEDOT_SUSPICIOUS_CHANNEL_IDS
        qc_status = sample_qc
        if qc_status == "ok" and is_suspicious_channel:
            qc_status = "suspicious_channel"
        rows.append(
            {
                "recording_id": f"{electrode}:{match.group('word')}:{match.group('session_token')}",
                "electrode": electrode,
                "word": match.group("word"),
                "session_token": match.group("session_token"),
                "label_token": match.group("label_token"),
                "channel_id": channel_id,
                "path": _path_relative_to_project(path),
                "sample_count": sample_count,
                "qc_status": qc_status,
                "is_suspicious_channel": is_suspicious_channel,
            }
        )
    return rows


def build_recording_manifest(
    electrode: str,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Build a per-channel manifest for one electrode source."""
    electrode_name = config.normalize_electrode_name(electrode)
    root = Path(data_dir) if data_dir is not None else config.get_default_source_dir(electrode_name)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found for {electrode_name}: {root}")

    if electrode_name == "agagcl":
        rows = _iter_agagcl_rows(root, electrode_name)
    elif electrode_name == "pedot":
        rows = _iter_pedot_rows(root, electrode_name)
    else:
        raise KeyError(f"Unsupported electrode '{electrode}'")

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        return pd.DataFrame(
            columns=[
                "recording_id",
                "electrode",
                "word",
                "session_token",
                "label_token",
                "channel_id",
                "path",
                "sample_count",
                "qc_status",
                "is_suspicious_channel",
            ]
        )
    manifest["channel_id"] = manifest["channel_id"].astype(str)
    manifest["path"] = manifest["path"].astype(str)
    return manifest.sort_values(["word", "session_token", "channel_id", "path"]).reset_index(drop=True)


def summarize_recordings(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-channel manifest rows into recording-level QC."""
    if manifest_df.empty:
        return pd.DataFrame(
            columns=[
                "recording_id",
                "electrode",
                "word",
                "session_token",
                "label_token",
                "channel_count",
                "available_channels",
                "valid_channels",
                "sample_count_min",
                "sample_count_max",
                "has_length_mismatch",
                "suspicious_channel_count",
                "load_error_count",
                "qc_status",
                "is_usable",
            ]
        )

    rows: list[dict[str, object]] = []
    for recording_id, group in manifest_df.groupby("recording_id", sort=True):
        valid_rows = group[group["qc_status"] == "ok"].copy()
        valid_channels = sorted(valid_rows["channel_id"].astype(str).unique().tolist(), key=int)
        available_channels = sorted(group["channel_id"].astype(str).unique().tolist(), key=int)
        sample_counts = [int(value) for value in valid_rows["sample_count"].dropna().tolist()]
        sample_count_min = min(sample_counts) if sample_counts else None
        sample_count_max = max(sample_counts) if sample_counts else None
        has_length_mismatch = bool(sample_counts and sample_count_min != sample_count_max)
        suspicious_count = int(group["is_suspicious_channel"].astype(bool).sum())
        load_error_count = int((group["qc_status"] != "ok").sum())

        issues: list[str] = []
        if suspicious_count:
            issues.append("suspicious_channels")
        if load_error_count:
            issues.append("channel_qc_failures")
        if has_length_mismatch:
            issues.append("length_mismatch")
        if not valid_channels:
            issues.append("no_valid_channels")

        rows.append(
            {
                "recording_id": recording_id,
                "electrode": group["electrode"].iloc[0],
                "word": group["word"].iloc[0],
                "session_token": group["session_token"].iloc[0],
                "label_token": group["label_token"].dropna().iloc[0] if group["label_token"].notna().any() else None,
                "channel_count": len(available_channels),
                "available_channels": ",".join(available_channels),
                "valid_channels": ",".join(valid_channels),
                "sample_count_min": sample_count_min,
                "sample_count_max": sample_count_max,
                "has_length_mismatch": has_length_mismatch,
                "suspicious_channel_count": suspicious_count,
                "load_error_count": load_error_count,
                "qc_status": "ok" if not issues else ";".join(issues),
                "is_usable": not issues,
            }
        )
    return pd.DataFrame(rows).sort_values(["word", "session_token"]).reset_index(drop=True)


def channels_from_summary_value(value: str | None) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    return tuple(part for part in value.split(",") if part)


def derive_stable_channel_subset(summary_df: pd.DataFrame) -> tuple[str, ...]:
    """Return the largest deterministic channel subset present in all usable recordings."""
    usable = summary_df[summary_df["is_usable"].astype(bool)].copy()
    if usable.empty:
        return ()
    channel_sets = [set(channels_from_summary_value(value)) for value in usable["valid_channels"]]
    stable = sorted(set.intersection(*channel_sets), key=int) if channel_sets else []
    return tuple(stable)


def filter_summary_for_channels(
    summary_df: pd.DataFrame,
    required_channels: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    required = tuple(str(channel) for channel in required_channels or ())
    filtered = summary_df[summary_df["is_usable"].astype(bool)].copy()
    if not required:
        return filtered.reset_index(drop=True)
    keep_mask = []
    required_set = set(required)
    for value in filtered["valid_channels"]:
        keep_mask.append(required_set.issubset(set(channels_from_summary_value(value))))
    return filtered.loc[keep_mask].reset_index(drop=True)


def load_block_from_manifest_rows(
    rows: pd.DataFrame,
    allowed_channels: tuple[str, ...] | list[str],
) -> tuple[np.ndarray | None, str | None]:
    """Load one multichannel recording block from manifest rows."""
    selected_channels = [str(channel) for channel in allowed_channels]
    by_channel = {
        str(row["channel_id"]): Path(config.PROJECT_ROOT / str(row["path"]))
        for _, row in rows.iterrows()
        if str(row["channel_id"]) in selected_channels and row["qc_status"] == "ok"
    }
    missing = [channel for channel in selected_channels if channel not in by_channel]
    if missing:
        return None, f"missing channel(s): {', '.join(missing)}"

    signals: list[np.ndarray] = []
    lengths: set[int] = set()
    for channel in selected_channels:
        try:
            signal = np.load(by_channel[channel]).squeeze()
        except Exception as exc:
            return None, f"corrupt channel file {by_channel[channel].name}: {exc}"
        if np.asarray(signal).dtype.kind not in "fiu":
            return None, f"invalid dtype for {by_channel[channel].name}: {np.asarray(signal).dtype}"
        signal_arr = np.asarray(signal, dtype=np.float32)
        if signal_arr.ndim != 1:
            return None, f"invalid signal shape for {by_channel[channel].name}: {signal_arr.shape}"
        signals.append(signal_arr)
        lengths.add(int(signal_arr.shape[-1]))

    if len(lengths) != 1:
        return None, "channel lengths do not match"
    return np.stack(signals, axis=0), None
