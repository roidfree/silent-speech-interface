"""A preprocessor thats follows Jayla's preprocessing, segmentation, and feature extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from .. import config


@dataclass(frozen=True)
class JaylaConfig:
    """Config values that mirror Jayla's checked-in code."""

    fs: int = config.SAMPLING_RATE
    notch_freqs: tuple[float, ...] = tuple(config.NOTCH_FREQS)
    notch_q: float = config.NOTCH_Q
    bp_low: float = config.BP_LOW
    bp_high: float = config.BP_HIGH
    bp_order: int = config.BP_ORDER
    window_ms: float = config.WINDOW_MS
    hop_ms: float = config.HOP_MS
    smoothing_cutoff_hz: float = config.SMOOTHING_CUTOFF_HZ
    threshold_decay: float = config.THRESHOLD_DECAY
    threshold_min_ratio: float = config.THRESHOLD_MIN_RATIO
    ignore_start_ms: float = config.IGNORE_START_MS
    ignore_end_ms: float = config.IGNORE_END_MS
    min_duration_ms: float = config.MIN_DURATION_MS
    merge_gap_ms: float = config.MERGE_GAP_MS
    onset_buffer_ms: float = config.ONSET_BUFFER_MS
    offset_buffer_ms: float = config.OFFSET_BUFFER_MS
    threshold_floor: float = config.THRESHOLD_FLOOR
    min_active_ch: int = config.MIN_ACTIVE_CH


class JaylaPreprocessor:
    """Jayla's exact single-channel preprocessing btw! 
    Also with a multichannel helper for block-level processing."""

    def __init__(self, params: JaylaConfig | None = None):
        self.params = params or JaylaConfig()

    def remove_dc(self, signal: np.ndarray) -> np.ndarray:
        x = np.asarray(signal).squeeze().astype(float)
        return x - np.mean(x)

    def notch(self, signal: np.ndarray) -> np.ndarray:
        x = np.asarray(signal).squeeze().astype(float).copy()
        for f0 in self.params.notch_freqs:
            b, a = iirnotch(w0=f0, Q=self.params.notch_q, fs=self.params.fs)
            x = filtfilt(b, a, x)
        return x

    def bandpass(self, signal: np.ndarray) -> np.ndarray:
        x = np.asarray(signal).squeeze().astype(float)
        b, a = butter(
            self.params.bp_order,
            [self.params.bp_low, self.params.bp_high],
            btype="bandpass",
            fs=self.params.fs,
        )
        return filtfilt(b, a, x)

    def process(self, raw_signal: np.ndarray) -> np.ndarray:
        signal = self.remove_dc(raw_signal)
        signal = self.notch(signal)
        return self.bandpass(signal)


class JaylaDetector:
    """Jayla-exact single-channel detector with a multichannel fusion helper."""

    def __init__(self, params: JaylaConfig | None = None, method: str = "spc"):
        self.params = params or JaylaConfig()
        self.method = method
        self.window_samples = int(self.params.window_ms * self.params.fs / 1000)
        self.hop_samples = int(self.params.hop_ms * self.params.fs / 1000)
        self.onset_buffer_samples = int(self.params.onset_buffer_ms * self.params.fs / 1000)
        self.offset_buffer_samples = int(self.params.offset_buffer_ms * self.params.fs / 1000)
        self.ignore_windows = int(self.params.ignore_start_ms / self.params.hop_ms)
        self.ignore_end_windows = int(self.params.ignore_end_ms / self.params.hop_ms)
        self.min_duration_frames = int(self.params.min_duration_ms / self.params.hop_ms)
        self.merge_gap_frames = int(self.params.merge_gap_ms / self.params.hop_ms)
        self.preprocessor = JaylaPreprocessor(self.params)

    def preprocess(self, raw_signal: np.ndarray) -> np.ndarray:
        return self.preprocessor.process(raw_signal)

    def compute_rms_envelope(self, signal: np.ndarray) -> np.ndarray:
        n_samples = len(signal)
        if n_samples < self.window_samples:
            return np.array([0.0], dtype=float)

        n_windows = (n_samples - self.window_samples) // self.hop_samples + 1
        rms_env = np.zeros(n_windows, dtype=float)
        for i in range(n_windows):
            start = i * self.hop_samples
            end = start + self.window_samples
            rms_env[i] = np.sqrt(np.mean(signal[start:end] ** 2))
        return rms_env

    def smooth_envelope(self, envelope: np.ndarray) -> np.ndarray:
        if len(envelope) < 2:
            return envelope

        frame_rate = self.params.fs / self.params.hop_ms
        nyquist = frame_rate / 2
        b, a = butter(
            self.params.bp_order,
            self.params.smoothing_cutoff_hz / nyquist,
            btype="low",
        )
        smoothed = filtfilt(b, a, envelope)
        return np.maximum(smoothed, 0.0)

    def _get_initial_threshold(self, metric: np.ndarray) -> float:
        if self.method == "spc":
            if len(metric) == 0:
                return 1e-10
            mu_b = np.percentile(metric, 10)
            baseline = metric[metric <= mu_b]
            sigma_b = np.std(baseline) if len(baseline) > 0 else 0.0
            thresh = mu_b + (5.5 * sigma_b)
        else:
            mu_b = np.mean(metric)
            sigma_b = np.std(metric)
            thresh = mu_b + (4.5 * sigma_b)
        return max(float(thresh), self.params.threshold_floor)

    def threshold_activity(self, smoothed_env: np.ndarray) -> tuple[np.ndarray, float]:
        if len(smoothed_env) < 2:
            return np.zeros_like(smoothed_env, dtype=bool), 0.0

        env = smoothed_env.copy()
        if self.ignore_windows < len(env):
            env[: self.ignore_windows] = 0.0
        if 0 < self.ignore_end_windows < len(env):
            env[-self.ignore_end_windows :] = 0.0

        current_threshold = self._get_initial_threshold(env)
        decay_floor = current_threshold * self.params.threshold_min_ratio

        active_mask = env > current_threshold
        for _ in range(100):
            if np.any(active_mask):
                break
            current_threshold *= self.params.threshold_decay
            if current_threshold < decay_floor:
                break
            active_mask = env > current_threshold

        if self.ignore_windows < len(active_mask):
            active_mask[: self.ignore_windows] = False
        if 0 < self.ignore_end_windows < len(active_mask):
            active_mask[-self.ignore_end_windows :] = False

        return active_mask.astype(bool), float(current_threshold)

    def segments_from_active_mask(self, active_mask: np.ndarray) -> list[tuple[int, int]]:
        if len(active_mask) < 2:
            return []

        segments: list[tuple[int, int]] = []
        state = "INACTIVE"
        onset = 0
        below_count = 0

        end_limit = len(active_mask) - self.ignore_end_windows if self.ignore_end_windows > 0 else len(active_mask)
        end_limit = max(end_limit, self.ignore_windows)

        for i in range(self.ignore_windows, end_limit):
            is_active = bool(active_mask[i])
            if state == "INACTIVE":
                if is_active:
                    state = "ACTIVE"
                    onset = i
                    below_count = 0
            else:
                if not is_active:
                    below_count += 1
                    if below_count >= self.min_duration_frames:
                        segments.append((onset, i - below_count))
                        state = "INACTIVE"
                else:
                    below_count = 0

        if state == "ACTIVE":
            segments.append((onset, end_limit - 1))

        if len(segments) > 1:
            merged = [segments[0]]
            for onset_next, offset_next in segments[1:]:
                prev_onset, prev_offset = merged[-1]
                gap = onset_next - prev_offset
                if gap <= self.merge_gap_frames:
                    merged[-1] = (prev_onset, offset_next)
                else:
                    merged.append((onset_next, offset_next))
            segments = merged
        return segments

    def convert_to_sample_space(self, segments_env: list[tuple[int, int]]) -> list[tuple[int, int]]:
        sample_segments: list[tuple[int, int]] = []
        for onset_env, offset_env in segments_env:
            t_start = max(0, onset_env * self.hop_samples - self.onset_buffer_samples)
            t_end = offset_env * self.hop_samples + self.offset_buffer_samples
            sample_segments.append((t_start, t_end))
        return sample_segments

    def detect_channel(self, signal: np.ndarray) -> dict[str, np.ndarray | float | list[tuple[int, int]]]:
        clean_signal = self.preprocess(signal)
        rms_env = self.compute_rms_envelope(clean_signal)
        smooth_env = self.smooth_envelope(rms_env)
        active_mask, final_threshold = self.threshold_activity(smooth_env)
        segments_env = self.segments_from_active_mask(active_mask)
        segments = self.convert_to_sample_space(segments_env)
        return {
            "clean_signal": clean_signal,
            "rms_envelope": rms_env,
            "smooth_envelope": smooth_env,
            "active_mask": active_mask,
            "final_threshold": final_threshold,
            "segments_env": segments_env,
            "segments": segments,
        }

    def detect(self, signal: np.ndarray, return_metadata: bool = False) -> dict[str, np.ndarray | float | int | list[tuple[int, int]]]:
        channel = self.detect_channel(signal)
        n_samples = len(np.asarray(signal).squeeze())
        labels = np.zeros(n_samples, dtype=int)
        for onset, offset in channel["segments"]:
            labels[onset : min(offset, n_samples)] = 1

        results = {
            "segments": channel["segments"],
            "labels": labels,
            "final_threshold": channel["final_threshold"],
            "n_segments": len(channel["segments"]),
            "clean_signal": channel["clean_signal"],
        }
        if return_metadata:
            results["rms_envelope"] = channel["rms_envelope"]
            results["smooth_envelope"] = channel["smooth_envelope"]
            results["derivative"] = np.abs(np.diff(channel["smooth_envelope"]))
            results["active_mask"] = channel["active_mask"]
        return results


def fuse_channel_masks(channel_masks: np.ndarray, min_active_ch: int = config.MIN_ACTIVE_CH) -> np.ndarray:
    """Fuse per-channel boolean masks into one block-level mask."""
    if channel_masks.ndim != 2:
        raise ValueError(f"Expected a 2D mask array, got shape {channel_masks.shape}")
    return np.sum(channel_masks.astype(int), axis=0) >= min_active_ch


def segment_multichannel_block(
    block: np.ndarray,
    detector: JaylaDetector | None = None,
) -> dict[str, np.ndarray | list[tuple[int, int]] | list[float]]:
    """Run Jayla-exact preprocessing per channel, then fuse active masks."""
    signal = np.asarray(block)
    if signal.ndim != 2:
        raise ValueError(f"Expected block shape (channels, time), got {signal.shape}")

    detector = detector or JaylaDetector()
    channel_results = [detector.detect_channel(signal[ch]) for ch in range(signal.shape[0])]
    filtered_block = np.stack([res["clean_signal"] for res in channel_results], axis=0)
    channel_masks = np.stack([res["active_mask"] for res in channel_results], axis=0)
    fused_mask = fuse_channel_masks(channel_masks, detector.params.min_active_ch)
    fused_segments_env = detector.segments_from_active_mask(fused_mask)
    fused_segments = detector.convert_to_sample_space(fused_segments_env)

    labels = np.zeros(signal.shape[1], dtype=int)
    for onset, offset in fused_segments:
        labels[onset : min(offset, signal.shape[1])] = 1

    return {
        "filtered_block": filtered_block,
        "channel_thresholds": [float(res["final_threshold"]) for res in channel_results],
        "channel_masks": channel_masks,
        "fused_mask": fused_mask,
        "segments_env": fused_segments_env,
        "segments": fused_segments,
        "labels": labels,
    }


def _frame_signal(signal: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    """Frame a 1D signal, keeping at least one frame for short trials."""
    x = np.asarray(signal).squeeze().astype(float)
    if len(x) == 0:
        return np.zeros((1, frame_len), dtype=float)
    if len(x) < frame_len:
        return x[np.newaxis, :]
    starts = range(0, len(x) - frame_len + 1, hop_len)
    return np.stack([x[start : start + frame_len] for start in starts], axis=0)


def _compute_window_features(frames: np.ndarray) -> dict[str, np.ndarray]:
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    mav = np.mean(np.abs(frames), axis=1)
    var = np.var(frames, axis=1)
    wl = np.sum(np.abs(np.diff(frames, axis=1)), axis=1)
    sd = np.std(frames, axis=1)
    return {"rms": rms, "mav": mav, "var": var, "wl": wl, "sd": sd}


def extract_features(
    trial: np.ndarray,
    preprocessor: JaylaPreprocessor | None = None,
    params: JaylaConfig | None = None,
) -> dict[str, float]:
    """Extract flattened XGBoost-ready features from one multichannel trial."""
    signal = np.asarray(trial)
    if signal.ndim != 2:
        raise ValueError(f"Expected trial shape (channels, time), got {signal.shape}")

    params = params or JaylaConfig()
    preprocessor = preprocessor or JaylaPreprocessor(params)
    frame_len = int(config.FEATURE_WINDOW_MS * params.fs / 1000)
    hop_len = int(config.FEATURE_HOP_MS * params.fs / 1000)

    features: dict[str, float] = {
        "length_samples": float(signal.shape[1]),
        "trial_duration_s": float(signal.shape[1] / params.fs),
    }

    for channel_idx in range(signal.shape[0]):
        filtered = preprocessor.process(signal[channel_idx])
        frames = _frame_signal(filtered, frame_len, hop_len)
        feature_map = _compute_window_features(frames)
        prefix = f"ch{channel_idx + 1}"
        for feature_name, values in feature_map.items():
            features[f"{prefix}_{feature_name}_mean"] = float(np.mean(values))
            features[f"{prefix}_{feature_name}_std"] = float(np.std(values))
            features[f"{prefix}_{feature_name}_max"] = float(np.max(values))
    return features
