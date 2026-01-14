"""Preprocessing helpers: filtering and feature extraction placeholders."""
from typing import Any
import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(signal: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def extract_features(epoch: np.ndarray) -> np.ndarray:
    """Extract simple features from an epoch (multi-channel or 1D).

    Currently computes RMS per channel and concatenates.
    """
    arr = np.asarray(epoch)
    if arr.ndim == 1:
        rms = np.sqrt(np.mean(arr ** 2))
        return np.array([rms])
    else:
        # assume shape (channels, samples) or (samples, channels)
        if arr.shape[0] < arr.shape[1]:
            # channels, samples
            chans = arr
        else:
            chans = arr.T
        feats = [np.sqrt(np.mean(ch ** 2)) for ch in chans]
        return np.asarray(feats)
