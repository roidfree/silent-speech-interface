"""Simple data loader for .mat EMG files.

Expect .mat files that contain arrays for EMG channels, or a key you can adapt.
"""
from typing import List, Tuple
import os
import scipy.io
import numpy as np


def load_mat_folder(folder: str, key: str = None) -> List[np.ndarray]:
    """Load all .mat files in a folder and return a list of arrays.

    Parameters
    - folder: path to folder containing .mat files
    - key: optional key inside the .mat structure to extract (if None, tries common keys)
    """
    arrays = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith('.mat'):
            continue
        path = os.path.join(folder, fname)
        mat = scipy.io.loadmat(path)
        if key and key in mat:
            arrays.append(np.asarray(mat[key]))
        else:
            # heuristics: pick the first ndarray value with 1-3 dims
            found = None
            for v in mat.values():
                if isinstance(v, np.ndarray):
                    found = v
                    break
            if found is None:
                raise ValueError(f"No ndarray found in {path}")
            arrays.append(np.asarray(found))
    return arrays


def load_dataset(processed_dir: str) -> Tuple[List[np.ndarray], List[int]]:
    """Load dataset arranged in subfolders `yes/` and `no/` under processed_dir.

    Returns (X, y) where X is list of arrays and y is list of labels (1 for yes, 0 for no).
    """
    X = []
    y = []
    yes_dir = os.path.join(processed_dir, 'yes')
    no_dir = os.path.join(processed_dir, 'no')
    if os.path.exists(yes_dir):
        for arr in load_mat_folder(yes_dir):
            X.append(arr)
            y.append(1)
    if os.path.exists(no_dir):
        for arr in load_mat_folder(no_dir):
            X.append(arr)
            y.append(0)
    return X, y
