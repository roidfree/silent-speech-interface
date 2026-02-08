import os
import numpy as np

def inspect_array(arr, name, preview=10):
    print(f"\n=== {name} ===")
    print(f"Type: {type(arr)}")
    print(f"Dtype: {arr.dtype}")
    print(f"Shape: {arr.shape}")
    print(f"Dimensions: {arr.ndim}")

    if arr.size == 0:
        print("Array is empty.")
        return

    print(f"Min: {np.min(arr)}")
    print(f"Max: {np.max(arr)}")
    print(f"Mean: {np.mean(arr)}")
    print(f"Std: {np.std(arr)}")

    # Preview values
    if arr.ndim == 1:
        print(f"First {preview} values:\n{arr[:preview]}")
    elif arr.ndim == 2:
        print(f"First {preview} rows:\n{arr[:preview, :]}")
    else:
        print(f"Preview (flattened):\n{arr.flatten()[:preview]}")


def load_np(path, friendly_name):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    try:
        return np.load(path)
    except Exception as e:
        print(f"Error loading {friendly_name} at {path}: {e}")
        return None


yes_raw_path = "/Users/dameer/Downloads/yes_CNC_ch1.npy"
yes_segmented_path = "/Users/dameer/Downloads/yes_CNC_ch1_s1_X.npy"

yes_raw = load_np(yes_raw_path, "Raw YES signal (ch1)")
yes_segmented = load_np(yes_segmented_path, "Segmented YES samples (X)")

if yes_raw is not None:
    inspect_array(yes_raw, "Raw YES signal (ch1)")

if yes_segmented is not None:
    inspect_array(yes_segmented, "Segmented YES samples (X)")