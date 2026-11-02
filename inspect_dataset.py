import os
import glob
import numpy as np

def inspect_image(path):
    # Load .npy file
    try:
        data = np.load(path)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

    return {
        "filename": os.path.basename(path),
        "shape": data.shape, # (H, W, C) or (H, W) or (C, H, W)
        "dtype": str(data.dtype),
        "min": float(data.min()),
        "max": float(data.max()),
        "mean": float(data.mean()),
        "std": float(data.std())
    }

gt_dir = "train/GT"
noisy_dir = "train/NoisyLR"

gt_images = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))
noisy_images = sorted(glob.glob(os.path.join(noisy_dir, "*.npy")))

if not gt_images or not noisy_images:
    print("No .npy images found!")
    exit(1)

print(f"Total GT images: {len(gt_images)}")
print(f"Total NoisyLR images: {len(noisy_images)}")

# Check first pair
count = 0
for i in range(min(5, len(gt_images))):
    gt_info = inspect_image(gt_images[i])
    noisy_info = inspect_image(noisy_images[i])
    
    if gt_info is None or noisy_info is None:
        continue

    print(f"\n--- Image Pair {i} ---")
    print(f"GT: {gt_info['filename']} | Shape: {gt_info['shape']} | DType: {gt_info['dtype']} | Range: [{gt_info['min']:.2f}, {gt_info['max']:.2f}]")
    print(f"LR: {noisy_info['filename']} | Shape: {noisy_info['shape']} | DType: {noisy_info['dtype']} | Range: [{noisy_info['min']:.2f}, {noisy_info['max']:.2f}]")
    
    # Calculate scale factor based on dimensions
    # Assuming shape is (H, W) or (H, W, C)
    gh, gw = gt_info['shape'][:2]
    nh, nw = noisy_info['shape'][:2]
    scale_h = gh / nh
    scale_w = gw / nw
    print(f"Scale Factor: H={scale_h:.1f}x, W={scale_w:.1f}x")
