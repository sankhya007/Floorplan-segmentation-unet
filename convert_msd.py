import os
import numpy as np
import cv2
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
INPUT_BASE = "used_datasets/msd/modified-swiss-dwellings-v2"
OUT_IMG    = "dataset/images"
OUT_MASK   = "dataset/masks"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_MASK, exist_ok=True)


def convert_split(split):

    struct_dir = os.path.join(INPUT_BASE, split, "struct_in")

    if not os.path.exists(struct_dir):
        print(f"Skipping {split} — struct_in not found")
        return

    files = sorted([f for f in os.listdir(struct_dir) if f.endswith(".npy")])
    print(f"\nProcessing MSD {split} — {len(files)} files...")

    for fname in tqdm(files):
        idx = fname.replace(".npy", "")

        struct_path = os.path.join(struct_dir, fname)
        stack = np.load(struct_path)

        wall_mask = stack[..., 0].astype(np.uint8)

        if np.sum(wall_mask) == 0:
            print(f"  EMPTY MASK: {fname}")
            continue

        h, w = wall_mask.shape

        img = np.ones((h, w, 3), dtype=np.uint8) * 255
        img[wall_mask == 1] = [0, 0, 0]

        binary_mask = (wall_mask == 0).astype(np.uint8) * 255

        out_name = f"msd_{split}_{idx}.png"

        cv2.imwrite(os.path.join(OUT_IMG,  out_name), img)
        cv2.imwrite(os.path.join(OUT_MASK, out_name), binary_mask)


for split in ["train", "test"]:
    convert_split(split)

print("\n MSD conversion DONE.")
print("Both datasets are now merged in dataset/images and dataset/masks")