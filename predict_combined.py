import torch
import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm
from model import UNet

"""
Auto-switching inference script for S.T.I.T.C.H floorplan segmentation.

Tries direct inference first (faster, no seams).
Falls back to tiled stitching if the image is too large for a single pass.

Usage:
  python predict_auto.py --image path/to/floorplan.jpg

Optional:
  --model      path to weights file (default: unet.pth)
  --stride     patch stride for tiled mode (default: 128)
  --output     output filename (default: stitched_mask.png)
  --threshold  binarization threshold 0.0-1.0 (default: 0.50)
  --max-direct max pixels for direct inference, e.g. 1500*1500 (default: 2250000)
"""

PATCH_SIZE  = 256
MAX_DIRECT  = 2_250_000   # ~1500x1500 — tune this to your VRAM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",      required=True,               help="Path to input floorplan image")
    p.add_argument("--model",      default="unet.pth",          help="Path to model weights")
    p.add_argument("--stride",     type=int, default=128,       help="Patch stride for tiled mode")
    p.add_argument("--output",     default="stitched_mask.png", help="Output filename")
    p.add_argument("--threshold",  type=float, default=0.50,    help="Binarization threshold")
    p.add_argument("--max-direct", type=int, default=MAX_DIRECT,
                   help="Max pixel count for direct inference (default: 2250000)")
    return p.parse_args()


def load_model(path, device):
    model = UNet()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def create_weight_map(size):
    y, x = np.ogrid[-1:1:size*1j, -1:1:size*1j]
    return np.exp(-(x**2 + y**2) * 4).astype(np.float32)


def preprocess(img_rgb, device):
    t = img_rgb.astype(np.float32) / 255.0
    t = torch.from_numpy(t).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def postprocess(binary_mask, H, W):
    """Remove small blobs that are likely text / symbols."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)
    min_area = (H * W) * 0.00005
    for i in range(1, num_labels):
        area  = stats[i, cv2.CC_STAT_AREA]
        w_box = stats[i, cv2.CC_STAT_WIDTH]
        h_box = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(w_box, h_box) / (min(w_box, h_box) + 1e-6)
        if area < min_area and aspect < 3.0:
            continue
        cleaned[labels == i] = 1
    return cleaned


# ------------------------------------------------------------------
# MODE 1 — direct inference (no tiling)
# ------------------------------------------------------------------
def infer_direct(model, img_rgb, device, threshold):
    H, W = img_rgb.shape[:2]

    # pad to multiples of 32 so the UNet encoder/decoder sizes stay aligned
    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    padded = cv2.copyMakeBorder(img_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

    tensor = preprocess(padded, device)
    with torch.no_grad():
        pred = model(tensor)
        pred = torch.sigmoid(pred).squeeze().cpu().numpy()

    pred = pred[:H, :W]   # strip padding
    return pred


# ------------------------------------------------------------------
# MODE 2 — tiled inference with Gaussian-weighted blending
# ------------------------------------------------------------------
def infer_tiled(model, img_rgb, device, stride, threshold):
    H, W = img_rgb.shape[:2]

    pad_h = int(0.05 * H)
    pad_w = int(0.05 * W)
    img_padded = cv2.copyMakeBorder(img_rgb, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REFLECT_101)
    pH, pW = img_padded.shape[:2]

    weight_map = create_weight_map(PATCH_SIZE)
    final_mask = np.zeros((pH, pW), dtype=np.float32)
    weight_sum = np.zeros((pH, pW), dtype=np.float32)

    y_pos = list(range(0, max(1, pH - PATCH_SIZE + 1), stride))
    x_pos = list(range(0, max(1, pW - PATCH_SIZE + 1), stride))
    if y_pos[-1] != max(0, pH - PATCH_SIZE):
        y_pos.append(max(0, pH - PATCH_SIZE))
    if x_pos[-1] != max(0, pW - PATCH_SIZE):
        x_pos.append(max(0, pW - PATCH_SIZE))

    total = len(y_pos) * len(x_pos)
    pbar  = tqdm(total=total, desc="Tiling", unit="patch")

    for y1 in y_pos:
        for x1 in x_pos:
            patch = img_padded[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE]
            if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
                ph = PATCH_SIZE - patch.shape[0]
                pw = PATCH_SIZE - patch.shape[1]
                patch = cv2.copyMakeBorder(patch, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)

            tensor = preprocess(patch, device)
            with torch.no_grad():
                pred = model(tensor)
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()
            pred = np.clip(pred, 0.01, 0.99)

            final_mask[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE] += pred * weight_map
            weight_sum[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE] += weight_map
            pbar.update(1)

    pbar.close()

    weight_sum[weight_sum == 0] = 1e-8
    final_mask = final_mask / weight_sum
    final_mask = final_mask[pad_h:pad_h + H, pad_w:pad_w + W]

    return final_mask


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Weights not found: {args.model}")
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    model = load_model(args.model, device)
    print(f"Model loaded: {args.model}")

    img = cv2.imread(args.image)
    if img is None:
        raise ValueError(f"Could not read image: {args.image}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W    = img_rgb.shape[:2]
    pixels  = H * W
    print(f"Image size: {W}x{H}  ({pixels:,} px)")

    # --- decide mode ---
    if pixels <= args.max_direct:
        print(f"Direct inference  (under {args.max_direct:,} px limit)")
        prob_map = infer_direct(model, img_rgb, device, args.threshold)
        mode = "direct"
    else:
        print(f"Image too large for direct inference — switching to tiled mode (stride={args.stride})")
        prob_map = infer_tiled(model, img_rgb, device, args.stride, args.threshold)
        mode = "tiled"

    # save raw probability map for debugging
    cv2.imwrite("debug_raw_mask.png", (prob_map * 255).astype(np.uint8))
    print("Debug mask saved: debug_raw_mask.png")

    # binarize + clean
    binary = (prob_map > args.threshold).astype(np.uint8)
    binary = postprocess(binary, H, W)

    cv2.imwrite(args.output, binary * 255)
    print(f"Saved: {args.output}  [{mode} mode]")


if __name__ == "__main__":
    main()