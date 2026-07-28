
import torch
import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm
import segmentation_models_pytorch as smp

"""
Tiled interference script for S.T.I.T.C.H, to parse floorplans 

Usage:
  python predict_tiled_tta.py --image path/to/floorplan.jpg 

Optional:
  --model   path to weights file (default: unet.pth)
  --stride  patch stride in pixels (default: 128, lower = slower but smoother)
  --output  output filename (default: stitched_mask.png)

Use case: 
    use this script when using the TTA varient trained weight to parse 
    the floorplan, model size would be ~90Mb. This model is a lot better
    in scanning the unnecessary noise and leave that out of the binary 
    mask. 

    basically what it doing is scanning a single image 4 times rather than
    scanning it conce and then giving out the weighted average of the 4 
    combined masks that it is getting from the parsing that is how this script
    can actually acheave better noise reduction from the final image. 

Model name in Hugging face: 
    unet.pth
    
Training sequence: 
    use cell - 1-4, 5.2, 6, 7.2, 8
"""

# ------------------------------ CONFIG ------------------------------
PATCH_SIZE = 256
# --------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",  required=True,               help="Path to input floorplan image")
    p.add_argument("--model",  default="unet.pth",          help="Path to model weights")
    p.add_argument("--stride", type=int, default=128,       help="Patch stride (lower = smoother)")
    p.add_argument("--output", default="stitched_mask.png", help="Output filename")
    return p.parse_args()


def get_model():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,   # no download needed — we load our own weights
        in_channels=3,
        classes=1,
    )


def preprocess_patch(patch, device):
    patch = patch.astype(np.float32) / 255.0
    patch = np.transpose(patch, (2, 0, 1))
    return torch.from_numpy(patch).unsqueeze(0).to(device)


def create_weight_map(size):
    y, x = np.ogrid[-1:1:size*1j, -1:1:size*1j]
    return np.exp(-(x**2 + y**2) * 4).astype(np.float32)


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model weights not found: {args.model}")
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    model = get_model()
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded: {args.model}")

    img = cv2.imread(args.image)
    if img is None:
        raise ValueError(f"Could not read image: {args.image}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    H, W, _ = img.shape
    print(f"Original H, W: {H}, {W}")

    # 5% reflective padding — reduces edge artifacts
    pad_h = int(0.05 * H)
    pad_w = int(0.05 * W)
    img = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REFLECT_101)
    padded_H, padded_W, _ = img.shape

    weight_map = create_weight_map(PATCH_SIZE)
    final_mask = np.zeros((padded_H, padded_W), dtype=np.float32)
    weight_sum = np.zeros((padded_H, padded_W), dtype=np.float32)

    STRIDE = args.stride
    y_positions = list(range(0, padded_H - PATCH_SIZE, STRIDE))
    x_positions = list(range(0, padded_W - PATCH_SIZE, STRIDE))
    if y_positions[-1] != padded_H - PATCH_SIZE:
        y_positions.append(padded_H - PATCH_SIZE)
    if x_positions[-1] != padded_W - PATCH_SIZE:
        x_positions.append(padded_W - PATCH_SIZE)

    total_patches = len(y_positions) * len(x_positions)

    # TTA: original + 3 flips, averaged
    tta_pairs = [
        (lambda x: x,                                           lambda x: x),
        (lambda x: np.flip(x, axis=0).copy(),                  lambda x: np.flip(x, axis=0).copy()),
        (lambda x: np.flip(x, axis=1).copy(),                  lambda x: np.flip(x, axis=1).copy()),
        (lambda x: np.flip(np.flip(x, axis=0), axis=1).copy(), lambda x: np.flip(np.flip(x, axis=0), axis=1).copy()),
    ]

    print(f"Running inference on {total_patches} patches x {len(tta_pairs)} TTA variants...")

    pbar = tqdm(total=total_patches, desc="Patching", unit="patch",
                bar_format="{l_bar}{bar:30}{r_bar}")

    for y1 in y_positions:
        for x1 in x_positions:
            patch = img[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE]
            if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
                ph = PATCH_SIZE - patch.shape[0]
                pw = PATCH_SIZE - patch.shape[1]
                patch = cv2.copyMakeBorder(patch, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)

            patch_pred = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)

            for aug, inv in tta_pairs:
                tensor = preprocess_patch(aug(patch), device)
                with torch.no_grad():
                    pred = model(tensor)
                pred = torch.sigmoid(pred).squeeze().cpu().numpy()
                pred = np.clip(pred, 0.05, 0.95)
                patch_pred += inv(pred)

            patch_pred /= len(tta_pairs)

            final_mask[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE] += patch_pred * weight_map
            weight_sum[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE] += weight_map

            pbar.set_postfix(y=y1, x=x1)
            pbar.update(1)

    pbar.close()

    weight_sum[weight_sum == 0] = 1e-8
    final_mask = final_mask / weight_sum

    # remove padding
    final_mask = final_mask[pad_h:pad_h + H, pad_w:pad_w + W]

    cv2.imwrite("debug_raw_mask.png", (final_mask * 255).astype(np.uint8))
    print("Debug mask saved: debug_raw_mask.png")

    # Otsu threshold — adapts to each image's brightness distribution
    raw_uint8 = (final_mask * 255).astype(np.uint8)
    otsu_thresh, binary_mask = cv2.threshold(raw_uint8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"Otsu threshold: {otsu_thresh}/255")

    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.dilate(binary_mask, np.ones((2, 2), np.uint8), iterations=1)

    # remove small blobs (text / symbols)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        area   = stats[i, cv2.CC_STAT_AREA]
        w_box  = stats[i, cv2.CC_STAT_WIDTH]
        h_box  = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(w_box, h_box) / (min(w_box, h_box) + 1e-6)
        if area < 80 and aspect < 3.0:
            continue
        cleaned[labels == i] = 1
    binary_mask = cleaned

    cv2.imwrite(args.output, binary_mask * 255)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()