import torch
import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm
from model import UNet

"""
Tiled inference script for S.T.I.T.C.H floorplan segmentation.

Usage:
  python predict_tiled.py --image path/to/floorplan.jpg

Optional:
  --model   path to weights file (default: unet.pth)
  --stride  patch stride in pixels (default: 128, lower = slower but smoother)
  --output  output filename (default: stitched_mask.png)

Use Case: 
    use this script while using the basic trained weight to parse 
    the floorplan, the trained weight size would be ~30Mb.

Model name in Hugging face: 
    unet_tta.pth

Training sequence: 
    use cell - 1-4, 5.1, 6, 7.1, 8
"""

# ------------------------------ CONFIG ------------------------------
PATCH_SIZE = 512
"""
this is the ammount of data the parser has access to while parsing 
a single portion/segment, the more the number the more time consuming 
it is gonna be and the smaller the littler context the parser has 
about its enviorment, so the parsing becomes shitty

N.B: make sure you change the default stride in the code or when parsing
an image using the code and keep that to 50% of the PATCH_SIZE or else
the parsing WILL BE INSANE...
"""
# --------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",  required=True,              help="Path to input floorplan image")
    p.add_argument("--model",  default="unet.pth",         help="Path to model weights")
    p.add_argument("--stride", type=int, default=128,      help="Patch stride (lower = smoother)")
    # here you can change the stride number by default it is 128, the bettre alternative is to use whatever you have in the patch size, the 50% of that(because that was kinda the benchmark to prove that this concept even works at the 1st place)
    p.add_argument("--output", default="stitched_mask.png", help="Output filename")
    return p.parse_args()


def preprocess_patch(patch, device):
    patch = patch.astype(np.float32) / 255.0
    patch = np.transpose(patch, (2, 0, 1))
    return torch.from_numpy(patch).unsqueeze(0).to(device)


def create_weight_map(size):
    y, x = np.ogrid[-1:1:size*1j, -1:1:size*1j]
    weight = np.exp(-(x**2 + y**2) * 4)
    return weight.astype(np.float32)


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model weights not found: {args.model}")
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    model = UNet()
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
    print(f"Padded H, W: {padded_H}, {padded_W}")

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
    print(f"Running inference on {total_patches} patches...")

    pbar = tqdm(total=total_patches, desc="Patching", unit="patch")

    for y1 in y_positions:
        for x1 in x_positions:
            patch = img[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE]
            if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
                ph = PATCH_SIZE - patch.shape[0]
                pw = PATCH_SIZE - patch.shape[1]
                patch = cv2.copyMakeBorder(patch, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)

            # here the segment f used to parse a single patch once and then give an output
            
            patch_tensor = preprocess_patch(patch, device)
            with torch.no_grad():
                pred = model(patch_tensor)
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()
            pred = np.clip(pred, 0.01, 0.99)

            # here we are trying out TTA(test time augmentaion) where it is going to parse the single patch more than once, 4 times to be exact and then give an weighted average of that parsing to make sure that the parsng is done right 

            # tta_pairs = [
            #     (lambda x: x,                                           lambda x: x),
            #     (lambda x: np.flip(x, axis=0).copy(),                  lambda x: np.flip(x, axis=0).copy()),
            #     (lambda x: np.flip(x, axis=1).copy(),                  lambda x: np.flip(x, axis=1).copy()),
            #     (lambda x: np.flip(np.flip(x, axis=0), axis=1).copy(), lambda x: np.flip(np.flip(x, axis=0), axis=1).copy()),
            # ]

            # patch_pred = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
            # for aug, inv in tta_pairs:
            #     tensor = preprocess_patch(aug(patch), device)
            #     with torch.no_grad():
            #         pred = model(tensor)
            #     pred = torch.sigmoid(pred).squeeze().cpu().numpy()
            #     pred = np.clip(pred, 0.01, 0.99)
            #     patch_pred += inv(pred)
            # patch_pred /= len(tta_pairs)

            final_mask[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE] += pred * weight_map
            weight_sum[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE] += weight_map      # non TTA 

            # final_mask[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE] += patch_pred * weight_map
            # weight_sum[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE] += weight_map        # TTA

            pbar.update(1)

    pbar.close()

    weight_sum[weight_sum == 0] = 1e-8
    final_mask = final_mask / weight_sum

    # remove padding
    final_mask = final_mask[pad_h:pad_h + H, pad_w:pad_w + W]

    cv2.imwrite("debug_raw_mask.png", (final_mask * 255).astype(np.uint8))
    print("Debug mask saved: debug_raw_mask.png")

    # binarize
    """
    "final_mask > 0.5" - this number is actually a threashold
    if the parser is not pickign up smaller walls in the given 
    floorplan than try to decrease the number oa bit and try 
    that out with the image.

    fundamentally the threashold is working as a filter so when 
    there is a wall in the floorplan that is thinner that it 
    should be it(because then it might not b a wall), the threashold
    limit filters it out of the binary image output.
    """

    # on some other universe this threasholding problem might have been solved by a automaion code, not here though(would have been nice)

    # redundant code (otsu trial)
    # raw_uint8 = (final_mask * 255).astype(np.uint8)
    # otsu_thresh, binary_mask = cv2.threshold(raw_uint8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # print(f"Otsu threshold: {otsu_thresh}/255")

    binary_mask = (final_mask > 0.50).astype(np.uint8)

    # this portion was not doing anything this is fully redundant 

    # morphological fill 
    # kernel = np.ones((3, 3), np.uint8)
    # binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    # binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    # morphological dialate
    # binary_mask = cv2.dilate(binary_mask, np.ones((2, 2), np.uint8), iterations=1)

    # remove text / symbols
    # text blobs are small AND roughly square; walls are large AND elongated
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        area   = stats[i, cv2.CC_STAT_AREA]
        w_box  = stats[i, cv2.CC_STAT_WIDTH]
        h_box  = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(w_box, h_box) / (min(w_box, h_box) + 1e-6)
        # if area < (H * W) * 0.00005 and aspect < 3.0:
        # trying to if removing the blob filter makes the parsing better, the main probel right now is thet the doors that are placed in an angle are actually not parsing properly making a blob 
        if area < (H * W) * 0.00005:
            # cant just assume the tickness of the text in the image it has to be dynamic thats why "area < (H * W) * 0.00005"
            continue
        cleaned[labels == i] = 1
    binary_mask = cleaned

    cv2.imwrite(args.output, binary_mask * 255)
    print(f" Saved: {args.output}")


if __name__ == "__main__":
    main()


