import torch
import cv2
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "unet.pth"   # change if needed
IMAGE_PATH = r"C:\Users\Asus\parser-model\test.jpg"

PATCH_SIZE = 256
STRIDE = 128   # 50% overlap (IMPORTANT)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD MODEL (EDIT THIS IF NEEDED)
# -----------------------------
from model import UNet   # import  model

model = UNet()          # create model

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

model.load_state_dict(state_dict)  # ✅ load weights

model.to(DEVICE)
model.eval()
# -----------------------------
# PREPROCESS (MATCH TRAINING)
# -----------------------------
def preprocess_patch(patch):
    # patch = patch / 255.0   # this one worked when the image was a bit smaller 
    patch = patch.astype(np.float32) / 255.0
    patch = np.transpose(patch, (2, 0, 1))
    patch = torch.from_numpy(patch).unsqueeze(0)
    return patch.to(DEVICE)

# -----------------------------
# CREATE WEIGHT MAP (smooth blending)
# -----------------------------
def create_weight_map(size):
    h, w = size, size
    y = np.linspace(-1, 1, h)
    x = np.linspace(-1, 1, w)
    xv, yv = np.meshgrid(x, y)

    weight = 1 - (xv**2 + yv**2)
    weight = np.clip(weight, 0, 1)

    return weight

weight_map = create_weight_map(PATCH_SIZE)

# -----------------------------
# LOAD IMAGE
# -----------------------------
img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

H, W, _ = img.shape
print("H, W:", H, W)

# -----------------------------
# OUTPUT CANVAS
# -----------------------------
final_mask = np.zeros((H, W), dtype=np.float32)
weight_sum = np.zeros((H, W), dtype=np.float32)

# -----------------------------
# SLIDING WINDOW
# -----------------------------
for y in range(0, H, STRIDE):
    for x in range(0, W, STRIDE):

        y1 = min(y, H - PATCH_SIZE)
        x1 = min(x, W - PATCH_SIZE)
        
        print("Processing patch at:", y1, x1)

        patch = img[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE]

        patch_tensor = preprocess_patch(patch)

        with torch.no_grad():
            pred = model(patch_tensor)

        pred = torch.sigmoid(pred).squeeze().cpu().numpy()

        # Apply weight
        weighted_pred = pred * weight_map

        final_mask[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE] += weighted_pred
        weight_sum[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE] += weight_map

print("Before normalize min/max:", final_mask.min(), final_mask.max())
print("Weight sum min/max:", weight_sum.min(), weight_sum.max())

# -----------------------------
# NORMALIZE
# -----------------------------
final_mask = np.divide(final_mask, weight_sum, where=weight_sum!=0)
# -----------------------------
# THRESHOLD
# -----------------------------
binary_mask = (final_mask > 0.5).astype(np.uint8)

# -----------------------------
# SAVE
# -----------------------------
print("Binary mask unique values:", np.unique(binary_mask))
cv2.imwrite("stitched_mask.png", binary_mask * 255)

print("✅ Saved: stitched_mask.png")