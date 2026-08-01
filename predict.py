import torch
import cv2
import numpy as np
from model import UNet

device     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "unet_best_till_now.pth"
IMAGE_PATH = "mall_more.png"

model = UNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

orig = cv2.imread(IMAGE_PATH)
if orig is None:
    raise FileNotFoundError(f"Could not read: {IMAGE_PATH}")

H, W = orig.shape[:2]
pad_h = (32 - H % 32) % 32
pad_w = (32 - W % 32) % 32
img = cv2.copyMakeBorder(orig, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

with torch.no_grad():
    pred = model(img)
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    pred = pred.squeeze().cpu().numpy()

pred = pred[:H, :W]  # strip padding
mask = np.zeros((H, W, 3), dtype=np.uint8)
mask[pred == 1] = [255, 255, 255]

cv2.imwrite("prediction.png", mask)
print("✅ Prediction saved as prediction.png")
print("Unique prediction values:", np.unique(pred))