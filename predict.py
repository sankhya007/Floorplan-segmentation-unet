# import cv2
# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# from model import UNet

# model = UNet()
# model.load_state_dict(torch.load("unet.pth", map_location="cpu"))
# model.eval()

# img = cv2.imread("test.jpg")
# img_resized = cv2.resize(img, (512, 512)) / 255.0

# img_tensor = torch.tensor(img_resized).permute(2,0,1).unsqueeze(0).float()

# with torch.no_grad():
#     pred = model(img_tensor)
#     pred = torch.sigmoid(pred)  # since we removed sigmoid earlier
#     pred = pred[0][0].numpy()

# plt.imshow(pred, cmap="gray")
# plt.title("Prediction")
# plt.axis('off')
# plt.show()











import torch
import cv2
import numpy as np
from model import UNet

# -----------------------------
# CONFIG
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "unet.pth"
IMAGE_PATH = "test.jpg"   # 👈 change this

# -----------------------------
# LOAD MODEL
# -----------------------------
#model = UNet(n_classes=4).to(device)
model = UNet().to(device)
#model = UNet(n_classes=2).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -----------------------------
# LOAD IMAGE
# -----------------------------
img = cv2.imread(IMAGE_PATH)
orig = img.copy()

img = cv2.resize(img, (256, 256))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img / 255.0

img = torch.tensor(img).permute(2, 0, 1).float()
img = img.unsqueeze(0).to(device)

# -----------------------------
# PREDICT
# -----------------------------

# with torch.no_grad():
#     pred = model(img)
#     pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

with torch.no_grad():
    pred = model(img)
    pred = torch.sigmoid(pred)          # convert logits → probability
    pred = (pred > 0.5).float()         # threshold
    pred = pred.squeeze().cpu().numpy() # shape: H x W

# -----------------------------
# COLOR MASK
# -----------------------------
#h, w = pred.shape
#color_mask = np.zeros((h, w, 3), dtype=np.uint8)

# wall → blue
#color_mask[pred == 2] = [255, 0, 0]

# door → green
#color_mask[pred == 1] = [0, 255, 0]

# window → white
#color_mask[pred == 3] = [255, 255, 255]

#color_mask[pred == 1] = [255, 255, 255]  # object (white)


pred = cv2.resize(pred.astype(np.uint8), (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_NEAREST)
color_mask = np.zeros((orig.shape[0], orig.shape[1], 3), dtype=np.uint8)
color_mask[pred == 1] = [255, 255, 255]

# -----------------------------
# SAVE OUTPUT
# -----------------------------
cv2.imwrite("prediction.png", color_mask)

print("✅ Prediction saved as prediction.png")
print("Unique prediction values:", np.unique(pred))
