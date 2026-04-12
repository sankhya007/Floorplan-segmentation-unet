# import torch
# import cv2
# import numpy as np

# #--------------------------CONFIG------------------------------------
# MODEL_PATH = "unet.pth"   # change if needed
# IMAGE_PATH = r"C:\Users\Asus\parser-model\test.jpg"

# PATCH_SIZE = 256
# STRIDE = 150   # 50% overlap (IMPORTANT)

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #---------------------------------------------------------------------

# from model import UNet   # import  model

# model = UNet()          # create model

# state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
# model.load_state_dict(state_dict)  # load weights
# model.to(DEVICE)
# model.eval()

# def preprocess_patch(patch):
#     # patch = patch / 255.0   # this one worked when the image was a bit smaller 
#     patch = patch.astype(np.float32) / 255.0
#     patch = np.transpose(patch, (2, 0, 1))
#     patch = torch.from_numpy(patch).unsqueeze(0)
#     return patch.to(DEVICE)

# #here we are using gaussian blur to blend thats why the corners are getting fucked 
# def create_weight_map(size):
#     h, w = size, size
#     y, x = np.ogrid[-1:1:h*1j, -1:1:w*1j]
#     weight = np.exp(-(x**2 + y**2) * 4)   # gaussian
#     return weight.astype(np.float32)

# weight_map = create_weight_map(PATCH_SIZE)

# img = cv2.imread(IMAGE_PATH)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# H, W, _ = img.shape
# print("H, W:", H, W)

# final_mask = np.zeros((H, W), dtype=np.float32)
# weight_sum = np.zeros((H, W), dtype=np.float32)

# y_positions = list(range(0, H - PATCH_SIZE, STRIDE))
# x_positions = list(range(0, W - PATCH_SIZE, STRIDE))

# if y_positions[-1] != H - PATCH_SIZE:
#     y_positions.append(H - PATCH_SIZE)

# if x_positions[-1] != W - PATCH_SIZE:
#     x_positions.append(W - PATCH_SIZE)

# for y1 in y_positions:
#     for x1 in x_positions:

#         print("Processing patch at:", y1, x1)

#         patch = img[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE]
#         if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
#             pad_h = PATCH_SIZE - patch.shape[0]
#             pad_w = PATCH_SIZE - patch.shape[1]
#             patch = cv2.copyMakeBorder(
#                 patch, 
#                 0, pad_h, 
#                 0, pad_w, 
#                 cv2.BORDER_REFLECT
#             )

#         patch_tensor = preprocess_patch(patch)

#         with torch.no_grad():
#             pred = model(patch_tensor)

#         pred = torch.sigmoid(pred).squeeze().cpu().numpy()
#         pred = np.clip(pred, 0.05, 0.95)

#         weighted_pred = pred * weight_map

#         final_mask[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE] += weighted_pred
#         weight_sum[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE] += weight_map
        

# print("Before normalize min/max:", final_mask.min(), final_mask.max())
# print("Weight sum min/max:", weight_sum.min(), weight_sum.max())

# weight_sum[weight_sum == 0] = 1e-8
# final_mask = final_mask / weight_sum

# # DEBUG (visualize raw probabilities)
# cv2.imwrite("debug_raw_mask.png", (final_mask * 255).astype(np.uint8))
# binary_mask = (final_mask > 0.5).astype(np.uint8)


# kernel = np.ones((3,3), np.uint8)
# binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
# binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
# binary_mask = cv2.dilate(binary_mask, np.ones((2,2), np.uint8), iterations=1)

# print("Binary mask unique values:", np.unique(binary_mask))
# cv2.imwrite("stitched_mask.png", binary_mask * 255)

# print("Saved: stitched_mask.png")












import torch
import cv2
import numpy as np

# -------------------------- CONFIG ------------------------------------
MODEL_PATH = "unet.pth"
IMAGE_PATH = r"C:\Users\Asus\parser-model\test.jpg"

PATCH_SIZE = 256
STRIDE = 128   #50% for each block 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------------------------------------------------

from model import UNet

model = UNet()
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()


def preprocess_patch(patch):
    patch = patch.astype(np.float32) / 255.0
    patch = np.transpose(patch, (2, 0, 1))
    patch = torch.from_numpy(patch).unsqueeze(0)
    return patch.to(DEVICE)


def create_weight_map(size):
    h, w = size, size
    y, x = np.ogrid[-1:1:h*1j, -1:1:w*1j]
    weight = np.exp(-(x**2 + y**2) * 4)
    return weight.astype(np.float32)


weight_map = create_weight_map(PATCH_SIZE)

# -------------------- LOAD IMAGE --------------------
img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

H, W, _ = img.shape
print("Original H, W:", H, W)

# -------------------- ADD 5% PADDING --------------------
pad_h = int(0.05 * H)
pad_w = int(0.05 * W)

img = cv2.copyMakeBorder(
    img,
    pad_h, pad_h,
    pad_w, pad_w,
    cv2.BORDER_REFLECT_101
)

padded_H, padded_W, _ = img.shape
print("Padded H, W:", padded_H, padded_W)

# -------------------- INIT MASKS --------------------
final_mask = np.zeros((padded_H, padded_W), dtype=np.float32)
weight_sum = np.zeros((padded_H, padded_W), dtype=np.float32)

# -------------------- PATCH GRID --------------------
y_positions = list(range(0, padded_H - PATCH_SIZE, STRIDE))
x_positions = list(range(0, padded_W - PATCH_SIZE, STRIDE))

if y_positions[-1] != padded_H - PATCH_SIZE:
    y_positions.append(padded_H - PATCH_SIZE)

if x_positions[-1] != padded_W - PATCH_SIZE:
    x_positions.append(padded_W - PATCH_SIZE)

# -------------------- INFERENCE LOOP --------------------
for y1 in y_positions:
    for x1 in x_positions:

        print("Processing patch at:", y1, x1)

        patch = img[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE]

        if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
            pad_h2 = PATCH_SIZE - patch.shape[0]
            pad_w2 = PATCH_SIZE - patch.shape[1]
            patch = cv2.copyMakeBorder(
                patch,
                0, pad_h2,
                0, pad_w2,
                cv2.BORDER_REFLECT_101
            )

        patch_tensor = preprocess_patch(patch)

        with torch.no_grad():
            pred = model(patch_tensor)

        pred = torch.sigmoid(pred).squeeze().cpu().numpy()
        pred = np.clip(pred, 0.05, 0.95)

        weighted_pred = pred * weight_map

        final_mask[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE] += weighted_pred
        weight_sum[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE] += weight_map

# -------------------- NORMALIZE --------------------
print("Before normalize min/max:", final_mask.min(), final_mask.max())
print("Weight sum min/max:", weight_sum.min(), weight_sum.max())

weight_sum[weight_sum == 0] = 1e-8
final_mask = final_mask / weight_sum

# -------------------- REMOVE PADDING --------------------
final_mask = final_mask[
    pad_h:pad_h + H,
    pad_w:pad_w + W
]

# -------------------- SAVE DEBUG --------------------
cv2.imwrite("debug_raw_mask.png", (final_mask * 255).astype(np.uint8))

# -------------------- BINARIZE --------------------
binary_mask = (final_mask > 0.5).astype(np.uint8)

kernel = np.ones((3,3), np.uint8)
binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
binary_mask = cv2.dilate(binary_mask, np.ones((2,2), np.uint8), iterations=1)

print("Binary mask unique values:", np.unique(binary_mask))

cv2.imwrite("stitched_mask.png", binary_mask * 255)

print("Saved: stitched_mask.png")