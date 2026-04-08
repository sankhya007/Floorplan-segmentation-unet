import cv2
import numpy as np
import imageio
import os

# -----------------------------
# CONFIG
# -----------------------------
IMAGE_PATH = "test.jpg"
PATCH_SIZE = 256
STRIDE = 128

os.makedirs("assets", exist_ok=True)

# -----------------------------
# LOAD IMAGE
# -----------------------------
img = cv2.imread(IMAGE_PATH)

if img is None:
    raise ValueError(f"❌ Could not load image: {IMAGE_PATH}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

H, W, _ = img.shape

frames = []

# -----------------------------
# SLIDING WINDOW VISUALIZATION
# -----------------------------
for y in range(0, H, STRIDE):
    for x in range(0, W, STRIDE):

        y1 = min(y, H - PATCH_SIZE)
        x1 = min(x, W - PATCH_SIZE)

        frame = img.copy()

        # darken background (focus effect)
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (0, 0),
            (W, H),
            (0, 0, 0),
            -1
        )
        alpha = 0.5
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # highlight current patch
        patch = img[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE]
        frame[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE] = patch

        # draw patch border
        cv2.rectangle(
            frame,
            (x1, y1),
            (x1 + PATCH_SIZE, y1 + PATCH_SIZE),
            (255, 0, 0),
            3
        )

        # text background
        cv2.rectangle(frame, (0, 0), (450, 50), (0, 0, 0), -1)

        # add text
        cv2.putText(
            frame,
            f"Patch: ({x1}, {y1})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        frames.append(frame)

# -----------------------------
# ADD FINAL FRAME (full image)
# -----------------------------
final_frame = img.copy()
cv2.putText(
    final_frame,
    "Final Stitched Output",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 0),
    2
)

frames.append(final_frame)

# -----------------------------
# SAVE GIF
# -----------------------------
gif_path = "assets/stitching.gif"

imageio.mimsave(
    gif_path,
    frames,
    fps=6
)

print(f"✅ GIF saved: {gif_path}")