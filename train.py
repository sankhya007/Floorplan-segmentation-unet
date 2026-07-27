import os
import cv2
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import UNet

"""
Local training script for S.T.I.T.C.H floorplan segmentation.

Setup:
  pip install -r requirements.txt

Dataset structure expected:
  dataset/
    images/   <- floorplan images (.png)
    masks/    <- binary wall masks (.png)

Run:
  python train.py

To generate the dataset first:
  python convert_cubicasa.py
  python convert_msd.py
"""

# ------------------------------ CONFIG ------------------------------
IMG_DIR    = "dataset/images"
MASK_DIR   = "dataset/masks"
EPOCHS     = 15
BATCH_SIZE = 4       # lower to 2 if you run out of VRAM
LR         = 1e-4
LIMIT      = None    # set to e.g. 3000 to cap training images, None = use all
NUM_WORKERS = 4      # set to 0 on Windows if you get multiprocessing errors
# --------------------------------------------------------------------

train_transform = A.Compose([
    A.Rotate(limit=180, p=0.8),
    A.ElasticTransform(alpha=120, sigma=6, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
], additional_targets={'mask': 'mask'})


def add_hard_negatives(img, mask):
    """Stamp synthetic furniture/symbol shapes as background so the model
    learns to ignore them."""
    h, w = mask.shape
    for _ in range(random.randint(2, 6)):
        cx = random.randint(20, w - 20)
        cy = random.randint(20, h - 20)
        sz = random.randint(8, 30)
        temp = np.zeros((h, w), np.uint8)
        if random.random() < 0.5:
            cv2.rectangle(temp, (cx - sz, cy - sz), (cx + sz, cy + sz), 1, -1)
        else:
            cv2.circle(temp, (cx, cy), sz, 1, -1)
        mask[temp == 1] = 0
    return img, mask


class FloorplanDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augment=False):
        self.img_names = sorted(os.listdir(img_dir))
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.augment   = augment

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name = self.img_names[idx]

        img = cv2.imread(os.path.join(self.img_dir, name))
        if img is None:
            print("BAD IMAGE:", name)
            return self.__getitem__(0)
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(os.path.join(self.mask_dir, name), cv2.IMREAD_UNCHANGED)
        if mask is None:
            print("BAD MASK:", name)
            return self.__getitem__(0)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)

        if self.augment:
            if random.random() < 0.5:
                img, mask = add_hard_negatives(img, mask)
            aug  = train_transform(image=img, mask=mask)
            img  = aug['image']
            mask = aug['mask']

        img  = img.astype(np.float32) / 255.0
        img  = torch.from_numpy(img).permute(2, 0, 1)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: training on CPU will be very slow. A GPU is strongly recommended.")

    dataset = FloorplanDataset(IMG_DIR, MASK_DIR, augment=True)
    if LIMIT:
        dataset.img_names = dataset.img_names[:LIMIT]
    print(f"Total images: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )
    print(f"Total batches per epoch: {len(loader)}")

    model     = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # pos_weight=3.0 — walls are sparse pixels; penalise missing them 3x more
    loss_fn   = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device))

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5, verbose=True
    )

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for imgs, masks in loop:
            imgs  = imgs.to(device)
            masks = masks.to(device)
            preds = model(imgs)
            loss  = loss_fn(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} — Avg Loss: {avg_loss:.4f}")

        scheduler.step(avg_loss)

        torch.save(model.state_dict(), "unet.pth")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "unet_best.pth")
            print(f"  ✅ Best model saved (loss: {best_loss:.4f})")

    print(f"\n✅ TRAINING COMPLETE — best loss: {best_loss:.4f}")
    print("Weights saved: unet.pth (last epoch), unet_best.pth (best epoch)")