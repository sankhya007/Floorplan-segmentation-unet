# import torch
# from torch.utils.data import DataLoader
# import torch.nn as nn
# import torch.optim as optim

# from dataset import FloorplanDataset
# from model import UNet

# device = "cuda" if torch.cuda.is_available() else "cpu"

# dataset = FloorplanDataset("dataset/images", "dataset/masks")
# loader = DataLoader(dataset, batch_size=1, shuffle=True)

# model = UNet().to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# loss_fn = nn.BCEWithLogitsLoss()

# epochs = 10

# for epoch in range(epochs):
#     total_loss = 0

#     for i, (imgs, masks) in enumerate(loader):
        
#         imgs = imgs.to(device)
#         masks = masks.to(device)

#         preds = model(imgs)
#         loss = loss_fn(preds, masks)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
        
#         if i % 10 == 0:
#             print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}")

#     print(f"Epoch {epoch+1} DONE, Total Loss: {total_loss:.4f}")
#     avg_loss = total_loss / len(loader)
#     print(f"Epoch {epoch+1} DONE, Avg Loss: {avg_loss:.4f}")
#     torch.save(model.state_dict(), "unet.pth")
#     print("Model Saved !")








import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# DATASET 
class FloorplanDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_names = sorted(os.listdir(img_dir))
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name = self.img_names[idx]

        # ---- IMAGE ----
        img_path = os.path.join(self.img_dir, name)
        img = cv2.imread(img_path)
        if img is None:
            print("BAD IMAGE:", img_path)
            return self.__getitem__(0)

        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        # ---- MASK ----
        mask_path = os.path.join(self.mask_dir, name)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        mask = (mask > 0).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask


# SIMPLE UNET
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = DoubleConv(3, 64)
        self.p1 = nn.MaxPool2d(2)

        self.d2 = DoubleConv(64, 128)
        self.p2 = nn.MaxPool2d(2)

        self.d3 = DoubleConv(128, 256)
        self.p3 = nn.MaxPool2d(2)

        self.b = DoubleConv(256, 512)

        self.u3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.c3 = DoubleConv(512, 256)

        self.u2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.c2 = DoubleConv(256, 128)

        self.u1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.c1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        d3 = self.d3(self.p2(d2))

        b = self.b(self.p3(d3))

        u3 = self.u3(b)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.c3(u3)

        u2 = self.u2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.c2(u2)

        u1 = self.u1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.c1(u1)

        return self.out(u1)

# TRAINING SETUP
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = FloorplanDataset("dataset/images", "dataset/masks")
print("Total images:", len(dataset))
dataset.img_names = dataset.img_names[:3000]  # if you want to limit the numer of images you are doing the training on use this line 
loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
print(f"Total batches: {len(loader)}")
model = UNet().to(device)

loss_fn = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 15


# TRAIN LOOP

print("Starting training...")

for epoch in range(epochs):
    model.train()
    total_loss = 0

    loop = tqdm(loader, desc=f"Epoch {epoch+1}", leave=True)

    for imgs, masks in loop:
        imgs = imgs.to(device)
        masks = masks.to(device)

        preds = model(imgs)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    print(f"\nEpoch {epoch+1} DONE")
    print(f"Total Loss: {total_loss:.4f}")
    print(f"Avg Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "unet.pth")
    print(" Model Saved")


print("\n TRAINING COMPLETE")