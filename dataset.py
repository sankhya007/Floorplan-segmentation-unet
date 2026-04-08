import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class FloorplanDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_names = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name = self.img_names[idx]

        img = cv2.imread(os.path.join(self.img_dir, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0

        # mask = cv2.imread(os.path.join(self.mask_dir, name), cv2.IMREAD_GRAYSCALE)
        # mask = (mask > 0).astype(np.float32)  # 🔥 IMPORTANT
        
        mask = cv2.imread(os.path.join(self.mask_dir, name), cv2.IMREAD_UNCHANGED)

        # ensure single channel (no implicit grayscale conversion)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        mask = (mask > 0).astype(np.float32)

        # to tensor
        img = torch.tensor(img).permute(2, 0, 1).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return img, mask