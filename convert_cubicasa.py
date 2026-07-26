import json
import os
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools import mask as coco_mask
import gc

INPUT_BASE = "used_datasets/cc5k"         # ← updated path
OUT_IMG = "dataset/images"
OUT_MASK = "dataset/masks"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_MASK, exist_ok=True)

WALL_ID = 2
DOOR_ID = 1
WINDOW_ID = 3


def draw_segmentation(mask, seg, class_id, h, w):
    if isinstance(seg, list) and len(seg) > 0:
        for poly in seg:
            if len(poly) < 6:
                continue
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            if np.max(pts) <= 1.0:
                pts[:, 0] *= w
                pts[:, 1] *= h
            pts = np.round(pts).astype(np.int32)
            if pts.shape[0] >= 3:
                cv2.fillPoly(mask, [pts], class_id)
        return True

    elif isinstance(seg, dict) and "counts" in seg:
        if isinstance(seg["counts"], list):
            rle = coco_mask.frPyObjects(seg, h, w)
            m = coco_mask.decode(rle)
        else:
            m = coco_mask.decode(seg)
        if len(m.shape) == 3:
            m = np.any(m, axis=2)
        mask[m > 0] = class_id
        return True

    return False


def draw_bbox(mask, bbox, class_id, h, w):
    x, y, bw, bh = bbox
    if x <= 1:
        x *= w; y *= h; bw *= w; bh *= h
    x, y, bw, bh = int(x), int(y), int(bw), int(bh)
    cv2.rectangle(mask, (x, y), (x + bw, y + bh), class_id, -1)


def convert_split(split):
    split_path = os.path.join(INPUT_BASE, split)
    json_path = os.path.join(split_path, "_annotations.coco.json")

    if not os.path.exists(json_path):
        print(f"Skipping {split} — no annotations found")
        return

    print(f"\nProcessing CubiCasa {split}...")

    with open(json_path, "r") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    ann_map = {}
    for ann in data["annotations"]:
        ann_map.setdefault(ann["image_id"], []).append(ann)

    del data
    gc.collect()

    for img_id, img_info in tqdm(images.items()):
        file_name = img_info["file_name"]
        img_path = os.path.join(split_path, file_name)

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)
        anns = ann_map.get(img_id, [])

        # PASS 1: walls + windows
        for ann in anns:
            category = ann["category_id"]
            seg = ann.get("segmentation", None)
            handled = False
            if category in [WALL_ID, WINDOW_ID]:
                if seg is not None:
                    handled = draw_segmentation(mask, seg, 1, h, w)
                if not handled and "bbox" in ann:
                    draw_bbox(mask, ann["bbox"], 1, h, w)

        # PASS 2: cut out doors
        for ann in anns:
            if ann["category_id"] != DOOR_ID:
                continue
            seg = ann.get("segmentation", None)
            handled = False
            temp = np.zeros_like(mask)
            if seg is not None:
                handled = draw_segmentation(temp, seg, 1, h, w)
            if not handled and "bbox" in ann:
                draw_bbox(temp, ann["bbox"], 1, h, w)
            kernel = np.ones((7, 7), np.uint8)
            temp = cv2.dilate(temp, kernel, iterations=1)
            mask[temp > 0] = 0

        if np.sum(mask) == 0:
            print(f"  EMPTY MASK: {file_name}")

        new_name = f"cc5k_{split}_{file_name}"
        cv2.imwrite(os.path.join(OUT_IMG, new_name), img)

        binary_mask = (mask > 0).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(OUT_MASK, new_name), binary_mask)

        del img, mask
        if img_id % 100 == 0:
            gc.collect()


for split in ["train", "valid", "test"]:
    convert_split(split)

print("\n CubiCasa conversion DONE.")