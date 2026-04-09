# import os
# import cv2
# import numpy as np
# import xml.etree.ElementTree as ET


# INPUT_DIR = "used_datasets/cubicasa5k/high_quality"
# OUT_IMG = "dataset/images"
# OUT_MASK = "dataset/masks"

# os.makedirs(OUT_IMG, exist_ok=True)
# os.makedirs(OUT_MASK, exist_ok=True)


# def draw_line(mask, x1, y1, x2, y2, value, thickness=3):
#     cv2.line(mask, (x1, y1), (x2, y2), value, thickness)


# from svgpathtools import svg2paths

# def parse_svg(svg_path, img_shape):

#     # Instead of parsing SVG, use the image directly
#     img_path = svg_path.replace("model.svg", "F1_scaled.png")

#     img = cv2.imread(img_path, 0)

#     # threshold to extract structure
#     _, mask = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    
#     # this portion is here to clear out the noise and any other white dorts in the image
#     # kernel = np.ones((3,3), np.uint8)
#     # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     # do not want to keep this because this actually fucked up the parsing

#     return mask


# for folder in os.listdir(INPUT_DIR):

#     folder_path = os.path.join(INPUT_DIR, folder)
    
#     print("processing:", folder_path)

#     img_path = os.path.join(folder_path, "F1_scaled.png")
#     svg_path = os.path.join(folder_path, "model.svg")

#     if not os.path.exists(img_path) or not os.path.exists(svg_path):
#         continue

#     img = cv2.imread(img_path)
#     mask = parse_svg(svg_path, img.shape)

#     cv2.imwrite(f"{OUT_IMG}/{folder}.png", img)
#     cv2.imwrite(f"{OUT_MASK}/{folder}.png", mask)

# print("Done")









# working code for jut one onject masks

# import json
# import os
# import cv2
# import numpy as np
# from tqdm import tqdm

# from pycocotools import mask as coco_mask

# INPUT_BASE = "used_datasets/cubicasa5k-2.v6i.coco"
# OUT_IMG = "dataset/images"
# OUT_MASK = "dataset/masks"

# os.makedirs(OUT_IMG, exist_ok=True)
# os.makedirs(OUT_MASK, exist_ok=True)


# # FINAL CLASS REMAP
# # CLASS_MAP = {
# #     0: 2,  # wall → blue
# #     1: 1,  # door → green
# #     2: 3   # window → white
# # }
# CLASS_MAP = {
#     1: 2,  # wall → blue
#     2: 1,  # door → green
#     3: 3   # window → white
# }


# def convert_split(split):

#     split_path = os.path.join(INPUT_BASE, split)
#     json_path = os.path.join(split_path, "_annotations.coco.json")

#     if not os.path.exists(json_path):
#         return

#     print(f"\nProcessing {split}...")

#     with open(json_path) as f:
#         data = json.load(f)
        
#     print("\nCATEGORIES:")
#     for cat in data["categories"]:
#         print(cat)

#     images = {}
#     for img in data["images"]:
#         images[img["id"]] = img

#     ann_map = {}
#     for ann in data["annotations"]:
#         ann_map.setdefault(ann["image_id"], []).append(ann)

#     for img_id, img_info in tqdm(images.items()):

#         file_name = img_info["file_name"]
#         img_path = os.path.join(split_path, file_name)

#         if not os.path.exists(img_path):
#             continue

#         img = cv2.imread(img_path)
#         h, w = img.shape[:2]

#         mask = np.zeros((h, w), dtype=np.uint8)
        
#         anns = ann_map.get(img_id, [])

#         print("\n--- DEBUG ---")
#         print("Image:", file_name)
#         print("Annotations:", len(anns))

#         for ann in anns:

#             class_id = CLASS_MAP.get(ann["category_id"], 0)
#             # if the class if was just 1 then every mask gets into just one thing 

#             # -------------------------
#             # HANDLE BBOX (RELIABLE)
#             # -------------------------
#             if "bbox" in ann:

#                 x, y, bw, bh = ann["bbox"]

#                 # 🔥 SCALE FROM NORMALIZED → PIXELS
#                 if x <= 1 and y <= 1 and bw <= 1 and bh <= 1:
#                     x = int(x * w)
#                     y = int(y * h)
#                     bw = int(bw * w)
#                     bh = int(bh * h)
#                 else:
#                     x, y, bw, bh = int(x), int(y), int(bw), int(bh)

#                 # 🔥 DRAW FILLED RECTANGLE
#                 cv2.rectangle(mask, (x, y), (x + bw, y + bh), class_id, -1)

#         print("Mask values:", np.unique(mask))

#         new_name = f"{split}_{file_name}"

#         cv2.imwrite(os.path.join(OUT_IMG, new_name), img)
#         color_mask = np.zeros((h, w, 3), dtype=np.uint8)

#         # wall → blue
#         color_mask[mask == 1] = [255, 0, 0]

#         # door → green
#         color_mask[mask == 2] = [0, 255, 0]

#         # window → white
#         color_mask[mask == 3] = [255, 255, 255]

#         cv2.imwrite(os.path.join(OUT_MASK, new_name), color_mask)


# for split in ["train", "valid", "test"]:
#     convert_split(split)

# print("\nDONE MULTI-CLASS ✅")









# code nof the newer dataset 
# we are doing the parsing in multilayer, the walls, doors and windows are differently parsed and not all in one 

# import json
# import os
# import cv2
# import numpy as np
# from tqdm import tqdm

# from pycocotools import mask as coco_mask

# INPUT_BASE = "used_datasets/cubicasa5k-2.v6i.coco"
# OUT_IMG = "dataset/images"
# OUT_MASK = "dataset/masks"

# os.makedirs(OUT_IMG, exist_ok=True)
# os.makedirs(OUT_MASK, exist_ok=True)


# # FINAL CLASS REMAP
# # CLASS_MAP = {
# #     0: 2,  # wall → blue
# #     1: 1,  # door → green
# #     2: 3   # window → white
# # }
# CLASS_MAP = {
#     1: 2,  # wall → blue
#     2: 1,  # door → green
#     3: 3   # window → white
# }


# def convert_split(split):

#     split_path = os.path.join(INPUT_BASE, split)
#     json_path = os.path.join(split_path, "_annotations.coco.json")

#     if not os.path.exists(json_path):
#         return

#     print(f"\nProcessing {split}...")

#     with open(json_path) as f:
#         data = json.load(f)
        
#     print("\nCATEGORIES:")
#     for cat in data["categories"]:
#         print(cat)

#     images = {}
#     for img in data["images"]:
#         images[img["id"]] = img

#     ann_map = {}
#     for ann in data["annotations"]:
#         ann_map.setdefault(ann["image_id"], []).append(ann)

#     for img_id, img_info in tqdm(images.items()):

#         file_name = img_info["file_name"]
#         img_path = os.path.join(split_path, file_name)

#         if not os.path.exists(img_path):
#             continue

#         img = cv2.imread(img_path)
#         h, w = img.shape[:2]

#         mask = np.zeros((h, w), dtype=np.uint8)
        
#         anns = ann_map.get(img_id, [])

#         print("\n--- DEBUG ---")
#         print("Image:", file_name)
#         print("Annotations:", len(anns))

#         for ann in anns:

#             class_id = CLASS_MAP.get(ann["category_id"], 0)
#             # if the class if was just 1 then every mask gets into just one thing 

#             # -------------------------
#             # HANDLE BBOX (RELIABLE)
#             # -------------------------
#             if "bbox" in ann:

#                 x, y, bw, bh = ann["bbox"]

#                 # 🔥 SCALE FROM NORMALIZED → PIXELS
#                 if x <= 1 and y <= 1 and bw <= 1 and bh <= 1:
#                     x = int(x * w)
#                     y = int(y * h)
#                     bw = int(bw * w)
#                     bh = int(bh * h)
#                 else:
#                     x, y, bw, bh = int(x), int(y), int(bw), int(bh)

#                 # draw only borders (thin walls)
#                 thickness = 1

#                 cv2.rectangle(mask, (x, y), (x + bw, y + bh), class_id, thickness)

#         print("Mask values:", np.unique(mask))

#         new_name = f"{split}_{file_name}"

#         cv2.imwrite(os.path.join(OUT_IMG, new_name), img)
#         color_mask = np.zeros((h, w, 3), dtype=np.uint8)

#         # wall → blue
#         color_mask[mask == 1] = [0, 0, 255]

#         # door → green
#         color_mask[mask == 2] = [0, 255, 0]

#         # window → white
#         color_mask[mask == 3] = [255, 255, 255]

#         # training mask (grayscale)
#         cv2.imwrite(os.path.join(OUT_MASK, new_name), mask)

#         # optional visualization
#         cv2.imwrite(os.path.join(OUT_MASK, "vis_" + new_name), color_mask)


# for split in ["train", "valid", "test"]:
#     convert_split(split)

# print("\nDONE MULTI-CLASS ✅")














# this is the mask maer that we are using to make the walls dense and the doors walk through
import json
import os
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools import mask as coco_mask
import gc

INPUT_BASE = "used_datasets/cubicasa5k-2.v6i.coco"
OUT_IMG = "dataset/images"
OUT_MASK = "dataset/masks"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_MASK, exist_ok=True)

WALL_ID = 2
DOOR_ID = 1
WINDOW_ID = 3

# Single class (binary)
CLASS_MAP = {
    1: 1, # these are the sold walls 
    2: 0, # doors become walk through
    3: 1  # this is the window i am not goint to turn it into 0 because then whaat the fuck man people are going to start jumping from the windows
}


def draw_segmentation(mask, seg, class_id, h, w):
    """Draw segmentation (polygon or RLE) onto mask."""

    # -------- POLYGON --------
    if isinstance(seg, list) and len(seg) > 0:

        for poly in seg:
            if len(poly) < 6:
                continue

            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)

            # scale if normalized
            if np.max(pts) <= 1.0:
                pts[:, 0] *= w
                pts[:, 1] *= h

            pts = np.round(pts).astype(np.int32)

            if pts.shape[0] >= 3:
                cv2.fillPoly(mask, [pts], class_id)

        return True  # segmentation worked

    # -------- RLE --------
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

    return False  # segmentation failed


def draw_bbox(mask, bbox, class_id, h, w):
    """Fallback: draw filled bbox."""
    x, y, bw, bh = bbox

    # normalize if needed
    if x <= 1:
        x *= w
        y *= h
        bw *= w
        bh *= h

    x, y, bw, bh = int(x), int(y), int(bw), int(bh)

    cv2.rectangle(mask, (x, y), (x + bw, y + bh), class_id, -1)


def convert_split(split):

    split_path = os.path.join(INPUT_BASE, split)
    json_path = os.path.join(split_path, "_annotations.coco.json")

    if not os.path.exists(json_path):
        return

    print(f"\nProcessing {split}...")   

    with open(json_path, "r") as f:
        data = json.load(f)

    # build image dict
    images = {img["id"]: img for img in data["images"]}

    # build annotation map
    ann_map = {}
    for ann in data["annotations"]:
        ann_map.setdefault(ann["image_id"], []).append(ann)

    # 🔥 free memory immediately
    del data
    gc.collect()

    for img_id, img_info in tqdm(images.items()):

        file_name = img_info["file_name"]
        img_path = os.path.join(split_path, file_name)

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            print("Failed to load:", img_path)
            continue
        h, w = img.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)
        anns = ann_map.get(img_id, [])

        # =========================
        # PASS 1: DRAW WALLS + WINDOWS (BLOCKED)
        # =========================
        for ann in anns:

            category = ann["category_id"]
            seg = ann.get("segmentation", None)

            handled = False

            # -------- WALL OR WINDOW --------
            if category in [WALL_ID, WINDOW_ID]:  # BOTH are blocked 

                if seg is not None:
                    handled = draw_segmentation(mask, seg, 1, h, w)

                if not handled and "bbox" in ann:
                    draw_bbox(mask, ann["bbox"], 1, h, w)


        # =========================
        # PASS 2: REMOVE DOORS (MAKE GAPS)
        # =========================
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

            # expanding door region before cutting, because or else there is a small line left here
            kernel = np.ones((7,7), np.uint8)
            temp = cv2.dilate(temp, kernel, iterations=1)

            mask[temp > 0] = 0
                    
                    


        if np.sum(mask) == 0:
            print(f"⚠️ EMPTY MASK: {file_name}")
            
        # THICKEN WALLS
        # kernel = np.ones((5,5), np.uint8)
        # mask = cv2.dilate(mask, kernel, iterations=1)

        # save
        new_name = f"{split}_{file_name}"

        cv2.imwrite(os.path.join(OUT_IMG, new_name), img)

        binary_mask = (mask > 0).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(OUT_MASK, new_name), binary_mask)

        # optional visualization
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        vis[mask == 1] = [255, 255, 255]
        
        del img, mask
        # temp may not always exist, so safe delete
        if 'temp' in locals():
            del temp
        if img_id % 100 == 0:
            gc.collect()
   


for split in ["train", "valid", "test"]:
    convert_split(split)

print("\nDONE ✅")