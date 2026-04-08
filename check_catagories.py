# temp file 

import json

with open("used_datasets/cubicasa5k-2.v6i.coco/train/_annotations.coco.json") as f:
    data = json.load(f)

for cat in data["categories"]:
    print(cat)