# S.T.I.T.C.H
### Segmentation & Tiled Inference for Topological Chart Handling

> Floorplan wall segmentation using UNet — built to feed structural data into **[T.R.A.G.I.C](https://github.com/sankhya007/T.R.A.G.I.C-Crowd-Evac)**, a real-time crowd evacuation simulation system.

---

## What This Does

You give it a floorplan image. It gives you back a binary mask — walls white, walkable space black. Clean, fast, and accurate enough to drive a simulation.

That mask feeds directly into T.R.A.G.I.C, where agents navigate the parsed layout in real time during evacuation scenarios.

No manual annotation. No CAD software. Just a floorplan image and a trained model.

---

## Why It Exists

Evacuation simulations need accurate spatial maps. Getting those maps from raw architectural drawings is a pain — especially when the drawings have diagonal walls, rotated layouts, or non-standard structure.

S.T.I.T.C.H solves that. It parses the floorplan automatically and hands off a usable wall mask to the simulation layer.

---

## Repo Contents

| File | What it is |
|---|---|
| `model.py` | UNet architecture definition |
| `train.py` | Training loop — run this to train from scratch |
| `predict.py` | Single image inference |
| `predict_tiled.py` | Tiled inference for large floorplans (use this one) |
| `dataset.py` | PyTorch dataset loader |
| `convert_cubicasa.py` | Converts CubiCasa5K COCO annotations → binary masks |
| `convert_msd.py` | Converts MSD `.npy` files → binary masks |
| `colab_train.ipynb` | Ready-to-run Google Colab notebook for training on GPU |
| `diagram_stitch.py` | Generates the stitching visualization GIF |

---

## Downloads

Everything you need is hosted externally — the repo stays lightweight.

### Trained Model Weights
The pretrained `unet.pth` weights, ready to run inference immediately.

🔗 **[Download from HuggingFace](https://huggingface.co/sankhya007/Floorplan_parser_STITCH/tree/main)**

Place `unet.pth` in the project root before running any predict script.

---

### Dataset
The fully converted and merged training dataset (~10,000 images + masks) from both CubiCasa5K and Modified Swiss Dwellings, preprocessed and ready to train on directly.

🔗 **[Download dataset.zip from HuggingFace](https://huggingface.co/sankhya007/Floorplan_parser_STITCH/tree/main)**

Unzip it into the project root so the structure looks like:
```
S.T.I.T.C.H/
├── dataset/
│   ├── images/
│   └── masks/
```

---

### Training Notebook
If you want to retrain on Google Colab (free T4 GPU), the notebook is in the repo as `colab_train.ipynb`. It handles everything — mounts Drive, unzips dataset, trains, saves weights back to Drive.

Steps:
1. Upload `dataset.zip` to your Google Drive root
2. Open `colab_train.ipynb` in Google Colab
3. Set runtime to **T4 GPU** (Runtime → Change runtime type)
4. Run cells top to bottom
5. `unet.pth` auto-saves to your Drive when done

---

## Quickstart — Run Inference

```bash
git clone https://github.com/sankhya007/S.T.I.T.C.H
cd S.T.I.T.C.H
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download `unet.pth` from HuggingFace and place it in the root, then:

**For a single small image:**
```bash
# set IMAGE_PATH in predict.py first
python predict.py
# output → prediction.png
```

**For large floorplans (recommended):**
```bash
# set IMAGE_PATH in predict_tiled.py first
python predict_tiled.py
# output → stitched_mask.png + debug_raw_mask.png
```

---

## Quickstart — Train From Scratch

1. Download `dataset.zip` from HuggingFace and unzip into project root
2. Or build your own dataset by running the converters:
```bash
python convert_cubicasa.py   # needs CubiCasa5K in used_datasets/cc5k/
python convert_msd.py        # needs MSD in used_datasets/msd/
```
3. Then train:
```bash
python train.py
```
Or use `colab_train.ipynb` on Google Colab for GPU training.

---

## How Tiled Inference Works

Large floorplans can't just be resized — you lose detail and break walls. Instead:

```
Input Image
    ↓
Add ~5% reflective padding
    ↓
Sliding window (256×256 patches, 50% overlap)
    ↓
UNet inference on each patch
    ↓
Gaussian-weighted blending (center = high confidence, edges = low)
    ↓
Remove padding
    ↓
Binary mask — walls white, walkable space black
```

The Gaussian blending eliminates seams and broken edges at patch boundaries.

<p align="center">
  <img src="assets/stitching.gif" width="50%"/>
</p>

---

## Model & Training Details

| | |
|---|---|
| Architecture | UNet with BatchNorm |
| Input size | 256×256 RGB |
| Output | Binary mask (walls=white, space=black) |
| Loss | BCEWithLogitsLoss |
| Optimizer | Adam (lr=1e-4) |
| Batch size | 16 |
| Epochs | 15 |
| Training images | ~10,000 |
| Trained on | Google Colab T4 GPU |

---

## Training Data

Trained on two datasets merged together:

**CubiCasa5K** — ~5,000 annotated residential floorplans (COCO format). Doors are removed from masks to create walkable gaps.
→ https://github.com/CubiCasa/CubiCasa5k

**Modified Swiss Dwellings (MSD)** — ~5,000+ multi-unit Swiss building floorplans. Covers diagonal walls and complex rotated layouts that CubiCasa doesn't have.
→ https://github.com/caspervanengelenburg/msd

---

## Results

| Original Floorplan | Parsed Wall Mask |
|---|---|
| <img src="assets/original.jpg" width="300"/> | <img src="assets/stitched.png" width="300"/> |

<p align="center">
  <img src="assets/debug_raw_mask.png" width="40%"/><br>
  <em>Raw probability map before thresholding</em>
</p>

---

## Part Of

S.T.I.T.C.H is the perception layer of a larger system:

**[T.R.A.G.I.C — Crowd Evacuation Simulation](https://github.com/sankhya007/T.R.A.G.I.C-Crowd-Evac)**
Real-time agent-based evacuation simulation that uses the wall masks from this model to define navigable space and run crowd flow analysis.

---

## Requirements

- Python 3.10+
- PyTorch
- OpenCV
- NumPy
- tqdm
- pycocotools *(dataset conversion only)*

```bash
pip install -r requirements.txt
```

---

## Limitations

- Trained on residential floorplans — may not generalize to industrial or highly irregular layouts
- Furniture and fine interior details are not segmented
- Very thin walls may be missed at low image resolution

---

## License

MIT — use it, build on it, give credit if it helped.

---

## Author

**Sankhyapriyo Dey**
Building tools that will make my job obsolete.