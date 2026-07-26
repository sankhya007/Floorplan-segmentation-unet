# S.T.I.T.C.H
### Segmentation & Tiled Inference for TopologicalChard Handling

> Floorplan wall segmentation using UNet — built to feed structural data into **[T.R.A.G.I.C](https://github.com/sankhya007/T.R.A.G.I.C-Crowd-Evac)**, a real-time crowd evacuation simulation system.

---

## What This Does

You give it a floorplan image. It gives you back a binary mask — walls white, walkable space black. Clean, fast, and accurate enough to drive a simulation.

That mask then feeds directly into T.R.A.G.I.C, where agents navigate the parsed layout in real time during evacuation scenarios.

No manual annotation. No CAD software. Just a floorplan image and a trained model.

---

## Why It Exists

Evacuation simulations need accurate spatial maps. Getting those maps from raw architectural drawings is a pain — especially when the drawings have diagonal walls, rotated layouts, or non-standard structure.

S.T.I.T.C.H solves that. It parses the floorplan automatically and hands off a usable wall mask to the simulation layer.

---

## How It Works

Large floorplans can't just be resized and thrown at a model — you lose detail and break walls. Instead:

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
Binary mask — walls white, space black
```

The Gaussian blending is the key bit. Without it you get seams, broken edges, and missing walls at patch boundaries. With it the stitching is seamless.

---

## Model

- **Architecture:** UNet with BatchNorm
- **Input:** 256×256 RGB floorplan image
- **Output:** Binary segmentation mask
- **Loss:** BCEWithLogitsLoss
- **Optimizer:** Adam (lr=1e-4)
- **Epochs:** 15
- **Training data:** ~10,000 floorplan images

---

## Training Data

Trained on a combined dataset of two sources:

**1. CubiCasa5K**
~5,000 annotated floorplans with wall, door, and window labels in COCO format.
Doors are cut out of the mask (dilated removal) to create clean walkable gaps.
→ https://github.com/CubiCasa/CubiCasa5k

**2. Modified Swiss Dwellings (MSD)**
~5,000+ floorplans of multi-unit Swiss residential buildings.
Covers diagonal walls, rotated layouts, and complex multi-apartment structures that CubiCasa doesn't have.
→ https://github.com/caspervanengelenburg/msd

Combining both datasets significantly improves performance on non-axis-aligned walls.

---

## Trained Weights

Download from HuggingFace and place in the project root:

🔗 https://huggingface.co/sankhya007/Floorplan_parser_STITCH/tree/main

---

## Results

| | |
|---|---|
| **Original floorplan** | **Parsed wall mask** |
| <img src="assets/original.jpg" width="300"/> | <img src="assets/stitched.png" width="300"/> |

### Debug — Raw probability map before thresholding
<p align="center">
  <img src="assets/debug_raw_mask.png" width="40%"/>
</p>

### Stitching in action
<p align="center">
  <img src="assets/stitching.gif" width="40%"/>
</p>

---

## Installation

```bash
git clone https://github.com/sankhya007/S.T.I.T.C.H
cd S.T.I.T.C.H
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

**Single image:**
```bash
python predict.py
```

**Large floorplan (tiled inference):**
```bash
python predict_tiled.py
```

Set `IMAGE_PATH` in either script to your input file. Output saves as `prediction.png` or `stitched_mask.png`.

---

## Project Structure

```
S.T.I.T.C.H/
├── model.py              # UNet architecture
├── train.py              # Training loop
├── predict.py            # Single image inference
├── predict_tiled.py      # Tiled inference for large images
├── dataset.py            # Dataset loader
├── convert_cubicasa.py   # CubiCasa → binary mask converter
├── convert_msd.py        # MSD → binary mask converter
├── diagram_stitch.py     # Generates stitching visualization GIF
├── assets/
│   ├── original.jpg
│   ├── stitched.png
│   ├── debug_raw_mask.png
│   └── stitching.gif
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Part Of

S.T.I.T.C.H is the perception layer of a larger system:

**[T.R.A.G.I.C — Crowd Evacuation Simulation](https://github.com/sankhya007/T.R.A.G.I.C-Crowd-Evac)**
Real-time agent-based evacuation simulation that uses the wall masks produced by this model to define navigable space and run crowd flow analysis.

---

## Requirements

- Python 3.10+
- PyTorch
- OpenCV
- NumPy
- tqdm
- pycocotools (for dataset conversion only)

```bash
pip install -r requirements.txt
```

---

## Limitations

- Trained on residential floorplans — may not generalize to industrial or highly irregular layouts
- Furniture and fine interior details are not segmented
- Very thin walls may be missed depending on image resolution

---

## License

MIT — do what you want, give credit if it helped.

---

## Author

**Sankhyapriyo Dey**
Building tools that will make my jobs obsolete.