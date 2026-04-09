# Floorplan Parser & Segmentation System

A deep learning-based system to **parse architectural floorplans**, extract structural layouts, and prepare them for **simulation and analysis**.

---

## Features

- UNet-based floorplan segmentation
- Patch-based inference for large images
- Overlapping tile stitching (edge-aware)
- Accurate wall extraction from floorplans
- Handles large-scale layouts efficiently

---

## Results

### Original Floorplan

<p align="center">
  <img src="assets/original.jpg" width="500"/>
</p>

---

### Raw Model Prediction (Before Fix)

<p align="center">
  <img src="assets/prediction.jpg" width="500"/>
</p>

---

### Final Stitched Output (After Fix)

<p align="center">
  <img src="assets/stitched.png" width="500"/>
</p>

---

## Problem Faced

When running segmentation on large floorplans:

- Model performed poorly on full-size images  
- Edges were broken or missing  
- Large layouts were not parsed correctly  
- Context loss at boundaries  

---

## Solution: Patch-Based Segmentation + Smart Stitching

To fix this, we implemented a **sliding window inference system with overlap and blending**.

---

### Step 1: Image Tiling

- Split large image into smaller patches (e.g., 256×256)
- Allows model to focus on local features

---

### Step 2: Overlapping Inference

- Use stride < patch size  
- Example:
  Patch Size = 256  
  Stride     = 128  

- Each region is predicted multiple times

---

### Step 3: Weighted Blending (Key Insight)

- Center of patch = high confidence  
- Edges = low confidence  

Weight map concept:

Center → Strong  
Edges  → Weak  

This removes:
- seams
- edge artifacts
- broken walls

---

### Step 4: Stitching

We combine all patch predictions using:

- weighted accumulation
- normalization

Final formula:

```
final_mask = sum(predictions * weights) / sum(weights)
```

---

## Pipeline

```
        Input Image  
            V 
Split into overlapping patches  
            V  
      UNet Prediction  
            V  
     Weighted blending  
            V  
    Final stitched mask  
```

---

## Project Structure

```
parser-model/
├── model.py
├── train.py
├── predict.py
├── predict_tiled.py
├── diagram_stitch.py
├── assets/
│ ├── original.png
│ ├── prediction.png
│ ├── stitched.png
│ ├── stitching_diagram.png
│ └── stitching.gif
├── README.md
├── LICENSE
└── .gitignore
```

---

## python

python version used: 3.10.x

---

## Dataset

This project was trained using the **CubiCasa5K** dataset.

CubiCasa5K is a large-scale annotated floorplan dataset containing detailed architectural layouts with semantic labels.

Due to size and licensing constraints, the dataset is not included in this repository.

You can access the dataset here:  
https://github.com/CubiCasa/CubiCasa5k

If you use this dataset in your work, please consider citing the original authors.

---

## Installation 

```
pip install torch torchvision opencv-python numpy imageio
```

---

## Requirements

```
 pip install -r requirements.txt
```

---

## How to Run / Usage

python predict_tiled.py

---

## Model Details

- Architecture: UNet  
- Training data: ~1000 floorplan images  
- Training epochs: 15  
- Input resolution: 256 × 256  

---

## Limitations

- Model is trained on limited data and may not generalize to all architectural styles  
- Fine details such as doors and furniture are not explicitly modeled  
- Performance depends on preprocessing consistency  

---

## How Stitching Works

Large floorplans are processed using a **sliding window approach with overlap**.

### Step-by-step:

1. The image is split into overlapping patches  
2. Each patch is passed through the model  
3. Predictions are combined using a weighted map  
4. Overlapping regions are averaged to remove seams  

---

### Visualization

<p align="center">
  <img src="assets/stitching_diagram.png" width="500"/><br>
  <em>Overlapping patch concept for stitching</em>
</p>

---

### Stitching in Action

<p align="center">
  <img src="assets/stitching.gif" width="500"/><br>
  <em>Sliding window stitching in action</em>
</p>

The diagram explains how overlapping patches work, while the GIF shows the sliding window processing across the image in real time.

---

### Why Overlap Matters

Without overlap:
- Broken edges
- Missing walls  

With overlap:
- Smooth transitions
- Accurate structure  

---

## Important Notes

- Full-image inference is unreliable for large layouts  
- Always use tiled inference for best results  
- Preprocessing must match training pipeline  
- Normalization is critical for correct predictions  

---

## Key Achievement

- Successfully solved large floorplan parsing issue  
- Eliminated edge artifacts using overlap + blending  
- Improved segmentation quality significantly  

---

## Future Improvements

- Multi-scale inference  
- Test-time augmentation  
- Vectorization of wall structures  
- Integration with evacuation simulation systems  
- Detection of semantic elements (doors, exits)  

---

## License

This project is licensed under the MIT License.

---

## Author 

Sankhyapriyo Dey
