# Crop Field Segmentation from Drone Imagery

A deep learning pipeline for segmenting and delineating crop fields from aerial/drone imagery using three state-of-the-art architectures: YOLO11n segmentation, FracTAL ResUNet, and SAM-FracTAL ResUNet.

## Overview

This project provides tools to:
- Segment crop fields from RGB orthoimagery using deep learning models
- Extract field boundaries with high precision
- Generate vector outputs (GeoJSON, CSV) with field geometries and statistics
- Process large images via tiled inference with seamless blending

## Project Structure

```
.
├── notebooks/
│   ├── YOLO/
│   │   ├── Module01_ModelTraining-YOLO11n.ipynb   # YOLO11n-seg training
│   │   └── Module02_ModelInference-YOLO11n.ipynb  # YOLO11n inference
│   ├── FractalResUNET/
│   │   ├── Module01_ModelTraining-FractalResUNET.ipynb  # FracTAL ResUNet training
│   │   └── Module02_ModelInference-FractalResUNET.ipynb # Tiled inference pipeline
│   └── SAM-FractalResUNET/
│       ├── Module01_ModelTraining-SAMFractalResUNET.ipynb  # SAM-FracTAL training
│       └── Module02_ModelInference-SAMFractalResUNET.ipynb # SAM-FracTAL inference
├── src/
│   └── data_tilling.py       # Raster image tiling utility
├── models/
│   └── YOLO11n/
│       ├── yolo11n-seg-cropfield-v1.pt  # Trained YOLO11n model
│       └── yolo11n-seg-cropfield-v2.pt  # Optimized model version
├── result/
│   ├── cropfield_fractal_segmentation.png  # Sample output visualization
│   ├── cropfield_fractal_fields.csv        # Field geometries (CSV)
│   └── cropfield_fractal_statistics.json   # Processing statistics
└── images/                                  # Sample input images
```

## Model Architectures

### 1. YOLO11n Segmentation
- **Framework**: Ultralytics YOLO
- **Architecture**: YOLOv11 segmentation variant
- **Input Size**: 640x640 pixels
- **Strengths**: Fast inference, good for real-time applications
- **Training**: 100 epochs, AdamW optimizer, cosine LR schedule
- **Parameters**: ~2.8M

### 2. FracTAL ResUNet
- **Architecture**: Custom PyTorch implementation based on Waldner et al. (2021)
- **Input Size**: 512x512 pixels
- **Outputs**: Three-head architecture
  - Segmentation mask
  - Boundary probability map
  - Distance-to-boundary map
- **Key Components**:
  - FracTAL blocks with channel + spatial attention
  - Residual connections
  - Multi-task loss (boundary-weighted CE + Dice + boundary BCE + distance MSE)
- **Training**: 120 epochs, ~21.8M parameters
- **Strengths**: Excellent boundary precision, handles field merging prevention

### 3. SAM-FracTAL ResUNet
- **Architecture**: Hybrid model combining SAM ViT-B encoder with FracTAL decoder
- **Frozen Components**: SAM ViT-B image encoder (89M params, frozen)
- **Trainable Components**: FracTAL decoder + SAMFusion cross-attention (~22M params)
- **SAMFusion**: Cross-attention module that injects SAM semantic features into FracTAL bottleneck via a learned residual gate (α)
- **Strengths**: Rich semantic features from SAM improve boundary detection in complex scenes

## Training Data

- **Source**: Roboflow Crop-Field dataset
- **Format**: COCO segmentation format
- **Splits**: 384 train / 30 validation / 14 test images
- **Annotations**: 8,610 field polygons

## Inference Pipeline

### FracTAL ResUNet Tiled Inference

The inference pipeline (`Module02_ModelInference-FractalResUNET.ipynb`) implements:

1. **Image Loading**: Rasterio-based reader with optional RAM caching
2. **Tiled Inference**:
   - 512px tiles with 128px overlap
   - Cosine blending window for seamless mosaicking
   - FP16 for memory efficiency
3. **Multi-head Feature Fusion**:
   - Segmentation and boundary predictions blended across tiles
   - Boundary map used to cut segmentation mask (prevents field merging at tile boundaries)
4. **Post-processing**:
   - Connected components for instance extraction
   - Morphological operations for gap filling
   - Polygon simplification
   - Non-maximum suppression (NMS) for overlapping detections
5. **Georeferencing**: Transform pixel coordinates to GIS coordinates
6. **Output**: GeoJSON, CSV, visualization PNG, statistics JSON

### Configuration Options

```python
config = {
    'tile_size': 512,
    'overlap': 128,
    'resolution': 1.0,
    'batch_size': 8,
    'bound_thresh': 0.30,      # Boundary threshold for mask cutting
    'min_field_px': 300,       # Minimum field area in pixels
    'smooth_epsilon': 0.5,     # Polygon simplification epsilon
    'polygon_nms_iou': 0.4,    # NMS IoU threshold
    'buffer_distance': 20,     # Polygon buffer distance
}
```

## Source Code

### data_tilling.py

Utility for chunking large raster images into tiles for training:

```python
# Chunk raster with normalization
chunk_with_normalization(
    image_path="ortho.tif",
    output_dir="tiles/",
    tile_size=1280,
    overlap=128,
    bands=[1, 2, 3],
    normalization_method="minmax"  # or "percentile", "clip", "shift"
)
```

## Sample Results

| Metric | Value |
|--------|-------|
| Total Fields Detected | 34 |
| Average Boundary Confidence | 0.043 |
| Average Field Area | 13,570 px |
| Processing Time | 0.05 min |

## Requirements

### Core Dependencies
- Python >= 3.8
- PyTorch >= 2.0
- Ultralytics (for YOLO)
- rasterio, geopandas (for geospatial I/O)
- segment-anything (for SAM model)
- kornia, pycocotools

### Installation
```bash
pip install torch torchvision
pip install ultralytics roboflow
pip install rasterio geopandas
pip install segment-anything kornia pycocotools
```

## Usage

### Training (YOLO)
```python
from ultralytics import YOLO
model = YOLO('yolo11n-seg.pt')
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    device=0,
    batch=16
)
```

### Inference (FracTAL)
```python
from notebooks.FractalResUNET.Module02_ModelInference import run_cropfield_segmentation

gdf, detections = run_cropfield_segmentation(
    image_path="drone_image.png",
    model=model,
    config=config
)
```

## Model Versions

| Model | File | Description |
|-------|------|-------------|
| YOLO v1 | `yolo11n-seg-cropfield-v1.pt` | Initial trained model |
| YOLO v2 | `yolo11n-seg-cropfield-v2.pt` | Optimized inference version |
| FracTAL | `best.pt` (in notebook output) | Best boundary F1 checkpoint |
| SAM-FracTAL | `best.pt` (in notebook output) | SAM fusion trained model |

## License

[Specify license if applicable]