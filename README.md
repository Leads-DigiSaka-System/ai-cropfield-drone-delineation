# Crop Field Segmentation from Drone Imagery

A deep learning pipeline for segmenting and delineating crop fields from aerial/drone imagery using four architectures: YOLO11n segmentation, FracTAL ResUNet, SAM-FracTAL ResUNet and DINO-FracTAL ResUNet.

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
│   ├── SAM-FractalResUNET/
│   │   ├── Module01_ModelTraining-SAMFractalResUNET.ipynb  # SAM-FracTAL training
│   │   └── Module02_ModelInference-SAMFractalResUNET.ipynb # SAM-FracTAL inference
│   └── DINO-FractalResUNET/
│       ├── Module01_ModelTraining-DINOFractalResUNET.ipynb  # DINOv3/DINOv2-FracTAL training
│       └── Module02_ModelInference-DINOFractalResUNET.ipynb # DINOv3/DINOv2-FracTAL inference
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
- **Architecture**: FracTAL ResUNet with a frozen SAM ViT-B image encoder (89M params)
- **Fusion**: `GatedFusion` at three U-Net levels (enc3 skip at 64×64, enc4 skip and bottleneck at 32×32).
  SAM features are projected, resized to the level's grid and added through a learned per-pixel gate:
  `out = x + γ · gate(x, e) · e`. γ starts at 0.5, trains at the normal learning rate and has no weight decay.
- **Trainable**: FracTAL + fusion modules; checkpoints exclude the SAM weights

> **Why the fusion was rebuilt (v03):** in the previous version (cross-attention at the bottleneck only),
> the SAM gate ended training at tanh(α) = 0.012, so SAM was effectively unused. The gate trained at a
> learning rate of 1e-5 with weight decay, so it could not move more than ~0.03. The attention also had no
> positional encoding, so spatial alignment with SAM's features was lost.

### 4. DINO-FracTAL ResUNet
- **Architecture**: same U-Net and `GatedFusion`, with a frozen DINO ViT. Patch tokens from three depths
  (½, ¾ and the last block) feed the three fusion levels: shallower layers carry more local detail, the
  last layer the most semantics.
- **Backbone**: the first one of these that loads is used and recorded in the checkpoint:
  1. `facebook/dinov3-vitl16-pretrain-sat493m`: DINOv3 ViT-L/16 pretrained on satellite imagery
  2. `facebook/dinov3-vitb16-pretrain-lvd1689m`: DINOv3 ViT-B/16 pretrained on web images
  3. `facebook/dinov2-with-registers-base`: DINOv2 ViT-B/14 (Apache-2.0, not gated)
- **DINOv3 access**: the weights are free to use under Meta's DINOv3 License but gated on Hugging Face.
  Accept the licence on the model page, then add an `HF_TOKEN` secret in Colab. Without it, the notebook
  falls back to DINOv2 automatically.

### Training changes shared by SAM- and DINO-FracTAL
- **Instance-aware targets**: boundaries are built from per-field instance ids, so the edge shared by two
  touching fields is marked. Previously all fields had the same label, so shared edges were never
  supervised, which was a main cause of merged fields.
- **Per-field distance target**: 0 on the edge and 1 at each field's centre, whatever its size.
- **Native-resolution crops**: random 512 crops (±25% scale, 90° rotations and flips) instead of squashing
  whole images to 512, so training matches the scale of tiled inference.
  Targets are computed after the geometric transforms.
- **Thinner boundaries**: boundary BCE `pos_weight` 20 → 3 plus soft Dice; boundary weight in the
  segmentation CE 12 → 3.
- **Field-level model selection**: validation runs the full watershed post-processing and matches fields
  to ground truth (IoU > 0.5). The best checkpoint is chosen on field F1, with early stopping.
  Boundary F1 uses a 2 px tolerance.
- **Post-processing tuning + test**: after training, the watershed thresholds are grid-searched on the
  validation set and stored in `best.pt`, and final metrics are reported on the test split.
- **Mixed precision**: bf16 on A100/L4, fp16 with a gradient scaler on T4.

## Training Data

- **Source**: Roboflow Crop-Field dataset
- **Format**: COCO segmentation format
- **Splits**: 384 train / 30 validation / 14 test images
- **Annotations**: 8,610 field polygons

## Inference Pipeline

### SAM- / DINO-FracTAL Inference

1. **Tiled inference**: 512 px tiles, 128 px overlap, raised-cosine blending, batched tiles.
   The image is reflect-padded first, so border pixels are not zeroed. For speed and low CPU load:
   - tiles with no valid pixels (nodata, alpha or mask) are skipped
   - normalisation and tile blending run on the GPU
   - the U-Net uses channels-last memory on GPU
   - h/v flip TTA is off by default (`tta=True` turns it on, at 3× the compute)
   - PyTorch/OpenCV threads are capped via `cpu_threads` (default 2 on GPU runtimes)
2. **Seeded watershed** (as in the FracTAL paper), after a light Gaussian blur of the maps
   (`smooth_sigma`, 1 px): field cores = high distance, low boundary probability. Flooding
   `boundary − distance` inside the field extent splits touching fields while each keeps its full extent.
3. **Polygons**: pixel edges are traced with `rasterio.features.shapes`, small background holes are
   filled, then all fields are simplified together with `shapely.coverage_simplify`. Pixel staircases
   become straight edges, and shared edges stay shared (no gaps or overlaps). `smooth_px` (default 6)
   sets the strength. Coordinates are in the image CRS (area in m² for projected CRSs);
   non-georeferenced images stay in pixel coordinates with `crs=None`.
4. **Field filters** (`CONFIG['filters']`), measured on the smoothed polygons:
   - **Circular objects**: there are no circular fields in the Philippine context. A field is removed
     only if it is round by *both* measures: Polsby-Popper circularity 4πA/P² ≥ 0.90 *and* it fills
     ≥ 85 % of its smallest enclosing circle. Squares (0.785 / 0.64), compact pentagons and hexagons,
     ellipses and irregular fields are kept.
   - **Implausibly small fields**: a field is removed only if its area is < 5 % of the scene's
     *median* field area, *and* the scene has ≥ 10 fields, *and* it is under 200 m² (georeferenced
     images), *and* it does not touch the image edge or nodata (such fields are often real but
     only partly visible).
   - Removed fields are saved to `<name>_removed_fields.geojson` with a `filter_reason`, and drawn in
     red in the result image, so every removal can be reviewed. Set `filter_circular` / `filter_small`
     to `False` to turn a filter off.
5. **Output**: GeoJSON with `n_vertices`, `circularity`, `circle_fill`, `area_rel` (area / median)
   and `touches_edge` per field, plus a visualisation PNG (downsampled for large images, optional via
   `save_visualization`). Watershed thresholds come from the checkpoint (tuned on validation) and can
   be overridden in `CONFIG['postprocess']`.

Benchmark on a 3000×3000 px synthetic orthomosaic (CPU, 4 cores, small stand-in network, 25% nodata):

| | Before | After |
|---|---|---|
| Tiled inference | 135.7 s wall / 476 s CPU | 29.8 s wall / 102 s CPU |
| Fields (110 true) | 116 | 110 |
| Vertices per polygon | 211 | 22 |
| Overlap between neighbouring polygons | 1.41 m² | 0 |
| IoU with the true field outlines | 0.972 | 0.973 |

> Checkpoints from the previous SAM-FracTAL version (`sam-fractal-resunet-512-v02`) are not compatible
> with the new architecture. Retrain with the updated notebook (outputs go to `...-v03`).

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
- transformers >= 4.56 (for DINOv3 / DINOv2)
- scikit-image, pycocotools

### Installation
```bash
pip install torch torchvision
pip install ultralytics roboflow
pip install rasterio geopandas
pip install segment-anything "transformers>=4.56" scikit-image pycocotools
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
| SAM-FracTAL | `best.pt` (in notebook output) | SAM fusion trained model (+ model config and tuned post-processing) |
| DINO-FracTAL | `best.pt` (in notebook output) | DINOv3/DINOv2 fusion trained model (+ backbone name and tuned post-processing) |

## License

[Specify license if applicable]