import os, random, shutil, rasterio, torch

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

from rasterio.windows import Window
from rasterio.features import shapes
from rasterio.transform import Affine
from shapely.geometry import Polygon, MultiPolygon, box, shape


def chunk_with_normalization(
    image_path,
    output_dir,
    tile_size=256,
    overlap=10,
    bands=[1, 2, 3],
    img_format="jpg",
    normalization_method="minmax"
):
    """
    Chunk a raster image into tiles with proper handling of negative values.

    Args:
        image_path: Path to input raster
        output_dir: Output directory for tiles
        tile_size: Size of each tile
        overlap: Overlap between tiles
        bands: List of bands to extract
        img_format: Output format (jpg, png)
        normalization_method: How to handle value ranges
            - "minmax": Normalize to 0-255 based on min/max values
            - "clip": Clip negative values to 0, scale positives to 255
            - "shift": Shift all values to positive range
            - "percentile": Use 2nd and 98th percentiles for robust normalization
    """
    os.makedirs(output_dir, exist_ok=True)

    # First pass: analyze the data range
    print("🔍 Analyzing data range...")
    with rasterio.open(image_path) as src:
        width, height = src.width, src.height
        transform = src.transform
        crs = src.crs

        # Sample data to understand value range
        sample_data = src.read(bands)
        data_min = float(sample_data.min())
        data_max = float(sample_data.max())
        data_mean = float(sample_data.mean())
        data_std = float(sample_data.std())

        print(f"Data range: [{data_min:.2f}, {data_max:.2f}]")
        print(f"Mean: {data_mean:.2f}, Std: {data_std:.2f}")
        print(f"Data type: {sample_data.dtype}")
        print(f"Has negative values: {data_min < 0}")

    # Calculate normalization parameters
    if normalization_method == "minmax":
        norm_min, norm_max = data_min, data_max
    elif normalization_method == "clip":
        norm_min, norm_max = 0, data_max
    elif normalization_method == "shift":
        norm_min, norm_max = data_min, data_max
    elif normalization_method == "percentile":
        p2, p98 = np.percentile(sample_data, [2, 98])
        norm_min, norm_max = float(p2), float(p98)
        print(f"Using percentile range: [{norm_min:.2f}, {norm_max:.2f}]")

    print(f"Normalization method: {normalization_method}")
    print(f"Normalization range: [{norm_min:.2f}, {norm_max:.2f}]")

    # Second pass: chunk the data
    with rasterio.open(image_path) as src:
        tile_id = 0

        for y in tqdm(range(0, height, tile_size - overlap), desc="Chunking rows"):
            for x in range(0, width, tile_size - overlap):
                win_w = min(tile_size, width - x)
                win_h = min(tile_size, height - y)

                window = Window(x, y, win_w, win_h)
                transform_tile = src.window_transform(window)

                # Read bands
                tile = src.read(bands, window=window)
                tile = np.transpose(tile, (1, 2, 0))  # Convert to HWC

                # Handle different data types and negative values
                tile = normalize_tile_data(tile, normalization_method, norm_min, norm_max)

                # Save as image
                tile_name = f"tile_{tile_id:05d}.{img_format}"
                tile_path = os.path.join(output_dir, tile_name)

                # Handle different numbers of bands
                if tile.shape[2] == 1:
                    # Grayscale
                    Image.fromarray(tile[:,:,0], mode='L').save(tile_path)
                elif tile.shape[2] == 3:
                    # RGB
                    Image.fromarray(tile, mode='RGB').save(tile_path)
                else:
                    # More than 3 bands - take first 3
                    Image.fromarray(tile[:,:,:3], mode='RGB').save(tile_path)

                # Save metadata
                with open(os.path.join(output_dir, f"tile_{tile_id:05d}.txt"), "w") as meta:
                    meta.write(f"{x},{y},{win_w},{win_h}\n")
                    meta.write(",".join([str(val) for val in list(transform_tile)[:6]]) + "\n")

                tile_id += 1

    print(f"✅ Created {tile_id} tiles in {output_dir}")
    print(f"📝 Metadata files created with spatial information")

def normalize_tile_data(tile, method, norm_min, norm_max):
    """
    Normalize tile data to 0-255 range for image output
    """
    tile = tile.astype(np.float64)  # Use float64 for precision

    if method == "minmax":
        # Standard min-max normalization
        if norm_max > norm_min:
            tile = (tile - norm_min) / (norm_max - norm_min) * 255
        else:
            tile = np.zeros_like(tile)

    elif method == "clip":
        # Clip negative values to 0
        tile = np.clip(tile, 0, norm_max)
        if norm_max > 0:
            tile = tile / norm_max * 255

    elif method == "shift":
        # Shift to positive range then normalize
        tile = tile - norm_min  # Shift so minimum becomes 0
        if norm_max > norm_min:
            tile = tile / (norm_max - norm_min) * 255

    elif method == "percentile":
        # Clip to percentile range then normalize
        tile = np.clip(tile, norm_min, norm_max)
        if norm_max > norm_min:
            tile = (tile - norm_min) / (norm_max - norm_min) * 255

    # Ensure values are in valid range and convert to uint8
    tile = np.clip(tile, 0, 255).astype(np.uint8)
    return tile

def chunk_preserve_data(
    image_path,
    output_dir,
    tile_size=256,
    overlap=10,
    bands=[1, 2, 3],
    preserve_original=True
):
    """
    Alternative: Chunk and save as GeoTIFF to preserve original data values
    """
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(image_path) as src:
        width, height = src.width, src.height
        transform = src.transform
        profile = src.profile.copy()

        # Update profile for tiles
        profile.update({
            'height': tile_size,
            'width': tile_size,
            'count': len(bands)
        })

        tile_id = 0

        for y in tqdm(range(0, height, tile_size - overlap), desc="Chunking rows"):
            for x in range(0, width, tile_size - overlap):
                win_w = min(tile_size, width - x)
                win_h = min(tile_size, height - y)

                window = Window(x, y, win_w, win_h)
                transform_tile = src.window_transform(window)

                # Read bands
                tile = src.read(bands, window=window)

                if preserve_original:
                    # Save as GeoTIFF to preserve data
                    tile_name = f"tile_{tile_id:05d}.tif"
                    tile_path = os.path.join(output_dir, tile_name)

                    # Update profile for actual tile size
                    tile_profile = profile.copy()
                    tile_profile.update({
                        'height': win_h,
                        'width': win_w,
                        'transform': transform_tile
                    })

                    with rasterio.open(tile_path, 'w', **tile_profile) as dst:
                        dst.write(tile)
                else:
                    # Normalize and save as image
                    tile_normalized = normalize_tile_data(
                        np.transpose(tile, (1, 2, 0)),
                        "minmax",
                        tile.min(),
                        tile.max()
                    )

                    tile_name = f"tile_{tile_id:05d}.png"
                    tile_path = os.path.join(output_dir, tile_name)
                    Image.fromarray(tile_normalized).save(tile_path)

                # Save metadata
                with open(os.path.join(output_dir, f"tile_{tile_id:05d}.txt"), "w") as meta:
                    meta.write(f"{x},{y},{win_w},{win_h}\n")
                    meta.write(",".join([str(val) for val in list(transform_tile)[:6]]) + "\n")

                tile_id += 1

    print(f"✅ Created {tile_id} tiles in {output_dir}")

# Example usage
if __name__ == "__main__":
    # Method 1: Normalize to handle negative values
    chunk_with_normalization(
        image_path="/content/drive/MyDrive/AGRI/Segment/patches/ne02_patch.tif",
        output_dir="/content/drive/MyDrive/AGRI/Segment/YOLOv12/test/patches",
        tile_size=256,
        overlap=10,
        bands=[1, 2, 3],
        img_format="jpg",  # PNG handles the conversion better than JPG
        normalization_method="minmax"  # or "percentile" for robust normalization
    )