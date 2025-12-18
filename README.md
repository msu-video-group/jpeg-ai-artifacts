# Artifact Detection Metrics

This repository contains a set of metrics for detecting artifacts in images. Each metric is implemented in a separate script and compares the quality of classical and neural network image processing methods relative to the original.

## Repository Structure

- `color.py` - metric based on color differences (CIEDE2000)
- `edge.py` - metric based on edge analysis
- `struct_metric.py` - metric for structural differences (MSSSIM)
- `text.py` - metric for detecting artifacts in text regions (requires MMOCR)
- `test_data/` - test data:
  - `orig/` - original images (ground truth)
  - `ai/` - images processed by neural network methods
  - `classic/` - images processed by classical methods

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For the `text` metric, MMOCR installation with a compatible mmcv version is required:
```bash
# Install compatible mmcv version (mmcv>=2.0.0rc4,<2.1.0)
pip install "mmcv>=2.0.0rc4,<2.1.0"
pip install mmengine mmdet mmocr
```

## Usage

### Run all metrics

```bash
python run_metrics.py --all
```

### Run a specific metric

```bash
python run_metrics.py --metric color
python run_metrics.py --metric edge
python run_metrics.py --metric struct
python run_metrics.py --metric text
```

### Additional parameters

```bash
# Specify device (cpu/cuda)
python run_metrics.py --metric color --device cuda

# Specify path to test data
python run_metrics.py --metric color --data_dir ./test_data

# Specify output directory
python run_metrics.py --metric color --output_dir ./results
```

## Metric Descriptions

### Color Metric (`color.py`)
The metric is based on calculating color differences between the original and processed images using the CIEDE2000 standard. The metric builds a heatmap of differences and finds regions with maximum differences between neural network and classical methods.

### Edge Metric (`edge.py`)
The metric analyzes image edges using the Canny operator. Compares gradients of original, neural network, and classical images to detect artifacts.

### Struct Metric (`struct_metric.py`)
The metric uses MSSSIM (Multi-Scale Structural Similarity Index) to calculate structural differences. Identifies regions with the greatest structural differences between processing methods using local window computations.

### Text Metric (`text.py`)
Specialized metric for detecting artifacts in text regions. Uses MMOCR for text detection and FSIM for quality assessment of text regions.

## Result Format

After execution, the results directory contains images with visualization:
- Drawn bounding boxes around detected artifacts
- Labeled metric values for each region
- Separate images for each original-processed pair

Results are saved in the following structure:
```
output_dir/
  metric_name/
    image_name_0.png
    image_name_1.png
    ...
```

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- Pillow
- NumPy
- Kornia (for edge metric)
- Piqa (for text metric)
- MMOCR (optional, for text metric)
- scikit-image (for struct metric)
