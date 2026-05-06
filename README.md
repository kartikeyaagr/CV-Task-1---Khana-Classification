# Thali Detection — Task 2

Detect individual food compartments in top-down thali images and classify each one into one of 80 Indian food categories.

## Overview

Two-stage pipeline:

1. **Detect** — YOLOv8n fine-tuned on annotated thali images finds each compartment bounding box. If YOLO finds nothing, two classical fallbacks kick in automatically: contour detection (auto-Canny + approxPolyDP) then k-means colour segmentation.
2. **Classify** — Each cropped compartment is passed through a ConvNeXt Small classifier (trained in Task 1) to produce a food label + confidence score.

## Setup

```bash
uv sync
```

Requires Python ≥ 3.13.

## Usage

### Full pipeline (detect + classify)

```bash
uv run run.py \
  --model   models/95.3-25t-8f.pt \
  --classes data/classes.txt \
  --yolo    models/thali_detector.pt \
  --images  data/images/ \
  --output  outputs/
```

Annotated images are saved to `outputs/` alongside `outputs/predictions.json`.

**Key flags:**

| Flag           | Default                           | Description                                                               |
| -------------- | --------------------------------- | ------------------------------------------------------------------------- |
| `--model`      | required                          | Path to Task 1 ConvNeXt `.pt` checkpoint                                  |
| `--classes`    | required                          | `classes.txt` — one class name per line, same order as training           |
| `--yolo`       | `yolov8x.pt`                      | YOLO weights (use fine-tuned `thali_detector.pt`)                         |
| `--model-name` | `convnext_small.fb_in22k_ft_in1k` | timm model name                                                           |
| `--img-size`   | `320`                             | Classifier input resolution; try `480` for better accuracy on small crops |
| `--det-conf`   | `0.25`                            | YOLO confidence threshold                                                 |

### Detection debug (YOLO only, no classifier)

```bash
uv run debug_detect.py --yolo models/thali_detector.pt
```

Saves annotated images with raw YOLO boxes to `outputs/debug/`. Useful for tuning detection thresholds without waiting for the classifier.

Additional flags: `--conf`, `--nms-iou`, `--max-box`, `--min-box`, `--clahe`, `--unsharp`.

### Evaluate label Precision/Recall

After running the pipeline, compare predictions against a ground truth file:

```bash
uv run evaluate.py \
  --predictions  outputs/predictions.json \
  --ground-truth data/ground_truth.json
```

Add `--verbose` to see a per-image breakdown of correct, missed, and extra labels.

**Ground truth JSON format** (either form is accepted):

```json
{ "image1.jpg": [{"label": "rice"}, {"label": "dal"}] }
```
```json
{ "image1.jpg": ["rice", "dal"] }
```

**Output:**

```
=== Micro-averaged (aggregate) ===
  Precision : 0.8571
  Recall    : 0.7500
  F1        : 0.8000
  TP=6  FP=1  FN=2

=== Macro-averaged (per-image mean) ===
  Precision : 0.8333
  Recall    : 0.7222
  F1        : 0.7738
```

**Key flags:**

| Flag               | Default  | Description                              |
| ------------------ | -------- | ---------------------------------------- |
| `--predictions`    | required | Path to `predictions.json` from `run.py` |
| `--ground-truth`   | required | Path to ground truth JSON                |
| `--output`         | —        | Save full per-image results to JSON      |
| `--verbose` / `-v` | off      | Print per-image label breakdown          |

> Only label presence is evaluated — bounding box coordinates are ignored.

### Re-train the detector

```bash
uv run finetune.py
```

Reads labelled images from `finetune_images/train/`, splits 80/20, offline-augments 10× with albumentations (full 360° rotation, flips, colour jitter), then trains for 150 epochs on MPS. Copies best weights to `models/thali_detector.pt`.

## Project Structure

```
.
├── run.py                  # Main entry point
├── evaluate.py             # Label Precision/Recall evaluation
├── finetune.py             # Fine-tune YOLOv8n on thali data
├── debug_detect.py         # Detection-only debug script
├── src/
│   ├── detector.py         # Three-stage detector (YOLO → contour → colour)
│   ├── classifier.py       # ConvNeXt wrapper; handles multiple checkpoint formats
│   ├── pipeline.py         # Chains detector + classifier per image
│   ├── preprocess.py       # Mean-shift smoothing + optional CLAHE/unsharp
│   └── visualize.py        # Draws bounding boxes + labels onto images
├── data/
│   ├── images/             # Input thali images
│   ├── classes.txt         # 80 Khana class names (one per line)
│   └── download_classes.py # Extracts class list from Task 1 checkpoint
├── finetune_images/        # Roboflow-exported YOLO annotations (19 images)
├── models/
│   ├── thali_detector.pt   # Fine-tuned YOLO detector
│   └── 95.3-25t-8f.pt      # ConvNeXt classifier
└── outputs/                # Annotated images + predictions.json
```

## Output Format

`predictions.json`:

```json
{
  "Plate 1.jpg": [
    { "label": "dal makhani", "conf": 0.9312, "bbox": [120, 85, 310, 275] },
    { "label": "aloo gobi", "conf": 0.8741, "bbox": [330, 90, 510, 270] }
  ]
}
```

## Detection Pipeline Detail

`ThaliDetector` runs three stages in order, stopping at the first that returns results:

1. **YOLO** — runs with `conf=0.05`, custom NMS, then filters by box size (1–25% of image area) and aspect ratio (0.25–4.0). Auto-selects class filter: fine-tuned models accept all their own classes; pretrained COCO models filter to `{bowl, cup, plate}`.
2. **Contour fallback** — auto-Canny thresholds from image median, morphological close, approxPolyDP polygon filter (4–8 sides), fill-ratio ≥ 0.45.
3. **Colour fallback** — k-means (k=10) in LAB space, connected components per cluster, same size/aspect filters.

## Dependencies

- `ultralytics` — YOLOv8
- `timm` — ConvNeXt model + pretrained weights
- `torch` / `torchvision` — inference
- `opencv-python` — image processing, fallback detection
- `albumentations` — offline augmentation for fine-tuning
- `Pillow` — crop extraction for classifier
