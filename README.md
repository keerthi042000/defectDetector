# Corseco Quality Inspector

Simple AI-powered quality inspection for product images using a YOLO detector. The system classifies each product as `Good Quality`, `Minor Defect`, or `Major Defect`, draws defect bounding boxes, and generates a short inspection report with confidence.

## What It Does

- Detects visible product defects from one or more images.
- Classifies inspection quality into `Good Quality`, `Minor Defect`, or `Major Defect`.
- Saves annotated images with defect bounding boxes.
- Produces a machine-readable `report.json`.
- Uses a local open-source LLM to turn structured findings into a short 3-line inspection report.

## Project Files

- `quality_inspector.py`: main inference and report generation script
- `models/best.pt`: trained YOLO weights
- `pipeline.ipynb`: original data-prep and training notebook
- `dataset_yolo/`: YOLO-formatted dataset used for the detector

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Inference

Single image:

```bash
python3 quality_inspector.py dataset_yolo/images/val/capsule_crack_009_323.png --product-id capsule_sample
```

Multiple images of the same product:

```bash
python3 quality_inspector.py \
  dataset_yolo/images/val/capsule_crack_009_323.png \
  dataset_yolo/images/val/capsule_faulty_imprint_001_341.png \
  --product-id capsule_batch
```

Open-source LLM report generation:

```bash
export HF_REPORT_MODEL=google/flan-t5-base
python3 quality_inspector.py dataset_yolo/images/val/capsule_crack_009_323.png --product-id llm_demo --use-llm
```

The default is now `google/flan-t5-base`, which is a stronger free open-source model than the smaller variant. The first run downloads the selected Hugging Face model locally.

Outputs are saved under `outputs/<product-id>_<timestamp>/`:

- `annotated/*.png`
- `report.json`

## Example Output

```text
Inspection Result: FAIL
Issue: Major structural defect detected
Confidence: 41%
```

## Edge Deployment

This model is a good fit for edge devices because it already uses a compact YOLO backbone. A production edge path would be:

1. Export the detector to `ONNX`, `TensorRT`, or `OpenVINO`.
2. Run batch size `1` at a fixed image size such as `640x640`.
3. Quantize to `FP16` or `INT8` to reduce latency and memory.
4. Deploy on Jetson, industrial PCs, or ARM devices with a lightweight Python or C++ runtime.
