# Real-Time-CV-Inference-System

A real-time object detection system built with OpenCV and YOLO11n. It detects objects from a webcam feed and draws a bounding box with a label and confidence score for each one.

The YOLO11n model has been exported to ONNX format, and ONNX is used for inference. This makes the model usable across multiple platforms, including edge devices.

## Features

- Object detection using YOLO11n
- ONNX-based inference
- Webcam input
- Bounding boxes with label and confidence for each detection
- Detects the object classes in the COCO dataset (the dataset YOLO11n was trained on)

## Tech Stack

- Python
- OpenCV
- ONNX
- YOLO (YOLO11n)
- NumPy

## Modular Design

The project follows a modular practice, with each task separated into its own module:

- **Preprocessing:** prepares the input frame for the model
- **Inference:** runs the ONNX model
- **Postprocessing:** processes the model output into detections (bounding box, label, confidence)
- **Profiling:** measures per-stage latency and pipeline FPS

## Project Structure

```
.
├── models/
├── src/
│   ├── __init__.py
│   ├── benchmark_model.py
│   ├── config.py
│   ├── inference.py
│   ├── main.py
│   ├── postprocessing.py
│   ├── preprocessing.py
│   └── profiling.py
├── tests/
├── .gitignore
├── labels.json
├── README.md
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

The ONNX model is located in the `models/` directory (`models/*.onnx`).

Run the project with:

```bash
python -m src.main
```

Press `q` to quit.

## Performance

Measured per-stage latency and overall pipeline speed:

| Stage        | Mean (ms) | p50 (ms) | p90 (ms) | p99 (ms) |
|--------------|-----------|----------|----------|----------|
| Capture      | 5.07      | 2.15     | 2.692    | 4.258    |
| Preprocess   | 4.24      | 4.02     | 5.205    | 6.808    |
| Inference    | 55.01     | 54.01    | 63.108   | 72.144   |
| Postprocess  | 1.76      | 1.66     | 2.236    | 2.968    |
| Display      | 8.78      | 8.39     | 15.436   | 16.848   |
| **Pipeline** | 74.86     | 76.81    | 79.728   | 95.192   |

**Pipeline FPS: 13.35**

## Goals

- **Now:** run object detection on a laptop.
- **Future:** deploy on edge devices, and potentially on smart glasses (like Meta's) if a model small enough to fit becomes available.