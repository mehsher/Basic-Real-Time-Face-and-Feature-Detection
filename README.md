# Real-Time Object Detection System

A real-time object detection application built with Python and OpenCV.
Captures live webcam input, detects faces, eyes, and full bodies using
Haar Cascade classifiers, and overlays live results (bounding boxes,
labels, FPS, detection counts) on the video feed.

## Features

- Live camera capture and per-frame processing
- Multi-class detection (face / eye / body) via Haar Cascades
- Real-time FPS counter (smoothed, not jittery)
- On-screen HUD with per-class detection counts
- Snapshot capture (`S`) and live recording to `.avi` (`R`)
- Pause/resume (`P`) without killing the camera stream
- Command-line flags for camera index, resolution, and sensitivity

## Project Structure

```
realtime-object-detection/
├── main.py              # Entry point: camera loop, controls, HUD
├── detector.py           # ObjectDetector class + FPS tracking
├── utils/
│   └── overlay.py        # Drawing: boxes, labels, HUD panel
├── output/                # Snapshots and recordings land here
└── requirements.txt
```


## Setup

```bash
pip install -r requirements.txt
python main.py
```

## Usage

```bash
python main.py                      # default webcam
python main.py --camera 1           # use a second camera
python main.py --width 1280 --height 720
python main.py --min-neighbors 8    # stricter detection, fewer false positives
```

### Controls (while the window is focused)

| Key | Action |
|-----|--------|
| Q   | Quit |
| S   | Save snapshot to `output/` |
| P   | Pause / resume feed |
| R   | Start / stop recording to `output/` |

## How it works

Each frame is converted to grayscale and histogram-equalized (evens out
lighting before detection). It's then run through Haar Cascade
classifiers for faces, eyes, and full bodies. Each classifier returns
bounding boxes for whatever it finds, which get drawn back onto the
original color frame with class-specific colors and labels.

Detection parameters (`scaleFactor`, `minNeighbors`, `minSize`) are
exposed as constructor arguments on `ObjectDetector` so accuracy vs.
speed can be tuned without touching the detection logic itself.

## Why Haar Cascades

They're bundled with OpenCV, require no external model downloads, and
run comfortably in real time on a CPU — good fit for a lightweight,
dependency-free demo. Swapping in a deep learning detector (YOLO via
OpenCV's DNN module) later would only mean changing `detector.py`;
`main.py` and the overlay code wouldn't need to change.

## Requirements

- Python 3.8+
- A connected webcam
