"""
detector.py

Core detection engine. Wraps OpenCV's Haar Cascade classifiers behind a
single class so main.py doesn't need to know how detection actually works
under the hood — it just asks for detections and gets boxes back.

Why Haar Cascades instead of a deep learning model:
- Zero external downloads, runs fully offline, works on any machine with
  a CPU (no GPU / CUDA setup required).
- Fast enough for real-time (30+ FPS on a laptop webcam at 640x480).
- Good enough accuracy for faces/eyes/bodies without the overhead of
  a full YOLO pipeline.
"""

import cv2
import os
import time


class ObjectDetector:
    """
    Loads one or more Haar Cascade classifiers and runs them against
    video frames. Each classifier is tagged with a label and a color
    so multiple object types can be detected and distinguished visually
    in the same frame.
    """

    # Each entry: (label, cascade filename, BGR color for the box)
    DEFAULT_TARGETS = [
        ("Face", "haarcascade_frontalface_default.xml", (0, 255, 0)),
        ("Eye", "haarcascade_eye.xml", (255, 200, 0)),
        ("Body", "haarcascade_fullbody.xml", (0, 140, 255)),
    ]

    def __init__(self, targets=None, scale_factor=1.1, min_neighbors=5, min_size=(30, 30),
                 eye_min_neighbors=10):
        """
        targets: optional custom list of (label, cascade_filename, color).
                 Defaults to face/eye/body detection.
        scale_factor: how much the image size is reduced at each image scale.
                      Lower = more accurate but slower.
        min_neighbors: how many neighbors each candidate rectangle needs
                       to retain it. Higher = fewer false positives.
        min_size: minimum possible object size; smaller objects are ignored.
        eye_min_neighbors: eyes are small, low-contrast targets that are
                           especially prone to false positives (shirt
                           texture, background clutter). They get their
                           own, stricter threshold rather than sharing
                           min_neighbors with everything else.
        """
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.eye_min_neighbors = eye_min_neighbors

        self.classifiers = {}  # label -> (cv2.CascadeClassifier, color)
        targets = targets or self.DEFAULT_TARGETS
        cascade_base = cv2.data.haarcascades

        for label, filename, color in targets:
            path = os.path.join(cascade_base, filename)
            classifier = cv2.CascadeClassifier(path)
            if classifier.empty():
                # Don't crash the whole app over one missing cascade —
                # just skip it and keep going with the rest.
                print(f"[warn] Could not load cascade for '{label}' at {path}, skipping.")
                continue
            self.classifiers[label] = (classifier, color)

        if not self.classifiers:
            raise RuntimeError("No cascade classifiers loaded. Check your OpenCV install.")

    def detect(self, frame):
        """
        Runs all loaded classifiers against a single BGR frame.

        Faces and bodies are detected directly against the full frame.
        Eyes are instead searched for ONLY inside already-detected face
        regions — this is the standard approach, since an eye cascade
        run against a whole scene will happily match shirt folds, wood
        grain, and other texture that vaguely resembles an eye. Scoping
        the search to "inside a face we already found" removes almost
        all of those false positives for free.

        Returns a list of detections: (label, x, y, w, h, color)
        Coordinates are in the frame's own pixel space.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # improves detection in uneven lighting

        detections = []
        face_boxes = []

        for label, (classifier, color) in self.classifiers.items():
            if label == "Eye":
                continue  # handled separately below, scoped to face regions

            boxes = classifier.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
            )
            for (x, y, w, h) in boxes:
                detections.append((label, x, y, w, h, color))
                if label == "Face":
                    face_boxes.append((x, y, w, h))

        if "Eye" in self.classifiers and face_boxes:
            eye_classifier, eye_color = self.classifiers["Eye"]
            for (fx, fy, fw, fh) in face_boxes:
                # Eyes sit in the upper ~60% of a face; restricting the
                # search vertically as well cuts out mouth/beard/chin
                # false positives that the face-only crop wouldn't.
                roi_h = int(fh * 0.6)
                roi = gray[fy:fy + roi_h, fx:fx + fw]

                eye_boxes = eye_classifier.detectMultiScale(
                    roi,
                    scaleFactor=self.scale_factor,
                    minNeighbors=self.eye_min_neighbors,
                    minSize=(20, 20),
                )
                for (ex, ey, ew, eh) in eye_boxes:
                    # Translate coordinates back from the face-local ROI
                    # into full-frame coordinates.
                    detections.append(("Eye", fx + ex, fy + ey, ew, eh, eye_color))

        return detections


class FPSCounter:
    """
    Tracks a rolling average FPS so the on-screen counter doesn't jitter
    wildly frame to frame (which raw 1/dt would do).
    """

    def __init__(self, smoothing=0.9):
        self.smoothing = smoothing
        self.fps = 0.0
        self._last_time = time.time()

    def tick(self):
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        if dt > 0:
            current_fps = 1.0 / dt
            self.fps = (self.fps * self.smoothing) + (current_fps * (1 - self.smoothing))
        return self.fps
