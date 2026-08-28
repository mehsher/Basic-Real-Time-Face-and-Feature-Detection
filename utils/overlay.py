"""
utils/overlay.py

Everything related to drawing on the frame and rendering the HUD
(heads-up display). Kept separate from detection logic so the drawing
style can change without touching detection code at all.
"""

import cv2
import time
from collections import Counter


def draw_detections(frame, detections):
    """
    Draws a bounding box + label for every detection on the frame,
    in place. Returns nothing — frame is mutated directly since that's
    the cheapest option for a real-time loop (no extra copies per frame).
    """
    for (label, x, y, w, h, color) in detections:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        text = label
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Filled label background so text stays readable over busy footage
        cv2.rectangle(frame, (x, y - text_h - 8), (x + text_w + 6, y), color, -1)
        cv2.putText(
            frame, text, (x + 3, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )


def draw_hud(frame, fps, detections, paused=False, recording=False):
    """
    Draws the top-left status panel: FPS, per-class counts, and
    current mode indicators (paused/recording).
    """
    h, w = frame.shape[:2]

    counts = Counter(label for (label, *_rest) in detections)
    summary = "  ".join(f"{label}: {count}" for label, count in counts.items()) or "No detections"

    lines = [
        f"FPS: {fps:.1f}",
        summary,
    ]

    if paused:
        lines.append("PAUSED")
    if recording:
        lines.append("REC")

    # Semi-transparent backing bar for readability regardless of scene
    overlay = frame.copy()
    bar_height = 24 * len(lines) + 16
    cv2.rectangle(overlay, (0, 0), (min(w, 420), bar_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    y = 22
    for line in lines:
        color = (0, 0, 255) if line in ("PAUSED", "REC") else (255, 255, 255)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y += 24


def draw_controls_hint(frame):
    """Small persistent hint in the bottom-left so controls aren't a guessing game."""
    h, w = frame.shape[:2]
    hint = "Q: quit   S: snapshot   P: pause   R: record"
    cv2.putText(
        frame, hint, (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA
    )


def timestamp():
    """Consistent filename-safe timestamp used by snapshots and recordings."""
    return time.strftime("%Y%m%d_%H%M%S")
