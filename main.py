"""
main.py

Real-Time Object Detection System
----------------------------------
Captures live video from a webcam, runs Haar Cascade-based detection
on every frame, and overlays results (bounding boxes, labels, FPS,
per-class counts) in a live window.

Controls (while the window is focused):
    Q — quit
    S — save a snapshot of the current frame to output/
    P — pause / resume the video feed
    R — start / stop recording the annotated feed to output/

Usage:
    python main.py                  # default webcam (index 0)
    python main.py --camera 1       # use a different camera index
    python main.py --width 1280 --height 720
"""

import argparse
import os
import sys
import cv2

from detector import ObjectDetector, FPSCounter
from utils.overlay import draw_detections, draw_hud, draw_controls_hint, timestamp

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time object detection via webcam.")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument(
        "--min-neighbors", type=int, default=5,
        help="Higher = fewer false positives but may miss real objects (default: 5)"
    )
    return parser.parse_args()


def open_camera(index, width, height):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {index}. "
            "Check that it's connected and not in use by another app."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading cascade classifiers...")
    detector = ObjectDetector(min_neighbors=args.min_neighbors)

    print(f"Opening camera {args.camera}...")
    cap = open_camera(args.camera, args.width, args.height)

    fps_counter = FPSCounter()
    window_name = "Real-Time Object Detection"

    paused = False
    video_writer = None
    last_frame = None  # kept around so pausing/snapshotting works cleanly

    print("Running. Press Q in the video window to quit.")

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    print("[error] Failed to read frame from camera. Exiting.")
                    break
                last_frame = frame
            else:
                frame = last_frame.copy()

            detections = detector.detect(frame)
            draw_detections(frame, detections)

            fps = fps_counter.tick() if not paused else fps_counter.fps
            draw_hud(frame, fps, detections, paused=paused, recording=video_writer is not None)
            draw_controls_hint(frame)

            if video_writer is not None:
                video_writer.write(frame)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('s'):
                filename = os.path.join(OUTPUT_DIR, f"snapshot_{timestamp()}.png")
                cv2.imwrite(filename, frame)
                print(f"[saved] {filename}")

            elif key == ord('p'):
                paused = not paused
                print("[paused]" if paused else "[resumed]")

            elif key == ord('r'):
                if video_writer is None:
                    out_path = os.path.join(OUTPUT_DIR, f"recording_{timestamp()}.avi")
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    h, w = frame.shape[:2]
                    video_writer = cv2.VideoWriter(out_path, fourcc, 20.0, (w, h))
                    print(f"[recording started] {out_path}")
                else:
                    video_writer.release()
                    video_writer = None
                    print("[recording stopped]")

    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        print("Shut down cleanly.")


if __name__ == "__main__":
    main()
