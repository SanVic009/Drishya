#!/usr/bin/env python3
"""
main_optimized.py
Optimized Exam Hall Anti-Cheat Detector using YOLOv8-Pose + DeepSORT
Detects:
  - Phone usage
  - Passing/interaction between candidates
  - Standing/walking (leaving seat)
"""

import torch
import argparse
import cv2
import csv
from pathlib import Path
from ultralytics import YOLO
from utils import draw_track_box
from zone_autodetector import create_zones_from_frame
from backend.pose_behavior_detector import YOLOPoseBehaviorDetector
from deep_sort_realtime.deepsort_tracker import DeepSort


# --------------------------------------------------------
#  CSV Logger
# --------------------------------------------------------
class AlertLogger:
    """Logs alerts to a CSV file."""
    def __init__(self, filepath="alerts.csv"):
        self.filepath = Path(filepath)
        self.file = self.filepath.open(mode='w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(["Timestamp_sec", "Track_ID", "Alert_Type", "Info"])

    def log(self, timestamp, track_id, alert_type, info=""):
        self.writer.writerow([f"{timestamp:.2f}", track_id, alert_type, info])
        self.file.flush()

    def close(self):
        if self.file:
            self.file.close()


# --------------------------------------------------------
#  CLI Arguments
# --------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Optimized Suspicious Behavior Detection using YOLOv8-Pose + DeepSORT")
    p.add_argument("--video", type=str, required=True, help="Path to input video")
    p.add_argument("--display", action="store_true", help="Show live detection window")
    p.add_argument("--auto-zone", action="store_true", help="Auto-detect bench zones on first frame")
    p.add_argument("--conf", type=float, default=0.5, help="YOLO detection confidence threshold")
    p.add_argument("--frame-skip", type=int, default=3, help="Process every Nth frame for speed (default=3)")
    p.add_argument("--fps", type=float, default=30.0, help="Fallback FPS if metadata missing")
    return p.parse_args()


# --------------------------------------------------------
#  Main
# --------------------------------------------------------
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Load YOLOv8 Pose Model
    yolo_net = YOLO("yolov8n-pose.pt").to(device)
    if device == "cuda":
        yolo_net.model.half()  # FP16 for GPU inference

    # Initialize DeepSORT tracker
    tracker = DeepSort(max_age=25, n_init=2, nms_max_overlap=1.0, max_cosine_distance=0.3, embedder="mobilenet")

    # Video setup
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    delay = int(1000 / fps) if args.display else 1
    frame_idx = 0

    # Zones (optional)
    zones = []
    if args.auto_zone:
        print("[INFO] Starting automatic zone detection...")
        ret, first_frame = cap.read()
        if not ret:
            raise IOError("Cannot read first frame for auto-zoning.")
        try:
            zones = create_zones_from_frame(first_frame, yolo_net)
            print(f"[INFO] Auto-zoning complete: {len(zones)} zones detected.")
        except Exception as e:
            print(f"[WARN] Automatic zone detection failed: {e}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Initialize detector + logger
    pose_behavior = YOLOPoseBehaviorDetector(fps=fps)
    alert_logger = AlertLogger()

    print("[INFO] Processing video... Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if frame_idx % args.frame_skip != 0:
                continue

            current_time = frame_idx / fps
            frame_small = cv2.resize(frame, (960, 540))

            with torch.inference_mode():
                results = yolo_net(frame_small, verbose=False, classes=[0, 67])[0]

            detections = []
            phone_boxes = []

            for box in results.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = results.names[cls]

                if conf < args.conf:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if label == "person":
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))
                elif label == "cell phone":
                    phone_boxes.append([x1, y1, x2, y2])

            tracks = tracker.update_tracks(detections, frame=frame_small)

            # Pose-based behavior detections
            pose_alerts = pose_behavior.update(frame_small, tracks, results)

            # Phone detection based on IoU overlap
            for track in tracks:
                if not track.is_confirmed():
                    continue
                tid = track.track_id
                tx1, ty1, tx2, ty2 = track.to_ltrb()
                for (px1, py1, px2, py2) in phone_boxes:
                    inter_x1 = max(tx1, px1)
                    inter_y1 = max(ty1, py1)
                    inter_x2 = min(tx2, px2)
                    inter_y2 = min(ty2, py2)
                    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                    boxA_area = (tx2 - tx1) * (ty2 - ty1)
                    boxB_area = (px2 - px1) * (py2 - py1)
                    iou = inter_area / float(boxA_area + boxB_area - inter_area + 1e-6)
                    if iou > 0.05:
                        pose_alerts.setdefault(tid, set()).add("Phone Detected")

            # Logging + Visualization
            for track in tracks:
                if not track.is_confirmed():
                    continue
                tid = track.track_id
                current_alerts = pose_alerts.get(tid, set())
                for alert in current_alerts:
                    alert_logger.log(current_time, tid, alert)
                if args.display:
                    draw_track_box(frame_small, track.to_ltrb(), tid, alerts=current_alerts)

            if args.display:
                cv2.imshow("Exam Anti-Cheat (DeepSORT Optimized)", frame_small)
                if cv2.waitKey(delay) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        alert_logger.close()
        if args.display:
            cv2.destroyAllWindows()
        print("[INFO] Finished. Alerts saved to alerts.csv")


if __name__ == "__main__":
    main()
