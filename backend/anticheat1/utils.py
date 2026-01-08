# utils.py
import cv2
from pathlib import Path

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def draw_track_box(img, bbox_ltrb, track_id, alerts=None):
    """
    Draw bounding box and alerts for a track.
    bbox_ltrb: (left, top, right, bottom)
    alerts: A set of strings with active alerts for this track.
    """
    alerts = alerts or set()
    left, top, right, bottom = map(int, bbox_ltrb)
    
    is_alert = bool(alerts)
    color = (0, 0, 255) if is_alert else (0, 255, 0)
    thickness = 2 if is_alert else 1
    
    cv2.rectangle(img, (left, top), (right, bottom), color, thickness=thickness)
    
    label = f"ID {track_id}"
    if alerts:
        label += f" - {', '.join(sorted(list(alerts)))}"
    
    # Put label background
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (left, top - h - 6), (left + w + 6, top), color, -1)
    cv2.putText(img, label, (left + 3, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
