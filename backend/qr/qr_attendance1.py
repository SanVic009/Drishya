"""
qr_attendance.py
--------------------------------------------------
Daily Attendance + Emergency QR System with Popup Alert

Features:
✅ Generates daily Attendance & Emergency QRs
✅ Auto-regenerates both at midnight
✅ YOLOv8 person detection + QR decoding
✅ Attendance + emergency logging with snapshots
✅ Popup + beep alert on emergency detection
✅ One snapshot per QR scan event
"""

import os
import cv2
import time
import pandas as pd
from datetime import datetime, date
from ultralytics import YOLO
from pyzbar import pyzbar
import qrcode
from PIL import Image
import winsound
import threading
import tkinter as tk
from tkinter import messagebox

# ---------------- CONFIG ----------------
OUTPUT_DIR = "snapshots"
QR_DIR = "daily_qr"
LOG_CSV = "attendance.csv"
ALERT_CSV = "alerts.csv"
YOLO_MODEL = "yolov8n.pt"
CONF_THRESH = 0.4
# ----------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(QR_DIR, exist_ok=True)


# ---------- QR GENERATION ----------
def generate_qr(qr_text, filename):
    path = os.path.join(QR_DIR, filename)
    if not os.path.exists(path):
        img = qrcode.make(qr_text)
        img.save(path)
    return path


def generate_daily_qrs(show=True):
    today = date.today().isoformat()

    att_text = f"ATTENDANCE-{today}"
    emer_text = f"EMERGENCY-{today}"

    att_path = generate_qr(att_text, f"QR_ATTENDANCE_{today}.png")
    emer_path = generate_qr(emer_text, f"QR_EMERGENCY_{today}.png")

    print(f"[INFO] Today's QRs:")
    print(f"  ✅ Attendance: {att_path}")
    print(f"  🚨 Emergency:  {emer_path}")

    if show:
        Image.open(att_path).show()
        Image.open(emer_path).show()

    return today, att_text, emer_text


# ---------- FILE HELPERS ----------
def ensure_csvs():
    if not os.path.exists(LOG_CSV):
        pd.DataFrame(columns=["timestamp", "qr_data", "snapshot"]).to_csv(LOG_CSV, index=False)
    if not os.path.exists(ALERT_CSV):
        pd.DataFrame(columns=["timestamp", "qr_data", "snapshot"]).to_csv(ALERT_CSV, index=False)


def log_attendance(qr_data, snapshot):
    df = pd.read_csv(LOG_CSV)
    entry = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             "qr_data": qr_data, "snapshot": snapshot}
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(LOG_CSV, index=False)
    print(f"[LOGGED] Attendance | {qr_data}")


def log_alert(qr_data, snapshot):
    df = pd.read_csv(ALERT_CSV)
    entry = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             "qr_data": qr_data, "snapshot": snapshot}
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(ALERT_CSV, index=False)
    print(f"[ALERT] 🚨 Emergency QR detected | Logged alert for {qr_data}")


# ---------- POPUP ALERT ----------
def show_popup_alert(qr_data):
    """Displays a popup alert window on emergency detection"""
    def popup():
        root = tk.Tk()
        root.withdraw()
        messagebox.showwarning("🚨 Emergency Alert 🚨",
                               f"Emergency QR detected!\n\nQR Data: {qr_data}")
        root.destroy()
    threading.Thread(target=popup).start()


# ---------- QR DETECTION ----------
def detect_qr_from_crop(crop):
    decoded = pyzbar.decode(crop)
    texts = []
    for obj in decoded:
        data = obj.data.decode("utf-8")
        x, y, w, h = obj.rect
        cv2.rectangle(crop, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(crop, data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        texts.append(data)
    return texts


# ---------- MIDNIGHT REFRESH ----------
def midnight_check(last_date):
    current = date.today().isoformat()
    if current != last_date:
        print("\n[INFO] New day detected! Regenerating QRs...")
        return generate_daily_qrs(show=True)
    return last_date, f"ATTENDANCE-{last_date}", f"EMERGENCY-{last_date}"


# ---------- MAIN ----------
def main():
    ensure_csvs()
    last_date, qr_attendance, qr_emergency = generate_daily_qrs(show=True)

    model = YOLO(YOLO_MODEL)
    names = model.names
    person_cls = [k for k, v in names.items() if v.lower() == "person"][0]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not access camera.")
        return

    seen_attendance = set()
    seen_emergency = set()

    print("[INFO] System running... Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        last_date, qr_attendance, qr_emergency = midnight_check(last_date)

        results = model(frame, imgsz=640, conf=CONF_THRESH, verbose=False)[0]

        for box in results.boxes:
            cls = int(box.cls[0])
            if cls == person_cls:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                qrs = detect_qr_from_crop(crop)

                for qr_data in qrs:
                    ts = datetime.now().strftime("%H%M%S")
                    snapshot_name = f"{ts}_{qr_data}.jpg"
                    snapshot_path = os.path.join(OUTPUT_DIR, snapshot_name)

                    # Emergency QR
                    if "EMERGENCY" in qr_data.upper() and qr_data not in seen_emergency:
                        seen_emergency.add(qr_data)
                        winsound.Beep(1000, 800)
                        cv2.imwrite(snapshot_path, frame)
                        log_alert(qr_data, snapshot_path)
                        show_popup_alert(qr_data)
                        cv2.putText(frame, "🚨 EMERGENCY ALERT 🚨", (60, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    # Attendance QR
                    elif "ATTENDANCE" in qr_data.upper() and qr_data not in seen_attendance:
                        seen_attendance.add(qr_data)
                        cv2.imwrite(snapshot_path, frame)
                        log_attendance(qr_data, snapshot_path)

        cv2.imshow("QR Attendance & Alert System", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
