
import cv2
import redis
import numpy as np
import time
import logging
import sys
import os
import pandas as pd
from threading import Thread, Lock
from flask import Flask, Response
from flask_cors import CORS
from pyzbar import pyzbar
from datetime import datetime, date
from ultralytics import YOLO
import qrcode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ---------------- CONFIG ----------------
OUTPUT_DIR = "qr_snapshots"
QR_DIR = "qr_codes_generated"
LOG_CSV = "qr_attendance.csv"
ALERT_CSV = "qr_alerts.csv"
YOLO_MODEL = "yolov8n.pt"
CONF_THRESH = 0.4
# ----------------------------------------

# Global state for daily QR codes
last_date = None
qr_attendance = None
qr_emergency = None


# ---------- QR GENERATION ----------
def generate_qr(qr_text, filename):
    """Generate QR code image (only if it doesn't exist)."""
    os.makedirs(QR_DIR, exist_ok=True)
    path = os.path.join(QR_DIR, filename)
    if not os.path.exists(path):
        img = qrcode.make(qr_text)
        img.save(path)
    return path


def generate_daily_qrs():
    """Generate daily attendance and emergency QR codes."""
    global last_date, qr_attendance, qr_emergency
    today = date.today().isoformat()

    att_text = f"ATTENDANCE-{today}"
    emer_text = f"EMERGENCY-{today}"

    att_path = generate_qr(att_text, f"QR_ATTENDANCE_{today}.png")
    emer_path = generate_qr(emer_text, f"QR_EMERGENCY_{today}.png")

    logger.info("[INFO] Today's QRs:")
    logger.info(f"  ✅ Attendance: {att_path}")
    logger.info(f"  🚨 Emergency:  {emer_path}")

    last_date = today
    qr_attendance = att_text
    qr_emergency = emer_text
    
    return today, att_text, emer_text


def midnight_check():
    """Check if date changed and regenerate QR codes if needed."""
    global last_date, qr_attendance, qr_emergency
    current = date.today().isoformat()
    if current != last_date:
        logger.info("\n[INFO] New day detected! Regenerating QRs...")
        return generate_daily_qrs()
    return last_date, qr_attendance, qr_emergency


# ---------- FILE HELPERS ----------
def ensure_csvs():
    """Ensure CSV files exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(LOG_CSV):
        pd.DataFrame(columns=["timestamp", "qr_data", "snapshot"]).to_csv(LOG_CSV, index=False)
    if not os.path.exists(ALERT_CSV):
        pd.DataFrame(columns=["timestamp", "qr_data", "snapshot"]).to_csv(ALERT_CSV, index=False)


def log_attendance(qr_data, snapshot):
    """Log attendance to CSV."""
    df = pd.read_csv(LOG_CSV)
    entry = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             "qr_data": qr_data, "snapshot": snapshot}
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(LOG_CSV, index=False)
    logger.info(f"[LOGGED] Attendance | {qr_data}")


def log_alert(qr_data, snapshot):
    """Log emergency alert to CSV."""
    df = pd.read_csv(ALERT_CSV)
    entry = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             "qr_data": qr_data, "snapshot": snapshot}
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(ALERT_CSV, index=False)
    logger.info(f"[ALERT] 🚨 Emergency QR detected | Logged alert for {qr_data}")


class RedisFrameSubscriber:
    """
    Subscribes to Redis stream and provides frames to consumers.
    """
    
    def __init__(self, redis_host='localhost', redis_port=6379, stream_name='camera:stream'):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=False
        )
        self.stream_name = stream_name
        self.last_id = '0-0'
        
        try:
            self.redis_client.ping()
            logger.info(f"✅ Subscriber connected to Redis stream: {stream_name}")
        except redis.ConnectionError as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
    
    def decode_frame(self, frame_data):
        """Decode JPEG frame from Redis stream."""
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    
    def read_frame(self, block_ms=1000):
        """
        Read next frame from Redis stream.
        Returns: (frame, metadata) or (None, None) if no frame available
        """
        try:
            messages = self.redis_client.xread(
                {self.stream_name: self.last_id},
                count=1,
                block=block_ms
            )
            
            if not messages:
                return None, None
            
            stream_name, stream_messages = messages[0]
            message_id, message_data = stream_messages[0]
            
            self.last_id = message_id
            
            frame_bytes = message_data[b'frame']
            width = int(message_data[b'width'])
            height = int(message_data[b'height'])
            timestamp = float(message_data[b'timestamp'])
            
            frame = self.decode_frame(frame_bytes)
            
            metadata = {
                'width': width,
                'height': height,
                'timestamp': timestamp,
                'stream_id': message_id.decode()
            }
            
            return frame, metadata
            
        except Exception as e:
            logger.error(f"Error reading frame from Redis: {e}")
            return None, None


class QRCodeDetector:
    """
    QR Code detection system using YOLOv8 + pyzbar.
    Detects persons first, then scans QR codes within person bounding boxes.
    Subscribes to Redis stream for frames.
    """
    
    def __init__(self, redis_host='localhost', redis_port=6379, stream_name='camera:stream'):
        self.subscriber = RedisFrameSubscriber(redis_host, redis_port, stream_name)
        
        # Load YOLO model for person detection
        logger.info("🔄 Loading YOLOv8 model for person detection...")
        self.model = YOLO(YOLO_MODEL)
        self.names = self.model.names
        self.person_cls = [k for k, v in self.names.items() if v.lower() == "person"][0]
        
        # Use GPU if available
        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            self.model.to('cuda')
            # Enable GPU optimizations
            torch.backends.cudnn.benchmark = True
            logger.info("✅ QR Detector using GPU with optimizations")
        else:
            logger.info("⚠️ QR Detector using CPU")
        
        # Current processed frame (for streaming)
        self.current_frame = None
        self.frame_lock = Lock()
        
        # Running state
        self.running = False
        self.thread = None
        
        # Stats
        self.frames_processed = 0
        self.qr_codes_detected = 0
        
        # Track seen QR codes to prevent duplicates
        self.seen_attendance = set()
        self.seen_emergency = set()
        
        logger.info("✅ QR code detector initialized with YOLOv8 person detection")
    
    def detect_qr_from_crop(self, crop):
        """Detect QR codes in cropped image (person bbox)."""
        decoded = pyzbar.decode(crop)
        texts = []
        for obj in decoded:
            data = obj.data.decode("utf-8")
            x, y, w, h = obj.rect
            cv2.rectangle(crop, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(crop, data, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            texts.append(data)
        return texts
    
    def process_frame(self, frame):
        """
        Process frame: detect persons with YOLO, scan QR codes in person bboxes.
        Handles attendance and emergency QR codes.
        """
        global qr_attendance, qr_emergency, last_date
        
        # Check for midnight refresh
        midnight_check()
        
        # Run YOLO person detection with FP16 for faster inference
        results = self.model(
            frame, 
            imgsz=640, 
            conf=CONF_THRESH, 
            verbose=False, 
            device=self.device,
            half=True if self.device == 'cuda' else False
        )[0]
        
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls == self.person_cls:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw person bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, "Person", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Crop to person bbox and detect QR codes
                crop = frame[y1:y2, x1:x2]
                qrs = self.detect_qr_from_crop(crop)
                
                for qr_data in qrs:
                    ts = datetime.now().strftime("%H%M%S")
                    snapshot_name = f"{ts}_{qr_data[:20]}.jpg"
                    snapshot_path = os.path.join(OUTPUT_DIR, snapshot_name)
                    
                    # Emergency QR handling
                    if "EMERGENCY" in qr_data.upper() and qr_data not in self.seen_emergency:
                        self.seen_emergency.add(qr_data)
                        cv2.imwrite(snapshot_path, frame)
                        log_alert(qr_data, snapshot_path)
                        self.qr_codes_detected += 1
                        
                        # Visual alert on frame
                        cv2.putText(frame, "EMERGENCY ALERT", (60, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        logger.warning(f"🚨 EMERGENCY QR DETECTED: {qr_data}")
                    
                    # Attendance QR
                    elif "ATTENDANCE" in qr_data.upper() and qr_data not in self.seen_attendance:
                        self.seen_attendance.add(qr_data)
                        cv2.imwrite(snapshot_path, frame)
                        log_attendance(qr_data, snapshot_path)
                        self.qr_codes_detected += 1
                        logger.info(f"✅ ATTENDANCE LOGGED: {qr_data}")
        
        # Add stats overlay
        self.add_stats_overlay(frame)
        
        return frame
    
    def add_stats_overlay(self, frame):
        """Add statistics overlay to frame."""
        height, width = frame.shape[:2]
        
        # Background for text
        cv2.rectangle(frame, (10, 10), (350, 130), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, 130), (255, 165, 0), 2)
        
        # Text
        cv2.putText(frame, "QR ATTENDANCE & ALERT SYSTEM", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
        cv2.putText(frame, f"Attendance Logged: {len(self.seen_attendance)}", (20, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Emergency Alerts: {len(self.seen_emergency)}", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, f"Total Detected: {self.qr_codes_detected}", (20, 115), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    
    def detection_loop(self):
        """
        Main detection loop - reads frames from Redis and detects QR codes.
        """
        logger.info("🚀 Starting QR code detection loop...")
        
        frame_count = 0
        start_time = time.time()
        last_log_time = start_time
        
        try:
            while self.running:
                # Read frame from Redis stream
                frame, metadata = self.subscriber.read_frame(block_ms=1000)
                
                if frame is None:
                    continue
                
                # Process frame with YOLO + QR detection
                processed_frame = self.process_frame(frame)
                
                # Update shared frame for streaming
                with self.frame_lock:
                    self.current_frame = processed_frame.copy()
                
                frame_count += 1
                self.frames_processed += 1
                
                # Log stats every 5 seconds
                current_time = time.time()
                if current_time - last_log_time >= 5.0:
                    elapsed = current_time - start_time
                    avg_fps = frame_count / elapsed
                    logger.info(
                        f"📊 QR Detector | Processed: {self.frames_processed} | "
                        f"Attendance: {len(self.seen_attendance)} | "
                        f"Alerts: {len(self.seen_emergency)} | FPS: {avg_fps:.2f}"
                    )
                    last_log_time = current_time
                    
        except KeyboardInterrupt:
            logger.info("⏹️  Stopping QR code detector...")
        except Exception as e:
            logger.error(f"❌ Error in detection loop: {e}", exc_info=True)
        finally:
            logger.info(f"✅ QR code detector stopped. Processed {frame_count} frames")

    
    def start(self):
        """Start detection in background thread."""
        if self.running:
            logger.warning("Detector already running")
            return
        
        self.running = True
        self.thread = Thread(target=self.detection_loop, daemon=True)
        self.thread.start()
        logger.info("🚀 QR code detector started")
    
    def stop(self):
        """Stop detection."""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        logger.info("⏹️  QR code detector stopped")
    
    def get_current_frame(self):
        """Get the current processed frame (thread-safe)."""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None
    
    def get_stats(self):
        """Get detection statistics."""
        return {
            'frames_processed': self.frames_processed,
            'qr_codes_detected': self.qr_codes_detected,
            'attendance_logged': len(self.seen_attendance),
            'emergency_alerts': len(self.seen_emergency),
            'running': self.running
        }



# Global detector instance
detector = None


def generate_frames():
    """Generate MJPEG stream for Flask endpoint."""
    global detector
    
    if detector is None or not detector.running:
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Detector not running", (50, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', error_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return
    
    while True:
        frame = detector.get_current_frame()
        
        if frame is None:
            time.sleep(0.03)
            continue
        
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.03)


@app.route('/api/qr_feed')
def video_feed():
    """Video streaming route for QR code detection."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/qr_stats')
def get_stats():
    """Get detection statistics."""
    global detector
    
    if detector is None:
        return {'error': 'Detector not initialized'}, 500
    
    return detector.get_stats()


@app.route('/api/qr_recent')
def get_recent():
    """Get recent attendance and alert logs."""
    global detector
    
    if detector is None:
        return {'error': 'Detector not initialized'}, 500
    
    # Read recent logs from CSV
    attendance_data = []
    alert_data = []
    
    try:
        if os.path.exists(LOG_CSV):
            df = pd.read_csv(LOG_CSV)
            attendance_data = df.tail(10).to_dict('records')
        
        if os.path.exists(ALERT_CSV):
            df = pd.read_csv(ALERT_CSV)
            alert_data = df.tail(10).to_dict('records')
    except Exception as e:
        logger.error(f"Error reading logs: {e}")
    
    return {
        'attendance': attendance_data,
        'alerts': alert_data
    }



@app.route('/health')
def health():
    """Health check endpoint."""
    return {'status': 'ok', 'service': 'qr_detector'}


def main():
    """Main entry point."""
    import signal
    
    global detector
    
    # Ensure directories and CSV files exist
    ensure_csvs()
    
    # Generate initial daily QR codes (attendance + emergency)
    generate_daily_qrs()
    
    # Initialize detector
    detector = QRCodeDetector(
        redis_host='localhost',
        redis_port=6379,
        stream_name='camera:stream'
    )
    
    # Start detection
    detector.start()
    
    # Graceful shutdown
    def signal_handler(sig, frame):
        logger.info("\n🛑 Shutting down QR code detector...")
        detector.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run Flask app
    logger.info("🌐 Starting Flask server on port 5003...")
    app.run(host='0.0.0.0', port=5003, debug=False, threaded=True)


if __name__ == '__main__':
    main()
