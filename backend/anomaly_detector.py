"""
Anomaly Detector Service
Subscribes to Redis camera stream and performs:
1. Suspicious object detection (phone, knife, scissors, bottle)
2. Uniform compliance detection (blue uniform check)
3. ID card detection (lanyard ID badge check)
Uses YOLOv8 for person/object detection + CLIP for uniform/ID classification.
"""

import cv2
import redis
import numpy as np
import time
import logging
import sys
import torch
from threading import Thread, Lock
from flask import Flask, Response, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image

try:
    import clip
except ImportError:
    logger = logging.getLogger(__name__)
    logger.error("CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


# ---------- CLIP Helper Functions ----------
def _avg_normalized(feats: torch.Tensor) -> torch.Tensor:
    """Average and normalize feature tensors."""
    feats = feats / feats.norm(dim=-1, keepdim=True)
    avg = feats.mean(dim=0, keepdim=True)
    return avg / avg.norm(dim=-1, keepdim=True)


def build_class_embeddings(model, device, prompts: dict):
    """Build CLIP text embeddings for classification."""
    with torch.no_grad():
        out = {}
        for cls, texts in prompts.items():
            tokens = clip.tokenize(texts).to(device)
            txt = model.encode_text(tokens)
            out[cls] = _avg_normalized(txt)
        return out


def classify_with_clip(model, preprocess, device, image_bgr, class_embs):
    """Classify image crop using CLIP."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    img_in = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        img_feat = model.encode_image(img_in)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        labels = list(class_embs.keys())
        txt_mat = torch.cat([class_embs[lbl] for lbl in labels], dim=0)
        logits = 100.0 * (img_feat @ txt_mat.t())
        probs = logits.softmax(dim=-1).squeeze(0)
    probs_np = probs.detach().float().cpu().numpy()
    idx = int(probs_np.argmax())
    return labels[idx], float(probs_np[idx])


def torso_crop(box, frame_h: int, frame_w: int):
    """Extract torso region from person bounding box."""
    x1, y1, x2, y2 = map(int, box)
    bw, bh = (x2 - x1), (y2 - y1)
    ty1 = int(y1 + 0.2 * bh)
    ty2 = int(y1 + 0.7 * bh)
    tx1 = x1
    tx2 = x2
    # Clamp to frame bounds
    tx1 = max(0, min(frame_w - 1, tx1))
    tx2 = max(0, min(frame_w - 1, tx2))
    ty1 = max(0, min(frame_h - 1, ty1))
    ty2 = max(0, min(frame_h - 1, ty2))
    if tx2 <= tx1 or ty2 <= ty1:
        return x1, y1, x2, y2
    return tx1, ty1, tx2, ty2


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


class AnomalyDetector:
    """
    Anomaly detection system using YOLOv8 + CLIP.
    Detects:
    1. Suspicious objects (phone, knife, scissors, bottle)
    2. Uniform compliance (blue uniform check)
    3. ID card presence (lanyard ID badge)
    Subscribes to Redis stream for frames.
    """
    
    def __init__(self, redis_host='localhost', redis_port=6379, stream_name='camera:stream'):
        self.subscriber = RedisFrameSubscriber(redis_host, redis_port, stream_name)
        
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️  Using device: {self.device}")
        
        # Load YOLO model
        logger.info("🔄 Loading YOLOv8 model...")
        self.yolo_model = YOLO('yolov8n.pt')
        
        if self.device.type == 'cuda':
            self.yolo_model.to('cuda')
            torch.backends.cudnn.benchmark = True
            logger.info("✅ YOLO model using GPU with optimizations")
        else:
            logger.info("⚠️ YOLO model using CPU")
        
        # Load CLIP model
        logger.info("🔄 Loading CLIP model for uniform/ID detection...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=str(self.device))
        self.clip_model.eval()
        logger.info("✅ CLIP model loaded")
        
        # Define CLIP prompts
        uniform_prompts = {
            "Uniform": [
                "a person wearing a blue uniform shirt",
                "a person in a blue collared uniform",
                "a student in a blue uniform shirt",
                "a worker wearing a blue uniform",
            ],
            "No Uniform": [
                "a person not wearing a blue uniform shirt",
                "a person in casual clothes without a blue uniform",
            ]
        }
        
        id_prompts = {
            "ID Card": [
                "a person wearing an ID card on a lanyard",
                "a person with an identity badge around the neck",
                "an employee with an ID badge visible",
            ],
            "No ID Card": [
                "a person without any ID card visible",
                "a person not wearing an ID badge",
            ]
        }
        
        # Build CLIP embeddings
        logger.info("🔄 Building CLIP embeddings...")
        self.uniform_embs = build_class_embeddings(self.clip_model, self.device, uniform_prompts)
        self.id_embs = build_class_embeddings(self.clip_model, self.device, id_prompts)
        logger.info("✅ CLIP embeddings ready")
        
        # Thresholds
        self.uniform_threshold = 0.5
        self.id_threshold = 0.5
        
        # Current processed frame (for streaming)
        self.current_frame = None
        self.frame_lock = Lock()
        
        # Running state
        self.running = False
        self.thread = None
        
        # Stats
        self.frames_processed = 0
        self.anomalies_detected = 0
        
        # Suspicious object classes
        self.suspicious_classes = {
            'cell phone': 'Phone usage detected',
            'knife': 'Dangerous object detected',
            'scissors': 'Sharp object detected',
            'bottle': 'Suspicious item'
        }
        
        logger.info("✅ Anomaly detector initialized")
    
    def detect_anomalies(self, frame):
        """
        Run YOLO + CLIP detection for:
        1. Suspicious objects
        2. Uniform compliance
        3. ID card presence
        Returns frame with overlays and anomaly count.
        """
        h, w = frame.shape[:2]
        
        # Run YOLO detection (persons + objects)
        results = self.yolo_model(frame, verbose=False, device=str(self.device), half=True, conf=0.35)
        
        result = results[0]
        anomaly_count = 0
        anomaly_messages = []
        
        # Create a copy for drawing
        annotated_frame = frame.copy()
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for box, conf, cls in zip(boxes, confs, classes):
                class_id = int(cls)
                class_name = self.yolo_model.names[class_id]
                confidence = float(conf)
                
                x1, y1, x2, y2 = map(int, box)
                
                # Check for person - run uniform and ID check
                if class_name == "person" and confidence > 0.4:
                    # Extract torso crop for CLIP
                    tx1, ty1, tx2, ty2 = torso_crop((x1, y1, x2, y2), h, w)
                    crop = frame[ty1:ty2, tx1:tx2]
                    
                    if crop.size > 0:
                        # Classify uniform
                        u_lbl, u_prob = classify_with_clip(
                            self.clip_model, self.clip_preprocess, self.device, crop, self.uniform_embs
                        )
                        
                        # Classify ID card
                        i_lbl, i_prob = classify_with_clip(
                            self.clip_model, self.clip_preprocess, self.device, crop, self.id_embs
                        )
                        
                        has_uniform = (u_lbl == "Uniform") and (u_prob >= self.uniform_threshold)
                        has_id = (i_lbl == "ID Card") and (i_prob >= self.id_threshold)
                        
                        # Determine color and alerts
                        if has_uniform and has_id:
                            color = (0, 200, 0)  # Green - compliant
                        elif has_uniform or has_id:
                            color = (0, 165, 255)  # Orange - partial compliance
                            anomaly_count += 1
                            if not has_uniform:
                                msg = "No Uniform"
                                if msg not in anomaly_messages:
                                    anomaly_messages.append(msg)
                            if not has_id:
                                msg = "No ID Card"
                                if msg not in anomaly_messages:
                                    anomaly_messages.append(msg)
                        else:
                            color = (0, 0, 220)  # Red - non-compliant
                            anomaly_count += 1
                            if "No Uniform & No ID" not in anomaly_messages:
                                anomaly_messages.append("No Uniform & No ID")
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label with background
                        label_lines = [
                            f"Uniform: {'Yes' if has_uniform else 'No'} ({u_prob:.2f})",
                            f"ID Card: {'Yes' if has_id else 'No'} ({i_prob:.2f})",
                        ]
                        
                        y_offset = y1 - 10
                        for line in reversed(label_lines):
                            (w_text, h_text), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.rectangle(annotated_frame, (x1, y_offset - h_text - 4), 
                                        (x1 + w_text + 4, y_offset), color, -1)
                            cv2.putText(annotated_frame, line, (x1 + 2, y_offset - 2),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            y_offset -= (h_text + 6)
                
                # Check for suspicious objects
                elif class_name in self.suspicious_classes and confidence > 0.5:
                    anomaly_count += 1
                    message = self.suspicious_classes[class_name]
                    if message not in anomaly_messages:
                        anomaly_messages.append(message)
                    
                    # Draw red box for suspicious objects
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(annotated_frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Add status overlay
        self.add_stats_overlay(annotated_frame, anomaly_count, anomaly_messages)
        
        return annotated_frame, anomaly_count
    
    def add_stats_overlay(self, frame, anomaly_count, messages):
        """Add statistics and alerts overlay to frame."""
        # Background for header
        cv2.rectangle(frame, (10, 10), (400, 80), (0, 0, 0), -1)
        
        if anomaly_count > 0:
            cv2.rectangle(frame, (10, 10), (400, 80), (0, 0, 255), 3)
        else:
            cv2.rectangle(frame, (10, 10), (400, 80), (0, 255, 0), 2)
        
        # Header text
        cv2.putText(frame, "ANOMALY DETECTION", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Anomalies: {anomaly_count}", (20, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Alert messages
        if messages:
            y_offset = 100
            for msg in messages[:3]:  # Show max 3 messages
                cv2.rectangle(frame, (10, y_offset - 25), (400, y_offset + 5), (0, 0, 255), -1)
                cv2.putText(frame, f"⚠️ {msg}", (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 35
    
    def detection_loop(self):
        """
        Main detection loop - reads frames from Redis and detects anomalies.
        """
        logger.info("🚀 Starting anomaly detection loop...")
        
        frame_count = 0
        start_time = time.time()
        last_log_time = start_time
        
        try:
            while self.running:
                # Read frame from Redis stream
                frame, metadata = self.subscriber.read_frame(block_ms=1000)
                
                if frame is None:
                    continue
                
                # Process frame
                processed_frame, anomaly_count = self.detect_anomalies(frame)
                
                # Update shared frame for streaming
                with self.frame_lock:
                    self.current_frame = processed_frame.copy()
                
                frame_count += 1
                self.frames_processed += 1
                if anomaly_count > 0:
                    self.anomalies_detected += 1
                
                # Log stats every 5 seconds
                current_time = time.time()
                if current_time - last_log_time >= 5.0:
                    elapsed = current_time - start_time
                    avg_fps = frame_count / elapsed
                    logger.info(
                        f"📊 Anomaly Detector | Processed: {self.frames_processed} | "
                        f"Anomalies: {self.anomalies_detected} | FPS: {avg_fps:.2f}"
                    )
                    last_log_time = current_time
                    
        except KeyboardInterrupt:
            logger.info("⏹️  Stopping anomaly detector...")
        except Exception as e:
            logger.error(f"❌ Error in detection loop: {e}", exc_info=True)
        finally:
            logger.info(f"✅ Anomaly detector stopped. Processed {frame_count} frames")
    
    def start(self):
        """Start detection in background thread."""
        if self.running:
            logger.warning("Detector already running")
            return
        
        self.running = True
        self.thread = Thread(target=self.detection_loop, daemon=True)
        self.thread.start()
        logger.info("🚀 Anomaly detector started")
    
    def stop(self):
        """Stop detection."""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        logger.info("⏹️  Anomaly detector stopped")
    
    def get_current_frame(self):
        """Get the current processed frame (thread-safe)."""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None


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


@app.route('/')
def index():
    return jsonify({
        'service': 'Drishya Anomaly Detection Server',
        'status': 'running',
        'version': '2.0',
        'endpoints': {
            'video_feed': '/api/video_feed',
            'stats': '/api/anomaly_stats',
            'health': '/health'
        }
    })


@app.route('/api/video_feed')
def video_feed():
    """Video streaming route for anomaly detection."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/anomaly_stats')
def get_stats():
    """Get detection statistics."""
    global detector
    
    if detector is None:
        return {'error': 'Detector not initialized'}, 500
    
    return jsonify({
        'frames_processed': detector.frames_processed,
        'anomalies_detected': detector.anomalies_detected,
        'running': detector.running
    })


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'service': 'anomaly_detector'})


def main():
    """Main entry point."""
    import signal
    
    global detector
    
    # Initialize detector
    detector = AnomalyDetector(
        redis_host='localhost',
        redis_port=6379,
        stream_name='camera:stream'
    )
    
    # Start detection
    detector.start()
    
    # Graceful shutdown
    def signal_handler(sig, frame):
        logger.info("\n🛑 Shutting down anomaly detector...")
        detector.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run Flask app
    logger.info("🌐 Starting Flask server on port 5001...")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)


if __name__ == '__main__':
    main()
