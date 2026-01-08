import cv2
import redis
import numpy as np
import torch
import time
import logging
from threading import Thread, Lock
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import sys
import os
import glob

# Add anticheat1 to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'anticheat1'))

from pose_behavior_detector import YOLOPoseBehaviorDetector
from utils import draw_track_box

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Video directory
TEST_VIDEOS_DIR = os.path.join(os.path.dirname(__file__), 'anticheat1', 'test_vids')


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
        self.last_id = '0-0'  # Start from beginning
        
        # Test connection
        try:
            self.redis_client.ping()
            logger.info(f"✅ Subscriber connected to Redis stream: {stream_name}")
        except redis.ConnectionError as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
    
    def decode_frame(self, frame_data):
        """
        Decode JPEG frame from Redis stream.
        """
        # Decode JPEG bytes to numpy array
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    
    def read_frame(self, block_ms=1000):
        """
        Read next frame from Redis stream.
        Returns: (frame, metadata) or (None, None) if no frame available
        """
        try:
            # Read from stream (blocking with timeout)
            messages = self.redis_client.xread(
                {self.stream_name: self.last_id},
                count=1,
                block=block_ms
            )
            
            if not messages:
                return None, None
            
            # Parse message
            stream_name, stream_messages = messages[0]
            message_id, message_data = stream_messages[0]
            
            # Update last_id for next read
            self.last_id = message_id
            
            # Extract frame and metadata
            frame_bytes = message_data[b'frame']
            width = int(message_data[b'width'])
            height = int(message_data[b'height'])
            timestamp = float(message_data[b'timestamp'])
            
            # Decode frame
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


class AntiCheatDetector:
    """
    Anti-cheat detection system using YOLOv8-Pose + DeepSORT.
    Subscribes to Redis stream for frames.
    """
    
    def __init__(self, redis_host='localhost', redis_port=6379, stream_name='camera:stream'):
        self.subscriber = RedisFrameSubscriber(redis_host, redis_port, stream_name)
        
        # Use GPU if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"🖥️  Using device: {self.device}")
        
        # Detection models
        logger.info("🔄 Loading YOLOv8-Pose model...")
        self.yolo_model = YOLO('anticheat1/yolov8n-pose.pt')
        
        # Move model to GPU with FP16 if available
        if self.device == 'cuda':
            self.yolo_model.to('cuda')
            # Enable FP16 inference for faster processing
            torch.backends.cudnn.benchmark = True
            logger.info("✅ YOLO model moved to GPU with optimizations")
        
        # DeepSORT tracker with GPU optimization
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=None,
            embedder="mobilenet",
            half=True if self.device == 'cuda' else False,
            embedder_gpu=True if self.device == 'cuda' else False
        )
        
        # Behavior detector
        self.pose_behavior = YOLOPoseBehaviorDetector(
            fps=30.0,
            walk_move_thresh=20.0,
            pass_dist_thresh=60.0,
            same_row_thresh=80.0,
            pass_min_frames=3
        )
        
        # Current processed frame (for streaming)
        self.current_frame = None
        self.frame_lock = Lock()
        
        # Running state
        self.running = False
        self.thread = None
        
        # Stats
        self.frames_processed = 0
        self.detections_count = 0
        
        logger.info("✅ Anti-cheat detector initialized")
    
    def detect_and_track(self, frame):
        """
        Run YOLO detection, DeepSORT tracking, and behavior analysis.
        Returns frame with overlays and alert information.
        """
        frame_small = cv2.resize(frame, (960, 540))
        
        # YOLO detection for person and cell phone with GPU optimization
        with torch.inference_mode():
            results = self.yolo_model(
                frame_small, 
                verbose=False, 
                classes=[0, 67], 
                device=self.device,
                half=True if self.device == 'cuda' else False,  # FP16 for faster inference
                imgsz=640
            )[0]
        
        detections = []
        phone_boxes = []
        
        # Parse YOLO results
        for box in results.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = results.names[cls]
            
            if conf < 0.4:  # Confidence threshold
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if label == "person":
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))
            elif label == "cell phone":
                phone_boxes.append([x1, y1, x2, y2])
        
        # Update DeepSORT tracker
        tracks = self.tracker.update_tracks(detections, frame=frame_small)
        
        # Pose-based behavior detection
        pose_alerts = self.pose_behavior.update(frame_small, tracks, results)
        
        # Phone detection based on IoU overlap with tracked persons
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            tid = track.track_id
            tx1, ty1, tx2, ty2 = track.to_ltrb()
            
            for (px1, py1, px2, py2) in phone_boxes:
                # Calculate intersection over union
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
        
        # Draw tracking boxes ONLY for people with alerts
        alert_count = 0
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            tid = track.track_id
            current_alerts = pose_alerts.get(tid, set())
            
            if current_alerts:
                alert_count += len(current_alerts)
                # Only draw box if there are alerts
                draw_track_box(frame_small, track.to_ltrb(), tid, alerts=current_alerts)
        
        # Add stats overlay
        self.add_stats_overlay(frame_small, len(tracks), alert_count)
        
        return frame_small, alert_count
    
    def add_stats_overlay(self, frame, track_count, alert_count):
        """Add statistics overlay to frame."""
        # Background for text
        cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 100), (0, 255, 0), 2)
        
        # Text
        cv2.putText(frame, "ANTI-CHEAT MONITOR", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Tracked: {track_count}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Alert status
        alert_color = (0, 0, 255) if alert_count > 0 else (0, 255, 0)
        cv2.putText(frame, f"Alerts: {alert_count}", (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, alert_color, 2)
    
    def detection_loop(self):
        """
        Main detection loop - reads frames from Redis and processes them.
        """
        logger.info("🚀 Starting anti-cheat detection loop...")
        
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
                processed_frame, alert_count = self.detect_and_track(frame)
                
                # Update shared frame for streaming
                with self.frame_lock:
                    self.current_frame = processed_frame.copy()
                
                frame_count += 1
                self.frames_processed += 1
                if alert_count > 0:
                    self.detections_count += 1
                
                # Log stats every 5 seconds
                current_time = time.time()
                if current_time - last_log_time >= 5.0:
                    elapsed = current_time - start_time
                    avg_fps = frame_count / elapsed
                    logger.info(
                        f"📊 Anti-Cheat | Processed: {self.frames_processed} | "
                        f"Alerts: {self.detections_count} | FPS: {avg_fps:.2f}"
                    )
                    last_log_time = current_time
                    
        except KeyboardInterrupt:
            logger.info("⏹️  Stopping anti-cheat detector...")
        except Exception as e:
            logger.error(f"❌ Error in detection loop: {e}", exc_info=True)
        finally:
            logger.info(f"✅ Anti-cheat detector stopped. Processed {frame_count} frames")
    
    def start(self):
        """Start detection in background thread."""
        if self.running:
            logger.warning("Detector already running")
            return
        
        self.running = True
        self.thread = Thread(target=self.detection_loop, daemon=True)
        self.thread.start()
        logger.info("🚀 Anti-cheat detector started")
    
    def stop(self):
        """Stop detection."""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        logger.info("⏹️  Anti-cheat detector stopped")
    
    def get_current_frame(self):
        """Get the current processed frame (thread-safe)."""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None


# Global detector instance
detector = None
video_processor = None


class VideoProcessor:
    """
    Process video files with anti-cheat detection.
    """
    def __init__(self):
        # Use GPU if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"🖥️  Video Processor using device: {self.device}")
        
        if self.device == 'cuda':
            logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Detection models
        logger.info("🔄 Loading YOLOv8-Pose model for video processing...")
        self.yolo_model = YOLO('anticheat1/yolov8n-pose.pt')
        
        # Move model to GPU with FP16 if available
        if self.device == 'cuda':
            self.yolo_model.to('cuda')
            # Enable GPU optimizations
            torch.backends.cudnn.benchmark = True
            logger.info("✅ YOLO model moved to GPU with optimizations")
            
            # Warmup the model with a dummy inference
            logger.info("🔥 Warming up GPU model...")
            dummy_frame = torch.zeros((1, 3, 640, 640)).cuda().half()
            _ = self.yolo_model(dummy_frame, verbose=False)
            logger.info("✅ GPU warmup complete")
        
        # DeepSORT tracker with GPU optimization and increased batch size
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=None,
            embedder="mobilenet",
            half=True if self.device == 'cuda' else False,
            embedder_gpu=True if self.device == 'cuda' else False
        )
        
        # Behavior detector
        self.pose_behavior = YOLOPoseBehaviorDetector(
            fps=30.0,
            walk_move_thresh=20.0,
            pass_dist_thresh=60.0,
            same_row_thresh=80.0,
            pass_min_frames=3
        )
        
        # Video state
        self.current_video = None
        self.current_frame = None
        self.frame_lock = Lock()
        self.is_processing = False
        self.processing_thread = None
        
        logger.info("✅ Video processor initialized")
    
    def process_video(self, video_path):
        """Process video file and stream frames with GPU optimization."""
        self.is_processing = True
        self.current_video = video_path
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            self.is_processing = False
            return
        
        logger.info(f"🎥 Processing video: {video_path}")
        
        # Reset tracker for new video with increased batch size
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=None,
            embedder="mobilenet",
            half=True if self.device == 'cuda' else False,
            embedder_gpu=True if self.device == 'cuda' else False
        )
        
        frame_count = 0
        
        try:
            while self.is_processing and self.current_video == video_path:
                ret, frame = cap.read()
                if not ret:
                    # Loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    # Reset tracker on loop with increased batch size
                    self.tracker = DeepSort(
                        max_age=30,
                        n_init=3,
                        nms_max_overlap=1.0,
                        max_cosine_distance=0.3,
                        nn_budget=None,
                        embedder="mobilenet",
                        half=True if self.device == 'cuda' else False,
                        embedder_gpu=True if self.device == 'cuda' else False
                    )
                    continue
                
                # Resize for processing
                frame_small = cv2.resize(frame, (960, 540))
                
                # YOLO detection with GPU and FP16 for faster inference
                results = self.yolo_model(
                    frame_small, 
                    verbose=False, 
                    classes=[0, 67], 
                    device=self.device,
                    half=True,
                    imgsz=640
                )[0]
                
                detections = []
                phone_boxes = []
                
                for box in results.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = results.names[cls]
                    
                    if conf < 0.4:
                        continue
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    if label == "person":
                        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))
                    elif label == "cell phone":
                        phone_boxes.append([x1, y1, x2, y2])
                
                # Update tracker
                tracks = self.tracker.update_tracks(detections, frame=frame_small)
                
                # Pose behavior detection
                pose_alerts = self.pose_behavior.update(frame_small, tracks, results)
                
                # Phone detection - add to alerts if phone detected near person
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    
                    tid = track.track_id
                    tx1, ty1, tx2, ty2 = track.to_ltrb()
                    
                    for (px1, py1, px2, py2) in phone_boxes:
                        # Calculate IoU
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
                
                # Draw boxes ONLY for people with alerts
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    
                    tid = track.track_id
                    current_alerts = pose_alerts.get(tid, set())
                    
                    # Only draw if there are alerts
                    if current_alerts:
                        draw_track_box(frame_small, track.to_ltrb(), tid, alerts=current_alerts)
                
                # Update current frame
                with self.frame_lock:
                    self.current_frame = frame_small.copy()
                
                frame_count += 1
                # No sleep - process as fast as GPU allows
                
        except Exception as e:
            logger.error(f"Error processing video: {e}", exc_info=True)
        finally:
            cap.release()
            logger.info(f"✅ Finished processing video. Frames: {frame_count}")
    
    def start_processing(self, video_path):
        """Start video processing in background thread."""
        # Stop any existing processing
        if self.is_processing:
            logger.info(f"Stopping current video processing...")
            self.stop_processing()
            time.sleep(0.5)  # Give time for cleanup
        
        logger.info(f"Starting new video: {video_path}")
        self.processing_thread = Thread(target=self.process_video, args=(video_path,), daemon=True)
        self.processing_thread.start()
    
    def stop_processing(self):
        """Stop video processing."""
        if self.is_processing:
            logger.info("Stopping video processing...")
            self.is_processing = False
            self.current_video = None
            if self.processing_thread:
                self.processing_thread.join(timeout=2.0)
            logger.info("Video processing stopped")
    
    def get_current_frame(self):
        """Get current processed frame."""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None


def generate_frames():
    """
    Generate MJPEG stream for Flask endpoint.
    """
    global detector
    
    if detector is None or not detector.running:
        # Return error frame
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
            time.sleep(0.03)  # ~30fps
            continue
        
        # Encode frame as JPEG
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            continue
        
        # Yield frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.03)  # ~30fps


@app.route('/api/anticheat_feed')
def video_feed():
    """Video streaming route for anti-cheat detection."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/anticheat_stats')
def get_stats():
    """Get detection statistics."""
    global detector
    
    if detector is None:
        return {'error': 'Detector not initialized'}, 500
    
    return {
        'frames_processed': detector.frames_processed,
        'detections_count': detector.detections_count,
        'running': detector.running
    }


@app.route('/api/videos/list')
def list_videos():
    """List available test videos."""
    try:
        if not os.path.exists(TEST_VIDEOS_DIR):
            return {'videos': [], 'error': 'Test videos directory not found'}
        
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_files.extend(glob.glob(os.path.join(TEST_VIDEOS_DIR, ext)))
        
        videos = []
        for video_path in sorted(video_files):
            filename = os.path.basename(video_path)
            file_size = os.path.getsize(video_path)
            videos.append({
                'filename': filename,
                'path': video_path,
                'size': file_size,
                'size_mb': round(file_size / (1024 * 1024), 2)
            })
        
        return {'videos': videos}
    except Exception as e:
        logger.error(f"Error listing videos: {e}")
        return {'error': str(e)}, 500


@app.route('/api/videos/process', methods=['POST'])
def process_video():
    """Start processing a selected video."""
    global video_processor
    
    try:
        data = request.get_json()
        video_filename = data.get('filename')
        
        if not video_filename:
            return {'error': 'No filename provided'}, 400
        
        video_path = os.path.join(TEST_VIDEOS_DIR, video_filename)
        
        if not os.path.exists(video_path):
            return {'error': 'Video file not found'}, 404
        
        # Initialize video processor if needed
        if video_processor is None:
            video_processor = VideoProcessor()
        
        # Start processing
        video_processor.start_processing(video_path)
        
        return {'status': 'processing', 'video': video_filename}
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return {'error': str(e)}, 500


@app.route('/api/videos/stop', methods=['POST'])
def stop_video():
    """Stop video processing."""
    global video_processor
    
    if video_processor:
        video_processor.stop_processing()
    
    return {'status': 'stopped'}


@app.route('/api/video_feed')
def video_feed_stream():
    """Stream processed video frames."""
    global video_processor
    
    def generate():
        while True:
            if video_processor is None or not video_processor.is_processing:
                error_frame = np.zeros((540, 960, 3), dtype=np.uint8)
                cv2.putText(error_frame, "No video processing", (300, 270),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, buffer = cv2.imencode('.jpg', error_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.1)
                continue
            
            frame = video_processor.get_current_frame()
            if frame is None:
                time.sleep(0.03)
                continue
            
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.03)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/health')
def health():
    """Health check endpoint."""
    return {'status': 'ok', 'service': 'anticheat_detector'}


def main():
    """
    Main entry point.
    """
    import signal
    
    global detector
    
    # Initialize detector
    detector = AntiCheatDetector(
        redis_host='localhost',
        redis_port=6379,
        stream_name='camera:stream'
    )
    
    # Start detection for camera feed
    detector.start()
    
    # Graceful shutdown
    def signal_handler(sig, frame):
        logger.info("\n🛑 Shutting down anti-cheat detector...")
        if detector.running:
            detector.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run Flask app
    logger.info("🌐 Starting Flask server on port 5002...")
    app.run(host='0.0.0.0', port=5002, debug=False, threaded=True)


if __name__ == '__main__':
    main()
