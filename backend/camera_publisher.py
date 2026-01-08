"""
Camera Publisher Service
Captures webcam frames and publishes them to Redis stream.
This is the single source that feeds all detection models (anticheat, QR, anomaly).
"""

import cv2
import redis
import time
import logging
import numpy as np
from threading import Thread, Lock
from flask import Flask, Response
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


class CameraPublisher:
    """
    Publishes camera frames to Redis stream for multiple consumers.
    Uses a compact encoding strategy to minimize memory usage.
    """
    
    def __init__(
        self,
        redis_host='localhost',
        redis_port=6379,
        stream_name='camera:stream',
        camera_id=0,
        max_stream_length=10,  # Keep only last 10 frames in Redis
        target_fps=30
    ):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=False  # We need binary data for images
        )
        self.stream_name = stream_name
        self.camera_id = camera_id
        self.max_stream_length = max_stream_length
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        
        self.running = False
        self.thread = None
        
        # For raw feed streaming
        self.current_frame = None
        self.frame_lock = Lock()
        
        # Test Redis connection
        try:
            self.redis_client.ping()
            logger.info(f"✅ Connected to Redis at {redis_host}:{redis_port}")
        except redis.ConnectionError as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
    
    def encode_frame(self, frame):
        """
        Encode frame to JPEG for efficient Redis storage.
        Returns bytes that can be stored in Redis.
        """
        # Encode frame as JPEG (quality 90 for good balance)
        success, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            raise ValueError("Failed to encode frame")
        return encoded.tobytes()
    
    def publish_frame(self, frame):
        """
        Publish a single frame to Redis stream with metadata.
        """
        try:
            # Encode frame
            encoded_frame = self.encode_frame(frame)
            
            # Get frame metadata
            height, width = frame.shape[:2]
            timestamp = time.time()
            
            # Publish to Redis stream
            message = {
                b'frame': encoded_frame,
                b'width': str(width).encode(),
                b'height': str(height).encode(),
                b'timestamp': str(timestamp).encode(),
                b'channels': str(frame.shape[2] if len(frame.shape) > 2 else 1).encode()
            }
            
            # Add to stream with automatic trimming
            stream_id = self.redis_client.xadd(
                self.stream_name,
                message,
                maxlen=self.max_stream_length,
                approximate=True  # Faster trimming
            )
            
            return stream_id
            
        except Exception as e:
            logger.error(f"Error publishing frame: {e}")
            return None
    
    def capture_loop(self):
        """
        Main capture loop - reads from camera and publishes to Redis.
        """
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            logger.error(f"❌ Failed to open camera {self.camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        logger.info(f"📹 Camera opened: {actual_width}x{actual_height} @ {actual_fps}fps")
        logger.info(f"🔴 Publishing to Redis stream: {self.stream_name}")
        
        frame_count = 0
        start_time = time.time()
        last_log_time = start_time
        
        try:
            while self.running:
                loop_start = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame, retrying...")
                    time.sleep(0.1)
                    continue
                
                # Store current frame for raw feed streaming
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                # Publish to Redis
                stream_id = self.publish_frame(frame)
                
                if stream_id:
                    frame_count += 1
                    
                    # Log stats every 5 seconds
                    current_time = time.time()
                    if current_time - last_log_time >= 5.0:
                        elapsed = current_time - start_time
                        avg_fps = frame_count / elapsed
                        logger.info(
                            f"📊 Published {frame_count} frames | "
                            f"Avg FPS: {avg_fps:.2f} | "
                            f"Stream: {self.stream_name}"
                        )
                        last_log_time = current_time
                
                # Frame rate control
                elapsed = time.time() - loop_start
                sleep_time = self.frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            logger.info("⏹️  Stopping camera publisher...")
        except Exception as e:
            logger.error(f"❌ Error in capture loop: {e}")
        finally:
            cap.release()
            logger.info(f"✅ Published {frame_count} frames total")
    
    def start(self):
        """Start publishing in a background thread."""
        if self.running:
            logger.warning("Publisher already running")
            return
        
        self.running = True
        self.thread = Thread(target=self.capture_loop, daemon=True)
        self.thread.start()
        logger.info("🚀 Camera publisher started")
    
    def stop(self):
        """Stop publishing."""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        logger.info("⏹️  Camera publisher stopped")
    
    def get_stream_info(self):
        """Get current stream information."""
        try:
            stream_length = self.redis_client.xlen(self.stream_name)
            return {
                'stream_name': self.stream_name,
                'length': stream_length,
                'max_length': self.max_stream_length
            }
        except Exception as e:
            logger.error(f"Error getting stream info: {e}")
            return None
    
    def get_current_frame(self):
        """Get the current raw frame (thread-safe)."""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None


# Global publisher instance
publisher = None


def generate_raw_feed():
    """Generate MJPEG stream for raw camera feed."""
    global publisher
    
    if publisher is None or not publisher.running:
        # Return error frame
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Camera not running", (50, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', error_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return
    
    while True:
        frame = publisher.get_current_frame()
        
        if frame is None:
            time.sleep(0.03)
            continue
        
        # Encode frame as JPEG
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            continue
        
        # Yield frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.03)  # ~30fps


@app.route('/api/raw_feed')
def raw_feed():
    """Raw camera feed without any detections."""
    return Response(
        generate_raw_feed(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/health')
def health():
    """Health check endpoint."""
    return {'status': 'ok', 'service': 'camera_publisher'}


def main():
    """
    Main entry point - runs the camera publisher with Flask server.
    """
    import signal
    import sys
    
    global publisher
    
    publisher = CameraPublisher(
        redis_host='localhost',
        redis_port=6379,
        stream_name='camera:stream',
        camera_id=0,
        max_stream_length=10,
        target_fps=30
    )
    
    def signal_handler(sig, frame):
        logger.info("\n🛑 Shutting down camera publisher...")
        publisher.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start camera publisher
    publisher.start()
    
    # Run Flask app
    logger.info("🌐 Starting Flask server on port 5004...")
    app.run(host='0.0.0.0', port=5004, debug=False, threaded=True)


if __name__ == '__main__':
    main()
