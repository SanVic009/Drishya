import cv2
from flask import Flask, Response, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from ultralytics import YOLO
import numpy as np

# Load environment variables from .env.python file
load_dotenv('.env.python')

app = Flask(__name__)
CORS(app)

# Global camera object
camera = None
# Load YOLO model
model = None

def get_camera():
    """Get or initialize the camera"""
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)  # 0 is the default camera
        # Set camera properties for better quality
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        camera.set(cv2.CAP_PROP_FPS, 30)
    return camera

def get_model():
    """Load YOLO model"""
    global model
    if model is None:
        try:
            # Try to load a custom model or use YOLOv8 for general detection
            # You can train a custom model for smoking detection
            model = YOLO('yolov8n.pt')  # Using nano model for speed
            print("✅ YOLO model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            model = None
    return model

def detect_smoking(frame):
    """
    Detect smoking in the frame using YOLO
    This is a placeholder - for actual smoking detection, you would need:
    1. A custom-trained YOLO model for smoking detection
    2. Or use object detection to find person + cigarette proximity
    """
    model = get_model()
    
    if model is None:
        # If no model, just return the original frame with a message
        cv2.putText(frame, "YOLO Model Not Loaded", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame, False
    
    # Run YOLO detection
    results = model(frame, verbose=False)
    
    # Get annotated frame
    annotated_frame = results[0].plot()
    
    # Check for smoking-related objects
    smoking_detected = False
    detected_objects = results[0].boxes
    
    if detected_objects is not None:
        for box in detected_objects:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            
            # For demonstration: flag if person is detected
            # In production, you'd check for cigarette/smoking specifically
            if class_name == 'person' and confidence > 0.5:
                # Add smoking detection logic here
                # For now, just showing detection is working
                pass
            
            # If you have a custom model trained for smoking detection:
            # if class_name == 'smoking' or class_name == 'cigarette':
            #     smoking_detected = True
    
    # Add status overlay
    status_text = "🚭 MONITORING: Smoking Detection Active"
    cv2.putText(annotated_frame, status_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if smoking_detected:
        cv2.putText(annotated_frame, "⚠️ ALERT: SMOKING DETECTED!", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    return annotated_frame, smoking_detected

def generate_frames():
    """Generate camera frames with YOLO detection"""
    camera = get_camera()
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Apply YOLO detection
        processed_frame, smoking_detected = detect_smoking(frame)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        # Yield frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return jsonify({
        'message': 'Drishya Python Video Stream Server',
        'status': 'running',
        'endpoints': {
            'health': '/api/health',
            'video_feed': '/api/video_feed',
            'cameras': '/api/cameras'
        }
    })

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'Drishya Video Stream API',
        'camera_available': get_camera().isOpened()
    })

@app.route('/api/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/api/cameras')
def cameras():
    """Get list of available cameras"""
    camera = get_camera()
    return jsonify({
        'cameras': [
            {
                'id': 1,
                'name': 'Camera 01',
                'location': 'Main Entrance',
                'status': 'active' if camera.isOpened() else 'inactive',
                'streamUrl': 'http://localhost:5001/api/video_feed',
                'type': 'live'
            },
            {
                'id': 2,
                'name': 'Camera 02',
                'location': 'Parking Lot',
                'status': 'active',
                'streamUrl': 'https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=600&h=400&fit=crop',
                'type': 'placeholder'
            },
            {
                'id': 3,
                'name': 'Camera 03',
                'location': 'Office Floor',
                'status': 'active',
                'streamUrl': 'https://images.unsplash.com/photo-1497366216548-37526070297c?w=600&h=400&fit=crop',
                'type': 'placeholder'
            }
        ]
    })

@app.route('/api/snapshot')
def snapshot():
    """Capture a single frame"""
    camera = get_camera()
    success, frame = camera.read()
    
    if success:
        ret, buffer = cv2.imencode('.jpg', frame)
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    else:
        return jsonify({'error': 'Failed to capture frame'}), 500

if __name__ == '__main__':
    PORT = int(os.getenv('PORT', 5001))
    print(f'🎥 Starting Drishya Video Stream Server on port {PORT}')
    print(f'📹 Camera feed: http://localhost:{PORT}/api/video_feed')
    
    try:
        app.run(host='0.0.0.0', port=PORT, debug=True, threaded=True)
    finally:
        # Release camera on shutdown
        if camera is not None:
            camera.release()
            cv2.destroyAllWindows()
