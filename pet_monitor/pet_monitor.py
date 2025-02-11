import os
import sys
import time
import numpy as np
import sounddevice as sd
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import threading
import queue
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO
import base64
import platform
import logging
import scipy.io.wavfile as wavfile
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

class PetMonitor:
    def __init__(self, 
                 video_output_dir='./recordings/video', 
                 audio_output_dir='./recordings/audio',
                 notification_method='print',
                 camera_id=0):
        
        # Create output directories with absolute paths
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.video_output_dir = os.path.join(self.base_dir, video_output_dir.lstrip('./'))
        self.audio_output_dir = os.path.join(self.base_dir, audio_output_dir.lstrip('./'))
        
        os.makedirs(self.video_output_dir, exist_ok=True)
        os.makedirs(self.audio_output_dir, exist_ok=True)
        
        # Device settings
        self.preferred_camera_id = camera_id
        self.preferred_audio_id = None
        self.camera = None
        self.init_camera(camera_id)
        
        # Motion detection parameters
        self.motion_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)
        self.min_motion_area = 500  # Minimum contour area to be considered motion
        self.motion_threshold = 20   # Threshold for motion detection
        self.motion_detected = False
        self.last_motion_time = time.time()
        self.motion_cooldown = 5  # Seconds between motion notifications
        
        # Bark detection parameters
        self.bark_sensitivity = 50  # General sensitivity (1-100)
        self.bark_duration = 300    # Minimum duration in ms
        self.bark_threshold = 50    # Amplitude threshold (1-100)
        self.bark_cooldown = 3      # Seconds between bark notifications
        self.last_bark_time = time.time()
        
        # Audio recording parameters
        self.sample_rate = 44100
        self.channels = 1
        self.audio_chunk_duration = 1.0
        self.audio_device = None
        
        # Initialize audio device with error handling
        self.init_audio()
        
        # Initialize bark detection model with dummy data
        self.init_bark_detection()
        
        self.notification_method = notification_method
        
        # Queues for thread communication
        self.video_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        
        # Frame for web interface
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Control flags
        self.running = True
        self.audio_enabled = True
    
    def init_bark_detection(self):
        """Initialize bark detection model with dummy data"""
        try:
            # Create some dummy data to initialize the scaler
            dummy_data = np.random.rand(100, 1000)  # 100 samples, 1000 features
            dummy_labels = np.random.choice([0, 1], 100)  # Binary classification
            
            self.scaler = StandardScaler()
            self.scaler.fit(dummy_data)
            
            self.classifier = SVC(kernel='rbf')
            self.classifier.fit(self.scaler.transform(dummy_data), dummy_labels)
            
            logger.info("Bark detection model initialized with dummy data")
        except Exception as e:
            logger.error(f"Failed to initialize bark detection model: {str(e)}")
            self.scaler = None
            self.classifier = None
    
    def init_audio(self, device_id=None):
        """Initialize audio with error handling and fallback"""
        try:
            if device_id is not None:
                device_info = sd.query_devices(device_id)
                if device_info['max_input_channels'] > 0:
                    self.audio_device = device_id
                    self.preferred_audio_id = device_id
                    self.sample_rate = int(device_info['default_samplerate'])
                    self.channels = min(device_info['max_input_channels'], 2)
                    logger.info(f"Successfully initialized audio device {device_id}")
                    return
            
            # If no device specified or specified device failed, find first input device
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    self.audio_device = i
                    self.preferred_audio_id = i
                    self.sample_rate = int(device['default_samplerate'])
                    self.channels = min(device['max_input_channels'], 2)
                    logger.info(f"Using default audio device {i}")
                    return
            
            raise Exception("No suitable audio input device found")
        except Exception as e:
            logger.error(f"Error initializing audio: {str(e)}")
            self.audio_device = None
            self.preferred_audio_id = None
    
    def detect_bark(self, audio_chunk):
        """Detect barking in audio chunk with configurable parameters"""
        try:
            if self.scaler is None or self.classifier is None:
                return False
            
            # Calculate audio features
            amplitude = np.abs(audio_chunk)
            mean_amplitude = np.mean(amplitude)
            
            # Apply sensitivity and threshold
            normalized_amplitude = (mean_amplitude * self.bark_sensitivity) / 50
            threshold_value = (self.bark_threshold * np.max(amplitude)) / 100
            
            # Check if the sound meets our criteria
            if normalized_amplitude > threshold_value:
                # Check duration
                duration_samples = int(self.bark_duration * self.sample_rate / 1000)
                if len(np.where(amplitude > threshold_value)[0]) >= duration_samples:
                    # Check cooldown
                    current_time = time.time()
                    if (current_time - self.last_bark_time) > self.bark_cooldown:
                        self.last_bark_time = current_time
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Bark detection error: {str(e)}")
            return False
    
    def save_audio(self, audio_chunk, filename):
        """Save audio chunk to file"""
        try:
            # Convert float32 to int16
            audio_int16 = (audio_chunk * 32767).astype(np.int16)
            wavfile.write(filename, self.sample_rate, audio_int16)
            return True
        except Exception as e:
            logger.error(f"Failed to save audio: {str(e)}")
            return False
    
    def record_audio(self):
        """Record audio from the microphone"""
        while self.running and self.audio_enabled:
            try:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                audio_filename = os.path.join(self.audio_output_dir, f'pet_audio_{timestamp}.wav')
                
                # Record audio
                audio_chunk = sd.rec(
                    int(self.sample_rate * self.audio_chunk_duration),
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    blocking=True
                )
                
                if self.detect_bark(audio_chunk):
                    self._send_notification("Bark detected!", 'bark')
                
                # Save audio chunk
                if self.save_audio(audio_chunk, audio_filename):
                    self.audio_queue.put(audio_filename)
                    
                    # Stream audio to web interface
                    audio_bytes = base64.b64encode(audio_chunk.tobytes()).decode('utf-8')
                    socketio.emit('audio_data', {'audio': audio_bytes})
                
            except Exception as e:
                logger.error(f"Audio recording error: {str(e)}")
                time.sleep(1)
    
    def _send_notification(self, message, notification_type='motion'):
        """Send a notification to the web interface with type"""
        try:
            logger.info(f"NOTIFICATION ({notification_type}): {message}")
            socketio.emit('notification', {
                'message': message,
                'type': notification_type
            })
        except Exception as e:
            logger.error(f"Failed to send notification: {str(e)}")

    def init_camera(self, camera_id):
        """Initialize camera with error handling and fallback"""
        try:
            if self.camera is not None:
                self.camera.release()
            
            self.camera = cv2.VideoCapture(camera_id)
            if not self.camera.isOpened():
                raise Exception(f"Failed to open camera {camera_id}")
            
            self.preferred_camera_id = camera_id
            logger.info(f"Successfully initialized camera {camera_id}")
        except Exception as e:
            logger.error(f"Error initializing camera {camera_id}: {str(e)}")
            # Try to fall back to first available camera
            if camera_id != 0:
                logger.info("Attempting to fall back to default camera (0)")
                try:
                    self.camera = cv2.VideoCapture(0)
                    if self.camera.isOpened():
                        self.preferred_camera_id = 0
                        logger.info("Successfully fell back to default camera")
                    else:
                        self.camera = None
                        logger.error("Failed to initialize fallback camera")
                except Exception as e2:
                    self.camera = None
                    logger.error(f"Error initializing fallback camera: {str(e2)}")

    def get_frame(self):
        """Safely get the current frame"""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None

    def detect_motion(self, frame):
        """Detect motion in the frame"""
        try:
            # Apply background subtraction
            fg_mask = self.motion_detector.apply(frame)
            
            # Threshold the mask
            _, thresh = cv2.threshold(fg_mask, self.motion_threshold, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Check if any contour is large enough
            motion_detected = False
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_motion_area:
                    motion_detected = True
                    # Draw rectangle around motion
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Update motion status and send notification if needed
            current_time = time.time()
            if motion_detected and not self.motion_detected and \
               (current_time - self.last_motion_time) > self.motion_cooldown:
                self._send_notification("Motion detected!")
                self.last_motion_time = current_time
            
            self.motion_detected = motion_detected
            return frame
            
        except Exception as e:
            logger.error(f"Motion detection error: {str(e)}")
            return frame

    def record_video(self):
        """Record video from the camera"""
        while self.running:
            try:
                ret, frame = self.camera.read()
                if ret:
                    # Apply motion detection
                    frame_with_motion = self.detect_motion(frame)
                    
                    # Update the current frame for web streaming
                    with self.frame_lock:
                        self.current_frame = frame_with_motion
                    
                    # Save frame if motion is detected
                    if self.motion_detected:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        video_filename = os.path.join(self.video_output_dir, f'motion_{timestamp}.jpg')
                        cv2.imwrite(video_filename, frame_with_motion)
                        self.video_queue.put(video_filename)
                
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Video recording error: {str(e)}")
                time.sleep(1)
    
    def start_monitoring(self):
        """Start the monitoring threads"""
        self.running = True
        video_thread = threading.Thread(target=self.record_video, daemon=True)
        video_thread.start()
        
        if self.audio_enabled:
            audio_thread = threading.Thread(target=self.record_audio, daemon=True)
            audio_thread.start()
        else:
            logger.warning("Audio recording is disabled due to initialization failure")
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.camera is not None:
            self.camera.release()
        logger.info("Resources cleaned up")

def get_available_devices():
    """Get lists of available video and audio devices"""
    try:
        # Get video devices
        video_devices = []
        index = 0
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            video_devices.append({
                'id': index,
                'name': f'Camera {index}'  # OpenCV doesn't provide device names
            })
            cap.release()
            index += 1

        # Get audio devices
        audio_devices = []
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:  # Input device
                audio_devices.append({
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'default_samplerate': device['default_samplerate']
                })

        return {
            'video_devices': video_devices,
            'audio_devices': audio_devices
        }
    except Exception as e:
        logger.error(f"Error getting devices: {str(e)}")
        return {'video_devices': [], 'audio_devices': []}

@app.route('/api/devices')
def list_devices():
    """API endpoint to list available devices"""
    return jsonify(get_available_devices())

@app.route('/api/parameters')
def get_parameters():
    """Get current parameter values"""
    try:
        if monitor:
            return jsonify({
                'motion': {
                    'min_area': monitor.min_motion_area,
                    'threshold': monitor.motion_threshold,
                    'cooldown': monitor.motion_cooldown
                },
                'bark': {
                    'sensitivity': monitor.bark_sensitivity,
                    'duration': monitor.bark_duration,
                    'threshold': monitor.bark_threshold,
                    'cooldown': monitor.bark_cooldown
                },
                'devices': {
                    'camera_id': monitor.preferred_camera_id,
                    'audio_id': monitor.preferred_audio_id
                }
            })
        else:
            return jsonify({'error': 'Monitor not initialized'}), 500
    except Exception as e:
        logger.error(f"Error getting parameters: {str(e)}")
        return jsonify({'error': str(e)}), 500

@socketio.on('select_video_device')
def handle_video_device(data):
    """Handle video device selection"""
    try:
        if monitor and 'device_id' in data:
            device_id = int(data['device_id'])
            # Try to initialize the camera
            monitor.init_camera(device_id)
            
            if monitor.camera and monitor.camera.isOpened():
                socketio.emit('device_update', {
                    'type': 'video',
                    'status': 'success',
                    'selected_id': monitor.preferred_camera_id
                })
            else:
                raise Exception(f"Failed to initialize camera {device_id}")
    except Exception as e:
        logger.error(f"Error switching video device: {str(e)}")
        socketio.emit('device_update', {
            'type': 'video',
            'status': 'error',
            'message': str(e),
            'selected_id': monitor.preferred_camera_id
        })

@socketio.on('select_audio_device')
def handle_audio_device(data):
    """Handle audio device selection"""
    try:
        if monitor and 'device_id' in data:
            device_id = int(data['device_id'])
            # Try to initialize the audio device
            monitor.init_audio(device_id)
            
            if monitor.audio_device is not None:
                socketio.emit('device_update', {
                    'type': 'audio',
                    'status': 'success',
                    'selected_id': monitor.preferred_audio_id
                })
            else:
                raise Exception(f"Failed to initialize audio device {device_id}")
    except Exception as e:
        logger.error(f"Error switching audio device: {str(e)}")
        socketio.emit('device_update', {
            'type': 'audio',
            'status': 'error',
            'message': str(e),
            'selected_id': monitor.preferred_audio_id
        })

# Create global monitor instance
monitor = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():
    """Get the current status of the monitor"""
    global monitor
    return jsonify({
        'camera_active': monitor is not None and monitor.camera is not None and monitor.camera.isOpened(),
        'audio_enabled': monitor is not None and monitor.audio_enabled,
        'os_type': platform.system(),
        'python_version': platform.python_version()
    })

def gen_frames():
    """Generate video frames for streaming"""
    while True:
        if monitor is None:
            time.sleep(1)
            continue
            
        frame = monitor.get_frame()
        if frame is not None:
            try:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                logger.error(f"Frame generation error: {str(e)}")
        time.sleep(0.033)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('update_motion_params')
def handle_motion_params(data):
    """Handle updates to motion detection parameters"""
    try:
        if monitor:
            updates = {}
            if 'min_area' in data:
                monitor.min_motion_area = int(data['min_area'])
                updates['min_area'] = monitor.min_motion_area
            if 'threshold' in data:
                monitor.motion_threshold = int(data['threshold'])
                updates['threshold'] = monitor.motion_threshold
            if 'cooldown' in data:
                monitor.motion_cooldown = int(data['cooldown'])
                updates['cooldown'] = monitor.motion_cooldown
            
            logger.info(f"Motion parameters updated: {updates}")
            # Confirm the updates back to the client
            socketio.emit('params_updated', {
                'type': 'motion',
                'values': updates
            })
    except Exception as e:
        logger.error(f"Error updating motion parameters: {str(e)}")
        socketio.emit('params_updated', {
            'type': 'motion',
            'status': 'error',
            'message': str(e)
        })

@socketio.on('update_bark_params')
def handle_bark_params(data):
    """Handle updates to bark detection parameters"""
    try:
        if monitor:
            updates = {}
            if 'sensitivity' in data:
                monitor.bark_sensitivity = int(data['sensitivity'])
                updates['sensitivity'] = monitor.bark_sensitivity
            if 'duration' in data:
                monitor.bark_duration = int(data['duration'])
                updates['duration'] = monitor.bark_duration
            if 'threshold' in data:
                monitor.bark_threshold = int(data['threshold'])
                updates['threshold'] = monitor.bark_threshold
            if 'cooldown' in data:
                monitor.bark_cooldown = int(data['cooldown'])
                updates['cooldown'] = monitor.bark_cooldown
            
            logger.info(f"Bark parameters updated: {updates}")
            # Confirm the updates back to the client
            socketio.emit('params_updated', {
                'type': 'bark',
                'values': updates
            })
    except Exception as e:
        logger.error(f"Error updating bark parameters: {str(e)}")
        socketio.emit('params_updated', {
            'type': 'bark',
            'status': 'error',
            'message': str(e)
        })

@socketio.on('toggle_audio')
def handle_toggle_audio(data):
    """Handle audio toggle"""
    try:
        if monitor:
            monitor.audio_enabled = data.get('enabled', True)
            logger.info(f"Audio {'enabled' if monitor.audio_enabled else 'disabled'}")
    except Exception as e:
        logger.error(f"Error toggling audio: {str(e)}")

def main():
    global monitor
    try:
        # Initialize the monitor
        monitor = PetMonitor()
        monitor.start_monitoring()
        
        # Get the hostname
        host = '127.0.0.1'  # Use localhost instead of hostname
        port = 5000
        
        logger.info(f"Starting web server at http://{host}:{port}")
        logger.info("Debug mode enabled - server will reload when files change")
        socketio.run(app, 
                    host=host, 
                    port=port, 
                    debug=True,  # Enable debug mode for development
                    use_reloader=True,  # Enable auto-reload
                    allow_unsafe_werkzeug=True)
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
    finally:
        if monitor is not None:
            monitor.cleanup()

if __name__ == "__main__":
    main()
