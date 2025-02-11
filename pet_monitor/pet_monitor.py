import os
import sys
import time
import platform
import numpy as np
import sounddevice as sd
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import threading
import queue
from flask import Flask, render_template, Response, jsonify, send_from_directory
from flask_socketio import SocketIO
import base64
import logging
import scipy.io.wavfile as wavfile
import tempfile
import soundfile as sf

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
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

        # Recording buffers
        self.video_buffer = []
        self.audio_buffer = []
        self.video_buffer_seconds = 5  # Keep 5 seconds of video
        self.audio_buffer_seconds = 5  # Keep 5 seconds of audio
        self.last_video_save = 0
        self.last_audio_save = 0
        self.min_save_interval = 2  # Minimum seconds between saves
        
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
            # List all audio devices for debugging
            devices = sd.query_devices()
            logger.debug("Available audio devices:")
            for i, device in enumerate(devices):
                logger.debug(f"  [{i}] {device['name']} (in={device['max_input_channels']}, "
                           f"out={device['max_output_channels']}, default={device['default_samplerate']}Hz)")
            
            if device_id is not None:
                device_info = sd.query_devices(device_id)
                if device_info['max_input_channels'] > 0:
                    self.audio_device = device_id
                    self.preferred_audio_id = device_id
                    self.sample_rate = int(device_info['default_samplerate'])
                    self.channels = min(device_info['max_input_channels'], 2)
                    logger.info(f"Successfully initialized audio device {device_id}: "
                              f"{device_info['name']} ({self.channels} channels @ {self.sample_rate}Hz)")
                    return
            
            # If no device specified or specified device failed, find first input device
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    self.audio_device = i
                    self.preferred_audio_id = i
                    self.sample_rate = int(device['default_samplerate'])
                    self.channels = min(device['max_input_channels'], 2)
                    logger.info(f"Using default audio device {i}: "
                              f"{device['name']} ({self.channels} channels @ {self.sample_rate}Hz)")
                    return
            
            raise Exception("No suitable audio input device found")
        except Exception as e:
            logger.error(f"Error initializing audio: {str(e)}")
            self.audio_device = None
            self.preferred_audio_id = None
    
    def detect_bark(self, audio_chunk):
        """Detect barking in audio chunk and save audio if detected"""
        try:
            # Convert to float32 if needed
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)
            
            # Calculate RMS amplitude with higher scaling for low inputs
            rms = np.sqrt(np.mean(np.square(audio_chunk))) * 10000
            
            # Calculate peak amplitude with higher scaling
            peak = np.max(np.abs(audio_chunk)) * 10000
            
            # Normalize to 0-100 range (values are already scaled)
            normalized_rms = min(100, rms)
            normalized_peak = min(100, peak)
            
            # Apply sensitivity factor
            sensitivity_factor = self.bark_sensitivity / 50.0  # Convert 0-100 to scaling factor
            detection_value = (normalized_rms + normalized_peak) / 2 * sensitivity_factor
            
            # Check if amplitude exceeds threshold and duration
            # Scale the input values before threshold comparison
            scaled_audio = np.abs(audio_chunk) * 10000
            samples_above_threshold = np.sum(scaled_audio > self.bark_threshold)
            
            # Reduce required duration based on sensitivity
            duration_samples = int((self.bark_duration / sensitivity_factor) * self.sample_rate / 1000)
            duration_samples = max(100, min(duration_samples, int(0.5 * self.sample_rate)))  # Between 100 samples and 0.5s
            
            # Log detection values for debugging
            if detection_value > self.bark_threshold / 2:  # Only log when there's significant sound
                logger.debug(f"Bark detection values - RMS: {normalized_rms:.2f}, Peak: {normalized_peak:.2f}, "
                           f"Detection: {detection_value:.2f}, Threshold: {self.bark_threshold}, "
                           f"Samples above threshold: {samples_above_threshold}, Required: {duration_samples}")
            
            # Detect bark based on amplitude and duration
            bark_detected = (detection_value > self.bark_threshold and 
                           samples_above_threshold > duration_samples)
            
            # Update audio buffer
            self.update_audio_buffer(audio_chunk)
            
            # Handle bark detection and notification
            current_time = time.time()
            if bark_detected and (current_time - self.last_bark_time) > self.bark_cooldown:
                logger.info(f"Bark detected! Detection value: {detection_value:.2f}, "
                          f"Samples above threshold: {samples_above_threshold}")
                # Save audio clip
                audio_file = self.save_audio_clip()
                if audio_file:
                    self._send_notification("Bark detected!", 'bark', {'audio': audio_file})
                self.last_bark_time = current_time
            
            return bark_detected
            
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
        logger.info("Audio recording thread started")
        while self.running and self.audio_enabled:
            try:
                if self.audio_device is None:
                    logger.warning("Audio device not available, attempting to reinitialize...")
                    self.init_audio(self.preferred_audio_id)
                    if self.audio_device is None:
                        time.sleep(5)  # Wait before retrying
                        continue

                # Log audio recording status
                logger.debug(f"Recording audio chunk (device={self.audio_device}, "
                           f"rate={self.sample_rate}, channels={self.channels})")

                # Record audio
                audio_chunk = sd.rec(
                    int(self.sample_rate * self.audio_chunk_duration),
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    blocking=True,
                    device=self.audio_device
                )
                
                # Log audio chunk properties
                if audio_chunk is not None:
                    logger.debug(f"Recorded audio chunk: shape={audio_chunk.shape}, "
                               f"dtype={audio_chunk.dtype}, "
                               f"min={np.min(audio_chunk):.3f}, "
                               f"max={np.max(audio_chunk):.3f}, "
                               f"rms={np.sqrt(np.mean(np.square(audio_chunk))):.3f}")
                else:
                    logger.warning("Failed to record audio chunk (None returned)")
                    continue
                
                # Update audio buffer
                self.update_audio_buffer(audio_chunk)
                
                # Check for bark
                if self.detect_bark(audio_chunk):
                    # Save audio clip and send notification
                    audio_file = self.save_audio_clip()
                    if audio_file:
                        self._send_notification("Bark detected!", 'bark', {'audio': audio_file})
                
            except Exception as e:
                logger.error(f"Audio recording error: {str(e)}")
                time.sleep(1)
        
        logger.info("Audio recording thread stopped")

    def _send_notification(self, message, notification_type='motion', data=None):
        """Send a notification to the web interface with type and optional data"""
        try:
            logger.info(f"NOTIFICATION ({notification_type}): {message}")
            notification = {
                'message': message,
                'type': notification_type
            }
            if data:
                notification['data'] = data
            socketio.emit('notification', notification)
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
        """Detect motion in frame and save video if detected"""
        try:
            # Create a copy of the frame for modification
            frame_with_motion = frame.copy()
            
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
                    cv2.rectangle(frame_with_motion, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Update video buffer
            self.update_video_buffer(frame_with_motion)
            
            # Handle motion detection and notification
            current_time = time.time()
            if motion_detected and not self.motion_detected and \
               (current_time - self.last_motion_time) > self.motion_cooldown:
                # Save video clip
                video_file = self.save_video_clip()
                if video_file:
                    self._send_notification("Motion detected!", 'motion', {'video': video_file})
                self.last_motion_time = current_time
            
            self.motion_detected = motion_detected
            
            # Return the frame with motion rectangles drawn
            return frame_with_motion
            
        except Exception as e:
            logger.error(f"Motion detection error: {str(e)}")
            return frame

    def record_video(self):
        """Record video from the camera"""
        while self.running:
            try:
                if self.camera is None or not self.camera.isOpened():
                    logger.warning("Camera not available, attempting to reinitialize...")
                    self.init_camera(self.preferred_camera_id)
                    if self.camera is None or not self.camera.isOpened():
                        time.sleep(5)  # Wait before retrying
                        continue

                ret, frame = self.camera.read()
                if ret:
                    # Apply motion detection and get frame with motion indicators
                    frame_with_motion = self.detect_motion(frame)
                    
                    # Update the current frame for web streaming
                    with self.frame_lock:
                        self.current_frame = frame_with_motion
                    
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Video recording error: {str(e)}")
                time.sleep(1)

    def save_video_clip(self, frames=None):
        """Save a video clip of the detection"""
        if time.time() - self.last_video_save < self.min_save_interval:
            return None
        
        try:
            timestamp = int(time.time())
            filename = f'motion_{timestamp}.mp4'
            filepath = os.path.join(self.video_output_dir, filename)
            
            # Use buffered frames or provided frames
            frames_to_save = frames if frames is not None else self.video_buffer
            
            if not frames_to_save:
                return None
                
            # Get frame properties
            height, width = frames_to_save[0].shape[:2]
            
            # Create video writer with H.264 codec
            if platform.system() == 'Darwin':  # macOS
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
            else:  # Linux/Windows
                fourcc = cv2.VideoWriter_fourcc(*'X264')
            
            out = cv2.VideoWriter(filepath, fourcc, 20.0, (width, height))
            
            # Write frames
            for frame in frames_to_save:
                out.write(frame)
            
            out.release()
            self.last_video_save = time.time()
            
            logger.info(f"Saved video clip: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving video clip: {str(e)}")
            return None

    def save_audio_clip(self, audio_data=None):
        """Save an audio clip of the detection"""
        if time.time() - self.last_audio_save < self.min_save_interval:
            return None
            
        try:
            timestamp = int(time.time())
            filename = f'bark_{timestamp}.wav'
            filepath = os.path.join(self.audio_output_dir, filename)
            
            # Use buffered audio or provided audio
            audio_to_save = audio_data if audio_data is not None else np.concatenate(self.audio_buffer)
            
            if audio_to_save.size == 0:
                return None
            
            # Convert float32 to int16
            audio_int16 = (audio_to_save * 32767).astype(np.int16)
            
            # Save as WAV file
            wavfile.write(filepath, self.sample_rate, audio_int16)
            self.last_audio_save = time.time()
            
            logger.info(f"Saved audio clip: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving audio clip: {str(e)}")
            return None

    def update_video_buffer(self, frame):
        """Update the video buffer with a new frame"""
        self.video_buffer.append(frame.copy())
        
        # Calculate max buffer size based on FPS
        fps = self.camera.get(cv2.CAP_PROP_FPS)
        max_frames = int(fps * self.video_buffer_seconds)
        
        # Trim buffer if needed
        if len(self.video_buffer) > max_frames:
            self.video_buffer = self.video_buffer[-max_frames:]

    def update_audio_buffer(self, audio_chunk):
        """Update the audio buffer with new audio data"""
        self.audio_buffer.append(audio_chunk)
        
        # Calculate max buffer size based on sample rate
        max_samples = int(self.sample_rate * self.audio_buffer_seconds)
        total_samples = sum(chunk.shape[0] for chunk in self.audio_buffer)
        
        # Trim buffer if needed
        while total_samples > max_samples and self.audio_buffer:
            removed_chunk = self.audio_buffer.pop(0)
            total_samples -= removed_chunk.shape[0]

    def start_monitoring(self):
        """Start the monitoring threads"""
        self.running = True
        
        # Start video recording thread
        video_thread = threading.Thread(target=self.record_video)
        video_thread.daemon = True
        video_thread.start()
        
        # Start audio recording thread if audio is enabled
        if self.audio_enabled:
            logger.info("Starting audio recording thread")
            audio_thread = threading.Thread(target=self.record_audio)
            audio_thread.daemon = True
            audio_thread.start()
        else:
            logger.warning("Audio recording is disabled")

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

@app.route('/recordings/video/<path:filename>')
def serve_video(filename):
    """Serve video recordings"""
    try:
        return send_from_directory(monitor.video_output_dir, filename)
    except Exception as e:
        logger.error(f"Error serving video {filename}: {str(e)}")
        return "Video not found", 404

@app.route('/recordings/audio/<path:filename>')
def serve_audio(filename):
    """Serve audio recordings"""
    try:
        return send_from_directory(monitor.audio_output_dir, filename)
    except Exception as e:
        logger.error(f"Error serving audio {filename}: {str(e)}")
        return "Audio not found", 404

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
