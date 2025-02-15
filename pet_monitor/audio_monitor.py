import argparse
import logging
import datetime
import os
import signal
import numpy as np
import json
from scipy import signal as scipy_signal
import alsaaudio
import time
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import wave
from flask import Flask, jsonify, send_file, request, render_template, send_from_directory
import threading
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audio_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Settings:
    def __init__(self, settings_file):
        self.settings_file = settings_file
        self.settings = self.load_settings()
    
    def load_settings(self):
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
            else:
                settings = self.get_default_settings()
                self.settings = settings  # Set settings before saving
                self.save_settings()  # Save default settings
                return settings
        except Exception as e:
            logger.error(f"Error loading settings: {str(e)}")
            settings = self.get_default_settings()
            self.settings = settings
            self.save_settings()
            return settings
    
    def save_settings(self):
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving settings: {str(e)}")
    
    def get_default_settings(self):
        return {
            'audio_device': None,  # Will be set to first available device
            'recording_interval': 10,  # minutes
            'last_known_devices': [],  # List of previously seen devices
            'recording_enabled': True  # Enable/disable recording
        }
    
    def update_settings(self, new_settings):
        self.settings.update(new_settings)
        self.save_settings()

class LevelDB:
    def __init__(self, db_file):
        self.db_file = db_file
        self.levels = {}
        self.load_db()

    def load_db(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r') as f:
                    self.levels = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error loading level database, starting fresh")
                self.levels = {}
        else:
            self.levels = {}

    def save_db(self):
        with open(self.db_file, 'w') as f:
            json.dump(self.levels, f)

    def add_level(self, filename, timestamp, overall_level, detailed_levels=None, duration=None):
        """Add both overall and detailed level data for a recording"""
        self.levels[filename] = {
            'timestamp': timestamp,
            'overall_level': overall_level,
            'detailed_levels': detailed_levels if detailed_levels else [],
            'duration': duration
        }
        self.save_db()

    def get_levels_since(self, since_dt):
        """Get all level data (both overall and detailed) since the given datetime"""
        return {
            filename: data 
            for filename, data in self.levels.items() 
            if datetime.datetime.strptime(data['timestamp'], "%Y-%m-%dT%H:%M:%S") > since_dt
        }

    def get_all_levels(self):
        """Get all level data including detailed levels"""
        return self.levels

    def update_detailed_levels(self, filename, detailed_levels, duration=None):
        """Update or add detailed level data for an existing recording"""
        if filename in self.levels:
            self.levels[filename]['detailed_levels'] = detailed_levels
            if duration:
                self.levels[filename]['duration'] = duration
            self.save_db()

class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, restart_callback):
        self.restart_callback = restart_callback
        self.last_reload = 0
        self.cooldown = 1  # Cooldown in seconds to prevent multiple reloads

    def on_modified(self, event):
        if event.is_directory:
            return
        if not event.src_path.endswith(('.py', '.html')):
            return
        current_time = time.time()
        if current_time - self.last_reload > self.cooldown:
            self.last_reload = current_time
            logger.info(f"Detected change in {event.src_path}, restarting...")
            self.restart_callback()

class AudioMonitor:
    def __init__(self, audio_output_dir="recordings/audio"):
        self.audio_output_dir = audio_output_dir
        os.makedirs(audio_output_dir, exist_ok=True)
        
        # Initialize settings
        self.settings = Settings(os.path.join(audio_output_dir, "settings.json"))
        
        # Initialize level database
        self.level_db = LevelDB(os.path.join(audio_output_dir, "levels.db"))
        
        # Audio parameters - using values known to work with the USB microphone
        self.sample_rate = 48000  # Known working sample rate
        self.channels = 1
        self.chunk_duration_minutes = self.settings.settings['recording_interval']
        self.chunk_duration_seconds = self.chunk_duration_minutes * 60
        self.format = alsaaudio.PCM_FORMAT_S16_LE
        self.device = self.settings.settings['audio_device'] or "hw:1,0"  # Use settings device or fallback
        self.period_size = 1024
        
        # Initialize state
        self.running = False
        self.current_recording = None
        self.current_recording_start = None
        self.recordings = []
        self.load_existing_recordings()
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # Start monitoring if enabled
        if self.settings.settings['recording_enabled']:
            self.start()

    def get_available_devices(self):
        """Get list of available audio input devices"""
        try:
            devices = alsaaudio.pcms(alsaaudio.PCM_CAPTURE)
            logger.info(f"Found {len(devices)} input devices")
            for device in devices:
                logger.info(f"Found device: {device}")
            return devices
        except Exception as e:
            logger.error(f"Error getting audio devices: {str(e)}")
            return []

    def select_audio_device(self):
        """Select the audio device to use"""
        try:
            # Always use the USB audio device (hw:1,0) as we know it works
            logger.info(f"Using USB audio device: {self.device}")
            return self.device
            
        except Exception as e:
            logger.error(f"Error selecting audio device: {str(e)}")
            return None

    def update_settings(self, new_settings):
        restart_required = False
        
        if 'audio_device' in new_settings:
            old_device = self.settings.settings['audio_device']
            if new_settings['audio_device'] != old_device:
                restart_required = True
                self.device = new_settings['audio_device']  # Update device immediately
        
        if 'recording_interval' in new_settings:
            old_interval = self.settings.settings['recording_interval']
            if new_settings['recording_interval'] != old_interval:
                restart_required = True
                self.chunk_duration_minutes = new_settings['recording_interval']
                self.chunk_duration_seconds = self.chunk_duration_minutes * 60
        
        if 'recording_enabled' in new_settings:
            old_enabled = self.settings.settings['recording_enabled']
            if new_settings['recording_enabled'] != old_enabled:
                if new_settings['recording_enabled']:
                    self.start()
                else:
                    self.stop()
        
        # Update settings
        self.settings.update_settings(new_settings)
        
        return restart_required
    
    def calculate_audio_levels(self, audio_data, num_segments=100):
        """Calculate audio levels for multiple segments of the recording"""
        # Convert to mono if stereo
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Normalize to float in range [-1, 1]
        audio_data = audio_data.astype(np.float32) / 32768.0
        
        # Split into segments
        segment_size = len(audio_data) // num_segments
        if segment_size == 0:
            segment_size = 1
        segments = np.array_split(audio_data, num_segments)
        
        levels = []
        for segment in segments:
            # Calculate RMS (root mean square)
            rms = np.sqrt(np.mean(np.square(segment)))
            
            # Convert to decibels relative to full scale (dBFS)
            if rms > 0:
                dbfs = 20 * np.log10(rms)
            else:
                dbfs = -96  # Noise floor
            
            # Normalize to 0-1 range
            # Map -60 dBFS (quiet) to 0 and 0 dBFS (max) to 1
            normalized = (dbfs + 60) / 60
            normalized = np.clip(normalized, 0, 1)
            
            levels.append(float(normalized))
        
        return levels

    def record_chunk(self):
        """Record a chunk of audio"""
        try:
            # Set up audio input with known working parameters
            inp = alsaaudio.PCM(
                alsaaudio.PCM_CAPTURE,
                alsaaudio.PCM_NORMAL,
                device=self.device,
                channels=self.channels,
                rate=self.sample_rate,
                format=self.format,
                periodsize=self.period_size
            )
            
            logger.info(f"Recording audio with: rate={self.sample_rate}, channels={self.channels}, format={self.format}")
            
            # Calculate number of frames needed
            total_frames = int(self.sample_rate * self.chunk_duration_seconds)
            frames = []
            
            # Record audio
            for _ in range(0, total_frames, self.period_size):
                if not self.running:
                    break
                length, data = inp.read()
                if length > 0:
                    frames.append(data)
            
            if not frames or not self.running:
                return
            
            # Convert to numpy array
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            
            # Save the recording
            timestamp = datetime.datetime.now()
            filename = f"recording_{timestamp.strftime('%Y-%m-%dT%H-%M-%S')}.wav"
            filepath = os.path.join(self.audio_output_dir, filename)
            
            # Save as WAV file
            with wave.open(filepath, 'wb') as w:
                w.setnchannels(self.channels)
                w.setsampwidth(2)  # 2 bytes for S16_LE format
                w.setframerate(self.sample_rate)
                w.writeframes(audio_data.tobytes())
            
            logger.info(f"Saved recording to {filepath}")
            
            # Calculate levels
            levels = self.calculate_audio_levels(audio_data)
            overall_level = float(np.mean(levels))
            
            # Store recording info
            recording_info = {
                'filename': filename,
                'timestamp': timestamp.isoformat(),
                'duration': self.chunk_duration_seconds,
                'level': overall_level,
                'detailed_levels': levels
            }
            
            # Save recording info
            self.save_recording_info(recording_info)
            
            # Add to level database
            self.level_db.add_level(
                filename,
                timestamp.isoformat(),
                overall_level,
                levels,
                self.chunk_duration_seconds
            )
            
        except Exception as e:
            logger.error(f"Error recording chunk: {str(e)}")

    def load_existing_recordings(self):
        """Load existing recordings from directory"""
        for filename in os.listdir(self.audio_output_dir):
            if filename.endswith(".wav"):
                try:
                    match = re.search(r'audio_(\d{8}_\d{6})', filename)
                    if not match:
                        continue
                    timestamp_str = match.group(1)
                    dt = datetime.datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    
                    # Try to load levels from corresponding JSON file
                    levels_filename = filename.replace(".wav", ".levels.json")
                    levels_filepath = os.path.join(self.audio_output_dir, levels_filename)
                    levels = []
                    if os.path.exists(levels_filepath):
                        try:
                            with open(levels_filepath, 'r') as f:
                                levels = json.load(f)['levels']
                        except Exception as e:
                            logger.error(f"Error loading levels from {levels_filename}: {str(e)}")
                    
                    self.recordings.append({
                        'timestamp': dt.strftime("%Y-%m-%dT%H:%M:%S"),
                        'filename': filename,
                        'duration': self.chunk_duration_minutes * 60,
                        'levels': levels,
                        'has_detailed_levels': os.path.exists(levels_filepath)
                    })
                except (IndexError, ValueError) as e:
                    logger.error(f"Error parsing timestamp from filename {filename}: {str(e)}")
                    continue

    def start_monitoring(self):
        """Start the audio monitoring loop"""
        logger.info("Starting audio monitoring")
        try:
            while self.running:
                if self.device is None:
                    logger.warning("No audio device available, retrying in 5 seconds...")
                    time.sleep(5)
                    self.device = self.select_audio_device()
                    continue
                
                try:
                    self.record_chunk()
                except Exception as e:
                    logger.error(f"Error in recording loop: {str(e)}")
                    time.sleep(1)  # Wait before retrying
        except Exception as e:
            logger.error(f"Error in monitoring loop: {str(e)}")
            self.running = False
    
    def start(self):
        """Start audio monitoring in a new thread"""
        if not self.running:
            self.running = True
            
            # Initialize current recording
            timestamp = datetime.datetime.now()
            self.current_recording_start = timestamp
            self.current_recording = {
                'timestamp': timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
                'status': 'recording',
                'filename': f'recording_{timestamp.strftime("%Y-%m-%dT%H:%M:%S")}.wav',
                'start_time': timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
                'duration': self.chunk_duration_minutes * 60
            }
            
            self.monitor_thread = threading.Thread(target=self.start_monitoring, daemon=True)
            self.monitor_thread.start()
            logger.info("Started audio monitoring")
    
    def stop(self):
        """Stop audio monitoring"""
        if self.running:
            self.running = False
            if hasattr(self, 'monitor_thread'):
                try:
                    self.monitor_thread.join(timeout=1)  # Wait for thread to finish
                except TimeoutError:
                    logger.warning("Monitor thread did not stop cleanly")
            self.current_recording = None
            logger.info("Stopped audio monitoring")
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False

    def get_recordings(self):
        recordings = []
        for recording in self.recordings:
            recording_data = {
                'filename': recording['filename'],
                'timestamp': recording['timestamp'],
                'level': recording.get('level', 0),
                'duration': recording.get('duration', self.settings.settings['recording_interval'] * 60)  # Convert minutes to seconds
            }
            recordings.append(recording_data)
        return recordings

    def signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)

    def save_recording_info(self, recording_info):
        """Save recording information to a JSON file."""
        try:
            # Convert datetime objects to strings
            info_to_save = recording_info.copy()
            if 'timestamp' in info_to_save and isinstance(info_to_save['timestamp'], datetime.datetime):
                info_to_save['timestamp'] = info_to_save['timestamp'].strftime("%Y-%m-%dT%H:%M:%S")
            if 'start_time' in info_to_save and isinstance(info_to_save['start_time'], datetime.datetime):
                info_to_save['start_time'] = info_to_save['start_time'].strftime("%Y-%m-%dT%H:%M:%S")
                
            recordings_file = os.path.join(self.audio_output_dir, 'recordings.json')
            recordings = []
            if os.path.exists(recordings_file):
                with open(recordings_file, 'r') as f:
                    recordings = json.load(f)
            recordings.append(info_to_save)
            with open(recordings_file, 'w') as f:
                json.dump(recordings, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving recording info: {e}")

# Create global monitor instance
monitor = None

# Add cache for timeline data
timeline_cache = {
    'data': None,
    'last_update': None,
    'update_interval': 60  # Update timeline every 60 seconds
}

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recordings')
def get_recordings():
    if not monitor:
        return jsonify([])
    
    # Get query parameters
    since = request.args.get('since')  # timestamp to get recordings after
    levels_only = request.args.get('levels_only', 'false').lower() == 'true'
    page = request.args.get('page', default=1, type=int)
    per_page = request.args.get('per_page', default=20, type=int)
    
    if levels_only:
        # If we only need levels, get them from the level database
        if since:
            try:
                since_dt = datetime.datetime.strptime(since, "%Y-%m-%dT%H:%M:%S")
                levels = monitor.level_db.get_levels_since(since_dt)
            except ValueError:
                levels = monitor.level_db.get_all_levels()
        else:
            levels = monitor.level_db.get_all_levels()
            
        # Convert to list format
        recordings = [
            {'filename': filename, 'timestamp': data['timestamp'], 'level': data['overall_level'], 'detailed_levels': data['detailed_levels'], 'duration': data['duration']}
            for filename, data in levels.items()
        ]
    else:
        recordings = monitor.get_recordings()
        
        # Filter recordings after the given timestamp
        if since:
            try:
                since_dt = datetime.datetime.strptime(since, "%Y-%m-%dT%H:%M:%S")
                recordings = [r for r in recordings if datetime.datetime.strptime(r['timestamp'], "%Y-%m-%dT%H:%M:%S") > since_dt]
            except ValueError:
                pass
    
    # Sort recordings by timestamp (newest first)
    recordings.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # Paginate results
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_recordings = recordings[start_idx:end_idx]
    
    return jsonify(paginated_recordings)

@app.route('/recordings/<path:filename>')
def serve_recording(filename):
    if not monitor:
        return "No audio monitor running", 404
        
    # Check if requesting levels data
    if filename.endswith('.levels.json'):
        audio_filename = filename.replace('.levels.json', '.wav')
        try:
            # Load audio file and analyze
            import wave
            import numpy as np
            
            with wave.open(os.path.join('recordings/audio', audio_filename), 'rb') as wf:
                # Read audio data
                signal = wf.readframes(wf.getnframes())
                signal = np.frombuffer(signal, dtype=np.int16)
                
                # Calculate levels for chunks of audio
                chunk_size = len(signal) // 100  # 100 data points
                if chunk_size == 0:
                    chunk_size = 1
                chunks = np.array_split(signal, len(signal) // chunk_size)
                levels = [float(np.sqrt(np.mean(chunk**2))) / 32768.0 for chunk in chunks]  # Normalize to 0-1
                
                return jsonify({'levels': levels})
        except Exception as e:
            return str(e), 404
    
    # Get client IP
    client_ip = request.remote_addr
    current_time = time.time()
    
    # Clean up old requests
    audio_rate_limit['requests'] = {
        ip: reqs for ip, reqs in audio_rate_limit['requests'].items()
        if current_time - reqs['timestamp'] < audio_rate_limit['time_window']
    }
    
    # Check rate limit
    if client_ip in audio_rate_limit['requests']:
        client_reqs = audio_rate_limit['requests'][client_ip]
        if client_reqs['count'] >= audio_rate_limit['max_requests']:
            return "Too many requests. Please wait before requesting more audio files.", 429
        client_reqs['count'] += 1
    else:
        audio_rate_limit['requests'][client_ip] = {
            'count': 1,
            'timestamp': current_time
        }
    
    try:
        return send_from_directory(
            'recordings/audio',
            filename,
            as_attachment=True,
            download_name=filename
        )
    except FileNotFoundError:
        return "Recording not found", 404

@app.route('/status')
def get_status():
    if not monitor:
        return jsonify({'error': 'Monitor not initialized'}), 500

    try:
        status = {
            'recording_enabled': monitor.settings.settings['recording_enabled'],
            'recording_interval': monitor.settings.settings['recording_interval'],
            'current_recording': None
        }

        if monitor.current_recording:
            status['current_recording'] = {
                'filename': monitor.current_recording,
                'start_time': monitor.current_recording_start.strftime('%Y-%m-%dT%H:%M:%S') if hasattr(monitor, 'current_recording_start') else None
            }

        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/daily_timeline')
def get_daily_timeline():
    if not monitor:
        return jsonify([])
    
    # Get query parameters for date range
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date', default=datetime.datetime.now().strftime("%Y-%m-%d"))
    
    current_time = datetime.datetime.now()
    
    # Check if we can use cached data
    if (timeline_cache['data'] is not None and 
        timeline_cache['last_update'] is not None and
        (current_time - timeline_cache['last_update']).total_seconds() < timeline_cache['update_interval']):
        return jsonify(timeline_cache['data'])
    
    try:
        if start_date:
            start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        else:
            # Default to start of current day
            start_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        
        end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d") + datetime.timedelta(days=1)
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
    
    # Filter recordings for the specified date range
    filtered_recordings = [
        r for r in monitor.recordings 
        if start_dt <= datetime.datetime.strptime(r['timestamp'], "%Y-%m-%dT%H:%M:%S") < end_dt
    ]
    
    # Sort recordings by timestamp
    filtered_recordings.sort(key=lambda x: x['timestamp'])
    
    # Process timeline data
    timeline_data = []
    for recording in filtered_recordings:
        dt = datetime.datetime.strptime(recording['timestamp'], "%Y-%m-%dT%H:%M:%S")
        
        # Calculate average level if levels exist
        avg_level = 0
        if 'levels' in recording and recording['levels']:
            levels = [float(l) for l in recording['levels']]
            avg_level = sum(levels) / len(levels)
        
        timeline_data.append({
            'timestamp': recording['timestamp'],
            'level': avg_level,
            'recording_id': recording['filename'] if avg_level > 0 else None
        })
    
    # Cache the results
    timeline_cache['data'] = [{
        'date': end_date,
        'timeline': timeline_data
    }]
    timeline_cache['last_update'] = current_time
    
    return jsonify(timeline_cache['data'])

@app.route('/settings', methods=['GET', 'POST'])
def get_settings():
    if not monitor:
        return jsonify({'error': 'Monitor not initialized'}), 500
    
    if request.method == 'POST':
        try:
            new_settings = request.json
            
            # Convert string 'true'/'false' to boolean if needed
            if 'recording_enabled' in new_settings:
                if isinstance(new_settings['recording_enabled'], str):
                    new_settings['recording_enabled'] = new_settings['recording_enabled'].lower() == 'true'
            
            # Update settings and save them
            monitor.settings.update_settings(new_settings)
            
            # Update monitor's internal state
            if 'recording_interval' in new_settings:
                monitor.chunk_duration_minutes = new_settings['recording_interval']
                monitor.chunk_duration_seconds = monitor.chunk_duration_minutes * 60
            
            if 'audio_device' in new_settings:
                monitor.device = new_settings['audio_device']
            
            return jsonify({'status': 'success'})
        except Exception as e:
            logger.error(f"Error updating settings: {str(e)}")
            return jsonify({'error': str(e)}), 400
    
    # GET request
    try:
        settings = monitor.settings.settings
        available_devices = monitor.get_available_devices()
        return jsonify({
            'recording_enabled': settings['recording_enabled'],
            'audio_device': settings['audio_device'],
            'recording_interval': settings['recording_interval'],
            'available_devices': available_devices,
            'last_known_devices': settings['last_known_devices']
        })
    except Exception as e:
        logger.error(f"Error getting settings: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/audio_devices')
def audio_devices():
    """Get list of available audio input devices."""
    try:
        if not monitor:
            return jsonify({'error': 'Monitor not initialized'}), 500
        devices = monitor.get_available_devices()
        # Return a list of device objects with id and name
        return jsonify([{'id': dev, 'name': dev, 'isDefault': dev == monitor.device} for dev in devices])
    except Exception as e:
        logger.error(f"Error getting audio devices: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Add rate limiting for audio file requests
audio_rate_limit = {
    'requests': {},
    'max_requests': 3,  # Maximum concurrent requests per client
    'time_window': 5    # Time window in seconds
}

def main():
    parser = argparse.ArgumentParser(description='Pet Monitor Audio Service')
    parser.add_argument('--dev', action='store_true', help='Run in development mode')
    args = parser.parse_args()

    is_dev = args.dev or os.environ.get('ENV') == 'development'
    is_production = os.environ.get('ENV') == 'production'

    if is_dev:
        logger.info("Running in development mode")
        # Set up file watching for auto-reload
        observer = Observer()
        restart_event = threading.Event()

        def restart_app():
            global monitor
            if monitor:
                monitor.stop()
            restart_event.set()

        event_handler = FileChangeHandler(restart_app)
        observer.schedule(event_handler, ".", recursive=True)
        observer.start()
    else:
        logger.info("Running in production mode")

    try:
        global monitor
        monitor = AudioMonitor()
        app.run(host='0.0.0.0', port=5000, use_reloader=False)
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        if monitor:
            monitor.stop()
        if is_dev:
            observer.stop()
            observer.join()
        sys.exit(1)

if __name__ == "__main__":
    main()
