import os
import time
import datetime
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.io import wavfile
import logging
import json
from flask import Flask, render_template, jsonify, request, send_from_directory
from datetime import timedelta
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
            if datetime.datetime.strptime(data['timestamp'], "%Y-%m-%d %H:%M:%S") > since_dt
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

class AudioMonitor:
    def __init__(self, audio_output_dir="recordings/audio"):
        self.audio_output_dir = audio_output_dir
        os.makedirs(audio_output_dir, exist_ok=True)
        
        # Initialize settings
        self.settings = Settings(os.path.join(audio_output_dir, "settings.json"))
        
        # Initialize level database
        self.level_db = LevelDB(os.path.join(audio_output_dir, "levels.db"))
        
        # Audio parameters
        self.sample_rate = 44100
        self.channels = 1
        self.chunk_duration_minutes = self.settings.settings['recording_interval']
        self.chunk_duration_seconds = self.chunk_duration_minutes * 60
        
        # Get available audio devices
        self.available_devices = self.get_available_devices()
        logger.info("Available audio devices:")
        for i, device in enumerate(self.available_devices):
            logger.info(f"  [{i}] {device['name']}")
        
        # Select audio device
        self.audio_device = self.select_audio_device()
        logger.info(f"Using audio device {self.audio_device}: {self.get_device_name(self.audio_device)}")
        
        # Initialize state
        self.running = False
        self.current_recording = None
        self.recordings = []
        self.load_existing_recordings()
        
        # Start monitoring if enabled
        if self.settings.settings['recording_enabled']:
            self.start()
    
    def get_available_devices(self):
        """Get list of available audio input devices"""
        devices = []
        try:
            device_list = sd.query_devices()
            for i, device in enumerate(device_list):
                if device['max_input_channels'] > 0:  # Only input devices
                    devices.append({
                        'id': i,
                        'name': device['name']
                    })
            logger.debug(f"Found {len(devices)} input devices")
        except Exception as e:
            logger.error(f"Error getting audio devices: {str(e)}")
        return devices

    def get_device_name(self, device_id):
        """Get the name of an audio device by its ID"""
        try:
            device = sd.query_devices(device_id)
            return device['name']
        except Exception as e:
            logger.error(f"Error getting device name for ID {device_id}: {str(e)}")
            return "Unknown Device"

    def select_audio_device(self):
        """Select the audio device to use"""
        try:
            # If we have a device set in settings and it's available, use it
            if self.settings.settings['audio_device'] is not None:
                device_id = self.settings.settings['audio_device']
                try:
                    sd.query_devices(device_id)
                    if device_id < len(self.available_devices):
                        return device_id
                except Exception:
                    logger.warning(f"Previously selected device {device_id} is no longer available")

            # If we have available devices, use the first one
            if self.available_devices:
                device_id = self.available_devices[0]['id']
                self.settings.settings['audio_device'] = device_id
                self.settings.save_settings()
                return device_id

            logger.error("No audio input devices available")
            return None
        except Exception as e:
            logger.error(f"Error selecting audio device: {str(e)}")
            return None
    
    def update_settings(self, new_settings):
        restart_required = False
        
        if 'audio_device' in new_settings:
            old_device = self.settings.settings['audio_device']
            if new_settings['audio_device'] != old_device:
                restart_required = True
        
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
        
        # If device changed, update it
        if 'audio_device' in new_settings:
            self.audio_device = self.select_audio_device()
        
        return restart_required
    
    def calculate_audio_levels(self, audio_data, num_segments=100):
        """Calculate audio levels for multiple segments of the recording"""
        # Convert to mono if stereo
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Split into segments
        segment_size = len(audio_data) // num_segments
        if segment_size == 0:
            segment_size = 1
        segments = np.array_split(audio_data, num_segments)
        
        levels = []
        for segment in segments:
            # Calculate RMS (root mean square)
            rms = np.sqrt(np.mean(np.square(segment)))
            
            # Convert to decibels
            if rms > 0:
                db = 20 * np.log10(rms)
            else:
                db = -96  # Noise floor
            
            # Normalize to 0-1 range
            # Typical values: -60 dB (quiet) to 0 dB (loud)
            normalized = (db + 60) / 60  # Shift and scale to 0-1
            normalized = np.clip(normalized, 0, 1)  # Ensure within bounds
            
            levels.append(float(normalized))
        
        return levels

    def record_chunk(self):
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"recording_{timestamp}.wav"
            filepath = os.path.join(self.audio_output_dir, filename)
            
            # Record audio
            audio_data = sd.rec(
                int(self.chunk_duration_seconds * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                device=self.audio_device
            )
            sd.wait()
            
            # Calculate detailed audio levels
            levels = self.calculate_audio_levels(audio_data)
            
            # Calculate overall level (average of all levels)
            level = float(np.mean(levels))
            
            # Save audio file
            sf.write(filepath, audio_data, self.sample_rate)
            
            # Save detailed level data to JSON
            levels_filepath = filepath.replace('.wav', '.levels.json')
            with open(levels_filepath, 'w') as f:
                json.dump({
                    'timestamp': timestamp,
                    'levels': levels,
                    'overall_level': level,
                    'duration': self.chunk_duration_seconds,
                    'num_segments': len(levels)
                }, f)
            
            # Save overall level to database
            timestamp_str = datetime.datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S").strftime("%Y-%m-%d %H:%M:%S")
            self.level_db.add_level(filename, timestamp_str, level, levels, self.chunk_duration_seconds)
            
            # Add to recordings list
            self.recordings.append({
                'filename': filename,
                'timestamp': timestamp_str,
                'level': level,
                'has_detailed_levels': True
            })
            
            logger.info(f"Recorded chunk: {filename} with {len(levels)} level segments")
            
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
                        'timestamp': dt.strftime("%Y-%m-%d %H:%M:%S"),
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
                if self.audio_device is None:
                    logger.warning("No audio device available, retrying in 5 seconds...")
                    time.sleep(5)
                    self.audio_device = self.select_audio_device()
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
            self.current_recording = {
                'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                'status': 'recording',
                'levels': [],
                'duration': self.chunk_duration_minutes * 60,
                'error_message': None,
                'current_level': 0
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
                since_dt = datetime.datetime.strptime(since, "%Y-%m-%d %H:%M:%S")
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
        recordings = monitor.recordings
        
        # Filter recordings after the given timestamp
        if since:
            try:
                since_dt = datetime.datetime.strptime(since, "%Y-%m-%d %H:%M:%S")
                recordings = [r for r in recordings if datetime.datetime.strptime(r['timestamp'], "%Y-%m-%d %H:%M:%S") > since_dt]
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
    if monitor:
        return jsonify({
            'recording': monitor.current_recording,
            'is_running': monitor.running
        })
    return jsonify({
        'recording': None,
        'is_running': False
    })

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
        if start_dt <= datetime.datetime.strptime(r['timestamp'], "%Y-%m-%d %H:%M:%S") < end_dt
    ]
    
    # Sort recordings by timestamp
    filtered_recordings.sort(key=lambda x: x['timestamp'])
    
    # Process timeline data
    timeline_data = []
    for recording in filtered_recordings:
        dt = datetime.datetime.strptime(recording['timestamp'], "%Y-%m-%d %H:%M:%S")
        
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
            
            restart_required = monitor.update_settings(new_settings)
            return jsonify({'restart_required': restart_required})
        except Exception as e:
            logger.error(f"Error updating settings: {str(e)}")
            return jsonify({'error': str(e)}), 400
    
    # GET request
    try:
        settings = monitor.settings.settings
        available_devices = monitor.available_devices
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

# Add rate limiting for audio file requests
audio_rate_limit = {
    'requests': {},
    'max_requests': 3,  # Maximum concurrent requests per client
    'time_window': 5    # Time window in seconds
}

def main():
    global monitor
    
    try:
        monitor = AudioMonitor()
        app.run(host='0.0.0.0', port=5001)
    except KeyboardInterrupt:
        if monitor:
            monitor.cleanup()

if __name__ == "__main__":
    main()
