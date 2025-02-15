import argparse
import logging
import datetime
import os
import signal
import numpy as np
import json
from scipy import signal as scipy_signal
import platform
import time
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import wave
from flask import Flask, jsonify, send_file, request, render_template, send_from_directory
import threading
import re
import queue
import lameenc

# Platform-specific imports
if platform.system() == 'Darwin':
    import pyaudio
else:
    import alsaaudio

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
        self.default_settings = {
            'recording_enabled': True,
            'recording_interval': 10,  # minutes
            'audio_device': None,
            'last_known_devices': [],  # List of previously detected devices
            'retention_hours': 48  # Default to 48 hours retention
        }
        self.settings = self.load_settings()
    
    def load_settings(self):
        """Load settings from file or create with defaults if not exists"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                # Ensure all default settings exist
                for key, value in self.default_settings.items():
                    if key not in settings:
                        settings[key] = value
                return settings
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
        
        # If loading fails or file doesn't exist, use defaults
        return self.default_settings.copy()
    
    def save_settings(self):
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving settings: {str(e)}")
    
    def get_retention_options(self):
        """Get available retention period options"""
        return [
            {'value': 24, 'label': '24 hours'},
            {'value': 48, 'label': '48 hours'},
            {'value': 72, 'label': '72 hours'}
        ]
    
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

    def remove_level(self, filename):
        """Remove a recording from the level database"""
        if filename in self.levels:
            del self.levels[filename]
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

class AudioInterface:
    def __init__(self, sample_rate, channels, period_size):
        self.sample_rate = sample_rate
        self.channels = channels
        self.period_size = period_size

    def open_stream(self, device=None):
        raise NotImplementedError()

    def read(self, stream):
        raise NotImplementedError()

    def close_stream(self, stream):
        raise NotImplementedError()

    def list_devices(self):
        raise NotImplementedError()

class LinuxAudioInterface(AudioInterface):
    def open_stream(self, device=None):
        # Handle both string and integer device specifications
        if device is None:
            device = "hw:1,0"
        elif isinstance(device, int):
            device = f"hw:{device},0"
        return alsaaudio.PCM(
            alsaaudio.PCM_CAPTURE,
            alsaaudio.PCM_NORMAL,
            device=device,
            channels=self.channels,
            rate=self.sample_rate,
            format=alsaaudio.PCM_FORMAT_S16_LE,
            periodsize=self.period_size
        )

    def read(self, stream):
        length, data = stream.read()
        return data if length > 0 else None

    def close_stream(self, stream):
        stream.close()

    def list_devices(self):
        # Convert card numbers to hw:X,0 format for consistency
        cards = alsaaudio.cards()
        return [f"hw:{i},0" for i in range(len(cards))]

class MacAudioInterface(AudioInterface):
    def open_stream(self, device=None):
        p = pyaudio.PyAudio()
        device_index = None
        if device is not None:
            # Handle both string and integer device specifications
            if isinstance(device, int):
                device_index = device
            else:
                # Find the device index by name
                for i in range(p.get_device_count()):
                    dev_info = p.get_device_info_by_index(i)
                    if str(device) in str(dev_info['name']):
                        device_index = i
                        break

        return p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.period_size
        )

    def read(self, stream):
        try:
            data = stream.read(self.period_size, exception_on_overflow=False)
            return data
        except Exception as e:
            logger.error(f"Error reading from audio stream: {e}")
            return None

    def close_stream(self, stream):
        stream.stop_stream()
        stream.close()
        p = stream._parent
        p.terminate()

    def list_devices(self):
        p = pyaudio.PyAudio()
        devices = []
        for i in range(p.get_device_count()):
            dev_info = p.get_device_info_by_index(i)
            if dev_info['maxInputChannels'] > 0:  # Only include input devices
                devices.append(dev_info['name'])
        p.terminate()
        return devices

class AudioMonitor:
    def __init__(self, audio_output_dir="recordings/audio"):
        self.audio_output_dir = audio_output_dir
        os.makedirs(audio_output_dir, exist_ok=True)
        
        # Initialize settings
        self.settings = Settings(os.path.join(audio_output_dir, "settings.json"))
        
        # Initialize level database
        self.level_db = LevelDB(os.path.join(audio_output_dir, "levels.db"))
        
        # Audio parameters
        self.sample_rate = 48000
        self.channels = 1
        self.chunk_duration_minutes = self.settings.settings['recording_interval']
        self.chunk_duration_seconds = self.chunk_duration_minutes * 60
        self.period_size = 1024
        
        # Initialize platform-specific audio interface
        if platform.system() == 'Darwin':
            self.audio_interface = MacAudioInterface(
                sample_rate=self.sample_rate,
                channels=self.channels,
                period_size=self.period_size
            )
        else:  # Linux/RPi
            self.audio_interface = LinuxAudioInterface(
                sample_rate=self.sample_rate,
                channels=self.channels,
                period_size=self.period_size
            )
        
        # Initialize device
        self.device = self.select_audio_device()
        
        # Initialize MP3 encoder with good quality settings
        self.encoder = lameenc.Encoder()
        self.encoder.set_bit_rate(128)  # 128kbps is good quality for voice
        self.encoder.set_in_sample_rate(self.sample_rate)
        self.encoder.set_channels(self.channels)
        self.encoder.set_quality(2)    # Quality range: 0 (best) to 9 (worst)
        
        # Initialize state
        self.running = False
        self.current_recording = None
        self.current_recording_start = None
        self.last_audio_data_time = None  # Track when we last got valid audio data
        self.recordings = []
        self.load_existing_recordings()
        
        # Initialize processing queue and worker thread
        self.processing_queue = queue.Queue()
        self.worker_thread = None
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # Start monitoring if enabled
        if self.settings.settings['recording_enabled']:
            self.start()

    def check_disk_space(self, required_mb=100):
        """Check if there's enough disk space available"""
        try:
            st = os.statvfs(self.audio_output_dir)
            free_mb = (st.f_bavail * st.f_frsize) / (1024 * 1024)
            return free_mb >= required_mb
        except Exception as e:
            logger.error(f"Error checking disk space: {str(e)}")
            return False

    def cleanup_old_recordings(self):
        """Remove recordings older than the retention period"""
        try:
            retention_hours = self.settings.settings['retention_hours']
            cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=retention_hours)
            
            # List all MP3 files
            mp3_files = glob.glob(os.path.join(self.audio_output_dir, "recording_*.mp3"))
            deleted_count = 0
            freed_space = 0
            
            for mp3_file in mp3_files:
                try:
                    # Extract timestamp from filename
                    filename = os.path.basename(mp3_file)
                    timestamp_str = filename.replace("recording_", "").replace(".mp3", "")
                    timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%dT%H-%M-%S")
                    
                    if timestamp < cutoff_time:
                        # Get file size before deleting
                        try:
                            size = os.path.getsize(mp3_file)
                        except:
                            size = 0
                            
                        try:
                            os.remove(mp3_file)
                            deleted_count += 1
                            freed_space += size
                            logger.info(f"Deleted old recording: {filename}")
                            
                            # Also remove from level database
                            self.level_db.remove_level(filename)
                        except Exception as e:
                            logger.error(f"Error deleting {filename}: {str(e)}")
                            
                except Exception as e:
                    logger.error(f"Error processing {mp3_file}: {str(e)}")
                    continue
            
            if deleted_count > 0:
                logger.info(f"Cleanup complete: deleted {deleted_count} files, freed {freed_space / (1024*1024):.1f}MB")
            
            return deleted_count, freed_space
            
        except Exception as e:
            logger.error(f"Error in cleanup: {str(e)}")
            return 0, 0

    def process_recording_worker(self):
        """Worker thread to process recordings asynchronously"""
        while self.running or not self.processing_queue.empty():
            try:
                # Get the next recording to process with a timeout
                try:
                    recording_data = self.processing_queue.get(timeout=1)
                except queue.Empty:
                    continue

                audio_data, timestamp = recording_data
                
                try:
                    # Check disk space before writing
                    if not self.check_disk_space(required_mb=100):
                        # Try to free up space
                        deleted_count, freed_space = self.cleanup_old_recordings()
                        if not self.check_disk_space(required_mb=100):
                            logger.error("Not enough disk space to write recording, even after cleanup")
                            self.processing_queue.task_done()
                            continue
                    
                    # Generate filename
                    filename = f"recording_{timestamp.strftime('%Y-%m-%dT%H-%M-%S')}.mp3"
                    filepath = os.path.join(self.audio_output_dir, filename)
                    
                    # Convert to MP3 and save
                    try:
                        # Encode to MP3
                        mp3_data = self.encoder.encode(audio_data)
                        mp3_data += self.encoder.flush()  # Get the last few frames
                        
                        # Write MP3 file
                        with open(filepath, 'wb') as f:
                            f.write(mp3_data)
                        
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
                        
                        logger.info(f"Finished processing recording {filename}")
                    except Exception as e:
                        logger.error(f"Error encoding MP3: {str(e)}")
                        raise
                        
                except Exception as e:
                    logger.error(f"Error processing recording: {str(e)}")
                
                self.processing_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in worker thread: {str(e)}")
                time.sleep(1)

    def start_monitoring(self):
        """Start the audio monitoring loop"""
        try:
            while self.running:
                if not self.settings.settings['recording_enabled']:
                    logger.info("Recording disabled in settings")
                    time.sleep(1)
                    continue

                # Check if audio device is still valid
                if self.last_audio_data_time and (datetime.datetime.now() - self.last_audio_data_time).total_seconds() > 30:
                    logger.warning("No audio data received for 30 seconds, attempting to reselect device...")
                    self.device = self.select_audio_device()
                    self.last_audio_data_time = datetime.datetime.now()  # Reset timer

                try:
                    # Try to record a chunk
                    self.record_chunk()
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    # If there's an audio device error, try to reselect the device
                    if "No such file or directory" in str(e) or "Invalid audio device" in str(e):
                        logger.warning("Audio device error, attempting to reselect device...")
                        self.device = self.select_audio_device()
                    time.sleep(5)  # Wait before retrying
                
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            self.running = False
    
    def record_chunk(self):
        """Record a chunk of audio"""
        try:
            stream = self.audio_interface.open_stream(self.device)
            
            # Calculate number of chunks needed for the desired duration
            chunks_needed = int((self.chunk_duration_seconds * self.sample_rate) / self.period_size)
            audio_data = []
            silence_threshold = 1  # Minimal amplitude to consider as non-silence
            had_audio = False

            # Set recording state
            timestamp = datetime.datetime.now()
            filename = f"recording_{timestamp.strftime('%Y-%m-%dT%H-%M-%S')}.mp3"
            self.current_recording = {
                'filename': filename,
                'timestamp': timestamp.isoformat(),
                'status': 'starting',
                'duration': self.chunk_duration_seconds
            }
            self.current_recording_start = timestamp

            for chunk_idx in range(chunks_needed):
                if not self.running:
                    break

                data = self.audio_interface.read(stream)
                if data is not None:
                    # Convert chunk to numpy array to check audio levels
                    chunk_data = np.frombuffer(data, dtype=np.int16)
                    if np.max(np.abs(chunk_data)) > silence_threshold:
                        had_audio = True
                        self.last_audio_data_time = datetime.datetime.now()
                    audio_data.append(data)
                    
                    # Update recording status
                    progress = (chunk_idx + 1) / chunks_needed * 100
                    self.current_recording['status'] = f'recording ({progress:.0f}%)'

            self.audio_interface.close_stream(stream)
            
            if not audio_data or not self.running:
                logger.warning("Recording stopped: no audio data or monitor stopped")
                self.current_recording['status'] = 'failed - no audio data'
                return
            
            if not had_audio:
                logger.warning("Recording contains only silence")
                self.current_recording['status'] = 'failed - silence only'
                return
            
            # Convert to numpy array
            audio_data = np.frombuffer(b''.join(audio_data), dtype=np.int16)
            
            # Create filepath
            filepath = os.path.join(self.audio_output_dir, filename)
            
            # Update status before processing
            self.current_recording['status'] = 'processing'
            
            # Add to processing queue
            self.processing_queue.put((audio_data, filepath, timestamp))
            
        except Exception as e:
            logger.error(f"Error recording chunk: {str(e)}")
            if self.current_recording:
                self.current_recording['status'] = f'failed - {str(e)}'
            raise

    def load_existing_recordings(self):
        """Load existing recordings from directory"""
        for filename in os.listdir(self.audio_output_dir):
            if filename.endswith(".mp3"):
                try:
                    match = re.search(r'audio_(\d{8}_\d{6})', filename)
                    if not match:
                        continue
                    timestamp_str = match.group(1)
                    dt = datetime.datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    
                    # Try to load levels from corresponding JSON file
                    levels_filename = filename.replace(".mp3", ".levels.json")
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

    def start(self):
        """Start audio monitoring in a new thread"""
        if not self.running:
            self.running = True
            
            # Start the worker thread
            self.worker_thread = threading.Thread(target=self.process_recording_worker, daemon=True)
            self.worker_thread.start()
            
            # Initialize current recording
            timestamp = datetime.datetime.now()
            self.current_recording_start = timestamp
            self.current_recording = {
                'timestamp': timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
                'status': 'recording',
                'filename': f'recording_{timestamp.strftime("%Y-%m-%dT%H-%M-%S")}.mp3',
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
            
            # Wait for processing queue to empty
            if hasattr(self, 'worker_thread'):
                try:
                    self.processing_queue.join()  # Wait for all tasks to complete
                    self.worker_thread.join(timeout=1)
                except TimeoutError:
                    logger.warning("Worker thread did not stop cleanly")
            
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

    def select_audio_device(self):
        """Select an appropriate audio device, with fallback options."""
        try:
            # Try to use the device from settings
            if self.settings.settings['audio_device']:
                logger.info(f"Using configured audio device: {self.settings.settings['audio_device']}")
                return self.settings.settings['audio_device']
            
            # Get available devices
            devices = self.audio_interface.list_devices()
            logger.info(f"Available audio devices: {devices}")
            
            if not devices:
                if platform.system() == 'Darwin':
                    logger.info("No devices found, using default macOS device")
                    return None  # PyAudio will use system default
                else:
                    logger.info("No devices found, using default ALSA device hw:0,0")
                    return "hw:0,0"  # Updated to use card 0
            
            # On Linux/RPi, prefer USB audio devices if available
            if platform.system() != 'Darwin':
                for device in devices:
                    if isinstance(device, str) and 'usb' in device.lower():
                        logger.info(f"Selected USB audio device: {device}")
                        return device
            
            # Otherwise use the first available device
            logger.info(f"Using first available device: {devices[0]}")
            return devices[0]
            
        except Exception as e:
            logger.error(f"Error selecting audio device: {e}")
            if platform.system() == 'Darwin':
                return None
            return "hw:0,0"  # Updated to use card 0

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
        audio_filename = filename.replace('.levels.json', '.mp3')
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
        current_time = datetime.datetime.now()
        status = {
            'recording_enabled': monitor.settings.settings['recording_enabled'],
            'recording_interval': monitor.settings.settings['recording_interval'],
            'current_recording': None,
            'monitor_running': monitor.running,
            'last_recording_check': current_time.strftime('%Y-%m-%dT%H:%M:%S'),
            'health': {
                'status': 'healthy',
                'issues': []
            }
        }

        # Add audio device health info
        if monitor.last_audio_data_time:
            audio_silence_duration = (current_time - monitor.last_audio_data_time).total_seconds()
            status['health']['last_audio_data'] = monitor.last_audio_data_time.strftime('%Y-%m-%dT%H:%M:%S')
            status['health']['audio_silence_duration'] = int(audio_silence_duration)
            
            if audio_silence_duration > 30:
                status['health']['status'] = 'warning'
                status['health']['issues'].append({
                    'type': 'audio_silence',
                    'message': f'No audio data received for {int(audio_silence_duration)} seconds',
                    'silence_duration': int(audio_silence_duration)
                })

        if monitor.current_recording:
            recording_start = monitor.current_recording_start
            expected_duration = monitor.settings.settings['recording_interval'] * 60  # in seconds
            current_duration = (current_time - recording_start).total_seconds()
            
            status['current_recording'] = {
                'filename': monitor.current_recording.get('filename'),
                'start_time': recording_start.strftime('%Y-%m-%dT%H:%M:%S'),
                'current_duration': int(current_duration),
                'expected_duration': expected_duration,
                'status': monitor.current_recording.get('status', 'unknown')
            }

            # Check for stale or failed recordings
            if 'failed' in monitor.current_recording.get('status', ''):
                status['health']['status'] = 'error'
                status['health']['issues'].append({
                    'type': 'recording_failed',
                    'message': monitor.current_recording.get('status', 'unknown failure')
                })
            elif current_duration > expected_duration + 30:  # Allow 30s grace period
                status['health']['status'] = 'warning'
                status['health']['issues'].append({
                    'type': 'stale_recording',
                    'message': f'Current recording duration ({int(current_duration)}s) exceeds expected duration ({expected_duration}s)',
                    'expected_duration': expected_duration,
                    'current_duration': int(current_duration)
                })

        elif monitor.settings.settings['recording_enabled'] and monitor.running:
            status['health']['status'] = 'warning'
            status['health']['issues'].append({
                'type': 'no_active_recording',
                'message': 'Recording is enabled but no active recording found'
            })

        # Check if monitor is running but recording is disabled
        if monitor.running and not monitor.settings.settings['recording_enabled']:
            status['health']['issues'].append({
                'type': 'info',
                'message': 'Monitor is running but recording is disabled'
            })

        # Check if monitor is not running but recording is enabled
        if not monitor.running and monitor.settings.settings['recording_enabled']:
            status['health']['status'] = 'error'
            status['health']['issues'].append({
                'type': 'error',
                'message': 'Recording is enabled but monitor is not running'
            })

        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify({
            'error': str(e),
            'health': {
                'status': 'error',
                'issues': [{
                    'type': 'error',
                    'message': f'Error getting status: {str(e)}'
                }]
            }
        }), 500

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
        available_devices = monitor.audio_interface.list_devices()
        return jsonify({
            'recording_enabled': settings['recording_enabled'],
            'audio_device': settings['audio_device'],
            'recording_interval': settings['recording_interval'],
            'retention_hours': settings['retention_hours'],
            'available_devices': available_devices,
            'last_known_devices': settings['last_known_devices'],
            'retention_options': monitor.settings.get_retention_options()
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
        devices = monitor.audio_interface.list_devices()
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
