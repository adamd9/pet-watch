# Pet Audio Monitor

A lightweight Python-based pet audio monitoring application designed for low-powered devices like the Raspberry Pi Zero 2W. The application records audio in configurable chunks and provides a web interface for reviewing recordings with audio level visualization.

## Features

- Real-time monitoring status
  - Live recording status indicator
  - Dynamic progress tracking
  - Time remaining display
  - Clear recording state visualization
  - Automatic status updates
- Daily timeline view showing audio activity across each day
  - Fixed 24-hour view with hourly grid lines
  - Detailed audio level visualization for each recording
  - Clear indication of recording gaps
  - Hover over points to see exact time and level
  - Click points with recordings to play them
- Smart recording previews and playback
  - Quick preview with mini level graphs for each recording
  - Full waveform visualization on demand
  - Interactive audio player with seeking
  - Real-time playback progress
  - Seamless transition from preview to player
- Audio level analysis
  - Real-time audio level monitoring
  - Detailed 100-point level analysis per recording
  - Automatic level scaling for clear visualization
  - Persistent storage of level data
  - Efficient data retrieval for timeline views
- Configurable settings
  - Audio input device selection with automatic fallback
  - Adjustable recording interval (1-60 minutes)
  - Settings persist between restarts
  - Real-time settings updates without restart
  - Automatic device detection and configuration
- Continuous audio recording in configurable chunks
- Live recording status and progress display
- Web-based playback interface
- Automatic file management
- Low resource usage optimized for Raspberry Pi

## Technical Details

### Audio Processing
- Records audio in configurable time chunks
- Analyzes audio in 100 segments per recording for detailed visualization
- Stores both overall and detailed level data for each recording
- Normalizes audio levels for consistent visualization
- Uses efficient numpy-based audio processing

### Data Storage
- Audio files stored as WAV format
- Detailed level data stored in JSON format
- Efficient level database for quick timeline access
- Automatic cleanup of old recordings

### User Interface
- Timeline shows fixed 24-hour view of current day
- Gaps in recording are clearly visible
- Two-stage visualization:
  - Quick preview with mini level graphs
  - Full waveform visualization with playback controls
- Dynamic level scaling for optimal visualization
- Interactive audio player with waveform seeking

## Hardware Requirements

### Raspberry Pi Zero 2W Setup
- Raspberry Pi Zero 2W
- USB Microphone or USB Sound Card
- Power supply (5V/2.5A recommended)
- MicroSD card (16GB+ recommended)

### PC Setup (for development)
- Microphone (built-in or external)

## Installation

### Development Setup

1. **Create and activate virtual environment**
   ```bash
   # Create new virtual environment
   python3 -m venv venv

   # Activate it
   source venv/bin/activate  # On Unix/macOS
   .\venv\Scripts\activate   # On Windows
   ```

2. **Install dependencies**
   ```bash
   # Upgrade pip
   pip install --upgrade pip

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Run in development mode**
   ```bash
   # This command will:
   # 1. Safely stop any existing instances
   # 2. Start a new instance in the background
   # 3. Show the live log output
   (kill $(pgrep -f "python.*audio_monitor.py") 2>/dev/null || true) && sleep 1 && (kill -9 $(pgrep -f "python.*audio_monitor.py") 2>/dev/null || true) && source venv/bin/activate && python audio_monitor.py > audio_monitor.log 2>&1 & tail -f audio_monitor.log
   ```

4. **Access the application**
   - Open web browser to `http://localhost:5001`
   - The interface shows:
     - Daily timeline view:
       - Complete 24-hour timeline for each day
       - Hour markers on the horizontal axis
       - Audio levels shown on vertical axis
       - Hover over any point to see time and level
       - Click on points with recordings to play them
     - Settings (click gear icon):
       - Audio Input Device: Select from available devices
       - Recording Interval: Choose 1-60 minutes
     - Current recording status and progress
     - Live audio level visualization
     - List of completed recordings with playback controls
     - Audio level visualization for each recording

5. **Stop the application**
   ```bash
   (kill $(pgrep -f "python.*audio_monitor.py") 2>/dev/null || true) && sleep 1 && (kill -9 $(pgrep -f "python.*audio_monitor.py") 2>/dev/null || true)
   ```

### Raspberry Pi Zero 2W Installation

1. **Prepare the Raspberry Pi**
   ```bash
   # Update system packages
   sudo apt-get update
   sudo apt-get upgrade -y

   # Install system dependencies
   sudo apt-get install -y \
       python3-dev \
       python3-pip \
       python3-venv \
       portaudio19-dev \
       libatlas-base-dev
   ```

2. **Set up the Python environment**
   ```bash
   # Create and activate virtual environment
   python3 -m venv venv
   source venv/bin/activate

   # Upgrade pip
   pip install --upgrade pip
   ```

3. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd pet_monitor
   ```

4. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Test the microphone**
   ```bash
   # List audio devices
   python3 -c "import sounddevice as sd; print(sd.query_devices())"
   ```

6. **Run the application**
   ```bash
   # This command will:
   # 1. Safely stop any existing instances
   # 2. Start a new instance in the background
   # 3. Show the live log output
   (kill $(pgrep -f "python.*audio_monitor.py") 2>/dev/null || true) && sleep 1 && (kill -9 $(pgrep -f "python.*audio_monitor.py") 2>/dev/null || true) && source venv/bin/activate && python audio_monitor.py > audio_monitor.log 2>&1 & tail -f audio_monitor.log
   ```

### Running as a Service

1. **Create a systemd service**
   ```bash
   sudo nano /etc/systemd/system/audio-monitor.service
   ```
   
   Add the following content:
   ```ini
   [Unit]
   Description=Audio Monitor Service
   After=network.target

   [Service]
   ExecStart=/home/pi/pet_monitor/venv/bin/python /home/pi/pet_monitor/audio_monitor.py
   WorkingDirectory=/home/pi/pet_monitor
   StandardOutput=inherit
   StandardError=inherit
   Restart=always
   User=pi

   [Install]
   WantedBy=multi-user.target
   ```

2. **Enable and start the service**
   ```bash
   sudo systemctl enable audio-monitor
   sudo systemctl start audio-monitor
   ```

## Development vs Production Mode

### Development Mode (macOS)
The application can be run in development mode, which provides additional features for developers:

1. Auto-reload on file changes:
   - The app automatically restarts when `.py` or `.html` files are modified
   - Changes are detected in real-time

2. To run in development mode:
```bash
# Option 1: Using command line flag
python audio_monitor.py --dev

# Option 2: Using environment variable
ENV=development python audio_monitor.py
```

### Production Mode (Raspberry Pi)
For production deployment on Raspberry Pi, the application runs as a systemd service:

1. Install the service:
```bash
# Copy files to Raspberry Pi
scp -r pet_monitor/* pi@your-raspberry-pi:/home/pi/pet-monitor/

# Set up Python virtual environment
ssh pi@your-raspberry-pi
cd /home/pi/pet-monitor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install systemd service
sudo cp pet-monitor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable pet-monitor
sudo systemctl start pet-monitor
```

2. Service features:
   - Auto-starts on boot
   - Restarts automatically on failure
   - Limited to 3 restart attempts within 60 seconds to prevent infinite restart loops
   - Logs accessible via journalctl:
     ```bash
     # View service logs
     sudo journalctl -u pet-monitor -f
     ```

3. Service management:
```bash
# Check service status
sudo systemctl status pet-monitor

# Stop service
sudo systemctl stop pet-monitor

# Start service
sudo systemctl start pet-monitor

# Restart service
sudo systemctl restart pet-monitor
```

## API Endpoints

### GET /recordings
Returns a paginated list of audio recordings.

Query Parameters:
- `since` (optional): ISO timestamp (YYYY-MM-DD HH:MM:SS) to get only recordings after this time
- `page` (optional, default: 1): Page number for pagination
- `per_page` (optional, default: 20): Number of recordings per page

Response:
```json
[
  {
    "id": "audio_20250212_074134",
    "timestamp": "2025-02-12 07:41:34",
    "duration": 60,
    "level": 45
  },
  ...
]
```

### GET /status
Returns the current recording status.

Response:
```json
{
  "recording": {
    "timestamp": "2025-02-12 07:41:34",
    "duration": 60
  },
  "is_running": true
}
```

### GET /daily_timeline
Returns audio level timeline data for a specific date range.

Query Parameters:
- `start_date` (optional): Start date in YYYY-MM-DD format. Defaults to same as end_date
- `end_date` (optional): End date in YYYY-MM-DD format. Defaults to current date

Response:
```json
[
  {
    "date": "2025-02-12",
    "timeline": [
      {
        "timestamp": "2025-02-12 07:41:34",
        "level": 45,
        "recording_id": "audio_20250212_074134"
      },
      ...
    ]
  }
]
```

## Usage

1. Start the application using one of the methods above
2. Open your web browser and navigate to:
   - Local access: `http://localhost:5001`
   - Remote access: `http://<raspberry-pi-ip>:5001`
3. The interface shows:
   - Daily timeline view:
     - Complete 24-hour timeline for each day
     - Hour markers on the horizontal axis
     - Audio levels shown on vertical axis
     - Hover over any point to see time and level
     - Click on points with recordings to play them
   - Settings (click gear icon):
     - Audio Input Device: Select from available devices
     - Recording Interval: Choose 1-60 minutes
   - Current recording status and progress
   - Live audio level visualization
   - List of completed recordings with playback controls
   - Audio level visualization for each recording

### Settings

The application includes persistent settings that can be configured through the web interface:

1. **Audio Input Device**
   - Select from a list of available audio input devices
   - Previously used devices that become unavailable are shown in gray
   - Automatically falls back to an available device if selected device is disconnected
   - Changes take effect after automatic restart

2. **Recording Interval**
   - Choose how long each recording chunk should be
   - Options: 1, 5, 10, 15, 30, or 60 minutes
   - Changes take effect after automatic restart

Settings are stored in `recordings/audio/settings.json` and persist between application restarts.

### Daily Timeline View

The timeline provides a comprehensive view of audio activity:

1. **Time Display**
   - Shows full 24-hour period for each day
   - Hour markers on horizontal axis
   - Current day always shown at top
   - Previous days shown below in reverse chronological order

2. **Audio Levels**
   - Vertical axis shows audio intensity
   - Gaps or zero levels indicate periods of silence
   - Higher levels indicate more audio activity

3. **Interaction**
   - Hover over any point to see exact time and audio level
   - Points with recordings show "Click to play" in tooltip
   - Click a point to play its associated recording
   - Timeline updates automatically every minute

### Audio Level Visualization

The application uses a decibel-based scale for audio levels:
- Very quiet sounds (-60 dB) appear as short bars
- Normal sounds (-30 dB) appear as medium bars
- Loud sounds (0 dB) appear as tall bars
- Levels are normalized to ensure consistent visualization
- Each recording stores one level measurement per second

## Audio File Format
Files must follow naming convention:
`recording_YYYY-MM-DD_HH-MM-SS.wav`

Example: `recording_2025-02-12_09-51-54.wav`

## Performance Optimizations

The application includes several optimizations to ensure efficient operation:

1. **Status Polling**
   - UI polls server status every 5 seconds instead of continuously
   - Reduces server load and network traffic

2. **Recording List Updates**
   - Incremental loading of new recordings using the `since` parameter
   - Pagination support with configurable page size
   - Only fetches new recordings when available

3. **Timeline View**
   - Server-side caching of timeline data (60-second cache duration)
   - Date-based filtering to load only relevant recordings
   - Efficient processing of audio levels for visualization

4. **Audio File Loading**
   - Client-side queue system for audio file requests
   - Maximum of 3 concurrent audio file downloads
   - Rate limiting to prevent server overload (3 requests per 5 seconds per client)
   - Automatic retry mechanism for rate-limited requests

5. **Resource Management**
   - Automatic cleanup of old requests and cache entries
   - Memory-efficient processing of large recording sets
   - Progressive loading of historical data

## Rate Limiting

To prevent server overload and ensure fair resource usage, the following rate limits are in place:

1. **Audio File Downloads**
   - Maximum 3 concurrent downloads per client
   - Maximum 3 requests per 5-second window per client IP
   - Automatic retry after 5 seconds if rate limit is exceeded

2. **Timeline Data**
   - Server-side caching with 60-second duration
   - Reduces processing load for frequently accessed data
   - Automatic cache refresh for new recordings

## Error Handling

The application includes robust error handling:

1. **Rate Limit Exceeded**
   - Returns HTTP 429 (Too Many Requests)
   - Client automatically retries after waiting period

2. **Invalid Requests**
   - Returns appropriate HTTP error codes
   - Provides descriptive error messages

3. **Missing Files**
   - Returns HTTP 404 for non-existent recordings
   - Graceful handling of deleted or moved files

## Storage Management

The application stores audio recordings in the `recordings/audio` directory. To prevent running out of storage:

1. **Implement automatic cleanup**
   - Add a cron job to remove old recordings:
   ```bash
   # Remove recordings older than 7 days
   0 0 * * * find /home/pi/pet_monitor/recordings/audio -type f -mtime +7 -delete
   ```

2. **Monitor storage usage**
   ```bash
   # Check storage usage
   du -sh /home/pi/pet_monitor/recordings/audio
   ```

## Troubleshooting

1. **Audio Issues**
   - Check audio device permissions
   - Verify ALSA configuration
   - Test recording: `arecord -d 5 test.wav`

2. **Performance Issues**
   - Monitor CPU usage: `top`
   - Check available storage: `df -h`
   - Monitor memory usage: `free -h`

## License

[Your chosen license]
