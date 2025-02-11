# Pet Monitor

A Python-based pet monitoring application that uses a USB camera and microphone to monitor your pet. Features include video streaming, audio monitoring, bark detection, and motion detection.

## Features

- Live video streaming via web interface
- Audio monitoring and recording
- Motion detection with adjustable sensitivity
- Bark detection (requires training)
- Web-based control interface
- Notification system

## Hardware Requirements

### Raspberry Pi Zero 2W Setup
- Raspberry Pi Zero 2W
- USB Camera (compatible with Linux UVC drivers)
- USB Microphone or USB Sound Card
- Power supply (5V/2.5A recommended)
- MicroSD card (16GB+ recommended)

### PC Setup
- USB Webcam
- Microphone (built-in or external)

## Installation

### Development Setup

1. **Check for existing virtual environment**
   ```bash
   # Check if venv directory exists
   ls -la venv

   # If it exists but you want to start fresh:
   rm -rf venv

   # If it exists and you want to use it:
   source venv/bin/activate  # On Unix/macOS
   .\venv\Scripts\activate   # On Windows
   ```

2. **Create new virtual environment (if needed)**
   ```bash
   # Create new virtual environment
   python3 -m venv venv

   # Activate it
   source venv/bin/activate  # On Unix/macOS
   .\venv\Scripts\activate   # On Windows
   ```

3. **Install/Update dependencies**
   ```bash
   # Upgrade pip
   pip install --upgrade pip

   # Install dependencies
   pip install -r requirements.txt
   ```

4. **Run in development mode**
   ```bash
   # Start server with auto-reload enabled
   python pet_monitor.py

   # The server will:
   # - Run in debug mode
   # - Auto-reload when code changes
   # - Show detailed error messages
   ```

5. **Access the application**
   - Open web browser to `http://localhost:5000`
   - Changes to Python files will trigger automatic server restart
   - Changes to templates will reload on next browser refresh

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
       libatlas-base-dev \
       libjasper-dev \
       libqtgui4 \
       libqt4-test \
       libhdf5-dev \
       libhdf5-serial-dev \
       libatlas-base-dev \
       libjpeg-dev \
       libilmbase-dev \
       libopenexr-dev \
       libgstreamer1.0-dev \
       git
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

5. **Test the camera**
   ```bash
   # List available video devices
   ls /dev/video*
   
   # If no devices are listed, check USB camera connection
   # and ensure it's compatible with Linux UVC drivers
   ```

6. **Test the microphone**
   ```bash
   # List audio devices
   arecord -l
   
   # If no devices are listed, check USB microphone connection
   # and ensure it's recognized by the system
   ```

7. **Run the application**
   ```bash
   python pet_monitor.py
   ```

   ```bash
   source venv/bin/activate && python pet_monitor.py > pet_monitor.log 2>&1 & tail -f pet_monitor.log
   ```

8. **Access the web interface**
   - Local access: `http://localhost:5000`
   - Remote access: `http://<raspberry-pi-ip>:5000`

## Running the Application

### Starting the Application
```bash
# Activate virtual environment
source venv/bin/activate

# Run the application
python pet_monitor.py > pet_monitor.log 2>&1 & tail -f pet_monitor.log
```

### Stopping the Application

```bash
# One-line command to kill all instances (safe kill, then force kill if needed)
(kill $(pgrep -f "python.*pet_monitor.py") 2>/dev/null || true) && sleep 1 && (kill -9 $(pgrep -f "python.*pet_monitor.py") 2>/dev/null || true)
```

The above command will:
1. Try to gracefully kill all pet_monitor.py processes
2. Wait 1 second for them to clean up
3. Force kill any remaining processes
4. Suppress any error messages if no processes are found
5. Always return success (won't break scripts)

For debugging, you can also:
```bash
# List all running instances
ps aux | grep "[p]ython.*pet_monitor.py"

# Check what's using port 5000
lsof -i :5000

# On macOS, you might need to disable AirPlay Receiver:
# System Settings -> AirDrop & Handoff -> AirPlay Receiver -> Off
```

### Configuration Tips

1. **Automatic startup**
   Create a systemd service to run at boot:
   ```bash
   sudo nano /etc/systemd/system/pet-monitor.service
   ```
   
   Add the following content:
   ```ini
   [Unit]
   Description=Pet Monitor Service
   After=network.target

   [Service]
   ExecStart=/home/pi/pet_monitor/venv/bin/python /home/pi/pet_monitor/pet_monitor.py
   WorkingDirectory=/home/pi/pet_monitor
   StandardOutput=inherit
   StandardError=inherit
   Restart=always
   User=pi

   [Install]
   WantedBy=multi-user.target
   ```

   Enable and start the service:
   ```bash
   sudo systemctl enable pet-monitor
   sudo systemctl start pet-monitor
   ```

2. **Motion Detection Settings**
   - Adjust motion sensitivity through the web interface
   - Default settings:
     - Minimum Motion Area: 500
     - Motion Threshold: 20
     - Notification Cooldown: 5 seconds

3. **Bark Detection**
   - The application uses an adaptive audio analysis approach to detect dog barks:

1. **Audio Processing**
   - Records continuous audio in chunks (1-second duration)
   - Analyzes both RMS (average) and peak amplitudes
   - Applies scaling to handle varying microphone input levels
   - Maintains a rolling buffer of recent audio for context

2. **Detection Parameters**
   - **Sensitivity** (0-100): Controls detection aggressiveness
     - Higher values make detection more sensitive to quiet sounds
     - Also affects the required duration of the sound
   - **Duration** (ms): Minimum time the sound must exceed threshold
     - Automatically adjusted based on sensitivity
     - Higher sensitivity reduces required duration
   - **Threshold** (0-100): Minimum amplitude to trigger detection
     - Applied to scaled audio values
     - Lower values detect quieter sounds
   - **Cooldown** (seconds): Minimum time between detections
     - Prevents rapid duplicate notifications

3. **Detection Algorithm**
   - Scales raw audio input to handle low microphone levels
   - Calculates both RMS and peak amplitudes for robust detection
   - Requires sufficient samples above threshold within the duration window
   - Adapts duration requirement based on sensitivity setting
   - Saves audio clips when barks are detected for review

4. **Tuning Tips**
   - Start with moderate settings: sensitivity=80, duration=50, threshold=5
   - Increase sensitivity if barks aren't detected
   - Decrease threshold if sounds are too quiet
   - Adjust duration if detections are too quick/slow
   - Check system input volume if detection is poor

### Troubleshooting

1. **Camera Issues**
   - Ensure camera is compatible with Linux UVC drivers
   - Try different USB ports
   - Check permissions: `sudo usermod -a -G video $USER`

2. **Audio Issues**
   - Check audio device permissions
   - Verify ALSA configuration: `arecord -l`
   - Test recording: `arecord -d 5 test.wav`

3. **Performance Issues**
   - Reduce video resolution in `pet_monitor.py`
   - Adjust motion detection area and threshold
   - Ensure good ventilation for the Raspberry Pi

### Security Notes

1. **Network Security**
   - The web interface is accessible to all devices on your network
   - Consider setting up authentication
   - Use a firewall to restrict access

2. **Updates**
   - Regularly update system packages
   - Keep Python dependencies up to date

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

[Your chosen license]
