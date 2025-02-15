#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Check if Python 3.9 is available through pyenv
PYTHON_PATH="$HOME/.pyenv/versions/3.9.18/bin/python"
if [ ! -f "$PYTHON_PATH" ]; then
    echo "Error: Python 3.9.18 not found at $PYTHON_PATH"
    exit 1
fi

# Create a temporary Python script to list audio devices
TMP_SCRIPT=$(mktemp)
cat > "$TMP_SCRIPT" << 'EOF'
import pyaudio

p = pyaudio.PyAudio()

print("\nAvailable Audio Input Devices:")
print("-----------------------------")

for i in range(p.get_device_count()):
    dev_info = p.get_device_info_by_index(i)
    if dev_info['maxInputChannels'] > 0:
        print(f"Device {i}: {dev_info['name']}")
        print(f"  Input channels: {dev_info['maxInputChannels']}")
        print(f"  Default sample rate: {dev_info['defaultSampleRate']}")
        print("-----------------------------")

p.terminate()
EOF

# Run the temporary script
$PYTHON_PATH "$TMP_SCRIPT"

# Clean up
rm "$TMP_SCRIPT"
