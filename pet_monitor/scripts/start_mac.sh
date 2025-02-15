#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Check if Python 3.9 is available through pyenv
PYTHON_PATH="$HOME/.pyenv/versions/3.9.18/bin/python"
if [ ! -f "$PYTHON_PATH" ]; then
    echo "Error: Python 3.9.18 not found at $PYTHON_PATH"
    echo "Please ensure you have Python 3.9.18 installed through pyenv"
    exit 1
fi

# Activate virtual environment if it exists, create if it doesn't
VENV_DIR="$PROJECT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    $PYTHON_PATH -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Install requirements if needed
echo "Checking/installing requirements..."
pip install -r "$PROJECT_DIR/requirements.txt"

# Start the audio monitor
echo "Starting Pet Watch Audio Monitor..."
echo "The web interface will be available at: http://localhost:5000"
echo "Press Ctrl+C to stop the monitor"
cd "$PROJECT_DIR"
python audio_monitor.py
