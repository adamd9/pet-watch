#!/bin/bash

# Find and kill the Python process running audio_monitor.py
PID=$(pgrep -f "python.*audio_monitor.py")
if [ -n "$PID" ]; then
    echo "Stopping Pet Watch Audio Monitor (PID: $PID)..."
    kill $PID
    echo "Stopped successfully."
else
    echo "No running Pet Watch Audio Monitor found."
fi
