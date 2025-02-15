#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default remote host
REMOTE_HOST="raspberrypizero.local"
REMOTE_USER="adam"
REMOTE_DIR="/home/adam/pet-monitor"

# Help message
show_help() {
    echo "Usage: $0 [-h <host>] [-u <user>] [-d <directory>]"
    echo
    echo "Clean up pet monitor recordings and restart the service"
    echo
    echo "Options:"
    echo "  -h <host>      Remote host (default: raspberrypizero.local)"
    echo "  -u <user>      Remote user (default: adam)"
    echo "  -d <dir>       Remote directory (default: /home/adam/pet-monitor)"
    echo "  --help         Show this help message"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h)
            REMOTE_HOST="$2"
            shift 2
            ;;
        -u)
            REMOTE_USER="$2"
            shift 2
            ;;
        -d)
            REMOTE_DIR="$2"
            shift 2
            ;;
        --help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

echo "Cleaning up pet monitor recordings on ${REMOTE_USER}@${REMOTE_HOST}..."

# Build SSH command
SSH_CMD="ssh ${REMOTE_USER}@${REMOTE_HOST}"

# Execute cleanup
$SSH_CMD "rm -f ${REMOTE_DIR}/recordings/audio/*.wav ${REMOTE_DIR}/recordings/audio/levels.db ${REMOTE_DIR}/recordings/audio/recordings.json && sudo systemctl restart pet-monitor"

if [ $? -eq 0 ]; then
    echo "Cleanup completed successfully"
    echo "Pet monitor service restarted"
else
    echo "Error during cleanup"
    exit 1
fi
