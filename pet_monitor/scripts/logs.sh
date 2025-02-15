#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load environment variables
if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
else
    echo "Error: .env file not found in $PROJECT_ROOT/. Please copy env.example to .env and configure it."
    exit 1
fi

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -f, --follow    Follow log output"
    echo "  -n N, --lines N Show last N lines"
    echo "  -b, --boot      Show logs since last boot"
    echo "  -h, --help      Show this help message"
}

# Parse command line arguments
FOLLOW=0
LINES=50
SINCE_BOOT=0

while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--follow)
            FOLLOW=1
            shift
            ;;
        -n|--lines)
            LINES="$2"
            shift 2
            ;;
        -b|--boot)
            SINCE_BOOT=1
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Build the journalctl command
CMD="$SCRIPT_DIR/ssh $RPI_USER@$RPI_HOST sudo journalctl -u pet-monitor"

if [ $SINCE_BOOT -eq 1 ]; then
    CMD="$CMD -b"
elif [ $FOLLOW -eq 1 ]; then
    CMD="$CMD -f"
else
    CMD="$CMD -n $LINES"
fi

# Execute the command
eval $CMD
