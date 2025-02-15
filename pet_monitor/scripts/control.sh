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
    echo "Usage: $0 COMMAND"
    echo "Commands:"
    echo "  start    Start the pet-monitor service"
    echo "  stop     Stop the pet-monitor service"
    echo "  restart  Restart the pet-monitor service"
    echo "  enable   Enable service to start on boot"
    echo "  disable  Disable service from starting on boot"
}

# Check if command is provided
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

# Execute the command
case $1 in
    start|stop|restart|enable|disable)
        ssh $RPI_USER@$RPI_HOST "sudo systemctl $1 pet-monitor"
        ;;
    *)
        echo "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac
