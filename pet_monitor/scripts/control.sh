#!/bin/bash

# Load environment variables
if [ -f ../.env ]; then
    source ../.env
else
    echo "Error: .env file not found. Please copy scripts/env.example to .env and configure it."
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
