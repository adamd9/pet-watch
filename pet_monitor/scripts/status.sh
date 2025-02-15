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
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -a, --all       Show all status information"
    echo "  -s, --service   Show only service status"
    echo "  -r, --resources Show resource usage"
    echo "  -t, --temp      Show CPU temperature"
    echo "  -h, --help      Show this help message"
}

# Parse command line arguments
SHOW_ALL=0
SHOW_SERVICE=0
SHOW_RESOURCES=0
SHOW_TEMP=0

if [ $# -eq 0 ]; then
    SHOW_SERVICE=1
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--all)
            SHOW_ALL=1
            shift
            ;;
        -s|--service)
            SHOW_SERVICE=1
            shift
            ;;
        -r|--resources)
            SHOW_RESOURCES=1
            shift
            ;;
        -t|--temp)
            SHOW_TEMP=1
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

# Show service status
if [ $SHOW_ALL -eq 1 ] || [ $SHOW_SERVICE -eq 1 ]; then
    echo "=== Service Status ==="
    ssh $RPI_USER@$RPI_HOST "sudo systemctl status pet-monitor"
    echo
fi

# Show resource usage
if [ $SHOW_ALL -eq 1 ] || [ $SHOW_RESOURCES -eq 1 ]; then
    echo "=== Resource Usage ==="
    ssh $RPI_USER@$RPI_HOST "ps aux | grep '[a]udio_monitor.py'"
    echo
fi

# Show CPU temperature
if [ $SHOW_ALL -eq 1 ] || [ $SHOW_TEMP -eq 1 ]; then
    echo "=== CPU Temperature ==="
    ssh $RPI_USER@$RPI_HOST "vcgencmd measure_temp"
    echo
fi
