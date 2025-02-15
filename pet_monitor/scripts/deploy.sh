#!/bin/bash

# Load environment variables
if [ -f ../.env ]; then
    source ../.env
else
    echo "Error: .env file not found. Please copy scripts/env.example to .env and configure it."
    exit 1
fi

# Deploy changes to Raspberry Pi
echo "Deploying to $RPI_USER@$RPI_HOST..."

# Sync files (excluding .git, venv, etc.)
rsync -avz --exclude '.git/' \
          --exclude 'venv/' \
          --exclude '.env' \
          --exclude '__pycache__/' \
          --exclude '*.pyc' \
          --exclude 'recordings/' \
          ../ $RPI_USER@$RPI_HOST:$RPI_APP_DIR/

# Update dependencies and restart service
ssh $RPI_USER@$RPI_HOST "cd $RPI_APP_DIR && \
    source venv/bin/activate && \
    pip install -r requirements.txt && \
    sudo systemctl restart pet-monitor"

echo "Deployment complete!"
