#!/bin/bash

# OpenAutoNote Setup Script for macOS

echo "Starting OpenAutoNote setup..."

# Check for ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed."
    echo "Please install it using Homebrew: brew install ffmpeg"
    exit 1
else
    echo "ffmpeg found."
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Setup complete! You can run the app using: streamlit run app.py"
