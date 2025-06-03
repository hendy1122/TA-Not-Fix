#!/bin/bash

# Setup script for Gas Detection System on Raspberry Pi
echo "=== Gas Detection System Setup ==="

# Update system
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
sudo apt install -y python3-pip python3-venv i2c-tools

# Enable I2C
echo "Enabling I2C interface..."
sudo raspi-config nonint do_i2c 0

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv gas_detection_env
source gas_detection_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python packages
echo "Installing Python packages..."
pip install -r requirements.txt

# Check I2C devices
echo "Checking I2C devices..."
sudo i2cdetect -y 1

echo "=== Setup Complete ==="
echo "To activate the virtual environment, run:"
echo "source gas_detection_env/bin/activate"
echo ""
echo "To run the gas detection system:"
echo "python gas_detection.py"
