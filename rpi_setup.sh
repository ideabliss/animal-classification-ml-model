#!/bin/bash
# Raspberry Pi 3B+ Setup Script for Farm Animal Detection

echo "Setting up Farm Animal Detection on Raspberry Pi 3B+"
echo "=================================================="

# Update system
echo "Updating system..."
sudo apt update && sudo apt upgrade -y

# Install Python dependencies
echo "Installing Python packages..."
sudo apt install -y python3-pip python3-venv

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv farm_env
source farm_env/bin/activate

# Install optimized packages for Pi
echo "Installing optimized packages..."
pip install --upgrade pip

# Install PyTorch CPU-only version for Pi
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip install opencv-python-headless
pip install numpy pillow requests

# Enable camera
echo "Enabling camera..."
sudo raspi-config nonint do_camera 0

# Increase GPU memory split for better performance
echo "Optimizing GPU memory..."
echo "gpu_mem=128" | sudo tee -a /boot/config.txt

# Set CPU governor to performance
echo "Setting CPU to performance mode..."
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils

echo ""
echo "Setup complete! Reboot your Pi and run:"
echo "source farm_env/bin/activate"
echo "python3 rpi_farm_classifier.py"