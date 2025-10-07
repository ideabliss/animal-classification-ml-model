# Farm Animal Detection for Raspberry Pi 3B+

## Quick Setup

### 1. Copy Files to Pi
```bash
scp -r "animal classification" pi@your-pi-ip:~/farm-detection/
```

### 2. Run Setup Script
```bash
cd ~/farm-detection
chmod +x rpi_setup.sh
./rpi_setup.sh
```

### 3. Reboot Pi
```bash
sudo reboot
```

### 4. Activate Environment
```bash
cd ~/farm-detection
source farm_env/bin/activate
```

## Usage

### Run Detection
```bash
python3 rpi_farm_classifier.py
```

### Train on Pi (Optional)
```bash
python3 rpi_train.py
```

## Pi Optimizations

### Performance Settings
- **CPU Threads**: Limited to 2 for stability
- **Camera FPS**: 10fps (vs 30fps on PC)
- **Frame Size**: 640x480 (vs 1080p)
- **Processing**: Every 30 frames (3 seconds)
- **Model**: MobileNet-V3 Small (ultra-lightweight)

### Memory Usage
- **Model Size**: ~10MB (vs 47MB EfficientNet)
- **RAM Usage**: ~200MB total
- **GPU Memory**: 128MB allocated

### Features
- ✅ **Real-time detection** (optimized for Pi)
- ✅ **Telegram alerts** (compressed images)
- ✅ **Human face filtering**
- ✅ **94% accuracy** (from trained model)
- ✅ **Low power consumption**
- ✅ **Headless operation** (no display needed)

## Headless Mode
Comment out these lines in `rpi_farm_classifier.py`:
```python
# cv2.imshow('Pi Farm Detection', frame)
# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break
```

## Troubleshooting

### Camera Issues
```bash
sudo raspi-config  # Enable camera
sudo reboot
```

### Performance Issues
```bash
# Check temperature
vcgencmd measure_temp

# Add cooling if >70°C
```

### Memory Issues
```bash
# Increase swap
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile  # Set CONF_SWAPSIZE=1024
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

## Hardware Requirements
- **Raspberry Pi 3B+** (minimum)
- **Camera Module** or USB camera
- **MicroSD**: 32GB+ (for dataset)
- **Power**: 5V 3A adapter
- **Cooling**: Heatsink recommended