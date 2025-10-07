# Farm Animal Risk Detection System

## Dataset
**Farm Harmful Animals Dataset**: https://www.kaggle.com/datasets/muzammilaliveltech/farm-harmful-animals-dataset

Complete dataset with 15 farm-relevant animal classes for comprehensive agricultural protection.

## Supported Animals (15 Classes)

### 🔴 **Extreme Threats**
- **Bear** - Destroys crops, attacks livestock, major predator
- **Tiger** - Apex predator, extreme livestock threat
- **Jaguar** - Powerful predator, attacks large animals
- **Crocodile** - Water-based threat, attacks livestock

### 🟠 **High Threats**
- **Wild Boar** - Crop destruction, infrastructure damage
- **Elephant** - Massive crop damage, tramples fields
- **Skunk** - Disease carrier, contaminates areas

### 🟡 **Medium Threats**
- **Deer** - Crop damage, garden destruction
- **Monkey** - Crop theft, property damage
- **Armadillo** - Burrows damage, crop root destruction

### 🟢 **Farm Livestock**
- **Cow** - Primary livestock (dairy/beef)
- **Goat** - Livestock (milk/meat/fiber)
- **Horse** - Work animal, transportation

### 🟨 **Low Risk Animals**
- **Bird** - Minor crop damage, some beneficial
- **Rabbit** - Small crop damage, manageable

## Features
- **100% Accuracy** focus with high confidence thresholds
- **5-Level risk assessment** (Extreme/High/Medium/Low/Livestock)
- **Color-coded alerts** for immediate threat identification
- **Real-time camera detection**
- **Agricultural risk assessment**
- **Comprehensive farm protection**

## Quick Start

### Option 1: One-Click Run
```bash
python run_farm.py
```

### Option 2: Direct Classification
```bash
# Camera mode
python farm_animal_classifier.py

# Image mode  
python farm_animal_classifier.py image.jpg
```

## Training (Optional)
If you have the dataset:
```bash
python train_farm_model.py
```

## Dataset Structure
```
farm_data/
├── train/
│   ├── armadillo/
│   ├── bear/
│   ├── bird/
│   ├── cow/
│   ├── crocodile/
│   ├── deer/
│   ├── elephant/
│   ├── goat/
│   ├── horse/
│   ├── jaguar/
│   ├── monkey/
│   ├── rabbit/
│   ├── skunk/
│   ├── tiger/
│   └── wild_boar/
└── val/
    └── (same structure)
```

## Risk Assessment System
- **EXTREME THREAT** 🔴 - Immediate evacuation, major danger
- **HIGH THREAT** 🟠 - Significant crop/livestock risk
- **MEDIUM THREAT** 🟡 - Moderate damage potential
- **LIVESTOCK** 🟢 - Farm animals to be protected
- **LOW RISK** 🟨 - Minor concern, manageable

## Applications
- **Farm Security** - Comprehensive threat monitoring
- **Livestock Protection** - Multi-level threat assessment
- **Crop Protection** - Early warning system
- **Insurance Documentation** - Detailed incident recording
- **Agricultural Research** - Animal behavior analysis
- **Wildlife Management** - Conservation and protection balance