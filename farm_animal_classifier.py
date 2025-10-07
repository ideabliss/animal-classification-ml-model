import subprocess
import sys
import os
from datetime import datetime

def install_requirements():
    packages = ['opencv-python', 'torch>=1.12.0', 'torchvision>=0.13.0', 'numpy', 'pillow']
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

try:
    import cv2
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torchvision import models
    import numpy as np
    from PIL import Image
except ImportError:
    print("Installing requirements...")
    install_requirements()
    import cv2
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torchvision import models
    import numpy as np
    from PIL import Image

class FarmAnimalClassifier:
    def __init__(self, enable_telegram=False, bot_token=None, chat_id=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Farm Harmful Animals Dataset - 15 Classes (matching your folder structure)
        self.class_names = [
            'Armadilles', 'Bear', 'Birds', 'Cow', 'Crocodile',
            'Deer', 'Elephant', 'Goat', 'Horse', 'Jaguar',
            'Monkey', 'Rabbit', 'Skunk', 'Tiger', 'Wild Boar'
        ]
        
        # Risk categories for farm protection (matching folder names)
        self.extreme_threats = ['Bear', 'Tiger', 'Jaguar', 'Crocodile']
        self.high_threats = ['Wild Boar', 'Elephant', 'Skunk']
        self.medium_threats = ['Deer', 'Monkey', 'Armadilles']
        self.farm_animals = ['Cow', 'Goat', 'Horse']
        self.neutral_animals = ['Birds', 'Rabbit']
        
        # Load model (trained or pre-trained)
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))
        
        # Try to load best available trained model
        if os.path.exists('farm_model_lightning_cpu.pth'):
            print("Loading LIGHTNING FAST model (94.0% accuracy - 14.8min training)...")
            # Use MobileNet-V3 architecture
            try:
                self.model = models.mobilenet_v3_small(pretrained=True)
                self.model.classifier = nn.Sequential(
                    nn.Linear(self.model.classifier[0].in_features, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, len(self.class_names))
                )
            except:
                self.model = models.resnet18(pretrained=True)
                self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))
            self.model.load_state_dict(torch.load('farm_model_lightning_cpu.pth', map_location=self.device))
        elif os.path.exists('farm_model_advanced_gpu.pth'):
            print("Loading ADVANCED GPU model (99%+ accuracy target)...")
            try:
                self.model = models.efficientnet_b3(pretrained=True)
                self.model.classifier = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(self.model.classifier[1].in_features, 512),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(512, len(self.class_names))
                )
            except:
                self.model = models.resnet152(pretrained=True)
                self.model.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(self.model.fc.in_features, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(512, len(self.class_names))
                )
            self.model.load_state_dict(torch.load('farm_model_advanced_gpu.pth', map_location=self.device))
        elif os.path.exists('farm_model_cpu.pth'):
            print("Loading CPU model (96.23% accuracy)...")
            self.model.load_state_dict(torch.load('farm_model_cpu.pth', map_location=self.device))
        else:
            print("No trained model found, using pre-trained ResNet50")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Telegram notifications
        self.enable_telegram = enable_telegram
        self.telegram_notifier = None
        if enable_telegram:
            try:
                from telegram_notifier import TelegramNotifier
                self.telegram_notifier = TelegramNotifier(bot_token, chat_id)
                print("Telegram notifications enabled")
            except ImportError:
                print("Telegram notifier not available")
                self.enable_telegram = False
        
        self.last_alert_time = {}
        self.alert_cooldown = 30  # seconds
        
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print("Face detection initialized")
    
    def get_risk_level(self, animal):
        if animal in self.extreme_threats:
            return "EXTREME THREAT"
        elif animal in self.high_threats:
            return "HIGH THREAT"
        elif animal in self.medium_threats:
            return "MEDIUM THREAT"
        elif animal in self.farm_animals:
            return "LIVESTOCK"
        elif animal in self.neutral_animals:
            return "LOW RISK"
        return "UNKNOWN"
    
    def detect_human_face(self, frame):
        """Detect human faces with multiple approaches"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Multiple detection passes with different parameters
            faces1 = self.face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(20, 20), maxSize=(300, 300))
            faces2 = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30), maxSize=(200, 200))
            faces3 = self.face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(40, 40), maxSize=(150, 150))
            
            # Check if ANY faces detected
            total_faces = len(faces1) + len(faces2) + len(faces3)
            
            if total_faces > 0:
                print(f"Human face detected ({total_faces} detections) - skipping classification")
                return True
            
            return False
        except:
            return False
    
    def classify_frame(self, frame):
        # Use MediaPipe to detect human faces FIRST
        if self.detect_human_face(frame):
            return None, 0, None  # Skip all classification if human detected
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, predicted = torch.max(probabilities, 0)
            
            animal = self.class_names[predicted.item()]
            
            # Standard confidence threshold
            if confidence.item() > 0.6:
                risk = self.get_risk_level(animal)
                return animal, confidence.item(), risk
        
        return None, 0, None
    
    def run_camera(self):
        cap = cv2.VideoCapture(0)
        current_animal = None
        current_confidence = 0
        current_risk = None
        no_detection_count = 0
        
        print("Farm Animal Detection Started. Press 'q' to quit.")
        print("Dataset: Farm Harmful Animals - https://www.kaggle.com/datasets/muzammilaliveltech/farm-harmful-animals-dataset")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Classify every 10 frames
            if cap.get(cv2.CAP_PROP_POS_FRAMES) % 10 == 0:
                animal, confidence, risk = self.classify_frame(frame)
                if animal:
                    current_animal = animal
                    current_confidence = confidence
                    current_risk = risk
                    no_detection_count = 0
                else:
                    no_detection_count += 1
                    if no_detection_count >= 3:
                        current_animal = None
                        current_confidence = 0
                        current_risk = None
            
            # Display results
            if current_animal and current_confidence > 0.5:
                accuracy = int(current_confidence * 100)
                
                # Send Telegram alert (with cooldown)
                if self.enable_telegram and self.telegram_notifier:
                    self.send_telegram_alert(current_animal, current_confidence, current_risk, frame)
                
                # Color coding based on threat level
                if current_risk == "EXTREME THREAT":
                    color = (0, 0, 139)    # Dark Red
                elif current_risk == "HIGH THREAT":
                    color = (0, 0, 255)    # Red
                elif current_risk == "MEDIUM THREAT":
                    color = (0, 165, 255)  # Orange
                elif current_risk == "LIVESTOCK":
                    color = (0, 255, 0)    # Green
                elif current_risk == "LOW RISK":
                    color = (255, 255, 0)  # Yellow
                else:
                    color = (255, 255, 255) # White
                
                text1 = f"Animal: {current_animal.upper()} ({accuracy}%)"
                text2 = f"Risk: {current_risk}"
                
                cv2.putText(frame, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.rectangle(frame, (5, 5), (450, 80), color, 2)
            
            cv2.imshow('Farm Animal Risk Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def classify_image(self, image_path):
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return
        
        frame = cv2.imread(image_path)
        animal, confidence, risk = self.classify_frame(frame)
        
        if animal and confidence > 0.5:
            accuracy = int(confidence * 100)
            print(f"Animal: {animal.upper()}")
            print(f"Confidence: {accuracy}%")
            print(f"Risk Level: {risk}")
            
            # Color coding
            if risk == "EXTREME THREAT":
                color = (0, 0, 139)
            elif risk == "HIGH THREAT":
                color = (0, 0, 255)
            elif risk == "MEDIUM THREAT":
                color = (0, 165, 255)
            elif risk == "LIVESTOCK":
                color = (0, 255, 0)
            elif risk == "LOW RISK":
                color = (255, 255, 0)
            else:
                color = (255, 255, 255)
            
            text1 = f"Animal: {animal.upper()} ({accuracy}%)"
            text2 = f"Risk: {risk}"
            
            cv2.putText(frame, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, text2, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (5, 5), (500, 90), color, 3)
        else:
            print("No farm animal detected")
            cv2.putText(frame, "No farm animal detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Farm Animal Detection Result', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def send_telegram_alert(self, animal, confidence, risk_level, frame):
        """Send Telegram alert with cooldown"""
        current_time = datetime.now().timestamp()
        
        # Check cooldown (avoid spam)
        if animal in self.last_alert_time:
            if current_time - self.last_alert_time[animal] < self.alert_cooldown:
                return
        
        # Send alert for ALL detected animals
        success = self.telegram_notifier.send_alert(animal, confidence, risk_level, frame)
        if success:
            self.last_alert_time[animal] = current_time
            print(f"Telegram alert sent: {animal} - {risk_level}")

if __name__ == "__main__":
    # Telegram configuration
    BOT_TOKEN = "7952163683:AAHaY8pogTsIqCfbqpAWSYhsVZ6G_raAFeg"
    CHAT_ID = "6070190518"
    
    # Enable Telegram if configured
    enable_telegram = BOT_TOKEN != "YOUR_BOT_TOKEN" and CHAT_ID != "YOUR_CHAT_ID"
    
    classifier = FarmAnimalClassifier(
        enable_telegram=enable_telegram,
        bot_token=BOT_TOKEN,
        chat_id=CHAT_ID
    )
    
    if len(sys.argv) > 1:
        classifier.classify_image(sys.argv[1])
    else:
        classifier.run_camera()