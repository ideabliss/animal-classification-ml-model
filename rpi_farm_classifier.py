import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from PIL import Image
import os
import time
from datetime import datetime

class RPiFarmClassifier:
    def __init__(self, enable_telegram=False, bot_token=None, chat_id=None):
        # Force CPU for Raspberry Pi
        self.device = torch.device("cpu")
        torch.set_num_threads(2)  # Optimize for Pi's 4 cores
        
        self.class_names = [
            'Armadilles', 'Bear', 'Birds', 'Cow', 'Crocodile',
            'Deer', 'Elephant', 'Goat', 'Horse', 'Jaguar',
            'Monkey', 'Rabbit', 'Skunk', 'Tiger', 'Wild Boar'
        ]
        
        # Risk categories
        self.extreme_threats = ['Bear', 'Tiger', 'Jaguar', 'Crocodile']
        self.high_threats = ['Wild Boar', 'Elephant', 'Skunk']
        self.medium_threats = ['Deer', 'Monkey', 'Armadilles']
        self.farm_animals = ['Cow', 'Goat', 'Horse']
        self.neutral_animals = ['Birds', 'Rabbit']
        
        # Load lightweight model for Pi
        self.model = self.create_pi_model()
        self.load_best_model()
        
        # Minimal transforms for Pi performance
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Telegram setup
        self.enable_telegram = enable_telegram
        self.telegram_notifier = None
        if enable_telegram:
            try:
                from rpi_telegram import RPiTelegramNotifier
                self.telegram_notifier = RPiTelegramNotifier(bot_token, chat_id)
                print("Telegram enabled")
            except:
                print("Telegram not available")
                self.enable_telegram = False
        
        self.last_alert_time = {}
        self.alert_cooldown = 30
        
        # Face detection for Pi
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print("Face detection ready")
        except:
            self.face_cascade = None
    
    def create_pi_model(self):
        """Create ultra-lightweight model for Pi"""
        # Use smallest possible model
        model = models.mobilenet_v3_small(pretrained=False)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, 128),
            nn.ReLU(),
            nn.Linear(128, len(self.class_names))
        )
        return model.to(self.device)
    
    def load_best_model(self):
        """Load best available model"""
        if os.path.exists('farm_model_lightning_cpu.pth'):
            print("Loading lightning model...")
            try:
                # Try MobileNet architecture first
                model = models.mobilenet_v3_small(pretrained=False)
                model.classifier = nn.Sequential(
                    nn.Linear(model.classifier[0].in_features, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, len(self.class_names))
                )
                model.load_state_dict(torch.load('farm_model_lightning_cpu.pth', map_location=self.device))
                self.model = model.to(self.device)
                print("Lightning model loaded (94% accuracy)")
            except:
                print("Using basic model")
        else:
            print("No trained model found")
        
        self.model.eval()
    
    def detect_human_face(self, frame):
        """Lightweight face detection for Pi"""
        if self.face_cascade is None:
            return False
        
        try:
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (320, 240))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            # Simple face detection
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
            
            if len(faces) > 0:
                print("Human detected - skipping")
                return True
            return False
        except:
            return False
    
    def classify_frame(self, frame):
        """Optimized classification for Pi"""
        # Check for humans first
        if self.detect_human_face(frame):
            return None, 0, None
        
        try:
            # Resize for faster processing
            frame_resized = cv2.resize(frame, (224, 224))
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Process with model
            input_tensor = self.transform(pil_image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                confidence, predicted = torch.max(probabilities, 0)
                
                if confidence.item() > 0.7:  # Higher threshold for Pi
                    animal = self.class_names[predicted.item()]
                    risk = self.get_risk_level(animal)
                    return animal, confidence.item(), risk
            
            return None, 0, None
        except Exception as e:
            print(f"Classification error: {e}")
            return None, 0, None
    
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
    
    def run_camera(self, camera_id=0):
        """Optimized camera loop for Pi"""
        cap = cv2.VideoCapture(camera_id)
        
        # Optimize camera settings for Pi
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 10)  # Lower FPS for Pi
        
        current_animal = None
        current_confidence = 0
        current_risk = None
        frame_count = 0
        
        print("Pi Farm Detection Started. Press Ctrl+C to quit.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every 30 frames (3 seconds at 10fps)
                if frame_count % 30 == 0:
                    animal, confidence, risk = self.classify_frame(frame)
                    if animal:
                        current_animal = animal
                        current_confidence = confidence
                        current_risk = risk
                        
                        # Send Telegram alert
                        if self.enable_telegram and self.telegram_notifier:
                            self.send_telegram_alert(animal, confidence, risk, frame)
                        
                        print(f"Detected: {animal} ({int(confidence*100)}%) - {risk}")
                    else:
                        current_animal = None
                
                # Simple display (optional - comment out for headless)
                if current_animal:
                    cv2.putText(frame, f"{current_animal} ({int(current_confidence*100)}%)", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, current_risk, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Show frame (comment out for headless operation)
                cv2.imshow('Pi Farm Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                time.sleep(0.1)  # Small delay to prevent overheating
                
        except KeyboardInterrupt:
            print("Stopping detection...")
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def send_telegram_alert(self, animal, confidence, risk_level, frame):
        """Send Telegram alert with cooldown"""
        current_time = datetime.now().timestamp()
        
        if animal in self.last_alert_time:
            if current_time - self.last_alert_time[animal] < self.alert_cooldown:
                return
        
        if self.telegram_notifier:
            success = self.telegram_notifier.send_alert(animal, confidence, risk_level, frame)
            if success:
                self.last_alert_time[animal] = current_time
                print(f"Telegram sent: {animal} - {risk_level}")

if __name__ == "__main__":
    # Telegram config
    BOT_TOKEN = "7952163683:AAHaY8pogTsIqCfbqpAWSYhsVZ6G_raAFeg"
    CHAT_ID = "6070190518"
    
    classifier = RPiFarmClassifier(
        enable_telegram=True,
        bot_token=BOT_TOKEN,
        chat_id=CHAT_ID
    )
    
    classifier.run_camera()