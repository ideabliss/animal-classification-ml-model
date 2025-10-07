import cv2
import numpy as np
import os
import time
from datetime import datetime

# Try to import torch for trained model
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torchvision import models
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available, using basic detection")

class SimplePiClassifier:
    def __init__(self, enable_telegram=False, bot_token=None, chat_id=None):
        print("Pi Farm Detection with Trained Model")
        
        # Farm animal classes (matching trained model)
        self.class_names = [
            'Armadilles', 'Bear', 'Birds', 'Cow', 'Crocodile',
            'Deer', 'Elephant', 'Goat', 'Horse', 'Jaguar',
            'Monkey', 'Rabbit', 'Skunk', 'Tiger', 'Wild Boar'
        ]
        
        # Load trained model if available
        self.model = None
        self.transform = None
        if TORCH_AVAILABLE and os.path.exists('farm_model_lightning_cpu.pth'):
            try:
                self.load_trained_model()
                print("Trained model loaded (94% accuracy)")
            except Exception as e:
                print(f"Failed to load trained model: {e}")
                print("Using basic OpenCV detection")
        
        # Risk categories (matching trained model)
        self.extreme_threats = ['Bear', 'Tiger', 'Jaguar', 'Crocodile']
        self.high_threats = ['Wild Boar', 'Elephant', 'Skunk']
        self.medium_threats = ['Deer', 'Monkey', 'Armadilles']
        self.farm_animals = ['Cow', 'Goat', 'Horse']
        self.neutral_animals = ['Birds', 'Rabbit']
        
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
        
        # Face detection
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print("Face detection ready")
        except:
            self.face_cascade = None
    
    def load_trained_model(self):
        """Load the trained MobileNet model"""
        # Create MobileNet model architecture
        self.model = models.mobilenet_v3_small(pretrained=False)
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier[0].in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, len(self.class_names))
        )
        
        # Load trained weights
        self.model.load_state_dict(torch.load('farm_model_lightning_cpu.pth', map_location='cpu'))
        self.model.eval()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def detect_human_face(self, frame):
        """Simple face detection"""
        if self.face_cascade is None:
            return False
        
        try:
            small_frame = cv2.resize(frame, (320, 240))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
            
            if len(faces) > 0:
                print("Human detected - skipping")
                return True
            return False
        except:
            return False
    
    def classify_with_trained_model(self, frame):
        """Use trained model for classification"""
        if self.model is None or self.transform is None:
            return None, 0, None
        
        try:
            # Prepare image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            input_tensor = self.transform(pil_image).unsqueeze(0)
            
            # Classify
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
            print(f"Model classification error: {e}")
            return None, 0, None
    
    def simple_animal_detection(self, frame):
        """Fallback basic detection if model fails"""
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_brown = np.array([10, 50, 20])
            upper_brown = np.array([20, 255, 200])
            mask = cv2.inRange(hsv, lower_brown, upper_brown)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 10000:
                    return "Unknown Animal", 0.60, "MEDIUM THREAT"
            
            return None, 0, None
        except:
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
        """USB camera detection with trained model"""
        # Try different backends for USB camera
        cap = None
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        
        for backend in backends:
            try:
                cap = cv2.VideoCapture(camera_id, backend)
                if cap.isOpened():
                    print(f"Camera opened with backend: {backend}")
                    break
                cap.release()
            except:
                continue
        
        if not cap or not cap.isOpened():
            cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("ERROR: Cannot open camera!")
            return
        
        # Optimize for Pi
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 10)
        
        frame_count = 0
        last_status_time = time.time()
        
        print("Pi Detection Started. Press Ctrl+C to quit.")
        if self.model:
            print("Using trained model (94% accuracy)")
        else:
            print("Using basic OpenCV detection")
        print("Processing frames... (status every 30 seconds)")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                current_time = time.time()
                
                # Status update every 30 seconds
                if current_time - last_status_time > 30:
                    print(f"Status: Running... Processed {frame_count} frames")
                    last_status_time = current_time
                
                # Process every 30 frames
                if frame_count % 30 == 0:
                    # Check for humans first
                    if self.detect_human_face(frame):
                        continue
                    
                    # Try trained model first, fallback to simple detection
                    animal, confidence, risk = self.classify_with_trained_model(frame)
                    if not animal:
                        animal, confidence, risk = self.simple_animal_detection(frame)
                    
                    if animal and confidence > 0.6:
                        print(f"*** DETECTION: {animal} ({int(confidence*100)}%) - {risk} ***")
                        
                        # Send Telegram alert
                        if self.enable_telegram and self.telegram_notifier:
                            self.send_telegram_alert(animal, confidence, risk, frame)
                    else:
                        # Show it's checking (every 10th processing cycle)
                        if (frame_count // 30) % 10 == 0:
                            print(f"Scanning... Frame {frame_count} (no animals detected)")
                        
                        # Display on frame
                        cv2.putText(frame, f"{animal} ({int(confidence*100)}%)", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, risk, (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Headless mode - no display (for SSH)
                # cv2.imshow('Simple Pi Detection', frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("Stopping detection...")
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def send_telegram_alert(self, animal, confidence, risk_level, frame):
        """Send Telegram alert"""
        current_time = datetime.now().timestamp()
        
        if animal in self.last_alert_time:
            if current_time - self.last_alert_time[animal] < self.alert_cooldown:
                return
        
        if risk_level in ["EXTREME THREAT", "HIGH THREAT", "MEDIUM THREAT"]:
            if self.telegram_notifier:
                success = self.telegram_notifier.send_alert(animal, confidence, risk_level, frame)
                if success:
                    self.last_alert_time[animal] = current_time
                    print(f"Telegram sent: {animal} - {risk_level}")

if __name__ == "__main__":
    # Telegram config
    BOT_TOKEN = "7952163683:AAHaY8pogTsIqCfbqpAWSYhsVZ6G_raAFeg"
    CHAT_ID = "6070190518"
    
    classifier = SimplePiClassifier(
        enable_telegram=True,
        bot_token=BOT_TOKEN,
        chat_id=CHAT_ID
    )
    
    classifier.run_camera()