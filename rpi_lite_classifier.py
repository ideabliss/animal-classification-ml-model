import cv2
import numpy as np
import os
import time
from datetime import datetime

# Try PyTorch Lite
try:
    import torch
    from PIL import Image
    import torchvision.transforms as transforms
    TORCH_LITE_AVAILABLE = True
    print("PyTorch Lite available")
except ImportError:
    TORCH_LITE_AVAILABLE = False
    print("PyTorch Lite not available, using basic detection")

class PiLiteClassifier:
    def __init__(self, enable_telegram=False, bot_token=None, chat_id=None):
        print("Pi Farm Detection with PyTorch Lite")
        
        # Farm animal classes
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
        
        # Load PyTorch Lite model
        self.lite_model = None
        self.transform = None
        if TORCH_LITE_AVAILABLE and os.path.exists('farm_model_lite.pt'):
            try:
                self.load_lite_model()
                print("✅ PyTorch Lite model loaded (94% accuracy)")
            except Exception as e:
                print(f"Failed to load lite model: {e}")
                print("Using basic detection")
        
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
    
    def load_lite_model(self):
        """Load PyTorch Lite model"""
        self.lite_model = torch.jit.load('farm_model_lite.pt', map_location='cpu')
        self.lite_model.eval()
        
        # Set to single thread for Pi
        torch.set_num_threads(1)
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def detect_human_face(self, frame):
        """Face detection"""
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
    
    def classify_with_lite_model(self, frame):
        """Use PyTorch Lite model"""
        if self.lite_model is None or self.transform is None:
            return None, 0, None
        
        try:
            # Prepare image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            input_tensor = self.transform(pil_image).unsqueeze(0)
            
            # Classify with lite model
            with torch.no_grad():
                outputs = self.lite_model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                confidence, predicted = torch.max(probabilities, 0)
                
                if confidence.item() > 0.75:  # High threshold for Pi
                    animal = self.class_names[predicted.item()]
                    risk = self.get_risk_level(animal)
                    return animal, confidence.item(), risk
            
            return None, 0, None
        except Exception as e:
            print(f"Lite model error: {e}")
            return None, 0, None
    
    def simple_detection_fallback(self, frame):
        """Fallback basic detection"""
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_brown = np.array([8, 50, 20])
            upper_brown = np.array([25, 255, 200])
            mask = cv2.inRange(hsv, lower_brown, upper_brown)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 8000:
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
        """Run detection with PyTorch Lite"""
        # Setup camera
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
        
        print("Pi Lite Detection Started. Press Ctrl+C to quit.")
        if self.lite_model:
            print("Using PyTorch Lite model (94% accuracy)")
        else:
            print("Using basic detection fallback")
        
        # Send startup message
        if self.enable_telegram and self.telegram_notifier:
            startup_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(startup_frame, 'LITE MODEL STARTED', (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            model_type = "PyTorch Lite (94%)" if self.lite_model else "Basic Detection"
            cv2.putText(startup_frame, model_type, (80, 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            success = self.telegram_notifier.send_alert("System", 1.0, "ONLINE", startup_frame)
            if success:
                print("Startup message sent to Telegram")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                current_time = time.time()
                
                # Status update
                if current_time - last_status_time > 30:
                    model_status = "Lite Model" if self.lite_model else "Basic"
                    print(f"Status: {model_status} - Processed {frame_count} frames")
                    last_status_time = current_time
                
                # Process every 30 frames (3 seconds)
                if frame_count % 30 == 0:
                    # Check for humans first
                    if self.detect_human_face(frame):
                        continue
                    
                    # Try lite model first, fallback to basic
                    animal, confidence, risk = self.classify_with_lite_model(frame)
                    if not animal:
                        animal, confidence, risk = self.simple_detection_fallback(frame)
                    
                    if animal and confidence > 0.6:
                        model_used = "LITE" if self.lite_model and confidence > 0.75 else "BASIC"
                        print(f"*** {model_used} DETECTION: {animal} ({int(confidence*100)}%) - {risk} ***")
                        
                        # Send Telegram alert
                        if self.enable_telegram and self.telegram_notifier:
                            self.send_telegram_alert(animal, confidence, risk, frame)
                    else:
                        if (frame_count // 30) % 10 == 0:
                            print(f"Scanning... Frame {frame_count} (no animals detected)")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("Stopping detection...")
        finally:
            cap.release()
    
    def send_telegram_alert(self, animal, confidence, risk_level, frame):
        """Send Telegram alert"""
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
    
    classifier = PiLiteClassifier(
        enable_telegram=True,
        bot_token=BOT_TOKEN,
        chat_id=CHAT_ID
    )
    
    classifier.run_camera()