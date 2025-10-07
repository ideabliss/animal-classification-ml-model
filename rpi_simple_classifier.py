import cv2
import numpy as np
import os
import time
from datetime import datetime

class SimplePiClassifier:
    def __init__(self, enable_telegram=False, bot_token=None, chat_id=None):
        print("Simple Pi Farm Detection (OpenCV-based)")
        
        # Animal detection using color/shape analysis
        self.class_names = [
            'Bear', 'Tiger', 'Wild Boar', 'Elephant', 'Deer',
            'Cow', 'Horse', 'Goat', 'Bird', 'Unknown Animal'
        ]
        
        # Risk categories
        self.extreme_threats = ['Bear', 'Tiger']
        self.high_threats = ['Wild Boar', 'Elephant']
        self.medium_threats = ['Deer']
        self.farm_animals = ['Cow', 'Horse', 'Goat']
        self.neutral_animals = ['Bird']
        
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
    
    def simple_animal_detection(self, frame):
        """Basic animal detection using motion and size"""
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Create mask for brown/dark colors (common in animals)
            lower_brown = np.array([10, 50, 20])
            upper_brown = np.array([20, 255, 200])
            mask1 = cv2.inRange(hsv, lower_brown, upper_brown)
            
            # Mask for darker colors
            lower_dark = np.array([0, 0, 0])
            upper_dark = np.array([180, 255, 50])
            mask2 = cv2.inRange(hsv, lower_dark, upper_dark)
            
            # Combine masks
            mask = cv2.bitwise_or(mask1, mask2)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter by size
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 5000:  # Minimum size for animal
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Simple classification based on size and shape
                    if area > 50000:  # Very large
                        if aspect_ratio > 1.5:
                            return "Elephant", 0.75, "HIGH THREAT"
                        else:
                            return "Bear", 0.80, "EXTREME THREAT"
                    elif area > 20000:  # Large
                        if aspect_ratio > 1.8:
                            return "Horse", 0.70, "LIVESTOCK"
                        else:
                            return "Wild Boar", 0.75, "HIGH THREAT"
                    elif area > 10000:  # Medium
                        if aspect_ratio > 1.5:
                            return "Deer", 0.65, "MEDIUM THREAT"
                        else:
                            return "Cow", 0.70, "LIVESTOCK"
                    else:  # Small
                        return "Bird", 0.60, "LOW RISK"
            
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
        """Simple camera detection"""
        cap = cv2.VideoCapture(camera_id)
        
        # Optimize for Pi
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 10)
        
        frame_count = 0
        
        print("Simple Pi Detection Started. Press Ctrl+C to quit.")
        print("Note: This uses basic OpenCV detection (no AI model)")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every 30 frames
                if frame_count % 30 == 0:
                    # Check for humans first
                    if self.detect_human_face(frame):
                        continue
                    
                    # Simple animal detection
                    animal, confidence, risk = self.simple_animal_detection(frame)
                    
                    if animal and confidence > 0.6:
                        print(f"Detected: {animal} ({int(confidence*100)}%) - {risk}")
                        
                        # Send Telegram alert
                        if self.enable_telegram and self.telegram_notifier:
                            self.send_telegram_alert(animal, confidence, risk, frame)
                        
                        # Display on frame
                        cv2.putText(frame, f"{animal} ({int(confidence*100)}%)", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, risk, (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Show frame (comment out for headless)
                cv2.imshow('Simple Pi Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
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