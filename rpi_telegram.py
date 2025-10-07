import requests
import cv2
import os
from datetime import datetime

class RPiTelegramNotifier:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
    
    def send_alert(self, animal, confidence, risk_level, frame):
        """Lightweight Telegram alert for Pi"""
        try:
            # Compress image for Pi's limited bandwidth
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_image = f"pi_detection_{timestamp}.jpg"
            
            # Resize and compress for faster upload
            small_frame = cv2.resize(frame, (320, 240))
            cv2.imwrite(temp_image, small_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            
            # Simple message
            accuracy = int(confidence * 100)
            message = f"FARM ALERT\\n\\nAnimal: {animal.upper()}\\nConfidence: {accuracy}%\\nRisk: {risk_level}\\nTime: {datetime.now().strftime('%H:%M:%S')}"
            
            # Send photo
            with open(temp_image, 'rb') as photo:
                files = {'photo': photo}
                data = {
                    'chat_id': self.chat_id,
                    'caption': message
                }
                
                response = requests.post(
                    f"{self.base_url}/sendPhoto",
                    files=files,
                    data=data,
                    timeout=30
                )
            
            # Clean up
            if os.path.exists(temp_image):
                os.remove(temp_image)
            
            return response.status_code == 200
            
        except Exception as e:
            print(f"Telegram error: {e}")
            return False