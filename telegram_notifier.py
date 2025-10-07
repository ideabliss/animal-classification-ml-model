import requests
import cv2
import os
from datetime import datetime

class TelegramNotifier:
    def __init__(self, bot_token=None, chat_id=None):
        self.bot_token = bot_token or "YOUR_BOT_TOKEN"
        self.chat_id = chat_id or "YOUR_CHAT_ID"
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
    
    def send_alert(self, animal, confidence, risk_level, frame):
        """Send Telegram alert with image"""
        try:
            # Save frame as temporary image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_image = f"temp_detection_{timestamp}.jpg"
            cv2.imwrite(temp_image, frame)
            
            # Create message
            accuracy = int(confidence * 100)
            risk_emoji = self.get_risk_emoji(risk_level)
            
            message = f"""FARM ALERT {risk_emoji}
            
Animal: {animal.upper()}
Confidence: {accuracy}%
Risk Level: {risk_level}
Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{self.get_risk_description(risk_level)}"""
            
            # Send photo with caption
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
                    timeout=10
                )
                
                print(f"Response status: {response.status_code}")
                if response.status_code != 200:
                    print(f"Response text: {response.text}")
            
            # Clean up temp file
            if os.path.exists(temp_image):
                os.remove(temp_image)
            
            return response.status_code == 200
            
        except Exception as e:
            print(f"Telegram notification failed: {e}")
            if 'response' in locals():
                print(f"Response status: {response.status_code}")
                print(f"Response text: {response.text}")
            return False
    
    def get_risk_emoji(self, risk_level):
        emojis = {
            "EXTREME THREAT": "[EXTREME]",
            "HIGH THREAT": "[HIGH]", 
            "MEDIUM THREAT": "[MEDIUM]",
            "LIVESTOCK": "[SAFE]",
            "LOW RISK": "[LOW]"
        }
        return emojis.get(risk_level, "[UNKNOWN]")
    
    def get_risk_description(self, risk_level):
        descriptions = {
            "EXTREME THREAT": "IMMEDIATE ACTION REQUIRED! Dangerous predator detected.",
            "HIGH THREAT": "High risk to crops/livestock. Monitor closely.",
            "MEDIUM THREAT": "Moderate threat. Take preventive measures.",
            "LIVESTOCK": "Farm animal detected. Ensure safety.",
            "LOW RISK": "Low risk animal. Normal monitoring."
        }
        return descriptions.get(risk_level, "Unknown risk level")
    
    def test_connection(self):
        """Test Telegram bot connection"""
        try:
            response = requests.get(f"{self.base_url}/getMe", timeout=5)
            return response.status_code == 200
        except:
            return False