import os
from farm_animal_classifier import FarmAnimalClassifier

def test_trained_model():
    print("🧪 Testing Your Trained Farm Animal Model")
    print("=" * 50)
    
    # Check if trained model exists
    if os.path.exists('farm_model_cpu.pth'):
        print("✅ Trained model found: farm_model_cpu.pth")
        print("🎯 Expected accuracy: 95.68%")
    else:
        print("❌ No trained model found!")
        print("💡 Run: python cpu_train.py to train first")
        return
    
    # Initialize classifier with Telegram (will auto-load trained model)
    print("\n🔄 Loading classifier with Telegram notifications...")
    BOT_TOKEN = "7952163683:AAHaY8pogTsIqCfbqpAWSYhsVZ6G_raAFeg"
    CHAT_ID = "6070190518"
    
    classifier = FarmAnimalClassifier(
        enable_telegram=True,
        bot_token=BOT_TOKEN,
        chat_id=CHAT_ID
    )
    
    print("\n📊 Model Details:")
    print(f"   Device: {classifier.device}")
    print(f"   Classes: {len(classifier.class_names)}")
    print(f"   Animals: {', '.join(classifier.class_names)}")
    
    print("\n🎮 Testing Options:")
    print("1. Camera test: python test_trained_model.py")
    print("2. Image test: python farm_animal_classifier.py image.jpg")
    print("3. Live demo: python run_farm.py")
    
    # Start camera test
    print("\n📹 Starting camera test...")
    print("Press 'q' to quit")
    classifier.run_camera()

if __name__ == "__main__":
    test_trained_model()