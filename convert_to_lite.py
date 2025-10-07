import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms

def convert_model_to_lite():
    """Convert trained model to PyTorch Lite for Pi"""
    print("Converting model to PyTorch Lite...")
    
    # Load original model
    model = models.mobilenet_v3_small(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 15)  # 15 classes
    )
    
    # Load trained weights
    model.load_state_dict(torch.load('farm_model_lightning_cpu.pth', map_location='cpu'))
    model.eval()
    
    # Convert to TorchScript (Lite)
    example_input = torch.randn(1, 3, 224, 224)
    traced_model = torch.jit.trace(model, example_input)
    
    # Optimize for mobile
    optimized_model = torch.jit.optimize_for_inference(traced_model)
    
    # Save lite model
    optimized_model.save('farm_model_lite.pt')
    print("✅ Lite model saved as 'farm_model_lite.pt'")
    
    # Test lite model
    with torch.no_grad():
        original_output = model(example_input)
        lite_output = optimized_model(example_input)
        
        print(f"Original model output shape: {original_output.shape}")
        print(f"Lite model output shape: {lite_output.shape}")
        print(f"Models match: {torch.allclose(original_output, lite_output, atol=1e-5)}")

if __name__ == "__main__":
    convert_model_to_lite()