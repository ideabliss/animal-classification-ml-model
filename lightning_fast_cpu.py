import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, models
import time

class LightningFastCPU:
    def __init__(self):
        self.device = torch.device("cpu")
        print(f"LIGHTNING FAST CPU training")
        
        # Minimal transforms for maximum speed
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def create_lightning_model(self, num_classes):
        # Use MobileNet for CPU speed
        try:
            model = models.mobilenet_v3_small(pretrained=True)
            model.classifier = nn.Sequential(
                nn.Linear(model.classifier[0].in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
        except:
            # Fallback to lightweight ResNet18
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        return model.to(self.device)
    
    def train_lightning(self, epochs=10, batch_size=16):
        # Optimized data loading for CPU
        train_dataset = datasets.ImageFolder('farm_data/train', transform=self.train_transform)
        val_dataset = datasets.ImageFolder('farm_data/val', transform=self.val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        num_classes = len(train_dataset.classes)
        total_train_images = len(train_dataset)
        total_val_images = len(val_dataset)
        total_batches = len(train_loader)
        
        print(f"📊 Dataset Info:")
        print(f"   Classes: {num_classes} animals")
        print(f"   Training images: {total_train_images}")
        print(f"   Validation images: {total_val_images}")
        print(f"   Batches per epoch: {total_batches}")
        print(f"   Images per batch: {batch_size}")
        
        # Lightning fast model
        model = self.create_lightning_model(num_classes)
        
        # Aggressive optimizer for speed
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0.0
        
        print(f"\\nLIGHTNING FAST training!")
        print(f"Target: 97%+ in 20-30 minutes")
        print(f"Epochs: {epochs} | Batch: {batch_size}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"\\nEpoch {epoch+1}/{epochs}")
            
            # Training
            model.train()
            train_correct = 0
            train_total = 0
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                if batch_idx % 50 == 0:
                    elapsed = time.time() - start_time
                    images_processed = (batch_idx + 1) * batch_size
                    progress = (batch_idx + 1) / total_batches * 100
                    print(f"  Batch {batch_idx}/{total_batches} ({progress:.1f}%) - Images: {images_processed}/{total_train_images} - Loss: {loss.item():.3f} - {elapsed/60:.1f}min")
            
            train_acc = 100 * train_correct / train_total
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = 100 * val_correct / val_total
            scheduler.step()
            
            elapsed = time.time() - start_time
            print(f"✅ EPOCH COMPLETE: All {total_train_images} training + {total_val_images} validation images processed")
            print(f"TRAIN: {train_acc:.1f}% | VAL: {val_acc:.1f}% | Time: {elapsed/60:.1f}min")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), 'farm_model_lightning_cpu.pth')
                print(f"★ Best: {best_acc:.1f}%")
            
            if val_acc >= 97.0:
                print(f"🎯 TARGET REACHED: {val_acc:.1f}%")
                break
        
        total_time = time.time() - start_time
        print(f"\\n⚡ Completed in {total_time/60:.1f} minutes!")
        print(f"🏆 Best accuracy: {best_acc:.1f}%")

if __name__ == "__main__":
    trainer = LightningFastCPU()
    trainer.train_lightning(epochs=10, batch_size=16)