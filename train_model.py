#!/usr/bin/env python3
"""
COVID-19 Chest X-Ray Classification Training Script
Author: Jordanaftermidnight
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class COVID19Dataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            # Create a blank image if file is corrupted
            image = Image.new('RGB', (224, 224), color='black')

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def load_data(covid_dir, normal_dir, max_samples_per_class=500):
    image_paths = []
    labels = []

    # Load COVID images (label = 1)
    if os.path.exists(covid_dir):
        covid_files = [f for f in os.listdir(covid_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:max_samples_per_class]
        for file in covid_files:
            image_paths.append(os.path.join(covid_dir, file))
            labels.append(1)

    # Load Normal images (label = 0)
    if os.path.exists(normal_dir):
        normal_files = [f for f in os.listdir(normal_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:max_samples_per_class]
        for file in normal_files:
            image_paths.append(os.path.join(normal_dir, file))
            labels.append(0)

    return image_paths, labels

# Data transforms
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class COVID19Classifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(COVID19Classifier, self).__init__()

        # Use ResNet18 as backbone (without pretrained weights due to SSL issue)
        self.resnet = models.resnet18(pretrained=False)

        # Replace final layer with custom classifier (removed BatchNorm for single-sample compatibility)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    return total_loss / len(train_loader), 100. * correct / total

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    accuracy = 100. * correct / total
    return total_loss / len(test_loader), accuracy, all_predictions, all_targets

def main():
    print("COVID-19 Chest X-Ray Classification")
    print("=" * 40)

    # Load data
    try:
        image_paths, labels = load_data('data/COVID', 'data/Normal')
        print(f"Loaded {len(image_paths)} images")
        print(f"COVID cases: {sum(labels)}")
        print(f"Normal cases: {len(labels) - sum(labels)}")

        if len(image_paths) == 0:
            print("No images found! Please check the data directory.")
            return

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            image_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Create datasets
        train_dataset = COVID19Dataset(X_train, y_train, transform=transform_train)
        test_dataset = COVID19Dataset(X_test, y_test, transform=transform_test)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Initialize model
    model = COVID19Classifier().to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Training loop (fewer epochs since no pre-trained weights)
    num_epochs = 15
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    print("\\nStarting training...")
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)

        # Evaluate
        test_loss, test_acc, _, _ = evaluate_model(model, test_loader, criterion, device)

        # Update learning rate
        scheduler.step()

        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f'Epoch [{epoch+1}/{num_epochs}]:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        print()

    # Final evaluation
    final_test_loss, final_test_acc, predictions, targets = evaluate_model(model, test_loader, criterion, device)

    print("Training completed!")
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")

    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_accuracy': final_test_acc,
        'epoch': num_epochs
    }, 'models/covid_classifier.pth')

    # Calculate detailed metrics
    print("\\nClassification Report:")
    print(classification_report(targets, predictions, target_names=['Normal', 'COVID']))

    # Confusion Matrix
    cm = confusion_matrix(targets, predictions)
    print("\\nConfusion Matrix:")
    print(cm)

    # Calculate sensitivity and specificity
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate

    print(f"\\nSensitivity (COVID Detection): {sensitivity:.3f}")
    print(f"Specificity (Normal Detection): {specificity:.3f}")

    # Save results
    results = {
        'final_accuracy': final_test_acc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'train_losses': train_losses,
        'test_losses': test_losses
    }

    import pickle
    with open('models/training_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("\\nModel and results saved successfully!")
    print(f"Achieved {final_test_acc:.2f}% accuracy (Target: >50%)")

    return final_test_acc > 50.0

if __name__ == "__main__":
    success = main()
    if success:
        print("\\n🎉 SUCCESS: Model achieved target accuracy!")
    else:
        print("\\n⚠️  Model did not reach target accuracy. Consider more training.")