#!/usr/bin/env python3
"""
Extended COVID-19 Classification Training - Additional 85 Epochs
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

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

class COVID19Classifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(COVID19Classifier, self).__init__()

        self.resnet = models.resnet18(pretrained=False)

        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
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
    print("COVID-19 Extended Training - Additional 85 Epochs")
    print("=" * 55)

    # Data transforms
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load data
    image_paths, labels = load_data('data/COVID', 'data/Normal')
    print(f"Loaded {len(image_paths)} images")

    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_dataset = COVID19Dataset(X_train, y_train, transform=transform_train)
    test_dataset = COVID19Dataset(X_test, y_test, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize model
    model = COVID19Classifier().to(device)

    # Load previous training state if exists
    previous_results = None
    if os.path.exists('models/covid_classifier.pth'):
        print("Loading previous model state...")
        checkpoint = torch.load('models/covid_classifier.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load previous results
        if os.path.exists('models/training_results.pkl'):
            with open('models/training_results.pkl', 'rb') as f:
                previous_results = pickle.load(f)

    # Setup optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  # Lower LR for fine-tuning
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Extended training for 85 more epochs
    additional_epochs = 85
    train_losses = previous_results['train_losses'] if previous_results else []
    train_accuracies = previous_results['train_accuracies'] if previous_results else []
    test_losses = previous_results['test_losses'] if previous_results else []
    test_accuracies = previous_results['test_accuracies'] if previous_results else []

    starting_epoch = len(train_accuracies) + 1
    total_epochs = starting_epoch + additional_epochs - 1

    print(f"Continuing training from epoch {starting_epoch} to {total_epochs}")
    print(f"Previous best accuracy: {max(test_accuracies) if test_accuracies else 0:.2f}%")
    print("\\nStarting extended training...")

    best_accuracy = max(test_accuracies) if test_accuracies else 0

    for epoch in range(additional_epochs):
        current_epoch = starting_epoch + epoch

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

        # Track best accuracy
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            print(f'Epoch [{current_epoch}/{total_epochs}]: New best accuracy {test_acc:.2f}%! 🎉')

        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == additional_epochs - 1:
            print(f'Epoch [{current_epoch}/{total_epochs}]:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
            print(f'  Best Accuracy So Far: {best_accuracy:.2f}%')
            print()

    # Final evaluation
    final_test_loss, final_test_acc, predictions, targets = evaluate_model(model, test_loader, criterion, device)

    print("Extended training completed!")
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")
    print(f"Best Accuracy Achieved: {best_accuracy:.2f}%")
    print(f"Total Training Epochs: {len(train_accuracies)}")

    # Calculate detailed metrics
    print("\\nFinal Classification Report:")
    print(classification_report(targets, predictions, target_names=['Normal', 'COVID']))

    cm = confusion_matrix(targets, predictions)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\\nFinal Metrics:")
    print(f"Sensitivity (COVID Detection): {sensitivity:.3f}")
    print(f"Specificity (Normal Detection): {specificity:.3f}")

    # Save updated model and results
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_accuracy': final_test_acc,
        'best_accuracy': best_accuracy,
        'epoch': len(train_accuracies)
    }, 'models/covid_classifier_extended.pth')

    # Save updated results
    results = {
        'final_accuracy': final_test_acc,
        'best_accuracy': best_accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'total_epochs': len(train_accuracies)
    }

    with open('models/training_results_extended.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("\\n🎉 Extended training completed successfully!")
    print(f"Model trained for {len(train_accuracies)} total epochs")
    print(f"Achieved {final_test_acc:.2f}% final accuracy")
    print(f"Peak accuracy: {best_accuracy:.2f}%")

    return final_test_acc, best_accuracy

if __name__ == "__main__":
    final_acc, best_acc = main()
    print(f"\\n🎯 EXTENDED TRAINING SUMMARY:")
    print(f"Final Accuracy: {final_acc:.2f}%")
    print(f"Best Accuracy: {best_acc:.2f}%")
    print(f"Training Target (>50%): {'✅ EXCEEDED' if final_acc > 50 else '❌ NOT MET'}")