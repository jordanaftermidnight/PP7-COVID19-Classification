#!/usr/bin/env python3
"""
Ensemble Model for COVID-19 Classification
Combines multiple CNN architectures for improved performance
Author: Jordanaftermidnight
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
from PIL import Image
import pickle
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class ResNet18Model(nn.Module):
    """ResNet-18 based model"""
    def __init__(self, num_classes=2):
        super(ResNet18Model, self).__init__()
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

class ResNet34Model(nn.Module):
    """ResNet-34 based model"""
    def __init__(self, num_classes=2):
        super(ResNet34Model, self).__init__()
        self.resnet = models.resnet34(pretrained=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

class DenseNetModel(nn.Module):
    """DenseNet-121 based model"""
    def __init__(self, num_classes=2):
        super(DenseNetModel, self).__init__()
        self.densenet = models.densenet121(pretrained=False)
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.densenet(x)

class EfficientNetModel(nn.Module):
    """EfficientNet-B0 based model (simplified version)"""
    def __init__(self, num_classes=2):
        super(EfficientNetModel, self).__init__()

        # Create a simple CNN that mimics EfficientNet structure
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class EnsembleModel(nn.Module):
    """Ensemble of multiple models"""
    def __init__(self, models_list, weights=None):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models_list)

        # Default equal weights if none provided
        if weights is None:
            self.weights = [1.0 / len(models_list)] * len(models_list)
        else:
            self.weights = weights

        # Learnable weights (optional)
        self.learnable_weights = nn.Parameter(torch.ones(len(models_list)))

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))

        # Weighted ensemble
        weighted_outputs = []
        for i, output in enumerate(outputs):
            weight = torch.softmax(self.learnable_weights, dim=0)[i]
            weighted_outputs.append(weight * output)

        ensemble_output = torch.stack(weighted_outputs).sum(dim=0)
        return ensemble_output

    def predict_with_individual_models(self, x):
        """Get predictions from individual models"""
        individual_predictions = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                output = model(x)
                probs = torch.softmax(output, dim=1)
                individual_predictions.append(probs)

        return individual_predictions

def load_data(covid_dir, normal_dir, max_samples_per_class=500):
    """Load dataset"""
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

def train_individual_model(model, train_loader, test_loader, model_name, epochs=10):
    """Train a single model"""
    print(f"Training {model_name}...")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    model.to(device)
    best_accuracy = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

        # Testing
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                test_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()

        train_acc = 100. * train_correct / train_total
        test_acc = 100. * test_correct / test_total

        if test_acc > best_accuracy:
            best_accuracy = test_acc

        scheduler.step()

        if epoch % 3 == 0:
            print(f'  Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')

    print(f'{model_name} best accuracy: {best_accuracy:.2f}%')
    return model, best_accuracy

def evaluate_ensemble(ensemble_model, test_loader):
    """Evaluate ensemble model"""
    ensemble_model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = ensemble_model(data)

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    accuracy = 100. * correct / total
    return accuracy, all_predictions, all_targets

def main():
    """Main ensemble training and evaluation"""
    print("COVID-19 Ensemble Model Training")
    print("=" * 40)

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

    # Initialize models
    models_info = [
        (ResNet18Model(), "ResNet-18"),
        (ResNet34Model(), "ResNet-34"),
        (DenseNetModel(), "DenseNet-121"),
        (EfficientNetModel(), "EfficientNet-like")
    ]

    trained_models = []
    individual_accuracies = []

    # Train individual models
    print("\\nTraining individual models...")
    for model, name in models_info:
        trained_model, accuracy = train_individual_model(
            model, train_loader, test_loader, name, epochs=8
        )
        trained_models.append(trained_model)
        individual_accuracies.append(accuracy)

    # Create ensemble
    print("\\nCreating ensemble model...")
    ensemble = EnsembleModel(trained_models)
    ensemble.to(device)

    # Fine-tune ensemble weights
    print("Fine-tuning ensemble weights...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ensemble.learnable_weights, lr=0.01)

    for epoch in range(5):
        ensemble.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = ensemble(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Evaluate ensemble
    ensemble_accuracy, predictions, targets = evaluate_ensemble(ensemble, test_loader)

    # Results
    print("\\n" + "="*50)
    print("ENSEMBLE MODEL RESULTS")
    print("="*50)

    print("Individual Model Accuracies:")
    for i, (_, name) in enumerate(models_info):
        print(f"  {name}: {individual_accuracies[i]:.2f}%")

    print(f"\\nEnsemble Accuracy: {ensemble_accuracy:.2f}%")
    print(f"Best Individual: {max(individual_accuracies):.2f}%")
    print(f"Improvement: {ensemble_accuracy - max(individual_accuracies):.2f}%")

    # Detailed metrics
    print("\\nEnsemble Classification Report:")
    print(classification_report(targets, predictions, target_names=['Normal', 'COVID']))

    cm = confusion_matrix(targets, predictions)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\\nEnsemble Medical Metrics:")
    print(f"Sensitivity (COVID Detection): {sensitivity:.3f}")
    print(f"Specificity (Normal Detection): {specificity:.3f}")

    # Save ensemble model
    os.makedirs('models', exist_ok=True)
    torch.save({
        'ensemble_state_dict': ensemble.state_dict(),
        'individual_models': [model.state_dict() for model in trained_models],
        'model_names': [name for _, name in models_info],
        'ensemble_accuracy': ensemble_accuracy,
        'individual_accuracies': individual_accuracies
    }, 'models/ensemble_model.pth')

    # Save results
    ensemble_results = {
        'ensemble_accuracy': ensemble_accuracy,
        'individual_accuracies': individual_accuracies,
        'model_names': [name for _, name in models_info],
        'sensitivity': sensitivity,
        'specificity': specificity,
        'improvement': ensemble_accuracy - max(individual_accuracies)
    }

    with open('models/ensemble_results.pkl', 'wb') as f:
        pickle.dump(ensemble_results, f)

    print("\\n🎉 Ensemble model training completed!")
    print(f"Ensemble model saved with {ensemble_accuracy:.2f}% accuracy")

    return ensemble, ensemble_accuracy

if __name__ == "__main__":
    ensemble, accuracy = main()
    print(f"\\n🏆 Final ensemble accuracy: {accuracy:.2f}%")