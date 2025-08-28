#!/usr/bin/env python3
"""
Grad-CAM Visualization for COVID-19 Classification Model
Shows which parts of chest X-rays the model focuses on for decision making
Author: Jordanaftermidnight
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    print("Warning: pytorch-grad-cam not installed. Install with: pip install pytorch-grad-cam")
    GradCAM = None
    GRADCAM_AVAILABLE = False
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
import random

class COVID19Classifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(COVID19Classifier, self).__init__()

        self.resnet = models.resnet18(pretrained=False)

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

class GradCAMVisualizer:
    def __init__(self, model_path='models/covid_classifier_extended.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the trained model
        self.model = COVID19Classifier()
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {model_path}")
        else:
            # Fallback to original model
            model_path = 'models/covid_classifier.pth'
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Model loaded from {model_path}")
            else:
                print("No trained model found! Please run training first.")
                return

        self.model.to(self.device)
        self.model.eval()

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Initialize Grad-CAM
        # Target the last convolutional layer in ResNet
        self.target_layers = [self.model.resnet.layer4[-1]]
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)

        print("Grad-CAM visualizer initialized successfully!")

    def load_and_preprocess_image(self, image_path):
        """Load and preprocess image for the model"""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')

            # Keep original for visualization
            original_image = np.array(image.resize((224, 224))) / 255.0

            # Transform for model
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            return input_tensor, original_image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None, None

    def generate_gradcam(self, image_path, target_class=None):
        """Generate Grad-CAM visualization for an image"""
        input_tensor, rgb_image = self.load_and_preprocess_image(image_path)
        if input_tensor is None:
            return None, None, None

        # Get model prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        # Use predicted class if no target specified
        if target_class is None:
            target_class = predicted_class

        # Generate Grad-CAM
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # Remove batch dimension

        # Overlay on original image
        visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

        class_names = ['Normal', 'COVID-19']
        prediction_info = {
            'predicted_class': predicted_class,
            'predicted_label': class_names[predicted_class],
            'confidence': confidence,
            'probabilities': {
                'Normal': probabilities[0][0].item(),
                'COVID-19': probabilities[0][1].item()
            }
        }

        return visualization, grayscale_cam, prediction_info

    def visualize_sample_images(self, covid_dir='data/COVID', normal_dir='data/Normal', num_samples=4):
        """Generate Grad-CAM for sample images from both classes"""
        sample_images = []

        # Get sample COVID images
        if os.path.exists(covid_dir):
            covid_files = [f for f in os.listdir(covid_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            covid_samples = random.sample(covid_files[:50], min(num_samples//2, len(covid_files)))
            for file in covid_samples:
                sample_images.append((os.path.join(covid_dir, file), 'COVID-19'))

        # Get sample Normal images
        if os.path.exists(normal_dir):
            normal_files = [f for f in os.listdir(normal_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            normal_samples = random.sample(normal_files[:50], min(num_samples//2, len(normal_files)))
            for file in normal_samples:
                sample_images.append((os.path.join(normal_dir, file), 'Normal'))

        if not sample_images:
            print("No sample images found!")
            return

        # Create visualization
        fig, axes = plt.subplots(len(sample_images), 3, figsize=(15, 5 * len(sample_images)))
        if len(sample_images) == 1:
            axes = axes.reshape(1, -1)

        for i, (image_path, true_label) in enumerate(sample_images):
            print(f"Processing {os.path.basename(image_path)}...")

            # Load original image
            original_image = Image.open(image_path).convert('RGB')
            original_array = np.array(original_image.resize((224, 224)))

            # Generate Grad-CAM
            cam_visualization, heatmap, prediction = self.generate_gradcam(image_path)

            if cam_visualization is not None:
                # Original image
                axes[i, 0].imshow(original_array)
                axes[i, 0].set_title(f'Original\\nTrue: {true_label}')
                axes[i, 0].axis('off')

                # Heatmap
                axes[i, 1].imshow(heatmap, cmap='jet')
                axes[i, 1].set_title('Attention Heatmap')
                axes[i, 1].axis('off')

                # Grad-CAM overlay
                axes[i, 2].imshow(cam_visualization)
                pred_label = prediction['predicted_label']
                confidence = prediction['confidence']
                title = f'Grad-CAM Overlay\\nPred: {pred_label}\\nConf: {confidence:.3f}'
                axes[i, 2].set_title(title)
                axes[i, 2].axis('off')

                print(f"  True: {true_label}, Predicted: {pred_label}, Confidence: {confidence:.3f}")
            else:
                for j in range(3):
                    axes[i, j].text(0.5, 0.5, 'Error loading image',
                                   ha='center', va='center', transform=axes[i, j].transAxes)
                    axes[i, j].axis('off')

        plt.tight_layout()
        plt.savefig('grad_cam_visualizations.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\\nGrad-CAM visualizations saved as 'grad_cam_visualizations.png'")

    def analyze_single_image(self, image_path):
        """Detailed analysis of a single image"""
        print(f"Analyzing: {os.path.basename(image_path)}")
        print("-" * 50)

        cam_visualization, heatmap, prediction = self.generate_gradcam(image_path)

        if cam_visualization is None:
            print("Error: Could not process image")
            return

        # Print prediction details
        print(f"Prediction: {prediction['predicted_label']}")
        print(f"Confidence: {prediction['confidence']:.4f}")
        print(f"Probabilities:")
        for class_name, prob in prediction['probabilities'].items():
            print(f"  {class_name}: {prob:.4f}")

        # Create detailed visualization
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # Original image
        original_image = Image.open(image_path).convert('RGB')
        original_array = np.array(original_image.resize((224, 224)))
        axes[0].imshow(original_array)
        axes[0].set_title('Original X-ray')
        axes[0].axis('off')

        # Heatmap
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Attention Heatmap')
        axes[1].axis('off')

        # Grad-CAM overlay
        axes[2].imshow(cam_visualization)
        axes[2].set_title('Grad-CAM Overlay')
        axes[2].axis('off')

        # Probability bar chart
        classes = list(prediction['probabilities'].keys())
        probs = list(prediction['probabilities'].values())
        colors = ['lightblue' if classes[i] != prediction['predicted_label'] else 'orange' for i in range(len(classes))]

        axes[3].bar(classes, probs, color=colors)
        axes[3].set_ylabel('Probability')
        axes[3].set_title('Class Probabilities')
        axes[3].set_ylim([0, 1])

        # Add confidence text
        for i, (class_name, prob) in enumerate(zip(classes, probs)):
            axes[3].text(i, prob + 0.02, f'{prob:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        filename = f'gradcam_{os.path.splitext(os.path.basename(image_path))[0]}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Analysis saved as '{filename}'")
        return prediction

def main():
    """Main function to demonstrate Grad-CAM functionality"""
    print("COVID-19 Chest X-ray Grad-CAM Visualization")
    print("=" * 50)

    # Initialize visualizer
    visualizer = GradCAMVisualizer()

    # Generate sample visualizations
    print("\\nGenerating Grad-CAM visualizations for sample images...")
    visualizer.visualize_sample_images(num_samples=4)

    # Analyze a specific image if available
    covid_dir = 'data/COVID'
    if os.path.exists(covid_dir):
        covid_files = [f for f in os.listdir(covid_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if covid_files:
            sample_covid = os.path.join(covid_dir, covid_files[0])
            print(f"\\nDetailed analysis of sample COVID image:")
            visualizer.analyze_single_image(sample_covid)

    print("\\n🔍 Grad-CAM visualization completed!")
    print("The model's attention areas show where it focuses to make COVID vs Normal decisions.")
    print("Red areas indicate high attention, blue areas indicate low attention.")

if __name__ == "__main__":
    main()