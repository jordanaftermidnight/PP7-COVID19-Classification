#!/usr/bin/env python3
"""
Streamlit Web Interface for COVID-19 Chest X-Ray Classification
Simple drag-and-drop interface with Grad-CAM visualization
Author: Jordanaftermidnight
"""

import streamlit as st
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
    st.warning("⚠️ Grad-CAM not available. Install with: pip install pytorch-grad-cam")
    GRADCAM_AVAILABLE = False
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
import io
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="COVID-19 X-Ray Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

@st.cache_resource
def load_model():
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = COVID19Classifier()

    # Try to load extended model first, then fallback to original
    model_paths = [
        'models/covid_classifier_extended.pth',
        'models/covid_classifier.pth'
    ]

    model_loaded = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
                model.eval()
                model_loaded = True
                st.success(f"✅ Model loaded from {model_path}")
                break
            except Exception as e:
                st.warning(f"Failed to load {model_path}: {e}")

    if not model_loaded:
        st.error("❌ No trained model found! Please run training first.")
        return None, None

    return model, device

@st.cache_data
def preprocess_image(image):
    """Preprocess uploaded image"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize for display
    display_image = image.resize((224, 224))

    # Transform for model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0)
    original_image = np.array(display_image) / 255.0

    return input_tensor, original_image, display_image

def predict_image(model, device, input_tensor):
    """Make prediction on image"""
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

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

    return prediction_info

def generate_gradcam(model, device, input_tensor, original_image, target_class=None):
    """Generate Grad-CAM visualization"""
    try:
        # Setup Grad-CAM
        target_layers = [model.resnet.layer4[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)

        # Generate CAM
        if target_class is None:
            # Use predicted class
            with torch.no_grad():
                output = model(input_tensor.to(device))
                target_class = torch.argmax(output, dim=1).item()

        targets = [ClassifierOutputTarget(target_class)]
        input_tensor = input_tensor.to(device)
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        # Create visualization
        visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)

        return visualization, grayscale_cam
    except Exception as e:
        st.error(f"Error generating Grad-CAM: {e}")
        return None, None

def main():
    # Title and description
    st.title("🔬 COVID-19 Chest X-Ray Classifier")
    st.markdown("""
    This AI model analyzes chest X-ray images to detect potential COVID-19 pneumonia patterns.

    **⚠️ Medical Disclaimer**: This is for research/educational purposes only.
    Not intended for clinical diagnosis. Always consult healthcare professionals.
    """)

    # Sidebar
    st.sidebar.header("📋 Model Information")
    st.sidebar.info("""
    **Architecture**: ResNet-18 based CNN
    **Training Accuracy**: 99.17%
    **Sensitivity**: 100% (COVID detection)
    **Specificity**: 95% (Normal detection)

    **Features**:
    - Deep learning classification
    - Grad-CAM attention visualization
    - Confidence scoring
    """)

    # Load model
    model, device = load_model()
    if model is None:
        st.stop()

    # File upload
    st.header("📤 Upload Chest X-Ray Image")
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image...",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload a chest X-ray image in JPG, PNG, or other common formats"
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded X-Ray Image", use_column_width=True)

        # Preprocess
        with st.spinner("Processing image..."):
            input_tensor, original_image, display_image = preprocess_image(image)

        # Make prediction
        with st.spinner("Making prediction..."):
            prediction = predict_image(model, device, input_tensor)

        # Display results
        st.header("🎯 Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Classification")

            # Prediction with confidence
            pred_label = prediction['predicted_label']
            confidence = prediction['confidence']

            if pred_label == 'COVID-19':
                st.error(f"🦠 **{pred_label}** detected")
            else:
                st.success(f"✅ **{pred_label}** classification")

            st.metric("Confidence", f"{confidence:.1%}")

            # Probability breakdown
            st.subheader("Probability Breakdown")
            for class_name, prob in prediction['probabilities'].items():
                st.progress(prob, text=f"{class_name}: {prob:.1%}")

        with col2:
            st.subheader("Risk Assessment")

            covid_prob = prediction['probabilities']['COVID-19']

            if covid_prob > 0.8:
                risk_level = "🔴 HIGH"
                risk_text = "Strong COVID-19 patterns detected"
                risk_color = "red"
            elif covid_prob > 0.5:
                risk_level = "🟡 MODERATE"
                risk_text = "Some COVID-19 patterns present"
                risk_color = "orange"
            else:
                risk_level = "🟢 LOW"
                risk_text = "Normal lung patterns observed"
                risk_color = "green"

            st.markdown(f"**Risk Level**: {risk_level}")
            st.markdown(f"*{risk_text}*")

            # Recommendations
            st.subheader("Recommendations")
            if covid_prob > 0.5:
                st.warning("""
                - Consult healthcare provider immediately
                - Consider RT-PCR testing
                - Follow isolation protocols
                - Monitor symptoms closely
                """)
            else:
                st.info("""
                - Continue regular health monitoring
                - Follow standard preventive measures
                - Consult doctor if symptoms develop
                """)

        # Grad-CAM Visualization
        st.header("🔍 AI Attention Visualization (Grad-CAM)")
        st.markdown("""
        This visualization shows which areas of the X-ray the AI model focused on to make its decision.
        Red areas indicate high attention, blue areas indicate low attention.
        """)

        with st.spinner("Generating attention map..."):
            cam_visualization, heatmap = generate_gradcam(
                model, device, input_tensor, original_image
            )

        if cam_visualization is not None:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Original X-Ray")
                st.image(display_image, use_column_width=True)

            with col2:
                st.subheader("Attention Heatmap")
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(heatmap, cmap='jet')
                ax.axis('off')
                st.pyplot(fig)
                plt.close()

            with col3:
                st.subheader("Overlay Visualization")
                st.image(cam_visualization, use_column_width=True)

        # Technical Details (Expandable)
        with st.expander("🔧 Technical Details"):
            st.json({
                "Model Architecture": "ResNet-18 based CNN",
                "Input Size": "224x224 RGB",
                "Predicted Class": prediction['predicted_label'],
                "Confidence Score": f"{confidence:.4f}",
                "COVID-19 Probability": f"{prediction['probabilities']['COVID-19']:.4f}",
                "Normal Probability": f"{prediction['probabilities']['Normal']:.4f}",
                "Device": str(device).upper()
            })

    else:
        # Instructions when no file uploaded
        st.info("""
        👆 **Upload a chest X-ray image to get started!**

        **How to use**:
        1. Click "Browse files" above
        2. Select a chest X-ray image (JPG, PNG, etc.)
        3. Wait for AI analysis
        4. View prediction results and attention visualization

        **Sample results you'll see**:
        - 🎯 Classification (COVID-19 or Normal)
        - 📊 Confidence scores and probabilities
        - 🔍 Grad-CAM visualization showing AI attention
        - 💡 Medical recommendations (for educational purposes)
        """)

        # Sample images (if available)
        sample_covid = "data/COVID"
        sample_normal = "data/Normal"

        if os.path.exists(sample_covid) or os.path.exists(sample_normal):
            st.subheader("📸 Try with Sample Images")
            st.info("You can test with images from the training dataset in the data/ folder")

if __name__ == "__main__":
    main()