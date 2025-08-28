#!/usr/bin/env python3
"""
Flask Web Interface for COVID-19 Chest X-Ray Classification
Simple upload and predict interface
Author: Jordanaftermidnight
"""

from flask import Flask, render_template, request, jsonify, send_file
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import base64
import io
import json
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

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

# Global variables
model = None
device = None
transform = None

def load_model():
    """Load the trained model"""
    global model, device, transform

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
                print(f"Model loaded from {model_path}")
                break
            except Exception as e:
                print(f"Failed to load {model_path}: {e}")

    if not model_loaded:
        print("No trained model found!")
        return False

    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return True

def preprocess_image(image):
    """Preprocess uploaded image"""
    if image.mode != 'RGB':
        image = image.convert('RGB')

    input_tensor = transform(image).unsqueeze(0)
    return input_tensor

def predict_image(input_tensor):
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

@app.route('/')
def index():
    """Main page"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>COVID-19 X-Ray Classifier</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .header { text-align: center; color: #2c3e50; margin-bottom: 30px; }
            .upload-area { border: 2px dashed #bdc3c7; border-radius: 10px; padding: 40px; text-align: center; margin: 20px 0; }
            .upload-area:hover { border-color: #3498db; }
            .btn { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            .btn:hover { background: #2980b9; }
            .results { margin-top: 30px; padding: 20px; border-radius: 10px; }
            .covid-positive { background: #ffeaa7; border-left: 5px solid #fdcb6e; }
            .covid-negative { background: #d1f2eb; border-left: 5px solid #00b894; }
            .disclaimer { background: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0; font-size: 14px; }
            .progress-bar { background: #ecf0f1; height: 20px; border-radius: 10px; margin: 5px 0; }
            .progress-fill { height: 100%; border-radius: 10px; }
            .covid-fill { background: #e74c3c; }
            .normal-fill { background: #2ecc71; }
            #loading { display: none; text-align: center; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔬 COVID-19 Chest X-Ray Classifier</h1>
            <p>AI-powered analysis of chest X-ray images for COVID-19 detection</p>
        </div>

        <div class="disclaimer">
            <strong>⚠️ Medical Disclaimer:</strong> This tool is for research and educational purposes only.
            It is not intended for clinical diagnosis. Always consult healthcare professionals for medical advice.
        </div>

        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-area">
                <h3>📤 Upload Chest X-Ray Image</h3>
                <input type="file" id="imageFile" name="image" accept="image/*" style="display: none;">
                <button type="button" onclick="document.getElementById('imageFile').click();" class="btn">
                    Choose X-Ray Image
                </button>
                <p>Supported formats: JPG, PNG, JPEG, BMP</p>
            </div>
            <div style="text-align: center;">
                <button type="submit" class="btn">🤖 Analyze Image</button>
            </div>
        </form>

        <div id="loading">
            <h3>🔄 Analyzing image...</h3>
            <p>Please wait while our AI model processes your X-ray.</p>
        </div>

        <div id="results" style="display: none;"></div>

        <script>
            document.getElementById('imageFile').addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        document.querySelector('.upload-area').innerHTML =
                            '<h3>Selected: ' + file.name + '</h3>' +
                            '<img src="' + e.target.result + '" style="max-width: 300px; max-height: 300px;">';
                    };
                    reader.readAsDataURL(file);
                }
            });

            document.getElementById('uploadForm').addEventListener('submit', function(e) {
                e.preventDefault();
                const formData = new FormData();
                const imageFile = document.getElementById('imageFile').files[0];

                if (!imageFile) {
                    alert('Please select an image file first!');
                    return;
                }

                formData.append('image', imageFile);

                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').style.display = 'none';

                fetch('/predict', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('loading').style.display = 'none';
                    displayResults(data);
                })
                .catch(error => {
                    document.getElementById('loading').style.display = 'none';
                    console.error('Error:', error);
                    alert('Error analyzing image. Please try again.');
                });
            });

            function displayResults(data) {
                const resultsDiv = document.getElementById('results');
                const isCovidPositive = data.predicted_label === 'COVID-19';

                resultsDiv.className = 'results ' + (isCovidPositive ? 'covid-positive' : 'covid-negative');
                resultsDiv.innerHTML =
                    '<h2>🎯 Analysis Results</h2>' +
                    '<h3>' + (isCovidPositive ? '🦠' : '✅') + ' Prediction: ' + data.predicted_label + '</h3>' +
                    '<p><strong>Confidence:</strong> ' + (data.confidence * 100).toFixed(1) + '%</p>' +
                    '<h4>📊 Probability Breakdown:</h4>' +
                    '<div>Normal: <div class="progress-bar"><div class="progress-fill normal-fill" style="width: ' + (data.probabilities.Normal * 100) + '%"></div></div> ' + (data.probabilities.Normal * 100).toFixed(1) + '%</div>' +
                    '<div>COVID-19: <div class="progress-bar"><div class="progress-fill covid-fill" style="width: ' + (data.probabilities["COVID-19"] * 100) + '%"></div></div> ' + (data.probabilities["COVID-19"] * 100).toFixed(1) + '%</div>' +
                    '<h4>💡 Recommendations:</h4>' +
                    '<p>' + (isCovidPositive ?
                        '• Consult healthcare provider immediately<br>• Consider RT-PCR testing<br>• Follow isolation protocols' :
                        '• Continue regular health monitoring<br>• Follow standard preventive measures<br>• Consult doctor if symptoms develop') + '</p>';

                resultsDiv.style.display = 'block';
            }
        </script>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Load and preprocess image
        image = Image.open(file.stream)
        input_tensor = preprocess_image(image)

        # Make prediction
        prediction = predict_image(input_tensor)

        return jsonify(prediction)

    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else 'not loaded'
    })

if __name__ == '__main__':
    print("Loading COVID-19 classification model...")
    if load_model():
        print("✅ Model loaded successfully!")
        print("🌐 Starting Flask server...")
        print("📱 Open http://localhost:5000 in your browser")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model. Please run training first.")