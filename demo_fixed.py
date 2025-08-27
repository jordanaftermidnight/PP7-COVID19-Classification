#!/usr/bin/env python3
"""
COVID-19 Classification Interactive Demo
Simple web interface for testing chest X-ray classification
Author: Jordanaftermidnight

Usage:
    python3 demo_fixed.py

Then open: http://localhost:8080
"""

from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

class COVID19Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(COVID19Classifier, self).__init__()
        self.resnet = models.resnet18(weights=None)
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

model = COVID19Classifier()
model.eval()

def demo_predict(image):
    """Demo prediction with realistic mock results"""
    img_array = np.array(image.convert('L'))
    darkness_factor = 1 - (img_array.mean() / 255.0)
    covid_base_prob = darkness_factor * 0.6 + random.uniform(0.1, 0.4)
    covid_prob = max(0.1, min(0.9, covid_base_prob))
    normal_prob = 1 - covid_prob
    
    predicted_class = 1 if covid_prob > 0.5 else 0
    confidence = covid_prob if predicted_class == 1 else normal_prob
    
    return {
        'predicted_class': predicted_class,
        'predicted_label': 'COVID-19' if predicted_class == 1 else 'Normal',
        'confidence': confidence,
        'probabilities': {
            'Normal': normal_prob,
            'COVID-19': covid_prob
        }
    }

@app.route('/')
def index():
    return '''<!DOCTYPE html>
<html>
<head>
    <title>COVID-19 X-Ray Classifier - DEMO</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f8f9fa; }
        .container { background: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .header { text-align: center; color: #2c3e50; margin-bottom: 30px; }
        .demo-banner { background: linear-gradient(45deg, #f39c12, #e67e22); color: white; padding: 15px; text-align: center; margin: 20px 0; border-radius: 10px; }
        .upload-section { border: 3px dashed #bdc3c7; border-radius: 15px; padding: 40px; text-align: center; margin: 20px 0; transition: all 0.3s; }
        .upload-section:hover { border-color: #3498db; background: #f8f9fa; }
        .btn { background: linear-gradient(45deg, #3498db, #2980b9); color: white; padding: 12px 25px; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; transition: all 0.3s; }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
        .results { margin-top: 30px; padding: 25px; border-radius: 15px; }
        .covid-positive { background: linear-gradient(135deg, #ffeaa7, #fdcb6e); border-left: 5px solid #e17055; }
        .covid-negative { background: linear-gradient(135deg, #d1f2eb, #81ecec); border-left: 5px solid #00b894; }
        .disclaimer { background: #fff3cd; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #ffc107; }
        .metrics { background: linear-gradient(135deg, #e8f4f8, #b2dfdb); padding: 20px; border-radius: 10px; margin: 20px 0; }
        .progress-bar { background: #ecf0f1; height: 25px; border-radius: 15px; margin: 8px 0; overflow: hidden; }
        .progress-fill { height: 100%; border-radius: 15px; transition: width 0.5s ease; }
        .covid-fill { background: linear-gradient(45deg, #e74c3c, #c0392b); }
        .normal-fill { background: linear-gradient(45deg, #2ecc71, #27ae60); }
        #loading { display: none; text-align: center; margin: 20px 0; color: #3498db; }
        .preview-img { max-width: 300px; max-height: 300px; border-radius: 10px; margin: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 COVID-19 Chest X-Ray Classifier</h1>
            <p>AI-powered analysis of chest X-ray images for COVID-19 detection</p>
        </div>
        
        <div class="demo-banner">
            <strong>🎬 DEMO MODE</strong> - Simulated predictions for demonstration purposes
        </div>
        
        <div class="metrics">
            <h3>📊 Model Performance Metrics</h3>
            <ul>
                <li><strong>Training Accuracy:</strong> 99.17%</li>
                <li><strong>COVID-19 Sensitivity:</strong> 100%</li>
                <li><strong>Normal Specificity:</strong> 95%</li>
                <li><strong>Architecture:</strong> ResNet-18 with Transfer Learning</li>
            </ul>
        </div>
        
        <div class="disclaimer">
            <strong>⚠️ Medical Disclaimer:</strong> This tool is for research and educational purposes only. 
            It is not intended for clinical diagnosis. Always consult healthcare professionals for medical advice.
        </div>
        
        <div class="upload-section" id="uploadSection">
            <h3>📤 Upload Chest X-Ray Image</h3>
            <input type="file" id="fileInput" accept="image/*" style="display: none;">
            <button type="button" id="uploadBtn" class="btn">Choose X-Ray Image</button>
            <p>Supported formats: JPG, PNG, JPEG, BMP</p>
            <div id="imagePreview"></div>
        </div>
        
        <div style="text-align: center; margin: 20px 0;">
            <button type="button" id="analyzeBtn" class="btn" disabled>🤖 Analyze Image</button>
        </div>
        
        <div id="loading">
            <h3>🔄 Analyzing image...</h3>
            <p>Please wait while our AI model processes your X-ray.</p>
        </div>
        
        <div id="results" style="display: none;"></div>
    </div>
    
    <script>
        let selectedFile = null;
        
        // File upload button click
        document.getElementById('uploadBtn').addEventListener('click', function() {
            document.getElementById('fileInput').click();
        });
        
        // File selection
        document.getElementById('fileInput').addEventListener('change', function(e) {
            selectedFile = e.target.files[0];
            console.log('File selected:', selectedFile ? selectedFile.name : 'none');
            
            if (selectedFile) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('imagePreview').innerHTML = 
                        '<h4>Selected: ' + selectedFile.name + '</h4>' +
                        '<img src="' + e.target.result + '" class="preview-img">';
                    
                    // Enable analyze button
                    document.getElementById('analyzeBtn').disabled = false;
                };
                reader.readAsDataURL(selectedFile);
            }
        });
        
        // Analyze button click
        document.getElementById('analyzeBtn').addEventListener('click', function() {
            console.log('Analyze button clicked!');
            
            if (!selectedFile) {
                alert('Please select an image file first!');
                return;
            }
            
            console.log('Processing file:', selectedFile.name);
            
            const formData = new FormData();
            formData.append('image', selectedFile);
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            
            console.log('Sending request to /predict');
            
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                console.log('Response status:', response.status);
                return response.json();
            })
            .then(data => {
                console.log('Response data:', data);
                document.getElementById('loading').style.display = 'none';
                displayResults(data);
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                console.error('Error:', error);
                alert('Error analyzing image: ' + error.message);
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
                    '• Continue regular health monitoring<br>• Follow standard preventive measures<br>• Consult doctor if symptoms develop') + '</p>' +
                '<div class="demo-banner" style="margin-top: 20px;"><small>⚠️ This is a DEMO prediction - not real medical analysis</small></div>';
            
            resultsDiv.style.display = 'block';
        }
    </script>
</body>
</html>'''

@app.route('/predict', methods=['POST'])
def predict():
    print("🔍 Predict endpoint called!")
    try:
        print(f"Request files: {list(request.files.keys())}")
        
        if 'image' not in request.files:
            print("❌ No image file in request")
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        print(f"📁 File received: {file.filename}")
        
        if file.filename == '':
            print("❌ Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        print("🖼️ Processing image...")
        image = Image.open(file.stream)
        print(f"Image size: {image.size}, mode: {image.mode}")
        
        prediction = demo_predict(image)
        print(f"✅ Prediction: {prediction}")
        
        return jsonify(prediction)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

if __name__ == '__main__':
    print("🎬 COVID-19 Classification DEMO Server - FIXED VERSION")
    print("=" * 60)
    print("✅ Demo server ready!")
    print("📱 Open http://localhost:8080 in your browser")
    print("🎯 Features: Upload images, get predictions, view metrics")
    print("⚠️  DEMO MODE: Uses simulated predictions")
    print("🔧 This version has fixed JavaScript issues")
    print()
    app.run(debug=True, host='0.0.0.0', port=8080)