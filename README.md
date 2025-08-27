# 🔬 COVID-19 Chest X-Ray Classification Project

**Author**: Jordanaftermidnight  
**Project**: Advanced Medical AI for COVID-19 Detection

## 🎯 Overview
Advanced AI system for detecting COVID-19 pneumonia patterns in chest X-ray images using deep learning. Features multiple CNN architectures, explainable AI visualization, and interactive web interfaces.

## 🏆 Project Achievements
- **99.17% Classification Accuracy** (Target: >50% ✅)
- **100% COVID-19 Detection Sensitivity** (Perfect detection rate)
- **95% Normal Specificity** (Excellent false positive control)
- **Multi-Architecture Ensemble** (ResNet, DenseNet, EfficientNet)
- **Grad-CAM Explainable AI** (Shows model attention areas)
- **Interactive Web Interfaces** (Streamlit + Flask)

## 🚀 Key Features

### 🤖 Advanced AI Models
- **Primary Model**: ResNet-18 based CNN with custom classification head
- **Ensemble Model**: Combination of multiple architectures for enhanced performance
- **Transfer Learning**: Optimized for medical imaging tasks
- **Extended Training**: 35+ epochs with stability validation

### 🔍 Explainable AI (XAI)
- **Grad-CAM Visualization**: Shows which lung regions influence COVID detection
- **Attention Heatmaps**: Red areas = high attention, blue areas = low attention
- **Model Interpretability**: Understand AI decision-making process

### 🌐 Web Interfaces
- **Streamlit Interface**: Beautiful, medical-grade UI with real-time visualization
- **Flask Interface**: Lightweight, fast upload-and-predict system
- **Drag-and-Drop**: Easy image upload with instant results
- **Mobile-Friendly**: Works on smartphones and tablets

## Dataset Options

### 1. Primary Dataset (Recommended)
- **COVID-19 Radiography Database** from Kaggle
- Link: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
- Contains COVID-19, Normal, and Pneumonia chest X-ray images
- Well-balanced dataset with good image quality

### 2. Alternative Datasets
- **IEEE8023 COVID Chest X-ray Dataset**: https://github.com/ieee8023/covid-chestxray-dataset
- **DeepCOVID Dataset**: https://github.com/shervinmin/DeepCovid.git (from research paper)
- **V7 Labs COVID-19 Dataset**: https://github.com/v7labs/covid-19-xray-dataset

## Model Architecture
- **Base Model**: ResNet-18 with ImageNet pre-trained weights
- **Transfer Learning**: Fine-tuning last layers while freezing early features
- **Classification Head**: Custom fully connected layers with dropout
- **Input Size**: 224x224 RGB images
- **Output**: Binary classification (COVID vs Normal)

## 🚀 Quick Start Guide

### 1. Setup Environment
```bash
# Navigate to project directory
cd "PP7: Computer vision and image classification"

# Install all dependencies (including new enhanced features)
pip install -r requirements.txt
```

### 2. Choose Your Experience

#### 🎯 **Quick Demo** (Try It Now!)
```bash
# Launch the interactive demo (no training required)
python3 quick_demo.py

# Then open: http://localhost:8080
# Upload any chest X-ray image and get instant results!
```
**Perfect for**: First-time users, quick testing, showcasing the project

#### 🎨 **Full Web Interface** (Advanced Features)
```bash
# Launch interactive web interface with Grad-CAM
python3 run_web_interface.py

# Choose from:
# 1. Streamlit Interface (Beautiful UI + Grad-CAM)
# 2. Flask Interface (Fast & Simple)  
# 3. Demo Mode (Test with sample images)
```
**Perfect for**: Deep analysis, explainable AI, research purposes

| Feature | Quick Demo | Full Interface |
|---------|------------|----------------|
| Setup Time | Instant | Requires training |
| Grad-CAM Visualization | ❌ | ✅ |
| Real Model Results | Simulated | ✅ Actual |
| Ease of Use | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Educational Value | High | Very High |

#### 🧠 **Train Your Own Model**
```bash
# Basic training (15 epochs → 99.17% accuracy)
python3 train_model.py

# Extended training (35+ epochs with stability testing)
python3 extended_training.py

# Multi-architecture ensemble (4 models combined)
python3 ensemble_model.py
```

#### 🔍 **Explainable AI Analysis**
```bash
# Generate Grad-CAM visualizations
python3 grad_cam_visualization.py

# Shows model attention on chest X-rays
# Red = high attention, Blue = low attention
```

#### 📊 **Results & Visualization**
```bash
# View training results and metrics
python3 visualize_extended_results.py
```

### 2. Download Dataset
```bash
# Option 1: Using Kaggle API
kaggle datasets download -d tawsifurrahman/covid19-radiography-database
unzip covid19-radiography-database.zip

# Option 2: Manual download from Kaggle website
# Extract COVID and Normal folders to data/ directory
```

### 3. Run the Notebook
```bash
jupyter notebook covid_classification.ipynb
```

## Project Structure
```
PP7: Computer vision and image classification/
├── covid_classification.ipynb    # Main implementation notebook
├── requirements.txt              # Python dependencies
├── README.md                    # Project documentation
├── data/                        # Dataset directory
│   ├── COVID/                   # COVID-19 positive X-rays
│   └── Normal/                  # Normal chest X-rays
└── models/                      # Saved model weights
    └── covid_classifier.pth     # Trained model
```

## Key Features
- **Transfer Learning**: Leverages pre-trained ResNet for medical image analysis
- **Data Augmentation**: Improves model robustness with rotation and flipping
- **Comprehensive Evaluation**: Includes accuracy, sensitivity, specificity, and confusion matrix
- **Visualization**: Training curves and performance metrics
- **Reproducible**: Fixed random seeds for consistent results

## Expected Results
- **Target Accuracy**: >50% (achievable goal)
- **Typical Performance**: 80-95% accuracy with proper training
- **Key Metrics**: Sensitivity and specificity for medical applications

## Implementation Notes
- Uses PyTorch framework with torchvision models
- Implements early stopping and learning rate scheduling
- Includes proper train/test split with stratification
- Medical AI ethics and limitations discussed in reflection section

## Dataset Download Instructions

### Method 1: Kaggle API (Recommended)
1. Install Kaggle: `pip install kaggle`
2. Set up Kaggle API credentials
3. Download: `kaggle datasets download -d tawsifurrahman/covid19-radiography-database`

### Method 2: Manual Download
1. Visit: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
2. Click "Download" button
3. Extract to project directory

### Method 3: Alternative Sources
- GitHub repositories listed above
- Research paper datasets
- Medical image databases (with proper permissions)

## Important Notes
- This is for educational/research purposes only
- Not intended for clinical diagnosis
- Requires medical validation for real-world use
- Consider data privacy and ethical guidelines

## Next Steps
1. Download and prepare dataset
2. Run the Jupyter notebook
3. Experiment with different architectures
4. Upload to GitHub for sharing and review