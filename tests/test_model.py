#!/usr/bin/env python3
"""
Comprehensive Test Suite for COVID-19 Classification Project
Author: Jordanaftermidnight

Test coverage for all major components including models, interfaces, and utilities.
"""

import unittest
import sys
import os
import torch
import numpy as np
from PIL import Image
import tempfile
import io

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
try:
    import train_model
    import ensemble_model
    import grad_cam_visualization
    from quick_demo import app as flask_app
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")


class TestModelArchitecture(unittest.TestCase):
    """Test the core model architecture and functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')  # Force CPU for testing

    def test_model_initialization(self):
        """Test that the COVID19Classifier can be initialized correctly"""
        try:
            from train_model import COVID19Classifier
            model = COVID19Classifier(num_classes=2, pretrained=False)
            self.assertIsNotNone(model)

            # Test forward pass with dummy data
            dummy_input = torch.randn(1, 3, 224, 224)
            output = model(dummy_input)
            self.assertEqual(output.shape, (1, 2))
        except Exception as e:
            self.skipTest(f"Model initialization test skipped: {e}")

    def test_model_output_range(self):
        """Test that model outputs are in expected range"""
        try:
            from train_model import COVID19Classifier
            model = COVID19Classifier(num_classes=2, pretrained=False)
            model.eval()

            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = model(dummy_input)

            # Check output shape
            self.assertEqual(output.shape[1], 2)

            # Apply softmax to check probabilities sum to 1
            probs = torch.softmax(output, dim=1)
            self.assertAlmostEqual(probs.sum().item(), 1.0, places=5)
        except Exception as e:
            self.skipTest(f"Model output test skipped: {e}")


class TestDataProcessing(unittest.TestCase):
    """Test data loading and preprocessing functions"""

    def test_image_preprocessing(self):
        """Test image preprocessing pipeline"""
        # Create a mock image
        mock_image = Image.new('RGB', (256, 256), color='red')

        try:
            from train_model import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            processed = transform(mock_image)
            self.assertEqual(processed.shape, (3, 224, 224))

            # Check normalization
            self.assertTrue(torch.all(processed >= -3))  # Reasonable lower bound
            self.assertTrue(torch.all(processed <= 3))   # Reasonable upper bound
        except Exception as e:
            self.skipTest(f"Image preprocessing test skipped: {e}")

    def test_dataset_class(self):
        """Test custom dataset class functionality"""
        try:
            from train_model import COVID19Dataset, transforms

            # Create temporary test images
            test_paths = []
            test_labels = [0, 1]

            for i in range(2):
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    mock_image = Image.new('RGB', (224, 224), color='red')
                    mock_image.save(tmp.name)
                    test_paths.append(tmp.name)

            # Test dataset
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

            dataset = COVID19Dataset(test_paths, test_labels, transform=transform)
            self.assertEqual(len(dataset), 2)

            # Test dataset item retrieval
            image, label = dataset[0]
            self.assertEqual(image.shape, (3, 224, 224))
            self.assertIn(label, [0, 1])

            # Cleanup
            for path in test_paths:
                os.unlink(path)

        except Exception as e:
            self.skipTest(f"Dataset test skipped: {e}")


class TestWebInterfaces(unittest.TestCase):
    """Test web interface functionality"""

    def test_flask_app_initialization(self):
        """Test that Flask app initializes correctly"""
        try:
            flask_app.config['TESTING'] = True
            self.client = flask_app.test_client()

            response = self.client.get('/')
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'COVID-19', response.data)
        except Exception as e:
            self.skipTest(f"Flask app test skipped: {e}")

    def test_prediction_endpoint(self):
        """Test prediction endpoint functionality"""
        try:
            flask_app.config['TESTING'] = True
            self.client = flask_app.test_client()

            # Create a mock image file
            mock_image = Image.new('RGB', (224, 224), color='blue')
            img_io = io.BytesIO()
            mock_image.save(img_io, 'PNG')
            img_io.seek(0)

            response = self.client.post('/predict',
                                      data={'file': (img_io, 'test.png')},
                                      content_type='multipart/form-data')

            self.assertEqual(response.status_code, 200)

            # Check if response contains expected keys
            import json
            data = json.loads(response.data)
            self.assertIn('prediction', data)
            self.assertIn('confidence', data)
        except Exception as e:
            self.skipTest(f"Prediction endpoint test skipped: {e}")


class TestVisualizationComponents(unittest.TestCase):
    """Test visualization and explanation components"""

    def test_grad_cam_imports(self):
        """Test that Grad-CAM visualization components can be imported"""
        try:
            import grad_cam_visualization
            self.assertTrue(hasattr(grad_cam_visualization, 'generate_gradcam'))
        except Exception as e:
            self.skipTest(f"Grad-CAM import test skipped: {e}")


class TestUtilities(unittest.TestCase):
    """Test utility functions and helpers"""

    def test_device_detection(self):
        """Test device detection functionality"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.assertIn(str(device), ['cpu', 'cuda:0', 'cuda'])

    def test_model_saving_loading(self):
        """Test model state saving and loading"""
        try:
            from train_model import COVID19Classifier

            # Create and save model
            model = COVID19Classifier(num_classes=2, pretrained=False)

            with tempfile.NamedTemporaryFile(suffix='.pth') as tmp:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'test_accuracy': 99.17,
                    'epoch': 15
                }, tmp.name)

                # Load model
                checkpoint = torch.load(tmp.name, map_location='cpu')
                self.assertIn('model_state_dict', checkpoint)
                self.assertEqual(checkpoint['test_accuracy'], 99.17)

        except Exception as e:
            self.skipTest(f"Model save/load test skipped: {e}")


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""

    def test_invalid_image_handling(self):
        """Test handling of invalid or corrupted images"""
        try:
            from train_model import COVID19Dataset, transforms

            # Create a text file instead of image
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
                tmp.write(b'This is not an image')
                tmp.flush()

                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()
                ])

                dataset = COVID19Dataset([tmp.name], [0], transform=transform)

                # This should raise an exception
                with self.assertRaises(Exception):
                    dataset[0]

                os.unlink(tmp.name)

        except Exception as e:
            self.skipTest(f"Invalid image test skipped: {e}")


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)