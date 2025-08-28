#!/usr/bin/env python3
"""
Script to download COVID-19 chest X-ray dataset from Kaggle
"""

import os
import sys
import subprocess
import zipfile

def check_kaggle_installation():
    """Check if Kaggle is installed"""
    try:
        import kaggle
        return True
    except ImportError:
        return False

def install_kaggle():
    """Install Kaggle package"""
    print("Installing Kaggle...")
    subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)
    print("Kaggle installed successfully!")

def download_dataset():
    """Download the COVID-19 radiography database"""
    dataset_name = "tawsifurrahman/covid19-radiography-database"
    print(f"Downloading dataset: {dataset_name}")

    try:
        # Download dataset
        result = subprocess.run([
            "kaggle", "datasets", "download", "-d", dataset_name
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error downloading dataset: {result.stderr}")
            return False

        # Extract dataset
        zip_file = "covid19-radiography-database.zip"
        if os.path.exists(zip_file):
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(".")

            # Clean up zip file
            os.remove(zip_file)
            print("Dataset extracted successfully!")

            # Create organized structure
            organize_dataset()
            return True
        else:
            print("Dataset zip file not found")
            return False

    except Exception as e:
        print(f"Error: {e}")
        return False

def organize_dataset():
    """Organize dataset into proper structure"""
    print("Organizing dataset structure...")

    # Create data directories
    os.makedirs("data", exist_ok=True)

    # Move folders if they exist
    source_dirs = ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]
    for dir_name in source_dirs:
        if os.path.exists(dir_name):
            if dir_name in ["COVID", "Normal"]:
                # These are our target classes
                target = f"data/{dir_name}"
                if not os.path.exists(target):
                    os.rename(dir_name, target)
                    print(f"Moved {dir_name} to {target}")
            else:
                print(f"Found additional class: {dir_name} (not using for binary classification)")

def main():
    """Main function"""
    print("COVID-19 Dataset Download Script")
    print("=" * 40)

    # Check if Kaggle is installed
    if not check_kaggle_installation():
        try:
            install_kaggle()
        except Exception as e:
            print(f"Failed to install Kaggle: {e}")
            print("Please install manually: pip install kaggle")
            return

    # Check if Kaggle credentials are set up
    kaggle_dir = os.path.expanduser("~/.kaggle")
    if not os.path.exists(f"{kaggle_dir}/kaggle.json"):
        print("Kaggle API credentials not found!")
        print("Please set up your Kaggle credentials:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Place kaggle.json in ~/.kaggle/")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return

    # Download and organize dataset
    if download_dataset():
        print("\nDataset setup complete!")
        print("You can now run the Jupyter notebook: covid_classification.ipynb")
    else:
        print("\nDataset download failed.")
        print("Please try downloading manually from:")
        print("https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database")

if __name__ == "__main__":
    main()