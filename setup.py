#!/usr/bin/env python3
"""
COVID-19 Classification Project Setup Script
Automated setup and launcher for the COVID-19 chest X-ray classification project
Author: Jordanaftermidnight

This script helps users quickly set up and run the project with minimal effort.
"""

import os
import sys
import subprocess
import importlib
import platform

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", platform.python_version())
        print("Please upgrade Python and try again.")
        return False
    print(f"✅ Python {platform.python_version()} - Compatible")
    return True

def check_package(package_name, install_name=None):
    """Check if a package is installed"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def install_minimal_requirements():
    """Install minimal requirements for quick demo"""
    minimal_packages = [
        "flask",
        "torch", 
        "torchvision",
        "pillow",
        "numpy"
    ]
    
    print("\n🔧 Installing minimal requirements for quick demo...")
    failed_packages = []
    
    for package in minimal_packages:
        print(f"📦 Installing {package}...", end=" ")
        if install_package(package):
            print("✅")
        else:
            print("❌")
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n⚠️  Failed to install: {', '.join(failed_packages)}")
        print("Try installing manually: pip install " + " ".join(failed_packages))
        return False
    
    print("\n✅ Minimal requirements installed successfully!")
    return True

def install_full_requirements():
    """Install all requirements"""
    requirements_file = "requirements.txt"
    
    if not os.path.exists(requirements_file):
        print(f"❌ {requirements_file} not found!")
        return False
    
    print(f"\n🔧 Installing all requirements from {requirements_file}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print("✅ All requirements installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def check_demo_files():
    """Check if demo files exist"""
    demo_files = ["quick_demo.py", "web_interface.py", "flask_app.py"]
    missing_files = []
    
    for file in demo_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All demo files found")
    return True

def run_quick_demo():
    """Launch the quick demo"""
    print("\n🚀 Launching Quick Demo...")
    print("📱 Opening http://localhost:8080 in your browser")
    print("⚠️  Demo uses simulated predictions for demonstration")
    print("🔄 Press Ctrl+C to stop the demo")
    print()
    
    try:
        subprocess.run([sys.executable, "quick_demo.py"])
    except KeyboardInterrupt:
        print("\n👋 Demo stopped. Thanks for testing!")
    except Exception as e:
        print(f"❌ Error running demo: {e}")

def run_web_interface():
    """Launch the full web interface"""
    print("\n🚀 Launching Full Web Interface...")
    print("🧠 This uses the real trained model (99.17% accuracy)")
    print("🔍 Includes Grad-CAM visualization")
    print("🔄 Press Ctrl+C to stop")
    print()
    
    try:
        subprocess.run([sys.executable, "web_interface.py"])
    except KeyboardInterrupt:
        print("\n👋 Interface stopped.")
    except Exception as e:
        print(f"❌ Error running interface: {e}")

def show_menu():
    """Show the main menu"""
    print("\n" + "="*60)
    print("🔬 COVID-19 Chest X-Ray Classification Project")
    print("   Advanced Medical AI for COVID-19 Detection")
    print("   Author: Jordanaftermidnight")
    print("="*60)
    
    print("\n🎯 Choose your setup option:")
    print("1. 🎬 Quick Demo (30 seconds setup)")
    print("   - Minimal dependencies")
    print("   - Instant browser demo")
    print("   - Perfect for first-time users")
    
    print("\n2. 🔬 Full Setup (Advanced features)")
    print("   - All dependencies")
    print("   - Real model with Grad-CAM")
    print("   - Research-grade interface")
    
    print("\n3. ⚡ Quick Launch (if already set up)")
    print("   - Launch quick demo")
    print("   - Launch full interface")
    
    print("\n4. ❓ Help & Information")
    print("5. 🚪 Exit")

def main():
    """Main setup function"""
    if not check_python_version():
        return
    
    while True:
        show_menu()
        choice = input("\n👆 Enter your choice (1-5): ").strip()
        
        if choice == "1":
            print("\n🎬 Setting up Quick Demo...")
            if install_minimal_requirements() and check_demo_files():
                run_quick_demo()
        
        elif choice == "2":
            print("\n🔬 Setting up Full Environment...")
            if install_full_requirements() and check_demo_files():
                print("\n✅ Full setup complete!")
                print("\n🚀 You can now run:")
                print("  python3 quick_demo.py          # Quick demo")
                print("  python3 web_interface.py       # Streamlit interface")
                print("  python3 flask_app.py           # Flask interface")
                print("  python3 train_model.py         # Train model")
        
        elif choice == "3":
            print("\n⚡ Quick Launch Options:")
            print("1. Quick Demo")
            print("2. Full Interface")
            launch_choice = input("Choose (1-2): ").strip()
            
            if launch_choice == "1":
                if check_demo_files():
                    run_quick_demo()
            elif launch_choice == "2":
                if check_demo_files():
                    run_web_interface()
            else:
                print("Invalid choice")
        
        elif choice == "4":
            print("\n📚 Help & Information:")
            print("\n🎯 Project Overview:")
            print("  This is an advanced AI system for COVID-19 detection")
            print("  in chest X-ray images using deep learning.")
            
            print("\n🏆 Key Features:")
            print("  • 99.17% classification accuracy")
            print("  • Real-time web interface")
            print("  • Explainable AI with Grad-CAM")
            print("  • Multiple CNN architectures")
            
            print("\n🔗 Useful Links:")
            print("  GitHub: https://github.com/jordanaftermidnight/-PP7-COVID19-Classification")
            print("  Dataset: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database")
            
            print("\n⚠️  Medical Disclaimer:")
            print("  This tool is for educational/research purposes only.")
            print("  NOT intended for clinical diagnosis.")
            
            input("\n📖 Press Enter to continue...")
        
        elif choice == "5":
            print("\n👋 Thank you for using the COVID-19 Classification Project!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your Python installation and try again.")