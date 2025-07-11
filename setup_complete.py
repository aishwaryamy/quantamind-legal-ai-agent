#!/usr/bin/env python3
"""
Quantamind Legal AI Agent - Complete Setup Script
Downloads all required data and models automatically
"""

import os
import subprocess
import sys
from pathlib import Path
import urllib.request

def print_step(step, description):
    print(f"\n{'='*60}")
    print(f"STEP {step}: {description}")
    print('='*60)

def check_auth():
    """Check if user is authenticated with Hugging Face"""
    try:
        result = subprocess.run(['huggingface-cli', 'whoami'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Hugging Face authentication verified")
            return True
        else:
            print("❌ Hugging Face authentication required")
            return False
    except FileNotFoundError:
        print("❌ huggingface-cli not found")
        return False

def setup_directories():
    """Create required directories"""
    dirs = ['data', 'models', 'results']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Created directory: {dir_name}/")

def download_cuad_dataset():
    """Download CUAD dataset"""
    if os.path.exists("data/CUAD_v1.json"):
        print("✅ CUAD dataset already exists")
        return True
    
    print("📥 Downloading CUAD dataset...")
    try:
        # Try to download from GitHub (if publicly available)
        url = "https://github.com/TheAtticusProject/cuad/raw/main/data/CUAD_v1.json"
        urllib.request.urlretrieve(url, "data/CUAD_v1.json")
        print("✅ CUAD dataset downloaded successfully")
        return True
    except Exception as e:
        print("⚠️  Automatic download failed")
        print("📖 Manual download required:")
        print("   1. Visit: https://github.com/TheAtticusProject/cuad")
        print("   2. Download CUAD_v1.json")
        print("   3. Place it in the data/ directory")
        print("   4. Rerun this setup script")
        return False

def run_setup():
    print("🤖 QUANTAMIND LEGAL AI AGENT - COMPLETE SETUP")
    print("This will download ~20GB of data and may take 30-60 minutes")
    print("Ensure you have sufficient disk space and stable internet connection")
    
    # Step 1: Check authentication
    print_step(1, "Authentication Check")
    if not check_auth():
        print("\n❌ Setup cannot continue without authentication")
        print("🔧 To fix this:")
        print("   1. pip install huggingface_hub")
        print("   2. huggingface-cli login")
        print("   3. Get your token from: https://huggingface.co/settings/tokens")
        print("   4. Rerun this setup script")
        return False
    
    # Step 2: Create directories
    print_step(2, "Creating Directories")
    setup_directories()
    
    # Step 3: Download dataset
    print_step(3, "Dataset Download")
    if not download_cuad_dataset():
        return False
    
    # Step 4: Preprocess data
    print_step(4, "Data Preprocessing")
    if not os.path.exists("data/processed_cuad.json"):
        print("🔄 Running preprocessing...")
        result = subprocess.run([sys.executable, "preprocess.py"])
        if result.returncode != 0:
            print("❌ Preprocessing failed")
            return False
        print("✅ Data preprocessing completed")
    else:
        print("✅ Processed data already exists")
    
    # Step 5: Download model
    print_step(5, "Model Download (~15GB)")
    if not os.path.exists("models/mistral-7b"):
        print("📥 Downloading Mistral 7B model...")
        print("⏳ This may take 15-30 minutes depending on your internet speed")
        result = subprocess.run([sys.executable, "download_model.py"])
        if result.returncode != 0:
            print("❌ Model download failed")
            return False
        print("✅ Model download completed")
    else:
        print("✅ Model already exists")
    
    # Step 6: Create search index
    print_step(6, "Creating Search Index")
    if not os.path.exists("data/faiss_index"):
        print("🔄 Building FAISS index...")
        result = subprocess.run([sys.executable, "create_faiss_index.py"])
        if result.returncode != 0:
            print("❌ Index creation failed")
            return False
        print("✅ Search index created")
    else:
        print("✅ Search index already exists")
    
    # Success!
    print_step("🎉", "SETUP COMPLETE!")
    print("🚀 Your Quantamind Legal AI Agent is ready!")
    print("\n🧪 Test the system:")
    print("   python3 inference_fixed.py --clause-type exclusivity")
    print("   python3 inference_fixed.py --clause-type termination")
    print("   python3 inference_fixed.py --clause-type liability")
    
    print("\n📊 What you can do now:")
    print("   • Analyze legal contracts for various clause types")
    print("   • Extract exclusivity, termination, and liability clauses")
    print("   • Run completely offline (no cloud uploads)")
    print("   • Process sensitive legal documents securely")
    
    return True

if __name__ == "__main__":
    print("Starting Quantamind Legal AI Agent setup...")
    success = run_setup()
    if not success:
        print("\n❌ Setup incomplete. Please resolve the issues above and rerun:")
        print("   python3 setup_complete.py")
        sys.exit(1)
    else:
        print("\n🎯 Setup successful! Repository is now fully functional.")
