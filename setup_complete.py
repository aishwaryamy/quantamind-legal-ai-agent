#!/usr/bin/env python3
"""
Automated setup script for Quantamind Legal AI Agent
This script handles all downloads and setup automatically.
"""

import os
import sys
import json
import requests
import zipfile
import shutil
from pathlib import Path
import subprocess
import time
from urllib.parse import urlparse
from tqdm import tqdm

def print_step(step_num, total_steps, description):
    """Print formatted step information"""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}/{total_steps}: {description}")
    print(f"{'='*60}")

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        sys.exit(1)
    print(f"✅ Python version: {sys.version}")

def check_disk_space():
    """Check available disk space"""
    try:
        statvfs = os.statvfs('.')
        available_space = statvfs.f_frsize * statvfs.f_bavail
        available_gb = available_space / (1024**3)
        
        if available_gb < 20:
            print(f"❌ Insufficient disk space. Available: {available_gb:.1f}GB, Required: 20GB")
            sys.exit(1)
        print(f"✅ Disk space: {available_gb:.1f}GB available")
    except:
        print("⚠️  Could not check disk space. Proceeding anyway...")

def create_directories():
    """Create required directories"""
    dirs = ['data', 'models', 'results', 'logs']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
    print("✅ Created directories:", ", ".join(dirs))

def install_requirements():
    """Install Python requirements"""
    print("Installing Python dependencies...")
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Requirements installed successfully")
        else:
            print(f"❌ Error installing requirements: {result.stderr}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error installing requirements: {e}")
        sys.exit(1)

def download_file_with_progress(url, filename, description):
    """Download file with progress bar"""
    print(f"Downloading {description}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as file, tqdm(
            desc=description,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✅ Downloaded {description}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {description}: {e}")
        return False

def download_cuad_dataset():
    """Download CUAD dataset automatically"""
    cuad_file = Path('data/CUAD_v1.json')
    
    if cuad_file.exists():
        print("✅ CUAD dataset already exists")
        return True
    
    # Try multiple sources for CUAD dataset
    cuad_urls = [
        "https://zenodo.org/records/4595826/files/CUAD_v1.json",
        "https://github.com/TheAtticusProject/cuad/raw/main/CUAD_v1.json"
    ]
    
    for url in cuad_urls:
        print(f"Attempting to download CUAD dataset from: {url}")
        if download_file_with_progress(url, cuad_file, "CUAD Dataset"):
            return True
    
    # If automatic download fails, create sample data
    print("⚠️  Could not download full CUAD dataset. Creating sample data...")
    return create_sample_data()

def create_sample_data():
    """Create sample data for testing"""
    sample_data = {
        "data": [
            {
                "title": "Software License Agreement",
                "paragraphs": [
                    {
                        "text": "This Agreement grants Company exclusive rights to distribute the Product in the Territory. During the Term, Company shall not compete with similar products or engage in any activities that would conflict with this exclusivity arrangement.",
                        "labels": ["Exclusivity", "Non-Compete"]
                    },
                    {
                        "text": "Either party may terminate this Agreement with thirty (30) days written notice. Upon termination, all rights and obligations shall cease, except for those provisions that by their nature should survive termination.",
                        "labels": ["Termination", "Notice Period"]
                    },
                    {
                        "text": "Company's liability under this Agreement shall not exceed the total amount paid by Client in the twelve (12) months preceding the claim. In no event shall Company be liable for indirect, incidental, or consequential damages.",
                        "labels": ["Liability", "Limitation of Liability"]
                    }
                ]
            }
        ]
    }
    
    with open('data/CUAD_v1.json', 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print("✅ Created sample CUAD dataset for testing")
    return True

def check_huggingface_auth():
    """Check Hugging Face authentication"""
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✅ Hugging Face authenticated as: {user_info['name']}")
        return True
    except Exception as e:
        print("❌ Hugging Face authentication failed.")
        print("Please run: huggingface-cli login")
        print("Get your token from: https://huggingface.co/settings/tokens")
        return False

def download_model():
    """Download Mistral model"""
    print("Downloading Mistral 7B model (this may take 15-30 minutes)...")
    
    try:
        result = subprocess.run([
            sys.executable, 'download_model.py'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Model downloaded successfully")
            return True
        else:
            print(f"❌ Error downloading model: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

def run_preprocessing():
    """Run data preprocessing"""
    print("Processing dataset...")
    
    try:
        result = subprocess.run([
            sys.executable, 'preprocess.py'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dataset processed successfully")
            return True
        else:
            print(f"❌ Error processing dataset: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing dataset: {e}")
        return False

def create_search_index():
    """Create FAISS search index"""
    print("Creating search index...")
    
    try:
        result = subprocess.run([
            sys.executable, 'create_faiss_index.py'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Search index created successfully")
            return True
        else:
            print(f"❌ Error creating search index: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error creating search index: {e}")
        return False

def run_validation():
    """Run validation test"""
    print("Running validation test...")
    
    try:
        result = subprocess.run([
            sys.executable, 'validate_installation.py'
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Validation test passed")
            print(result.stdout)
            return True
        else:
            print(f"❌ Validation test failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Validation test timed out")
        return False
    except Exception as e:
        print(f"❌ Error running validation: {e}")
        return False

def main():
    """Main setup function"""
    start_time = time.time()
    
    print("🚀 Quantamind Legal AI Agent - Automated Setup")
    print("This will download ~15GB of data and may take 30-60 minutes.")
    
    response = input("\nContinue with setup? (y/n): ")
    if response.lower() != 'y':
        print("Setup cancelled.")
        sys.exit(0)
    
    steps = [
        ("System Check", lambda: (check_python_version(), check_disk_space())),
        ("Create Directories", create_directories),
        ("Install Requirements", install_requirements),
        ("Download CUAD Dataset", download_cuad_dataset),
        ("Check Hugging Face Auth", check_huggingface_auth),
        ("Download Mistral Model", download_model),
        ("Process Dataset", run_preprocessing),
        ("Create Search Index", create_search_index),
        ("Run Validation", run_validation)
    ]
    
    failed_steps = []
    
    for i, (description, func) in enumerate(steps, 1):
        print_step(i, len(steps), description)
        
        try:
            result = func()
            if result is False:
                failed_steps.append(description)
        except Exception as e:
            print(f"❌ Error in {description}: {e}")
            failed_steps.append(description)
    
    # Final summary
    duration = time.time() - start_time
    print(f"\n{'='*60}")
    print("SETUP COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {duration/60:.1f} minutes")
    
    if failed_steps:
        print(f"❌ Failed steps: {', '.join(failed_steps)}")
        print("Please check the logs above and retry failed steps manually.")
    else:
        print("✅ All steps completed successfully!")
        print("\nTo test the system, run:")
        print("python3 inference_fixed.py --clause-type exclusivity")
        print("\nFor CPU-only inference, use:")
        print("python3 inference_fixed.py --clause-type exclusivity --cpu-only")

if __name__ == "__main__":
    main()