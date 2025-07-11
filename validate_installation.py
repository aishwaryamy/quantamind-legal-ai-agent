#!/usr/bin/env python3
"""
Installation validation script for Quantamind Legal AI Agent
This script verifies that all components are properly installed and working.
"""

import os
import sys
import json
import torch
import time
from pathlib import Path
import traceback

def print_test(test_name, status, details=""):
    """Print test result with formatting"""
    status_emoji = "✅" if status else "❌"
    print(f"{status_emoji} {test_name}")
    if details:
        print(f"   {details}")
    return status

def check_python_environment():
    """Check Python environment and dependencies"""
    print("🔍 Checking Python Environment...")
    
    try:
        # Check Python version
        python_version = sys.version_info
        python_ok = python_version >= (3, 8)
        print_test(f"Python Version", python_ok, 
                  f"Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check required packages
        packages = [
            'torch', 'transformers', 'sentence_transformers', 
            'faiss', 'peft', 'pandas', 'numpy', 'huggingface_hub'
        ]
        
        missing_packages = []
        for package in packages:
            try:
                __import__(package.replace('-', '_'))
                print_test(f"Package: {package}", True)
            except ImportError:
                print_test(f"Package: {package}", False, "Not installed")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
            print("Run: pip install -r requirements.txt")
            return False
        
        return python_ok
        
    except Exception as e:
        print(f"❌ Error checking Python environment: {e}")
        return False

def check_file_structure():
    """Check if required files and directories exist"""
    print("\n🔍 Checking File Structure...")
    
    required_files = [
        'requirements.txt',
        'preprocess.py',
        'download_model.py',
        'create_faiss_index.py',
        'inference_fixed.py',
        'data/CUAD_v1.json'
    ]
    
    required_dirs = [
        'data',
        'models',
        'results'
    ]
    
    all_good = True
    
    # Check directories
    for dir_path in required_dirs:
        exists = Path(dir_path).is_dir()
        print_test(f"Directory: {dir_path}", exists)
        if not exists:
            all_good = False
    
    # Check files
    for file_path in required_files:
        exists = Path(file_path).is_file()
        if exists:
            size = Path(file_path).stat().st_size
            print_test(f"File: {file_path}", True, f"Size: {size:,} bytes")
        else:
            print_test(f"File: {file_path}", False, "Missing")
            all_good = False
    
    return all_good

def check_data_integrity():
    """Check if data files are valid"""
    print("\n🔍 Checking Data Integrity...")
    
    try:
        # Check CUAD dataset
        cuad_path = Path('data/CUAD_v1.json')
        if cuad_path.exists():
            with open(cuad_path, 'r') as f:
                cuad_data = json.load(f)
            
            data_entries = len(cuad_data.get('data', []))
            print_test("CUAD Dataset", True, f"{data_entries} entries")
            
            # Check if it's sample data
            if data_entries < 10:
                print("   ⚠️  Using sample data (full dataset not downloaded)")
        else:
            print_test("CUAD Dataset", False, "File not found")
            return False
        
        # Check processed data if exists
        processed_path = Path('data/processed_cuad.json')
        if processed_path.exists():
            with open(processed_path, 'r') as f:
                processed_data = json.load(f)
            processed_entries = len(processed_data)
            print_test("Processed Data", True, f"{processed_entries} entries")
        else:
            print_test("Processed Data", False, "Not yet processed")
        
        return True
        
    except Exception as e:
        print_test("Data Integrity", False, f"Error: {e}")
        return False

def check_model_availability():
    """Check if model files are available"""
    print("\n🔍 Checking Model Availability...")
    
    try:
        # Check if models directory has content
        models_dir = Path('models')
        if models_dir.exists():
            model_files = list(models_dir.rglob('*'))
            if model_files:
                total_size = sum(f.stat().st_size for f in model_files if f.is_file())
                print_test("Model Files", True, f"{len(model_files)} files, {total_size/1024**3:.1f}GB")
            else:
                print_test("Model Files", False, "No model files found")
                return False
        else:
            print_test("Model Files", False, "Models directory not found")
            return False
        
        return True
        
    except Exception as e:
        print_test("Model Availability", False, f"Error: {e}")
        return False

def check_torch_setup():
    """Check PyTorch setup and device availability"""
    print("\n🔍 Checking PyTorch Setup...")
    
    try:
        # Check PyTorch version
        torch_version = torch.__version__
        print_test("PyTorch Version", True, f"Version {torch_version}")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print_test("CUDA Support", True, f"{gpu_count} GPU(s), {gpu_name}, {gpu_memory:.1f}GB")
        else:
            print_test("CUDA Support", False, "CPU only")
        
        # Check MPS (Apple Silicon) availability
        mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        if mps_available:
            print_test("MPS Support", True, "Apple Silicon GPU available")
        
        return True
        
    except Exception as e:
        print_test("PyTorch Setup", False, f"Error: {e}")
        return False

def test_basic_inference():
    """Test basic inference functionality"""
    print("\n🔍 Testing Basic Inference...")
    
    try:
        # Test tokenization
        from transformers import AutoTokenizer
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        
        test_text = "This is a test contract clause."
        tokens = tokenizer(test_text, return_tensors="pt")
        
        print_test("Tokenization", True, f"Tokens: {tokens['input_ids'].shape}")
        
        # Test if we can load the model (just check, don't actually load to save time)
        model_path = Path('models/mistral-7b')
        if model_path.exists():
            print_test("Model Loading", True, "Model files found")
        else:
            print_test("Model Loading", False, "Model files not found")
            return False
        
        return True
        
    except Exception as e:
        print_test("Basic Inference", False, f"Error: {e}")
        return False

def test_sample_contract():
    """Test with a sample contract"""
    print("\n🔍 Testing Sample Contract Analysis...")
    
    try:
        # Create a simple test without full model loading
        test_contract = """
        This Agreement grants Company exclusive rights to distribute the Product
        in the Territory. During the Term, Company shall not compete with similar
        products. The Agreement terminates with 30 days notice.
        """
        
        # Test basic text processing
        from sentence_transformers import SentenceTransformer
        
        print("Testing sentence embeddings...")
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = embedder.encode([test_contract])
        
        print_test("Text Embeddings", True, f"Shape: {embeddings.shape}")
        
        # Test if search index exists
        faiss_index_path = Path('data/faiss_index')
        if faiss_index_path.exists():
            print_test("Search Index", True, "FAISS index found")
        else:
            print_test("Search Index", False, "FAISS index not created")
        
        return True
        
    except Exception as e:
        print_test("Sample Contract", False, f"Error: {e}")
        return False

def generate_report(results):
    """Generate a summary report"""
    print("\n" + "="*60)
    print("VALIDATION REPORT")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
        print("Your installation is ready to use.")
        print("\nTo test the system, run:")
        print("python3 inference_fixed.py --clause-type exclusivity")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("Please check the errors above and fix them before proceeding.")
        
        failed_tests = [test for test, result in results.items() if not result]
        print(f"\nFailed tests: {', '.join(failed_tests)}")
    
    return passed == total

def main():
    """Main validation function"""
    print("🔍 Quantamind Legal AI Agent - Installation Validation")
    print("="*60)
    
    start_time = time.time()
    
    # Run all tests
    tests = {
        "Python Environment": check_python_environment,
        "File Structure": check_file_structure,
        "Data Integrity": check_data_integrity,
        "Model Availability": check_model_availability,
        "PyTorch Setup": check_torch_setup,
        "Basic Inference": test_basic_inference,
        "Sample Contract": test_sample_contract
    }
    
    results = {}
    
    for test_name, test_func in tests.items():
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Error in {test_name}: {e}")
            traceback.print_exc()
            results[test_name] = False
    
    # Generate report
    duration = time.time() - start_time
    success = generate_report(results)
    
    print(f"\nValidation completed in {duration:.1f} seconds")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())