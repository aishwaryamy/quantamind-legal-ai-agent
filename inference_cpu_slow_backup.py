#!/usr/bin/env python3
"""
CPU-optimized inference script for Quantamind Legal AI Agent
This version is optimized for CPU-only environments with memory efficiency.
"""

import os
import json
import argparse
import torch
import gc
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import warnings
warnings.filterwarnings("ignore")

class CPULegalAIAgent:
    def __init__(self, model_path="mistralai/Mistral-7B-Instruct-v0.1", cpu_only=True):
        """Initialize the CPU-optimized Legal AI Agent"""
        self.cpu_only = cpu_only
        self.model_path = model_path
        self.device = "cpu"
        
        print(f"🚀 Initializing Legal AI Agent (CPU Mode)")
        print(f"Device: {self.device}")
        
        # Force CPU usage
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.set_num_threads(os.cpu_count())
        
        self.tokenizer = None
        self.model = None
        self.embedder = None
        self.faiss_index = None
        self.clause_metadata = None
        
        self._load_components()
    
    def _load_components(self):
        """Load all necessary components with CPU optimization"""
        try:
            # Load tokenizer
            print("📝 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with CPU optimizations
            print("🧠 Loading model (this may take a few minutes on CPU)...")
            
            # Use smaller model or quantized version for CPU
            try:
                # Try to load a quantized version first
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,  # Use float32 for CPU
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"⚠️  Could not load full model: {e}")
                print("🔄 Falling back to pipeline mode...")
                
                # Fallback to pipeline which handles memory better
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model_path,
                    tokenizer=self.tokenizer,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    model_kwargs={"low_cpu_mem_usage": True}
                )
            
            # Load sentence embedder (lightweight model for CPU)
            print("🔍 Loading sentence embedder...")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            
            # Load search index if available
            self._load_search_index()
            
            print("✅ All components loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading components: {e}")
            raise
    
    def _load_search_index(self):
        """Load FAISS search index and metadata"""
        try:
            faiss_path = Path('data/faiss_index')
            metadata_path = Path('data/clause_metadata.json')
            
            if faiss_path.exists() and metadata_path.exists():
                print("📚 Loading search index...")
                self.faiss_index = faiss.read_index(str(faiss_path))
                
                with open(metadata_path, 'r') as f:
                    self.clause_metadata = json.load(f)
                
                print(f"✅ Search index loaded: {len(self.clause_metadata)} clauses")
            else:
                print("⚠️  Search index not found. RAG functionality disabled.")
                
        except Exception as e:
            print(f"⚠️  Could not load search index: {e}")
    
    def search_similar_clauses(self, query, top_k=3):
        """Search for similar clauses using FAISS"""
        if not self.faiss_index or not self.clause_metadata:
            return []
        
        try:
            # Encode query
            query_embedding = self.embedder.encode([query])
            
            # Search in FAISS index
            scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.clause_metadata):
                    clause_info = self.clause_metadata[idx]
                    clause_info['similarity_score'] = float(score)
                    results.append(clause_info)
            
            return results
            
        except Exception as e:
            print(f"⚠️  Search error: {e}")
            return []
    
    def create_prompt(self, clause_type, contract_text, similar_clauses=None):
        """Create optimized prompt for CPU inference"""
        # Shorter, more focused prompts for CPU efficiency
        prompt = f"""<s>[INST] You are a legal AI assistant. Find and extract the {clause_type} clause from this contract.

Contract text:
{contract_text[:1000]}...

Task: Identify the {clause_type} clause and explain its key terms briefly.

Response format:
Clause: [extracted text]
Analysis: [brief explanation]
[/INST]"""
        
        return prompt
    
    def generate_response(self, prompt, max_length=200):
        """Generate response with CPU optimizations"""
        try:
            if hasattr(self, 'pipeline'):
                # Use pipeline mode
                result = self.pipeline(
                    prompt,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
                return result[0]['generated_text'][len(prompt):].strip()
            
            else:
                # Use model directly
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512
                )
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=inputs['input_ids'].shape[1] + max_length,
                        do_sample=True,
                        temperature=0.1,
                        pad_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=1
                    )
                
                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                )
                
                return response.strip()
                
        except Exception as e:
            print(f"⚠️  Generation error: {e}")
            return f"Error generating response: {e}"
    
    def analyze_contract(self, contract_text, clause_type):
        """Analyze contract for specific clause type"""
        print(f"🔍 Analyzing contract for: {clause_type}")
        
        # Search for similar clauses (if available)
        similar_clauses = self.search_similar_clauses(f"{clause_type} clause", top_k=2)
        
        # Create prompt
        prompt = self.create_prompt(clause_type, contract_text, similar_clauses)
        
        # Generate response
        print("💭 Generating analysis...")
        response = self.generate_response(prompt, max_length=150)
        
        # Clean up memory
        gc.collect()
        
        return {
            'clause_type': clause_type,
            'analysis': response,
            'similar_clauses_found': len(similar_clauses),
            'method': 'CPU Inference'
        }

def load_sample_contract():
    """Load a sample contract for testing"""
    return """
    SOFTWARE LICENSE AGREEMENT
    
    This Agreement grants Company exclusive rights to distribute the Product
    in the Territory during the Term. Company shall not compete with similar
    products or engage in conflicting business activities.
    
    Either party may terminate this Agreement with thirty (30) days written
    notice. Upon termination, all rights cease except those that survive.
    
    Company's liability shall not exceed amounts paid by Client in the
    preceding twelve months. No liability for indirect or consequential damages.
    
    This Agreement is governed by California law. Disputes resolved in
    San Francisco County courts.
    
    Payment terms: Net 30 days. Late payments incur 1.5% monthly service charge.
    """

def main():
    """Main function with CPU-optimized arguments"""
    parser = argparse.ArgumentParser(description="Legal AI Agent - CPU Mode")
    parser.add_argument("--clause-type", required=True, 
                       help="Type of clause to extract (e.g., 'exclusivity', 'termination')")
    parser.add_argument("--contract-file", 
                       help="Path to contract file (optional)")
    parser.add_argument("--model-path", default="mistralai/Mistral-7B-Instruct-v0.1",
                       help="Model path or name")
    parser.add_argument("--max-length", type=int, default=150,
                       help="Maximum response length")
    parser.add_argument("--use-sample", action="store_true",
                       help="Use built-in sample contract")
    
    args = parser.parse_args()
    
    try:
        # Initialize agent
        agent = CPULegalAIAgent(model_path=args.model_path, cpu_only=True)
        
        # Get contract text
        if args.contract_file and Path(args.contract_file).exists():
            with open(args.contract_file, 'r') as f:
                contract_text = f.read()
            print(f"📄 Loaded contract from: {args.contract_file}")
        else:
            contract_text = load_sample_contract()
            print("📄 Using sample contract")
        
        # Analyze contract
        result = agent.analyze_contract(contract_text, args.clause_type)
        
        # Display results
        print("\n" + "="*60)
        print("ANALYSIS RESULT (CPU MODE)")
        print("="*60)
        print(f"Clause Type: {result['clause_type']}")
        print(f"Method: {result['method']}")
        print(f"Similar clauses found: {result['similar_clauses_found']}")
        print("\nAnalysis:")
        print(result['analysis'])
        print("="*60)
        
        # Performance note
        print("\n💡 CPU Performance Tips:")
        print("- CPU inference is slower but uses less memory")
        print("- Consider using GPU for production workloads")
        print("- Results may vary from GPU version due to different optimizations")
        
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()