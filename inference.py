import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import torch

class LegalContractAnalyzer:
    def __init__(self, model_path, use_finetuned=False):
        self.model_path = model_path
        self.use_finetuned = use_finetuned
        
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        print("Loading FAISS index and embedder...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index("data/faiss_index")
        
        with open("data/clause_metadata.json", 'r') as f:
            self.clause_metadata = json.load(f)
    
    def retrieve_similar_clauses(self, query, k=3):
        """Retrieve similar clauses using FAISS"""
        query_embedding = self.embedder.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.clause_metadata):
                results.append({
                    'clause': self.clause_metadata[idx],
                    'similarity_score': float(distances[0][i])
                })
        return results
    
    def analyze_contract(self, contract_text, clause_type="general"):
        """Analyze contract for specific clause types"""
        
        # Step 1: Retrieve similar clauses
        query = f"Find {clause_type} clause in contract"
        similar_clauses = self.retrieve_similar_clauses(query, k=2)
        
        # Step 2: Generate analysis using the model
        if self.use_finetuned:
            prompt = f"[INST] Identify the {clause_type} clause in this contract: {contract_text[:500]}... [/INST]"
        else:
            prompt = f"Analyze this contract excerpt for {clause_type} clauses:\n\n{contract_text[:500]}...\n\nKey clauses found:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            'analysis': response,
            'similar_clauses': similar_clauses,
            'method': 'fine-tuned' if self.use_finetuned else 'prompt-engineering'
        }

def main():
    parser = argparse.ArgumentParser(description='Legal Contract Analysis')
    parser.add_argument('--method', choices=['lora', 'prompt'], default='prompt', 
                       help='Analysis method: lora (fine-tuned) or prompt (prompt engineering)')
    parser.add_argument('--clause-type', default='exclusivity', 
                       help='Type of clause to analyze (e.g., exclusivity, termination, liability)')
    
    args = parser.parse_args()
    
    # Determine model path
    if args.method == 'lora':
        model_path = "models/mistral-7b-finetuned"
        use_finetuned = True
        print("Using LoRA fine-tuned model")
    else:
        model_path = "models/mistral-7b"
        use_finetuned = False
        print("Using prompt engineering approach")
    
    try:
        # Initialize analyzer
        analyzer = LegalContractAnalyzer(model_path, use_finetuned)
        
        # Example contract text (you can replace this with actual contract text)
        example_contract = """
        This Agreement grants Company exclusive rights to distribute the Product 
        in the Territory. During the Term, Company shall not compete with similar 
        products or enter agreements with competitors. The Agreement may be terminated 
        by either party with 30 days written notice. Company agrees to maintain 
        liability insurance of $2,000,000.
        """
        
        print(f"\nAnalyzing contract for {args.clause_type} clauses...")
        print("-" * 50)
        
        # Perform analysis
        result = analyzer.analyze_contract(example_contract, args.clause_type)
        
        print(f"Method: {result['method']}")
        print(f"\nAnalysis:\n{result['analysis']}")
        
        print(f"\nSimilar clauses found in training data:")
        for i, clause in enumerate(result['similar_clauses']):
            print(f"{i+1}. {clause['clause']['question']}: {clause['clause']['answer'][:100]}...")
            print(f"   Similarity score: {clause['similarity_score']:.4f}")
        
        # Performance comparison placeholder
        print(f"\n" + "="*50)
        print("PERFORMANCE METRICS (Simulated)")
        print("="*50)
        if args.method == 'lora':
            print("LoRA Fine-tuned Model:")
            print("- Accuracy: 92.5%")
            print("- F1-Score: 0.89")
            print("- Training Time: 2 hours")
            print("- Resource Usage: 12GB GPU memory")
        else:
            print("Prompt Engineering:")
            print("- Accuracy: 85.3%")
            print("- F1-Score: 0.82")
            print("- Training Time: None")
            print("- Resource Usage: 8GB GPU memory")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")

if __name__ == "__main__":
    main()
