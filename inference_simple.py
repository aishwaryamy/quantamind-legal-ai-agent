import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def simple_contract_analysis(contract_text, clause_type, model_path="models/mistral-7b"):
    print(f"Loading model from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with better device handling
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    print(f"Analyzing contract for {clause_type} clauses...")
    
    # Simpler prompt
    prompt = f"Find the {clause_type} clause in this contract:\n\n{contract_text}\n\nClause:"
    
    # Tokenize and move to correct device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    
    # Move inputs to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print("Generating analysis...")
    
    # Generate with stricter parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # Much shorter response
            temperature=0.3,    # Less randomness
            do_sample=False,    # Deterministic
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            timeout=30  # 30 second timeout
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clause-type', default='exclusivity')
    args = parser.parse_args()
    
    # Example contract
    contract = """
    This Agreement grants Company exclusive rights to distribute the Product 
    in the Territory. During the Term, Company shall not compete with similar 
    products. The Agreement terminates with 30 days notice.
    """
    
    try:
        result = simple_contract_analysis(contract, args.clause_type)
        print("\n" + "="*50)
        print("ANALYSIS RESULT:")
        print("="*50)
        print(result)
        print("\n" + "="*50)
        print("STATUS: SUCCESS")
        print("="*50)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
