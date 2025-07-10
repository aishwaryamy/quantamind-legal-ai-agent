from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def download_model():
    try:
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        
        print("Downloading Mistral 7B model...")
        # Download model with device_map for efficient loading
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto",
            torch_dtype=torch.float16  # Use half precision to save memory
        )
        
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Fix padding token issue
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("Saving model and tokenizer locally...")
        model.save_pretrained("models/mistral-7b")
        tokenizer.save_pretrained("models/mistral-7b")
        
        print("Model and tokenizer saved to models/mistral-7b")
        print(f"Model size: ~7B parameters")
        
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        print("Make sure you have enough disk space (~15GB) and stable internet connection")

if __name__ == "__main__":
    download_model()
