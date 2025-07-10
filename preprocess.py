import json
import pandas as pd
from transformers import AutoTokenizer

def preprocess_cuad(data_path, output_path):
    try:
        # Load CUAD dataset
        with open(data_path, 'r') as f:
            data = json.load(f)
        print("Loaded CUAD_v1.json successfully")

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        
        # Fix padding token issue - Mistral tokenizer doesn't have a pad_token by default
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("Tokenizer initialized with padding token")

        # Process contracts
        processed_data = []
        for contract in data['data']:
            for paragraph in contract['paragraphs']:
                context = paragraph['context']
                qas = paragraph['qas']
                # Tokenize contract text
                tokenized = tokenizer(context, truncation=True, max_length=512, padding='max_length')
                # Extract labels (clause types and answers)
                labels = [
                    {
                        'question': qa['question'].split('__')[-1],  # Extract clause type (e.g., "Exclusivity")
                        'answers': qa['answers'],
                        'is_impossible': qa.get('is_impossible', False)
                    }
                    for qa in qas
                ]
                processed_data.append({
                    'input_ids': tokenized['input_ids'],
                    'attention_mask': tokenized['attention_mask'],
                    'labels': labels
                })
        
        # Save processed data
        with open(output_path, 'w') as f:
            json.dump(processed_data, f, indent=2)
        print(f"Processed data saved to {output_path}")

    except FileNotFoundError:
        print(f"Error: {data_path} not found. Ensure CUAD_v1.json is in the data/ directory.")
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")

if __name__ == "__main__":
    preprocess_cuad("data/CUAD_v1.json", "data/processed_cuad.json")
