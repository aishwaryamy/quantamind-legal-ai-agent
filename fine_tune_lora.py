from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import torch
import json
from torch.utils.data import Dataset

class CUADDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Create input text from the first label's question and answer
        if item['labels'] and item['labels'][0]['answers']:
            question = item['labels'][0]['question']
            answer = item['labels'][0]['answers'][0]['text']
            
            # Format as instruction-following task
            input_text = f"[INST] Identify the {question} clause in this contract. [/INST] {answer}"
        else:
            input_text = "No valid clause found."
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()  # For causal LM, labels = input_ids
        }

def fine_tune_with_lora():
    try:
        print("Loading processed data...")
        with open("data/processed_cuad.json", 'r') as f:
            data = json.load(f)
        
        # Take a subset for faster training (adjust as needed)
        subset_size = min(100, len(data))
        data_subset = data[:subset_size]
        print(f"Using {subset_size} samples for training")
        
        print("Loading base model and tokenizer...")
        model_name = "models/mistral-7b"  # Use locally saved model
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        print("Configuring LoRA...")
        lora_config = LoraConfig(
            r=8,  # Rank
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Attention layers
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        print("Applying LoRA to model...")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        print("Creating dataset...")
        dataset = CUADDataset(data_subset, tokenizer)
        
        print("Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=2,
            per_device_train_batch_size=1,  # Small batch size for memory efficiency
            gradient_accumulation_steps=8,
            warmup_steps=10,
            logging_steps=10,
            save_steps=50,
            evaluation_strategy="no",
            learning_rate=2e-4,
            fp16=True,  # Use mixed precision
            remove_unused_columns=False,
        )
        
        print("Creating trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
        
        print("Starting fine-tuning...")
        trainer.train()
        
        print("Saving fine-tuned model...")
        model.save_pretrained("models/mistral-7b-finetuned")
        tokenizer.save_pretrained("models/mistral-7b-finetuned")
        
        print("Fine-tuning completed!")
        print("Model saved to models/mistral-7b-finetuned")
        
    except Exception as e:
        print(f"Error during fine-tuning: {str(e)}")
        print("Note: Fine-tuning requires significant memory. Consider reducing batch size or using a smaller subset.")

if __name__ == "__main__":
    fine_tune_with_lora()
