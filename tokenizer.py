from transformers import AutoTokenizer
from datasets import load_dataset
import os
hug_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

if not hug_token:
    raise ValueError("Hugging Face token is not set in the environment variables.")

dataset = load_dataset("squad") # Stanford Question Answering Dataset

# Tokenize the dataset
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token=hug_token)

# Check if the tokenizer has a pad_token. If not, set it to eos_token or add a new one.
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

def tokenize_squad(examples):
    '''Specifically tokenize the squad dataset.'''
    return tokenizer(
        examples["context"],
        examples["question"],
        padding="max_length",
        truncation=True,
        return_tensors="pt"  # Ensure tensors are returned for compatibility with PyTorch
    )


tokenized_dataset = dataset.map(tokenize_squad, batched=True, batch_size=100)