from transformers import AutoTokenizer
from datasets import load_from_disk
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

import os
hug_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

if not hug_token:
    raise ValueError("Hugging Face token is not set in the environment variables.")

data_dir = '~/.cache/huggingface/datasets/squad/plain_text/0.0.0/7b6d24c440a36b6815f21b70d25016731768db1f/'

tokenized_dataset = load_from_disk(data_dir)
from transformers import AutoModelForQuestionAnswering, Trainer, TrainingArguments

model = AutoModelForQuestionAnswering.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token=hug_token)
model.to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

if __name__ == '__main__':
    trainer.train()

    import datetime
    model_dir = f"./fine-tuned-model-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

