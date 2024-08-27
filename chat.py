from transformers import AutoTokenizer, AutoModelForCausalLM

import os
hug_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token=hug_token)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token=hug_token)


input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")

max_new_tokens = 50
outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
