from transformers import AutoTokenizer, AutoModelForCausalLM

from huggingface_hub import login

login(token="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")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", use_auth_token=True)


input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
