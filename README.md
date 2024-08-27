# LLaMA3 models with HuggingFace's Transformer library

Playing with Llama using [HuggingFace's transformers library](https://huggingface.co/docs/transformers/main/en/model_doc/llama).

1. Set up venv and download from `requirements.txt`
2. Request for access to the Llama3.1-8B model, and generate an access token on HuggingFace
3. Run `huggingface.py`, which will download weights (15G of disk space) to `~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct`