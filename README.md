# LLaMA3 models with HuggingFace's Transformer library

Playing with Llama using [HuggingFace's transformers library](https://huggingface.co/docs/transformers/main/en/model_doc/llama).

1. Set up venv and download from `requirements.txt`
2. Request for access to the [Llama3.1-8B model](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct), and generate an access token on [HuggingFace](https://huggingface.co/settings/tokens)
3. Set your access token as environmental variable by running
```bash
export HUGGINGFACE_HUB_TOKEN=your_access_token
```
4. Run `chat.py`, which will download weights (15G of disk space) to `~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct`

Can play with `input_text` and `max_new_tokens` for how long you want the response prompt to be
