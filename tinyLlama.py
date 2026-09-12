# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

# Set the Hugging Face cache directory to a local path within the project
import os
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.environ["HF_HUB_CACHE"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

import argparse
import torch
from transformers import pipeline

parser = argparse.ArgumentParser(description="Run a local Hugging Face text-generation model.")
parser.add_argument(
    "--model",
    default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    help="Model name or local path (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)",
)
args = parser.parse_args()

pipe = pipeline("text-generation", model=args.model, dtype=torch.bfloat16, device_map="auto")

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])
# <|system|>
# You are a friendly chatbot who always responds in the style of a pirate.</s>
# <|user|>
# How many helicopters can a human eat in one sitting?</s>
# <|assistant|>
# ...
