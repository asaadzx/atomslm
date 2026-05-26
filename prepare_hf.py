import os
import json
from transformers import AutoTokenizer

MODEL_DIR = "model"

print("Loading and saving Gemma tokenizer to the model folder...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
tokenizer.save_pretrained(MODEL_DIR)

# Create the standard config.json so llama.cpp knows our architecture sizes
config = {
    "architectures": ["GemmaForCausalLM"],
    "hidden_size": 256,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "intermediate_size": 682, # Our SwiGLU hidden_dim calculation
    "vocab_size": 256000,
    "rms_norm_eps": 1e-6,
    "torch_dtype": "float32"
}

with open(os.path.join(MODEL_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=4)

print("Hugging Face structure prepared inside the 'model/' directory!")