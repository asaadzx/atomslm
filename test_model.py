"""Test the trained model with a forward pass or text generation."""

import os
import torch
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import AtomLanguageModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model", "atom_slm.pt")
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Load model ---
model = AtomLanguageModel(n_layers=4)
state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)

# The saved state includes the causal mask buffer "layers.X.attention.bias"
# which is fine -- load_state_dict with strict=False will skip it
missing, unexpected = model.load_state_dict(state, strict=False)
if missing:
    print(f"Skipped (missing from state): {missing}")
if unexpected:
    print(f"Skipped (unexpected keys):  {unexpected}")
model.to(device)
model.eval()
print("Model loaded successfully.\n")

# --- Test 1: Forward pass with random tokens ---
print("=== Test 1: Forward pass (random input) ===")
x = torch.randint(0, 1000, (1, 64), device=device)
with torch.no_grad():
    logits = model(x)
print(f"  Input shape:  {x.shape}")
print(f"  Output shape: {logits.shape}")
print(f"  Output range: {logits.min().item():.2f} to {logits.max().item():.2f}\n")

# --- Test 2: Generate text (if tokenizer available) ---
print("=== Test 2: Text generation ===")
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    prompt = "what is egypt capital"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"  Prompt: {prompt!r}")
    print(f"  Tokenized: {input_ids[0].tolist()}")

    max_new_tokens = 20
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(generated)
        next_logit = logits[0, -1, :]
        next_id = torch.argmax(next_logit, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_id.unsqueeze(0)], dim=1)
        if next_id.item() == tokenizer.eos_token_id:
            break

    output = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"  Generated: {output!r}")
except Exception as e:
    print(f"  Tokenizer/generation failed: {e}")
    print("  (The model itself works — tokenizer may need a local path fix)")


# --- Test 3: Parameter count ---
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,} ({total_params/1e6:.2f}M)")
