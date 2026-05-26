import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# Configuration
DATA_DIR = "dataset"
CONTEXT_LEN = 512  # Our model's maximum view window
BATCH_ROWS = 1000  # How many web documents to pull

print("Loading Gemma tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
EOS_TOKEN = 1  # Gemma's End of Sequence token

print("Opening FineWeb-Edu stream...")
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

all_tokens = []
block_count = 0

print(f"Processing the first {BATCH_ROWS} documents...")
for i, example in enumerate(dataset):
    if i >= BATCH_ROWS:
        break
        
    # Tokenize text and append the critical EOS token
    tokens = tokenizer.encode(example['text'])
    tokens.append(EOS_TOKEN)
    all_tokens.extend(tokens)

    # Once we have enough tokens to fill whole blocks, save them
    while len(all_tokens) >= CONTEXT_LEN:
        chunk = all_tokens[:CONTEXT_LEN]
        all_tokens = all_tokens[CONTEXT_LEN:]
        
        # Convert to 32-bit unsigned integers
        np_chunk = np.array(chunk, dtype=np.uint32)
        
        # Save as a binary file
        np.save(os.path.join(DATA_DIR, f"block_{block_count}.npy"), np_chunk)
        block_count += 1

print(f"Done! Saved {block_count} training blocks to the '{DATA_DIR}' directory.")