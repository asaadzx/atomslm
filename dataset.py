import os
from datasets import load_dataset
from transformers import AutoTokenizer

# 1. Setup paths and tokenizer
DATA_DIR = "dataset"
os.makedirs(DATA_DIR, exist_ok=True)

print("Loading Gemma tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

# 2. Stream the dataset
print("Opening FineWeb-Edu stream...")
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

# Let's look at how we process one document first
iterable_dataset = iter(dataset)
first_document = next(iterable_dataset)
text_content = first_document['text']

# Tokenize the text
tokens = tokenizer.encode(text_content)
print(f"The first document has {len(tokens)} tokens.")