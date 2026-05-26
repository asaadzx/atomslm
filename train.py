import os
import torch
import torch.nn as nn
import numpy as np
import time
from model import AtomLanguageModel

# 1. Setup Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Hyperparameters
BATCH_SIZE = 4
CONTEXT_LEN = 512
LEARNING_RATE = 5e-4
DATA_DIR = "dataset"
MODEL_DIR = "model"  # New directory for our weights
STEPS = 2000         # Your 2000-step training target

os.makedirs(MODEL_DIR, exist_ok=True)

# 3. Initialize Model
print("Loading model...")
model = AtomLanguageModel(n_layers=4).to(device)

# 4. Setup Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

# Data Loader Helper
def get_batch(data_dir, batch_size=4):
    files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    chosen_files = np.random.choice(files, batch_size, replace=False)
    x_batch, y_batch = [], []
    for f in chosen_files:
        tokens = np.load(os.path.join(data_dir, f))
        x_batch.append(tokens[:-1])
        y_batch.append(tokens[1:])
    return torch.tensor(np.array(x_batch), dtype=torch.long).to(device), \
           torch.tensor(np.array(y_batch), dtype=torch.long).to(device)

# --- STARTING TRAINING ---
print(f"\n--- Starting 2000-Step Training Run ---")
model.train()

for step in range(STEPS):
    start_time = time.time()
    
    x, y = get_batch(DATA_DIR, BATCH_SIZE)
    optimizer.zero_grad()
    logits = model(x)
    
    B, T, C = logits.shape
    loss = loss_fn(logits.view(B*T, C), y.view(B*T))
    
    loss.backward()
    optimizer.step()
    
    step_time = time.time() - start_time
    
    # Print status every step, or every 10 steps to keep the terminal clean
    if (step + 1) % 10 == 0 or step == 0:
        print(f"Step {step+1}/{STEPS} | Loss: {loss.item():.4f} | Time: {step_time:.2f}s")

# --- SAVE THE WEIGHTS PERMANENTLY ---
print("\nTraining complete! Saving weights...")
save_path = os.path.join(MODEL_DIR, "atom_slm.pt")
torch.save(model.state_dict(), save_path)
print(f"Weights successfully saved to: {save_path}")