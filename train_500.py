import os
import time
import math
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from model import GPTConfig, GPT

# Load config overrides
out_dir = 'out'
max_iters = 500

eval_interval = 100
log_interval = 1
eval_iters = 10
batch_size = 4
block_size = 128

# Load metadata
data_dir = os.path.join('data', 'shakespeare_char')
meta_path = os.path.join(data_dir, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']
itos = meta['itos']
stoi = meta['stoi']

# Set up model
config = GPTConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    n_layer=4,
    n_head=4,
    n_embd=128,
    dropout=0.0,
    bias=False
)
model = GPT(config).to("cpu")
print(f"number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# Load data
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

# Get batch
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.to(torch.long), y.to(torch.long)

# Training loop
train_losses = []
val_losses = []
model.train()
for iter in range(1, max_iters + 1):
    x, y = get_batch('train')
    logits, _ = model(x)
    B, T, C = logits.shape
    loss = criterion(logits.view(B * T, C), y.view(B * T))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % log_interval == 0:
        print(f"Iter {iter}: Train Loss = {loss.item():.4f}")

    if iter % eval_interval == 0 or iter == 1:
        model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            for _ in range(eval_iters):
                vx, vy = get_batch('val')
                v_logits, _ = model(vx)
                vB, vT, vC = v_logits.shape
                v_loss = criterion(v_logits.view(vB * vT, vC), vy.view(vB * vT))
                val_loss_sum += v_loss.item()
            avg_val_loss = val_loss_sum / eval_iters
            val_losses.append(avg_val_loss)
            train_losses.append(loss.item())
            print(f"Step {iter}: Val Loss = {avg_val_loss:.4f}")
        model.train()

# Plot loss
plt.plot(np.arange(len(train_losses)) * eval_interval, train_losses, label='Train Loss')
plt.plot(np.arange(len(val_losses)) * eval_interval, val_losses, label='Val Loss')
plt.title("Training Loss (500 iterations)")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve_500_iters.png")
plt.show()

# Sampling
model.eval()
def sample(start="ROMEO:", max_new_tokens=200):
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    idx = torch.tensor([encode(start)], dtype=torch.long)
    for _ in range(max_new_tokens):
        logits, _ = model(idx)
        logits = logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)
    return decode(idx[0].tolist())

print("\n=== SAMPLE OUTPUT ===")
print(sample())
