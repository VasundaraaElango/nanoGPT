import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from model import GPT, GPTConfig

# Load metadata
data_dir = "data/shakespeare_char"
with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
vocab_size = meta["vocab_size"]
itos = meta["itos"]
stoi = meta["stoi"]

# Load val data
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode='r')
block_size = 128
batch_size = 4

# Untrained model config
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
model.eval()

# Evaluate real loss on val set
losses = []
criterion = torch.nn.CrossEntropyLoss()
for i in range(10):  # 10 validation batches
    ix = torch.randint(len(val_data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((val_data[j:j + block_size]).astype(np.int64)) for j in ix])
    y = torch.stack([torch.from_numpy((val_data[j + 1:j + 1 + block_size]).astype(np.int64)) for j in ix])
    with torch.no_grad():
        logits, _ = model(x)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = y.view(B * T)
        loss = criterion(logits, targets)
        losses.append(loss.item())

# Plot loss
plt.plot(range(len(losses)), losses, marker='o')
plt.title("Validation Loss (Untrained nanoGPT)")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("untrained_validation_loss.png")
plt.show()

# Generate text from untrained model
print("\n=== Untrained Output ===")
start = "ROMEO:"
input_ids = torch.tensor([stoi[s] for s in start], dtype=torch.long)[None, ...]

with torch.no_grad():
    for _ in range(100):
        logits, _ = model(input_ids)
        logits = logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

output = ''.join([itos[i] for i in input_ids[0].tolist()])
print(output)
