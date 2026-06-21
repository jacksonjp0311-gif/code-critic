"""Quick training script - runs inline to avoid spawn issues."""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from code_features import extract_features, FEATURE_DIM
from code_critic_tokenizer import CodeTokenizer, VOCAB_SIZE, MAX_SEQ_LEN
from data_generator import generate_dataset, BUGGY_CODE, STYLE_VIOLATIONS, PERFORMANCE_ISSUES
from data_generator import SECURITY_ISSUES, MAINTAINABILITY_ISSUES, NON_PYTHONIC, GOOD_CODE
from model_v2 import CodeCriticV2, save_model, model_size_params

# Generate data
print("Generating 5000 labeled samples...")
features, qualities, issue_labels = generate_dataset(n_samples=5000, augment=True)

# Tokenize code samples
all_base = BUGGY_CODE + STYLE_VIOLATIONS + PERFORMANCE_ISSUES + \
           SECURITY_ISSUES + MAINTAINABILITY_ISSUES + NON_PYTHONIC + GOOD_CODE
tokenizer = CodeTokenizer(max_length=MAX_SEQ_LEN)

all_tokens = []
for i in range(len(features)):
    base = all_base[i % len(all_base)]
    tokens = tokenizer.encode(base["code"])
    all_tokens.append(tokens)

tokens_arr = np.stack(all_tokens)
print(f"Tokenized {len(all_tokens)} samples, shape: {tokens_arr.shape}")

# Split
n_train = int(0.8 * len(features))
device = "cpu"

train_data = TensorDataset(
    torch.from_numpy(tokens_arr[:n_train]),
    torch.from_numpy(features[:n_train]).float(),
    torch.from_numpy(qualities[:n_train]).float(),
    torch.from_numpy(issue_labels[:n_train]).float(),
)
val_data = TensorDataset(
    torch.from_numpy(tokens_arr[n_train:]),
    torch.from_numpy(features[n_train:]).float(),
    torch.from_numpy(qualities[n_train:]).float(),
    torch.from_numpy(issue_labels[n_train:]).float(),
)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

# Create model
model = CodeCriticV2()
n_params = model_size_params(model)
print(f"Model: {n_params:,} params ({n_params/1e6:.2f}M)")

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
quality_loss_fn = nn.MSELoss()
issue_loss_fn = nn.BCEWithLogitsLoss()

print(f"Training for 50 epochs on {n_train} samples...")
start = time.time()

for epoch in range(50):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for bt, bf, bq, bi in train_loader:
        optimizer.zero_grad()
        outputs = model(bt, bf, mask_ratio=0.0)
        loss_q = quality_loss_fn(outputs["quality_score"], bq)
        loss_i = issue_loss_fn(outputs["issue_logits"], bi)
        loss = loss_q + 0.5 * loss_i
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    scheduler.step()
    avg_loss = total_loss / max(n_batches, 1)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for vt, vf, vq, vi in val_loader:
                vo = model(vt, vf, mask_ratio=0.0)
                vl = quality_loss_fn(vo["quality_score"], vq).item()
                vl += 0.5 * issue_loss_fn(vo["issue_logits"], vi).item()
                val_loss += vl
                val_batches += 1
        val_loss /= max(val_batches, 1)
        print(f"  Epoch {epoch+1:3d}/50  train={avg_loss:.4f}  val={val_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

elapsed = time.time() - start
print(f"Training complete in {elapsed:.1f}s")

# Save
save_model(model, "code_critic_v2.pt")
print("Done!")
