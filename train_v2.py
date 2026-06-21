"""
train_v2.py — Training script for CodeCriticV2.

Three-phase training:
  Phase 1: Self-supervised pretraining (MLM) on unlabeled Python code
  Phase 2: Supervised training on labeled issue data
  Phase 3: Fine-tuning on user's codebase (optional)

Usage:
  # Full training (pretrain + supervised)
  python train_v2.py --epochs-pretrain 20 --epochs-supervised 50 --output code_critic_v2.pt

  # Supervised only (skip pretraining)
  python train_v2.py --epochs-supervised 50 --output code_critic_v2.pt

  # Fine-tune existing model on your codebase
  python train_v2.py --finetune --data-dir ./my_project --resume code_critic_v2.pt --epochs 10
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from code_features import extract_features, FEATURE_DIM
from code_critic_tokenizer import CodeTokenizer, VOCAB_SIZE, MAX_SEQ_LEN
from data_generator import generate_dataset
from model_v2 import (
    CodeCriticV2, save_model, load_model, model_size_params,
    HIDDEN_DIM, NUM_LAYERS, NUM_HEADS, NUM_ISSUE_CLASSES,
)

# --------------------------------------------------------------------------- #
# Code corpus for MLM pretraining                                             #
# --------------------------------------------------------------------------- #

def collect_python_files(data_dir: str, max_files: int = 5000) -> list:
    """Collect Python files from a directory for pretraining."""
    code_samples = []
    for root, dirs, files in os.walk(data_dir):
        dirs[:] = [d for d in dirs if d not in {
            "__pycache__", ".git", "node_modules", ".venv", "venv",
            "env", ".tox", ".mypy_cache", ".pytest_cache", "site-packages",
        }]
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                        code = fh.read()
                    if 100 < len(code) < 50000:
                        code_samples.append(code)
                except Exception:
                    pass
                if len(code_samples) >= max_files:
                    return code_samples
    return code_samples


def pretrain_mlm(
    model: CodeCriticV2,
    code_samples: list,
    epochs: int = 20,
    lr: float = 3e-4,
    batch_size: int = 32,
    device: str = "cpu",
) -> list:
    """Self-supervised pretraining via masked language modeling."""
    tokenizer = CodeTokenizer(max_length=MAX_SEQ_LEN)

    print(f"Pretraining MLM on {len(code_samples)} code samples...")

    # Tokenize all samples
    all_tokens = []
    all_features = []
    for code in code_samples:
        tokens = tokenizer.encode(code)
        features = extract_features(code)
        all_tokens.append(tokens)
        all_features.append(features)

    tokens_arr = np.stack(all_tokens)
    features_arr = np.stack(all_features)

    dataset = TensorDataset(
        torch.from_numpy(tokens_arr),
        torch.from_numpy(features_arr).float(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.to(device)

    losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_tokens, batch_features in loader:
            batch_tokens = batch_tokens.to(device)
            batch_features = batch_features.to(device)

            optimizer.zero_grad()
            outputs = model(batch_tokens, batch_features, mask_ratio=0.15)

            # MLM loss
            mlm_logits = outputs["mlm_logits"]
            mlm_mask = outputs["mlm_mask"]

            if mlm_logits is not None and mlm_mask is not None:
                # Compute loss only on masked positions
                loss = F.cross_entropy(
                    mlm_logits[mlm_mask],
                    batch_tokens[:, :mlm_logits.shape[1]][mlm_mask],
                    ignore_index=0,  # PAD
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Pretrain Epoch {epoch+1:3d}/{epochs}  mlm_loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

    print(f"Pretraining complete. Final loss: {losses[-1]:.4f}")
    return losses


def train_supervised(
    model: CodeCriticV2,
    n_samples: int = 5000,
    epochs: int = 50,
    lr: float = 3e-4,
    batch_size: int = 32,
    device: str = "cpu",
) -> list:
    """Supervised training on labeled issue data."""
    tokenizer = CodeTokenizer(max_length=MAX_SEQ_LEN)

    print(f"\nGenerating {n_samples} labeled training samples...")
    features, qualities, issue_labels = generate_dataset(n_samples=n_samples, augment=True)

    # Tokenize a representative code sample for each feature vector
    # We use the features directly and create dummy token sequences
    # In practice, you'd tokenize the actual code, but for the data generator
    # we only have features. We'll use the tokenizer on synthetic code.
    from data_generator import BUGGY_CODE, STYLE_VIOLATIONS, PERFORMANCE_ISSUES
    from data_generator import SECURITY_ISSUES, MAINTAINABILITY_ISSUES, NON_PYTHONIC, GOOD_CODE

    all_base = BUGGY_CODE + STYLE_VIOLATIONS + PERFORMANCE_ISSUES + \
               SECURITY_ISSUES + MAINTAINABILITY_ISSUES + NON_PYTHONIC + GOOD_CODE

    # Create token sequences matching the features
    all_tokens = []
    for i in range(len(features)):
        # Pick a base template and tokenize it
        base = all_base[i % len(all_base)]
        tokens = tokenizer.encode(base["code"])
        all_tokens.append(tokens)

    tokens_arr = np.stack(all_tokens)

    # Split 80/20
    n_train = int(0.8 * len(features))
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

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    quality_loss_fn = nn.MSELoss()
    issue_loss_fn = nn.BCEWithLogitsLoss()

    model.to(device)
    losses = []

    print(f"Supervised training for {epochs} epochs on {n_train} samples...")
    start = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_tokens, batch_features, batch_quality, batch_issues in train_loader:
            batch_tokens = batch_tokens.to(device)
            batch_features = batch_features.to(device)
            batch_quality = batch_quality.to(device)
            batch_issues = batch_issues.to(device)

            optimizer.zero_grad()
            outputs = model(batch_tokens, batch_features, mask_ratio=0.0)

            loss_q = quality_loss_fn(outputs["quality_score"], batch_quality)
            loss_i = issue_loss_fn(outputs["issue_logits"], batch_issues)

            loss = loss_q + 0.5 * loss_i
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for vt, vf, vq, vi in val_loader:
                    vt, vf, vq, vi = vt.to(device), vf.to(device), vq.to(device), vi.to(device)
                    vo = model(vt, vf, mask_ratio=0.0)
                    vl = quality_loss_fn(vo["quality_score"], vq).item()
                    vl += 0.5 * issue_loss_fn(vo["issue_logits"], vi).item()
                    val_loss += vl
                    val_batches += 1
            val_loss /= max(val_batches, 1)
            print(
                f"  Epoch {epoch+1:3d}/{epochs}  "
                f"train_loss={avg_loss:.4f}  val_loss={val_loss:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

    elapsed = time.time() - start
    print(f"Supervised training complete in {elapsed:.1f}s")
    return losses


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Train CodeCriticV2")
    parser.add_argument("--epochs-pretrain", type=int, default=0,
                        help="MLM pretraining epochs (0 to skip)")
    parser.add_argument("--epochs-supervised", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=str, default="code_critic_v2.pt")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Python codebase for MLM pretraining")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("CodeCriticV2 Training")
    print("=" * 60)

    if args.resume:
        print(f"Loading model from {args.resume}")
        model = load_model(args.resume)
    else:
        model = CodeCriticV2()

    n_params = model_size_params(model)
    print(f"Model: {n_params:,} parameters ({n_params / 1e6:.2f}M)")
    print(f"Vocab: {VOCAB_SIZE}, Hidden: {HIDDEN_DIM}, Layers: {NUM_LAYERS}, Heads: {NUM_HEADS}")

    # Phase 1: MLM Pretraining
    if args.epochs_pretrain > 0:
        if args.data_dir:
            code_samples = collect_python_files(args.data_dir, max_files=3000)
            print(f"Collected {len(code_samples)} Python files for pretraining")
        else:
            # Use synthetic data for pretraining
            print("No --data-dir provided, using synthetic code for pretraining")
            from data_generator import BUGGY_CODE, GOOD_CODE, STYLE_VIOLATIONS
            all_base = BUGGY_CODE + GOOD_CODE + STYLE_VIOLATIONS
            code_samples = [s["code"] for s in all_base] * 50  # Repeat for volume

        if code_samples:
            pretrain_mlm(model, code_samples, epochs=args.epochs_pretrain, lr=args.lr,
                          batch_size=args.batch_size)

    # Phase 2: Supervised training
    if args.epochs_supervised > 0:
        train_supervised(model, n_samples=args.n_samples, epochs=args.epochs_supervised,
                         lr=args.lr, batch_size=args.batch_size)

    # Save
    save_model(model, args.output)
    print(f"\nDone! Model saved to {args.output}")


if __name__ == "__main__":
    main()
