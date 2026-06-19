"""
train.py — Training script for CodeFeedbackModel.

Supports three training modes:
1. Synthetic pretraining — generates good/bad code pairs with known labels
2. Self-supervised — masked feature prediction on any code corpus
3. User repo fine-tuning — ingest a local codebase for personalization

Usage:
    python train.py --mode synthetic --epochs 50 --output model.pt
    python train.py --mode selfsupervised --data-dir ./my_project --epochs 20
    python train.py --mode finetune --data-dir ./my_project --epochs 10 --lr 1e-4

The synthetic pretraining mode requires no external data and produces a
reasonable starting model in ~2 minutes on CPU.
"""

import argparse
import os
import random
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from code_features import extract_features, FEATURE_DIM
from code_feedback_model import (
    CodeFeedbackModel,
    save_model,
    load_model,
    model_size_params,
    HIDDEN_DIM,
    NUM_LAYERS,
    NUM_HEADS,
    NUM_ISSUE_CLASSES,
)

# --------------------------------------------------------------------------- #
# Synthetic data generation                                                   #
# --------------------------------------------------------------------------- #
GOOD_CODE_SAMPLES = [
    # Clean, well-structured code
    '''
def fibonacci(n: int) -> int:
    """Return the n-th Fibonacci number."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
''',
    '''
class Stack:
    """A simple stack implementation."""
    def __init__(self):
        self._items: list = []

    def push(self, item):
        self._items.append(item)

    def pop(self):
        if not self._items:
            raise IndexError("pop from empty stack")
        return self._items.pop()

    @property
    def is_empty(self) -> bool:
        return len(self._items) == 0
''',
    '''
from typing import List, Optional

def find_max(numbers: List[int]) -> Optional[int]:
    """Find the maximum value in a list."""
    if not numbers:
        return None
    max_val = numbers[0]
    for n in numbers[1:]:
        if n > max_val:
            max_val = n
    return max_val
''',
    '''
import json
from pathlib import Path

def load_config(path: str) -> dict:
    """Load a JSON configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(config_path) as f:
        return json.load(f)
''',
    '''
def process_items(items: list, *, verbose: bool = False) -> list:
    """Process a list of items, filtering and transforming."""
    results = []
    for item in items:
        if item is None:
            continue
        try:
            processed = item.strip().lower()
            if processed:
                results.append(processed)
        except AttributeError:
            if verbose:
                print(f"Skipping non-string item: {item}")
    return results
''',
]

BAD_CODE_SAMPLES = [
    # Buggy code
    '''
def divide(a, b):
    return a / b
''',
    # Security issues
    '''
def run_command(user_input):
    import os
    os.system("echo " + user_input)
''',
    # Poor style
    '''
def f(x):
    if x>0:
      y=x+1
    else:
        y=x-1
    return y
''',
    # Overly complex
    '''
def process(data):
    result = []
    for i in range(len(data)):
        if data[i] != None:
            if type(data[i]) == int:
                if data[i] > 0:
                    if data[i] % 2 == 0:
                        result.append(data[i] * 2)
                    else:
                        result.append(data[i] * 3)
                else:
                    result.append(0)
            elif type(data[i]) == str:
                if len(data[i]) > 0:
                    result.append(data[i].upper())
    return result
''',
    # No error handling, no types
    '''
def get_data(url):
    import requests
    r = requests.get(url)
    return r.json()
''',
    # Hardcoded secrets
    '''
API_KEY = "sk-1234567890abcdef"
DB_PASSWORD = "supersecret123"

def connect():
    return connect_db(password=DB_PASSWORD)
''',
    # Not pythonic
    '''
def contains_needle(haystack, needle):
    found = False
    for i in range(len(haystack)):
        if haystack[i] == needle:
            found = True
    return found
''',
    # Bare except
    '''
def safe_divide(a, b):
    try:
        return a / b
    except:
        pass
''',
]


def _label_for_code(code: str, is_good: bool) -> Tuple[float, np.ndarray]:
    """
    Create synthetic labels for a code sample.

    Returns:
        (quality_score, issue_probs) where issue_probs is a 6-dim array
    """
    if is_good:
        quality = random.uniform(0.75, 0.98)
        issue_probs = np.array([
            random.uniform(0.0, 0.15),   # bugs
            random.uniform(0.0, 0.25),   # style
            random.uniform(0.0, 0.15),   # performance
            random.uniform(0.0, 0.05),   # security
            random.uniform(0.0, 0.20),   # maintainability
            random.uniform(0.0, 0.15),   # pythonic
        ])
    else:
        quality = random.uniform(0.15, 0.55)
        # Pick 1-3 issue categories to be prominent
        num_issues = random.randint(1, 3)
        issue_indices = random.sample(range(6), num_issues)
        issue_probs = np.random.uniform(0.0, 0.2, size=6)
        for idx in issue_indices:
            issue_probs[idx] = random.uniform(0.5, 0.95)

    return quality, issue_probs.astype(np.float32)


def generate_synthetic_dataset(
    n_samples: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic dataset of good/bad code pairs.

    Returns:
        features: (n, FEATURE_DIM) float32
        qualities: (n,) float32
        issue_labels: (n, 6) float32
    """
    all_codes = GOOD_CODE_SAMPLES + BAD_CODE_SAMPLES
    features_list = []
    qualities_list = []
    issues_list = []

    for _ in range(n_samples):
        # Pick a random code sample and optionally mutate it
        is_good = random.random() > 0.4  # 60% bad, 40% good for balance
        pool = GOOD_CODE_SAMPLES if is_good else BAD_CODE_SAMPLES
        code = random.choice(pool)

        # Add slight variation by repeating or combining
        if random.random() > 0.7 and len(code) < 500:
            code = code + "\n" + random.choice(pool)

        feat = extract_features(code)
        quality, issue_probs = _label_for_code(code, is_good)

        features_list.append(feat)
        qualities_list.append(quality)
        issues_list.append(issue_probs)

    return (
        np.stack(features_list),
        np.array(qualities_list, dtype=np.float32),
        np.stack(issues_list),
    )


# --------------------------------------------------------------------------- #
# Training loop                                                               #
# --------------------------------------------------------------------------- #
def train_synthetic(
    model: CodeFeedbackModel,
    epochs: int = 50,
    lr: float = 3e-4,
    batch_size: int = 32,
    device: str = "cpu",
) -> List[float]:
    """Train on synthetic data. Returns list of epoch losses."""
    print(f"Generating synthetic dataset...")
    features, qualities, issue_labels = generate_synthetic_dataset(n_samples=600)

    # Split 80/20
    n_train = int(0.8 * len(features))
    train_data = TensorDataset(
        torch.from_numpy(features[:n_train]),
        torch.from_numpy(qualities[:n_train]),
        torch.from_numpy(issue_labels[:n_train]),
    )
    val_data = TensorDataset(
        torch.from_numpy(features[n_train:]),
        torch.from_numpy(qualities[n_train:]),
        torch.from_numpy(issue_labels[n_train:]),
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    quality_loss_fn = nn.MSELoss()
    issue_loss_fn = nn.BCEWithLogitsLoss()

    model.to(device)
    epoch_losses = []

    print(f"Training for {epochs} epochs on {n_train} samples...")
    start = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_feats, batch_quality, batch_issues in train_loader:
            batch_feats = batch_feats.to(device)
            batch_quality = batch_quality.to(device)
            batch_issues = batch_issues.to(device)

            optimizer.zero_grad()
            outputs = model(batch_feats, mask_ratio=0.15)

            # Multi-task loss
            loss_q = quality_loss_fn(outputs["quality_score"], batch_quality)
            loss_i = issue_loss_fn(outputs["issue_logits"], batch_issues)
            # Self-supervised reconstruction loss
            loss_r = nn.functional.mse_loss(outputs["reconstruction"], batch_feats)

            loss = loss_q + 0.5 * loss_i + 0.1 * loss_r
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        epoch_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            # Quick validation
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for vf, vq, vi in val_loader:
                    vf, vq, vi = vf.to(device), vq.to(device), vi.to(device)
                    vo = model(vf, mask_ratio=0.0)
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
    print(f"Training complete in {elapsed:.1f}s")
    return epoch_losses


def train_selfsupervised(
    model: CodeFeedbackModel,
    data_dir: str,
    epochs: int = 20,
    lr: float = 3e-4,
    batch_size: int = 32,
    device: str = "cpu",
) -> List[float]:
    """Self-supervised training via masked feature prediction on a user's codebase."""
    print(f"Scanning {data_dir} for Python files...")
    code_samples = []
    for root, dirs, files in os.walk(data_dir):
        # Skip common non-project dirs
        dirs[:] = [d for d in dirs if d not in {
            "__pycache__", ".git", "node_modules", ".venv", "venv",
            "env", ".tox", ".mypy_cache", ".pytest_cache",
        }]
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                        code = fh.read()
                    if len(code) > 50:  # skip tiny files
                        code_samples.append(code)
                except Exception:
                    pass

    if not code_samples:
        print("No Python files found! Falling back to synthetic pretraining.")
        return train_synthetic(model, epochs=epochs, lr=lr, device=device)

    print(f"Extracting features from {len(code_samples)} files...")
    features_list = [extract_features(code) for code in code_samples]
    features = np.stack(features_list)
    dataset = TensorDataset(torch.from_numpy(features))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.to(device)

    print(f"Self-supervised training for {epochs} epochs...")
    epoch_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for (batch_feats,) in loader:
            batch_feats = batch_feats.to(device)
            optimizer.zero_grad()
            outputs = model(batch_feats, mask_ratio=0.20)
            loss = nn.functional.mse_loss(outputs["reconstruction"], batch_feats)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        epoch_losses.append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  recon_loss={avg_loss:.4f}")

    print("Self-supervised training complete.")
    return epoch_losses


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Train CodeFeedbackModel")
    parser.add_argument(
        "--mode", choices=["synthetic", "selfsupervised", "finetune"],
        default="synthetic", help="Training mode"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=str, default="code_feedback_model.pt")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to user codebase (for selfsupervised/finetune)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to existing model to resume from")
    args = parser.parse_args()

    print("=" * 60)
    print("CodeFeedbackModel Training")
    print("=" * 60)

    # Create or load model
    if args.resume:
        print(f"Resuming from {args.resume}")
        model = load_model(args.resume)
    else:
        model = CodeFeedbackModel()

    n_params = model_size_params(model)
    print(f"Model: {n_params:,} parameters ({n_params / 1e6:.2f}M)")
    print(f"Feature dim: {FEATURE_DIM}, Hidden: {HIDDEN_DIM}, "
          f"Layers: {NUM_LAYERS}, Heads: {NUM_HEADS}")

    if args.mode == "synthetic":
        losses = train_synthetic(model, epochs=args.epochs, lr=args.lr,
                                 batch_size=args.batch_size)
    elif args.mode in ("selfsupervised", "finetune"):
        if not args.data_dir:
            print(f"--data-dir is required for mode '{args.mode}'")
            sys.exit(1)
        losses = train_selfsupervised(
            model, args.data_dir, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size,
        )
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)

    # Save
    save_model(model, args.output)
    print(f"\nFinal training loss: {losses[-1]:.4f}")
    print(f"Done! Model saved to {args.output}")


if __name__ == "__main__":
    main()
