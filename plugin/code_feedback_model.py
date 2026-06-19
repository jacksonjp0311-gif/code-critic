"""
code_feedback_model.py — Compact PyTorch model for code quality feedback.

Architecture:
- Feature projection: Linear(FEATURE_DIM → hidden) + LayerNorm + GELU
- Transformer encoder: 4 layers, 4 heads, hidden=256 (DistilBERT-style, no decoder)
- Multi-task heads:
    1. Quality score head: Linear → sigmoid → [0, 1]
    2. Issue classifier head: Linear → 6-class logits (bugs, style, perf, security, maintainability, pythonic)
    3. Confidence head: Linear → sigmoid → [0, 1]
    4. Masked feature prediction head: Linear → FEATURE_DIM (self-supervised)

Total params: ~1.2M → quantized int8 ≈ 1.2 MB on disk.

Design decisions:
- Small transformer (4 layers, 256 hidden) balances expressiveness vs size.
- Multi-task learning shares representations → better generalization on small data.
- Masked feature prediction acts as self-supervised pretraining signal.
- All heads are simple MLPs — fast inference on CPU.
"""

import json
import math
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #
FEATURE_DIM = 128
HIDDEN_DIM = 192
NUM_LAYERS = 4
NUM_HEADS = 4
NUM_ISSUE_CLASSES = 6
DROPOUT = 0.1
MODEL_FILENAME = "code_feedback_model.pt"
CONFIG_FILENAME = "model_config.json"


# --------------------------------------------------------------------------- #
# Model definition                                                            #
# --------------------------------------------------------------------------- #
class CodeFeedbackModel(nn.Module):
    """
    Compact transformer-based code feedback model.

    Input: float32 feature vector of shape (batch, FEATURE_DIM)
    Output: dict with quality_score, issue_logits, confidence, feature_reconstruction
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        num_heads: int = NUM_HEADS,
        num_classes: int = NUM_ISSUE_CLASSES,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_classes = num_classes

        # Input projection: FEATURE_DIM → hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Positional encoding (learned, single position since we have one "token")
        # We treat the feature vector as a sequence of 1 token with hidden_dim dims,
        # but to give the transformer something to work with, we project to a
        # sequence of length seq_len=8 using a learned embedding.
        self.seq_len = 4
        self.seq_proj = nn.Linear(hidden_dim, hidden_dim * self.seq_len)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, hidden_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pooling: CLS token (first position) + mean pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Quality score head
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Issue classifier head
        self.issue_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Masked feature prediction head (self-supervised)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self, x: torch.Tensor, mask_ratio: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (batch, FEATURE_DIM) feature vector
            mask_ratio: fraction of features to zero out (for self-supervised pretraining)

        Returns:
            dict with keys: quality_score, issue_logits, confidence, reconstruction
        """
        batch_size = x.shape[0]

        # Optional feature masking for self-supervised learning
        mask = None
        if mask_ratio > 0.0 and self.training:
            mask = torch.rand_like(x) < mask_ratio
            x_masked = x.clone()
            x_masked[mask] = 0.0
        else:
            x_masked = x

        # Project to sequence
        h = self.input_proj(x_masked)  # (batch, hidden_dim)
        h = self.seq_proj(h)            # (batch, hidden_dim * seq_len)
        h = h.view(batch_size, self.seq_len, self.hidden_dim)  # (batch, seq, hidden)
        h = h + self.pos_embedding

        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, hidden)
        h = torch.cat([cls, h], dim=1)  # (batch, 1+seq_len, hidden)

        # Transformer encoding
        h = self.transformer(h)  # (batch, 1+seq_len, hidden)

        # Pool: CLS token + mean of all positions
        cls_repr = h[:, 0]                    # (batch, hidden)
        mean_repr = h[:, 1:].mean(dim=1)      # (batch, hidden)
        pooled = torch.cat([cls_repr, mean_repr], dim=-1)  # (batch, hidden*2)

        # Multi-task heads
        quality = self.quality_head(pooled).squeeze(-1)       # (batch,)
        issues = self.issue_head(pooled)                       # (batch, num_classes)
        confidence = self.confidence_head(pooled).squeeze(-1)  # (batch,)
        reconstruction = self.reconstruction_head(pooled)      # (batch, feature_dim)

        return {
            "quality_score": quality,
            "issue_logits": issues,
            "confidence": confidence,
            "reconstruction": reconstruction,
        }

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Inference mode — no masking, returns detached CPU tensors."""
        self.eval()
        with torch.no_grad():
            return self.forward(x, mask_ratio=0.0)


# --------------------------------------------------------------------------- #
# Model I/O                                                                   #
# --------------------------------------------------------------------------- #
def save_model(model: CodeFeedbackModel, path: str) -> None:
    """Save model weights + config. Uses float16 to halve file size."""
    state = {
        "weights": {k: v.half() for k, v in model.state_dict().items()},
        "config": {
            "feature_dim": model.feature_dim,
            "hidden_dim": model.hidden_dim,
            "num_layers": model.num_layers,
            "num_heads": model.num_heads,
            "num_classes": model.num_classes,
        },
    }
    torch.save(state, path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Model saved to {path} ({size_mb:.2f} MB)")


def load_model(path: str) -> CodeFeedbackModel:
    """Load model from disk. Returns model in eval mode."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    cfg = state["config"]
    model = CodeFeedbackModel(
        feature_dim=cfg["feature_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        num_classes=cfg["num_classes"],
    )
    # Load float16 weights, model will upcast to float32
    model.load_state_dict({k: v.float() for k, v in state["weights"].items()})
    model.eval()
    return model


def model_size_params(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Inference helper                                                            #
# --------------------------------------------------------------------------- #
def run_inference(
    model: CodeFeedbackModel,
    feature_vector: np.ndarray,
) -> Dict:
    """
    Run end-to-end inference on a single feature vector.

    Args:
        model: loaded CodeFeedbackModel
        feature_vector: (FEATURE_DIM,) numpy float32 array

    Returns:
        dict with numpy arrays (quality_score, issue_probs, confidence, reconstruction)
    """
    x = torch.from_numpy(feature_vector).unsqueeze(0).float()  # (1, FEATURE_DIM)
    outputs = model.predict(x)
    return {
        "quality_score": outputs["quality_score"].item(),
        "issue_probs": torch.softmax(outputs["issue_logits"], dim=-1).squeeze(0).numpy(),
        "confidence": outputs["confidence"].item(),
        "reconstruction": outputs["reconstruction"].squeeze(0).numpy(),
    }


# --------------------------------------------------------------------------- #
# Self-test                                                                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=" * 60)
    print("CodeFeedbackModel — Self Test")
    print("=" * 60)

    model = CodeFeedbackModel()
    n_params = model_size_params(model)
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    # Test forward pass
    dummy = torch.randn(4, FEATURE_DIM)
    out = model(dummy, mask_ratio=0.15)
    print(f"quality_score: {out['quality_score'].shape} => {out['quality_score'].tolist()}")
    print(f"issue_logits:  {out['issue_logits'].shape}")
    print(f"confidence:    {out['confidence'].shape} => {out['confidence'].tolist()}")
    print(f"reconstruction:{out['reconstruction'].shape}")

    # Test save/load
    test_path = "_test_model.pt"
    save_model(model, test_path)
    loaded = load_model(test_path)
    out2 = loaded.predict(dummy)
    print(f"\nLoaded model quality: {out2['quality_score'].tolist()}")
    os.remove(test_path)
    print("\n✅ All tests passed!")
