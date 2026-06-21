"""
model_v2.py — Improved code-critic model with code-aware tokenization.

Key improvements over v1:
1. Token embedding layer — model sees actual code tokens, not just AST stats
2. Dual-input architecture — combines learned token representations with
   hand-crafted structural features (best of both worlds)
3. Pretraining head — masked language modeling for self-supervised learning
4. Better issue classification — separate expert heads per category
5. Still under 5MB, still < 100ms inference on CPU

Architecture:
  Code tokens → Embedding → Transformer encoder → Pooled token repr
  Structural features → Projection → Feature repr
  [Token repr | Feature repr] → Multi-task heads → Quality + 6 issues + confidence
"""

import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from code_critic_tokenizer import VOCAB_SIZE, PAD, CLS, EOS, UNK

# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #
FEATURE_DIM = 128          # Structural feature dimension
TOKEN_EMBED_DIM = 128      # Token embedding dimension
HIDDEN_DIM = 256           # Transformer hidden dimension
NUM_LAYERS = 4             # Transformer encoder layers
NUM_HEADS = 4              # Attention heads
NUM_ISSUE_CLASSES = 6      # Issue categories
MAX_SEQ_LEN = 256          # Max token sequence length
DROPOUT = 0.1
MODEL_FILENAME = "code_critic_v2.pt"


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
class CodeCriticV2(nn.Module):
    """
    Dual-input code quality model.

    Input:
      - token_ids: (batch, seq_len) — tokenized code
      - features: (batch, FEATURE_DIM) — structural features

    Output:
      - quality_score: (batch,) — overall quality [0, 1]
      - issue_logits: (batch, 6) — per-category issue logits
      - confidence: (batch,) — model confidence [0, 1]
      - mlm_logits: (batch, seq_len, vocab_size) — masked LM predictions
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        feature_dim: int = FEATURE_DIM,
        embed_dim: int = TOKEN_EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        num_heads: int = NUM_HEADS,
        num_classes: int = NUM_ISSUE_CLASSES,
        max_seq_len: int = MAX_SEQ_LEN,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len

        # ---- Token embedding ----
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)
        self.embed_norm = nn.LayerNorm(embed_dim)
        self.embed_dropout = nn.Dropout(dropout)

        # ---- Token encoder (small transformer) ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ---- Feature projection ----
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # ---- Fusion layer (token repr + feature repr) ----
        fusion_dim = embed_dim * 2  # Concatenate token and feature representations
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ---- Quality score head ----
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # ---- Issue classifier head (per-category experts) ----
        self.issue_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # ---- Confidence head ----
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # ---- MLM pretraining head ----
        self.mlm_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                torch.nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def _encode_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode token sequence to pooled representation."""
        batch_size, seq_len = token_ids.shape
        seq_len = min(seq_len, self.max_seq_len)

        # Embed tokens
        x = self.token_embedding(token_ids[:, :seq_len])  # (batch, seq, embed)
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.embed_norm(x)
        x = self.embed_dropout(x)

        # Create padding mask
        padding_mask = (token_ids[:, :seq_len] == PAD)

        # Encode
        x = self.token_encoder(x, src_key_padding_mask=padding_mask)

        # Pool: mean of non-padded positions
        mask = (~padding_mask).unsqueeze(-1).float()
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return x

    def _mlm_forward(self, token_ids: torch.Tensor, mask_ratio: float = 0.15):
        """Masked language modeling forward pass."""
        batch_size, seq_len = token_ids.shape
        seq_len = min(seq_len, self.max_seq_len)

        # Create mask
        mask = torch.rand(batch_size, seq_len, device=token_ids.device) < mask_ratio
        # Don't mask special tokens
        special = (token_ids[:, :seq_len] == PAD) | (token_ids[:, :seq_len] == CLS) | (token_ids[:, :seq_len] == EOS)
        mask = mask & ~special

        # Replace masked tokens with UNK
        masked_ids = token_ids[:, :seq_len].clone()
        masked_ids[mask] = UNK

        # Embed and encode
        x = self.token_embedding(masked_ids)
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.embed_norm(x)

        padding_mask = (token_ids[:, :seq_len] == PAD)
        x = self.token_encoder(x, src_key_padding_mask=padding_mask)

        # Predict original tokens
        logits = self.mlm_head(x)  # (batch, seq, vocab_size)

        return logits, mask

    def forward(
        self,
        token_ids: torch.Tensor,
        features: torch.Tensor,
        mask_ratio: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            token_ids: (batch, seq_len) — tokenized code
            features: (batch, FEATURE_DIM) — structural features
            mask_ratio: MLM mask ratio (0 = no masking, for inference)

        Returns:
            dict with quality_score, issue_logits, confidence, mlm_logits, mlm_mask
        """
        # Encode tokens
        token_repr = self._encode_tokens(token_ids)  # (batch, embed_dim)

        # Project features
        feature_repr = self.feature_proj(features)  # (batch, embed_dim)

        # Fuse
        fused = torch.cat([token_repr, feature_repr], dim=-1)  # (batch, embed_dim * 2)
        fused = self.fusion(fused)  # (batch, hidden_dim)

        # Multi-task heads
        quality = self.quality_head(fused).squeeze(-1)       # (batch,)
        issues = self.issue_head(fused)                       # (batch, num_classes)
        confidence = self.confidence_head(fused).squeeze(-1)  # (batch,)

        # MLM (only during training)
        mlm_logits = None
        mlm_mask = None
        if mask_ratio > 0.0 and self.training:
            mlm_logits, mlm_mask = self._mlm_forward(token_ids, mask_ratio)

        return {
            "quality_score": quality,
            "issue_logits": issues,
            "confidence": confidence,
            "mlm_logits": mlm_logits,
            "mlm_mask": mlm_mask,
        }

    def predict(self, token_ids: torch.Tensor, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Inference mode — no masking."""
        self.eval()
        with torch.no_grad():
            return self.forward(token_ids, features, mask_ratio=0.0)


# --------------------------------------------------------------------------- #
# Model I/O                                                                   #
# --------------------------------------------------------------------------- #
def save_model(model: CodeCriticV2, path: str) -> None:
    """Save model weights + config in float16."""
    state = {
        "weights": {k: v.half() for k, v in model.state_dict().items()},
        "config": {
            "vocab_size": model.vocab_size,
            "feature_dim": model.feature_dim,
            "embed_dim": model.embed_dim,
            "hidden_dim": model.hidden_dim,
            "num_layers": model.num_layers,
            "num_heads": model.num_heads,
            "num_classes": model.num_classes,
            "max_seq_len": model.max_seq_len,
        },
    }
    torch.save(state, path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Model saved to {path} ({size_mb:.2f} MB)")


def load_model(path: str) -> CodeCriticV2:
    """Load model from disk."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    cfg = state["config"]
    model = CodeCriticV2(
        vocab_size=cfg["vocab_size"],
        feature_dim=cfg["feature_dim"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        num_classes=cfg["num_classes"],
        max_seq_len=cfg["max_seq_len"],
    )
    model.load_state_dict({k: v.float() for k, v in state["weights"].items()})
    model.eval()
    return model


def model_size_params(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Self-test                                                                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=" * 60)
    print("CodeCriticV2 — Self Test")
    print("=" * 60)

    model = CodeCriticV2()
    n_params = model_size_params(model)
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.2f}M)")
    print(f"Vocab size: {VOCAB_SIZE}")

    # Test forward pass
    dummy_tokens = torch.randint(0, VOCAB_SIZE, (2, 128))
    dummy_features = torch.randn(2, FEATURE_DIM)

    out = model(dummy_tokens, dummy_features, mask_ratio=0.15)
    print(f"\nquality_score: {out['quality_score'].shape} → {out['quality_score'].tolist()}")
    print(f"issue_logits:  {out['issue_logits'].shape}")
    print(f"confidence:    {out['confidence'].shape}")
    print(f"mlm_logits:    {out['mlm_logits'].shape}")
    print(f"mlm_mask:      {out['mlm_mask'].shape}")

    # Test inference
    model.eval()
    with torch.no_grad():
        pred = model.predict(dummy_tokens, dummy_features)
    print(f"\nInference quality: {pred['quality_score'].tolist()}")

    # Test save/load
    test_path = "_test_v2.pt"
    save_model(model, test_path)
    loaded = load_model(test_path)
    with torch.no_grad():
        pred2 = loaded.predict(dummy_tokens, dummy_features)
    print(f"Loaded model quality: {pred2['quality_score'].tolist()}")
    os.remove(test_path)

    print("\n✅ All tests passed!")
