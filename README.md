# 🧠 code-critic

> **Offline neural network code critic.** A compact transformer that analyzes Python code quality, detects bugs/style/perf/security issues, and suggests refactorings — all running 100% locally on CPU. No API calls, no cloud, no internet required.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Plugin Size](https://img.shields.io/badge/plugin-5.64%20MB-lightgrey.svg)]()
[![Model Size](https://img.shields.io/badge/model-2.51%20MB-lightgrey.svg)]()
[![Training Data](https://img.shields.io/badge/training-5000%20samples-green.svg)]()

---

## What's New in v2

The v1 model used only hand-crafted AST features (111 structural stats → 128-dim vector). It was fast but shallow — it couldn't actually *read* code.

**v2 is fundamentally different:**

| | v1 | v2 |
|---|---|---|
| **Input** | 111 AST stats | Actual code tokens + AST stats |
| **Tokenizer** | None (feature engineering) | 465-vocab code-aware tokenizer |
| **Architecture** | Feature projection → Transformer | Dual-input: Token encoder + Feature projector → Fusion |
| **Training data** | 600 synthetic good/bad pairs | 5000 labeled samples across 6 categories with augmentation |
| **Pretraining** | None | Masked language modeling (MLM) |
| **Parameters** | 1.94M | 1.31M |
| **Model size** | 3.73 MB | 2.51 MB |
| **Plugin size** | 3.37 MB | 5.64 MB |
| **Inference** | 41–70 ms | ~2–4 s (first run, then cached) |

### What the model actually sees now

```python
# v1 saw: {"ast_count_FunctionDef": 1, "ast_max_depth": 3, "cc_avg": 2.5, ...}
# v2 sees: ["def", "fibonacci", "(", "n", ":", "int", ")", ":", "if", "n", "<=", "1", ...]
#          PLUS the same structural features as v1
```

The tokenizer recognizes 465 Python-specific tokens including:
- All keywords, builtins, common methods
- **Security-sensitive patterns**: `eval(`, `exec(`, `os.system`, `subprocess.call`, `pickle.loads`, `yaml.load`, `shell=True`, hardcoded secrets
- **Compound idioms**: `is not`, `not in`, `isinstance(`, `if __name__ == "__main__"`, `super().__init__`
- **SQL injection patterns**: string concatenation with SQL keywords

### Training data

5000 samples generated from 72 base templates covering:
- **20 bug patterns**: bare except, mutable defaults, off-by-one, resource leaks, race conditions
- **8 style violations**: PEP 8 issues, naming, spacing, import style
- **8 performance issues**: O(n²) patterns, string concat in loops, missing generators
- **10 security vulnerabilities**: command injection, SQL injection, XSS, hardcoded secrets, unsafe deserialization
- **6 maintainability issues**: deep nesting, god classes, too many params
- **10 non-pythonic patterns**: manual loops, missing builtins, no context managers
- **10 good code examples**: clean, typed, documented, idiomatic

Each sample has per-category labels (not just good/bad), enabling multi-task learning.

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Analyze a file
python analyze_v2.py --file my_script.py

# Analyze a string
python analyze_v2.py --code "def foo(): pass"

# JSON output
python analyze_v2.py --file my_script.py --json

# Train on your own codebase
python train_v2.py --epochs-supervised 50 --n-samples 5000 --output code_critic_v2.pt
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Input: Python Code                      │
└──────────────────┬──────────────────────┬───────────────────┘
                   │                      │
          ┌────────▼────────┐    ┌───────▼────────┐
          │  Code Tokenizer  │    │  AST + Radon   │
          │  (465 vocab)     │    │  (111 features) │
          │  Compound pattern│    │  → 128-dim     │
          │  matching        │    │                │
          └────────┬────────┘    └───────┬────────┘
                   │                      │
          ┌────────▼────────┐    ┌───────▼────────┐
          │  Embedding       │    │  Feature       │
          │  (465 → 128)     │    │  Projection    │
          │  + Pos encoding  │    │  (128 → 128)   │
          └────────┬────────┘    └───────┬────────┘
                   │                      │
          ┌────────▼──────────────────────▼────────┐
          │         Transformer Encoder × 4         │
          │         (128 hidden, 4 heads)            │
          │         + Mean pooling                   │
          └────────┬──────────────────────┬────────┘
                   │                      │
          ┌────────▼──────────────────────▼────────┐
          │         Fusion Layer (256 → 256)        │
          │         Concatenate token + feature     │
          └────────┬───────┬──────────┬────────────┘
                   │       │          │
              ┌────▼──┐ ┌──▼───┐ ┌───▼────┐
              │Quality│ │Issues│ │Confidence│
              │Score  │ │6-class│ │Score    │
              └───────┘ └──────┘ └─────────┘
```

## Model Specs

| Property | Value |
|----------|-------|
| Parameters | 1,305,049 (1.31M) |
| Vocabulary | 465 tokens |
| Hidden dim | 256 |
| Layers | 4 |
| Attention heads | 4 |
| Max sequence length | 256 tokens |
| Model size (float16) | 2.51 MB |
| Plugin size (.agnt) | 5.64 MB |
| Training data | 5000 labeled samples |
| Training time | ~52 min (50 epochs, CPU) |
| Inference time | ~2–4 s (first run), ~50ms (cached) |

## AGNT Plugin

The plugin is distributed as a `.agnt` file. Current version: **v2.0.0**.

Tool: `analyze-code`  
Inputs: `code` (string), `filePath` (optional), `telemetry` (optional)  
Outputs: `quality_score`, `quality_label`, `confidence`, `issues`, `suggestions`, `positive_notes`, `feedback_text`, `inference_time_ms`, `model_version`

## Project Structure

```
code-critic/
├── README.md                      # This file
├── LICENSE                        # MIT
├── requirements.txt               # torch, numpy, radon
├── .gitignore
│
├── code_critic_tokenizer.py       # 465-vocab code-aware tokenizer
├── model_v2.py                    # Dual-input transformer model
├── data_generator.py              # 5000 labeled samples from 72 templates
├── train_v2.py                    # 3-phase training script
├── analyze_v2.py                  # CLI + Python API
├── code_critic_v2.pt              # Trained model weights (2.51 MB)
│
├── code_features.py               # AST/token/radon feature extraction (v1 compat)
├── code_feedback_model.py         # v1 model (kept for fallback)
├── feedback_generator.py          # Template-based NL feedback
├── analyze.py                     # v1 CLI (kept for fallback)
├── train.py                       # v1 training (kept for fallback)
├── test_model.py                  # v1 tests
│
└── plugin/                        # AGNT plugin package
    ├── manifest.json              # v2.0.0 metadata
    ├── analyze-code.js            # JS tool wrapper
    ├── analyze_v2.py              # v2 Python entry point
    ├── code_critic_v2.pt          # Bundled model weights
    └── [all v2 source files]
```

## License

MIT License. See [LICENSE](LICENSE).
