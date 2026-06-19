# 🧠 code-critic

> **Offline neural network code critic for AGNT.** A compact 1.94M-parameter transformer that analyzes Python code quality, detects bugs/style/perf/security issues, and suggests refactorings — all running 100% locally on CPU in under 70ms. Ships as a first-class AGNT plugin (3.37 MB) with workflow node support and fine-tuning on your own codebase.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Size](https://img.shields.io/badge/plugin%20size-3.37%20MB-lightgrey.svg)](#)
[![Tests](https://img.shields.io/badge/tests-43%2F%20passing-brightgreen.svg)](#)

---

## Table of Contents

- [What It Is](#what-it-is)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Command Line](#command-line)
  - [Python API](#python-api)
  - [AGNT Plugin](#agnt-plugin)
- [Training](#training)
  - [Synthetic Pretraining](#synthetic-pretraining)
  - [Self-Supervised on Your Codebase](#self-supervised-on-your-codebase)
  - [Fine-Tuning](#fine-tuning)
- [Output Format](#output-format)
- [Model Specs](#model-specs)
- [Project Structure](#project-structure)
- [AGNT Workflow Example](#agnt-workflow-example)
- [Benchmarks](#benchmarks)
- [Roadmap](#roadmap)
- [License](#license)

---

## What It Is

`code-critic` is a **production-ready neural network** that acts as an always-available "code critic." Feed it a Python code snippet or full file, and it returns:

- **Overall quality score** (0–1)
- **Categorized issues** (bugs, style, performance, security, maintainability, pythonic-ness)
- **Ranked refactoring suggestions**
- **Positive notes** when code is strong
- **Confidence score** on its own feedback

It runs **100% offline** — no API calls, no cloud, no internet required. The entire plugin is **3.37 MB** and inference takes **41–70 ms** on CPU.

---

## Key Features

- 🧠 **Real neural network** — 1.94M-parameter transformer, not a prompt wrapper
- ⚡ **Fast** — 41–70 ms inference on consumer CPU
- 📦 **Tiny** — 3.37 MB plugin (model + code + weights)
- 🔒 **Fully offline** — zero network calls after installation
- 🔧 **Fine-tunable** — learn from your own codebase
- 🔌 **AGNT plugin** — first-class workflow node integration
- ✅ **Tested** — 43/43 unit tests passing

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Input: Python Code                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Feature Extraction                         │
│  ┌──────────┐  ┌──────────┐  ┌────────┐  ┌──────────────┐  │
│  │ AST Walk │  │ Tokenize │  │  Radon │  │  Telemetry   │  │
│  │ 40 feats │  │ 20 feats │  │ 18 feat │  │  8 feats     │  │
│  └──────────┘  └──────────┘  └────────┘  └──────────────┘  │
│                    → 111 raw features                        │
│                    → log-scale + L2 normalize                │
│                    → 128-dim float32 vector                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Transformer Encoder (1.94M params)              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Input Projection: Linear(128 → 192) + GELU        │    │
│  │  Sequence Projection: Linear(192 → 192×4)          │    │
│  │  Positional Embeddings (learned, length 4)          │    │
│  │  CLS Token + Transformer × 4 layers                 │    │
│  │  (hidden=192, heads=4, ff_dim=576, pre-norm)        │    │
│  │  Pooling: CLS + Mean → 384-dim                      │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────┼────────────┬──────────────┐
              ▼            ▼            ▼              ▼
┌──────────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
│  Quality Score   │ │  Issue   │ │  Conf.   │ │ Reconstruct  │
│  Head → [0,1]   │ │ 6-class  │ │ → [0,1]  │ │ Head (SSL)   │
└──────────────────┘ └──────────┘ └──────────┘ └──────────────┘
              │            │            │              │
              └────────────┴────────────┴──────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Feedback Generator                         │
│  Template-based NL generation from model outputs             │
│  → Quality label + emoji                                    │
│  → Categorized issues with severity (critical/high/med/low) │
│  → Ranked refactoring suggestions                           │
│  → Positive notes for strong code                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model (Synthetic Pretraining)

```bash
python train.py --mode synthetic --epochs 50 --output code_feedback_model.pt
```

This takes ~2 minutes on CPU and produces a reasonable starting model.

### 3. Analyze Code

```bash
# Analyze a file
python analyze.py --file my_script.py

# Analyze a code string
python analyze.py --code "def foo(): pass"

# JSON output for scripting
python analyze.py --file my_script.py --json

# With git/edit telemetry
python analyze.py --file my_script.py --telemetry '{"num_edits": 5, "num_additions": 20}'
```

### 4. Run Tests

```bash
python test_model.py
```

---

## Usage

### Command Line

```bash
# Basic file analysis
python analyze.py --file src/my_module.py

# Code string
python analyze.py --code "import os\ndef run(cmd): os.system(cmd)"

# JSON output (for CI/CD integration)
python analyze.py --file src/my_module.py --json

# With telemetry context
python analyze.py --file src/my_module.py --telemetry '{"num_edits": 15, "num_authors": 3, "file_age_days": 30}'

# Custom model path
python analyze.py --file src/my_module.py --model path/to/custom_model.pt
```

### Python API

```python
from analyze import analyze_code

result = analyze_code(
    code="def hello():\n    print('world')",
    file_path="greetings.py",
    telemetry={"num_edits": 3}
)

print(result["quality_score"])   # 0.8723
print(result["quality_label"])   # "Good"
print(result["confidence"])      # 0.936
print(result["feedback_text"])   # Full human-readable report

for issue in result["issues"]:
    print(f"[{issue['severity']}] {issue['category']}: {issue['description']}")
```

### AGNT Plugin

The plugin is distributed as a `.agnt` file. To install:

1. Copy `code-critic.agnt` to your AGNT plugins folder
2. Or use AGNT's plugin installer: `/api/plugins/install-file`
3. Hot-reload: `/api/plugins/reload`

The `analyze-code` tool accepts:
- `code` (string, required) — Python source code
- `filePath` (string, optional) — file path for context
- `telemetry` (string/JSON, optional) — git diffs, edit counts, etc.

Returns: `quality_score`, `quality_label`, `confidence`, `issues`, `suggestions`, `positive_notes`, `feedback_text`, `inference_time_ms`.

---

## Training

### Synthetic Pretraining

Generates 600 labeled good/bad code pairs and trains all heads simultaneously:

```bash
python train.py --mode synthetic --epochs 50 --output code_feedback_model.pt
```

Training converges in ~24 seconds on CPU. Loss typically drops from ~0.50 to ~0.23.

### Self-Supervised on Your Codebase

Scans your project for `.py` files and trains via masked feature prediction (no labels needed):

```bash
python train.py --mode selfsupervised --data-dir ./my_project --epochs 20
```

This learns the structural patterns of your codebase, making the critic more calibrated to your style.

### Fine-Tuning

Continue training a pretrained model on your codebase with a lower learning rate:

```bash
python train.py --mode finetune --data-dir ./my_project \
    --resume code_feedback_model.pt --epochs 10 --lr 1e-4
```

---

## Output Format

```json
{
  "quality_score": 0.8723,
  "quality_label": "Good",
  "quality_emoji": "✅",
  "confidence": 0.936,
  "issues": [
    {
      "category": "style",
      "severity": "low",
      "probability": 0.15,
      "description": "Minor style nits: consider consistent quote usage."
    }
  ],
  "suggestions": [
    {
      "priority": "low",
      "category": "style",
      "suggestion": "Minor style nits: consider consistent quote usage.",
      "confidence": 0.14
    }
  ],
  "positive_notes": [
    "Clean, well-structured code. 👍",
    "Well-documented with clear docstrings."
  ],
  "feedback_text": "✅ Code Quality: Good (87.2%)\n   Model Confidence: 93.6%\n\n📋 Issues Found:\n   ...",
  "inference_time_ms": 70
}
```

---

## Model Specs

| Property | Value |
|----------|-------|
| Architecture | Transformer encoder (4 layers, 192 hidden, 4 heads) |
| Parameters | 1,942,984 (~1.94M) |
| Feature dimension | 128 |
| Issue categories | 6 (bugs, style, perf, security, maintainability, pythonic) |
| Model format | PyTorch state dict, float16 quantized |
| Disk size | 3.73 MB |
| Plugin size (.agnt) | 3.37 MB |
| Inference time | 41–70 ms (CPU) |
| Training time | ~24s (50 epochs, synthetic, CPU) |
| Python | 3.10+ |
| PyTorch | 2.1+ (CPU) |

### Size Comparison

| Model | Parameters | Disk Size |
|-------|-----------|-----------|
| DistilBERT | 66M | ~250 MB |
| BERT-base | 110M | ~440 MB |
| **code-critic** | **1.94M** | **3.73 MB** |

---

## Project Structure

```
code-critic/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── code_features.py             # Feature extraction (AST, tokenize, radon)
├── code_feedback_model.py       # PyTorch model definition
├── feedback_generator.py        # NL feedback generation
├── train.py                     # Training script (3 modes)
├── analyze.py                   # CLI + Python API entry point
├── test_model.py                # 43 unit tests
├── code_feedback_model.pt       # Pretrained model weights
│
└── plugin/                      # AGNT plugin package
    ├── manifest.json            # Plugin metadata + tool schema
    ├── package.json             # ES module config
    ├── analyze-code.js          # JS tool (spawns Python subprocess)
    ├── code_features.py         # (copied)
    ├── code_feedback_model.py   # (copied)
    ├── code_feedback_model.pt   # (copied)
    ├── feedback_generator.py    # (copied)
    ├── analyze.py               # (copied)
    └── train.py                 # (copied)
```

---

## AGNT Workflow Example

Use `code-critic` in an automated code review loop:

```
[Code Generation Agent]
        │
        ▼
[code-critic: analyze-code] ──→ quality_score, issues, suggestions
        │
        ▼
[Condition: quality >= 0.75?]
    │           │
   YES          NO
    │           │
    ▼           ▼
 [Done]    [Refactor Agent]
              │ (feedback + code)
              ▼
         [code-critic: analyze-code] ──→ final feedback
              │
              ▼
            [Done]
```

See [WORKFLOW_EXAMPLE.md](WORKFLOW_EXAMPLE.md) for detailed setup instructions.

---

## Benchmarks

Tested on CPU (Intel/AMD, no GPU):

| Metric | Value |
|--------|-------|
| Feature extraction (100-line file) | ~15 ms |
| Model inference | ~30 ms |
| End-to-end (extract + infer + generate) | 41–70 ms |
| Memory footprint | ~50 MB RAM |
| Training (30 epochs, 600 samples) | 24 seconds |

---

## Roadmap

- [ ] LoRA fine-tuning support for efficient personalization
- [ ] Multi-language support (JavaScript, TypeScript, Rust)
- [ ] VS Code extension wrapper
- [ ] CI/CD GitHub Action integration
- [ ] AST-based diff analysis for PR reviews
- [ ] Incremental learning from AGNT agent edit telemetry

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

Built with ❤️ for the AGNT ecosystem.
