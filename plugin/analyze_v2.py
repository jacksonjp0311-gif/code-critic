"""
analyze_v2.py — CLI for CodeCriticV2.

Uses the improved model with code-aware tokenization.
Falls back to v1 model if v2 not available.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from code_features import extract_features, FEATURE_DIM
from code_critic_tokenizer import CodeTokenizer, MAX_SEQ_LEN
from feedback_generator import generate_feedback


def _find_model():
    candidates = [
        SCRIPT_DIR / "code_critic_v2.pt",
        SCRIPT_DIR / "code_feedback_model.pt",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return str(candidates[0])


def _load_model(model_path):
    """Load v2 or v1 model depending on file."""
    if "v2" in model_path:
        from model_v2 import load_model, CodeCriticV2
        return load_model(model_path), "v2"
    else:
        from code_feedback_model import load_model, CodeFeedbackModel
        return load_model(model_path), "v1"


def analyze_code(code, file_path=None, telemetry=None, model_path=None):
    if model_path is None:
        model_path = _find_model()

    if not os.path.exists(model_path):
        return {"error": f"Model not found: {model_path}", "quality_score": 0,
                "issues": [], "suggestions": [], "positive_notes": [],
                "confidence": 0, "feedback_text": "⚠️ Model not found."}

    model, version = _load_model(model_path)

    if not hasattr(analyze_code, "_tokenizer"):
        analyze_code._tokenizer = CodeTokenizer(max_length=MAX_SEQ_LEN)
    tokenizer = analyze_code._tokenizer

    features = extract_features(code, file_path=file_path, telemetry=telemetry)

    import torch
    if version == "v2":
        token_ids = tokenizer.encode(code)
        tokens_tensor = torch.from_numpy(token_ids).unsqueeze(0)
        features_tensor = torch.from_numpy(features).unsqueeze(0).float()
        outputs = model.predict(tokens_tensor, features_tensor)
    else:
        features_tensor = torch.from_numpy(features).unsqueeze(0).float()
        outputs = model.predict(features_tensor)

    issue_probs = torch.softmax(outputs["issue_logits"], dim=-1).squeeze(0).numpy()
    inference_output = {
        "quality_score": outputs["quality_score"].item(),
        "issue_probs": issue_probs,
        "confidence": outputs["confidence"].item(),
        "reconstruction": np.zeros(FEATURE_DIM),
    }

    feedback = generate_feedback(inference_output, code=code, file_path=file_path)
    feedback["model_path"] = model_path
    feedback["model_version"] = version
    feedback["feature_dim"] = FEATURE_DIM
    feedback["code_length"] = len(code)
    feedback["file_path"] = file_path
    return feedback


def main():
    parser = argparse.ArgumentParser(description="Code Critic v2 — Analyze Python code")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str)
    group.add_argument("--code", type=str)
    parser.add_argument("--telemetry", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()

    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            code = f.read()
        file_path = args.file
    else:
        code = args.code
        file_path = None

    telemetry = None
    if args.telemetry:
        try:
            telemetry = json.loads(args.telemetry)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid telemetry JSON: {e}", file=sys.stderr)
            sys.exit(1)

    start = time.time()
    result = analyze_code(code, file_path=file_path, telemetry=telemetry, model_path=args.model)
    elapsed = (time.time() - start) * 1000

    if "error" in result and result["error"]:
        print(f"⚠️ {result['error']}", file=sys.stderr)

    if args.json_output:
        output = {
            "quality_score": result.get("quality_score", 0),
            "quality_label": result.get("quality_label", "Unknown"),
            "quality_emoji": result.get("quality_emoji", ""),
            "confidence": result.get("confidence", 0),
            "issues": result.get("issues", []),
            "suggestions": result.get("suggestions", []),
            "positive_notes": result.get("positive_notes", []),
            "model_version": result.get("model_version", "unknown"),
            "inference_time_ms": round(elapsed, 1),
        }
        print(json.dumps(output, indent=2))
    else:
        print(result.get("feedback_text", "No feedback generated."))
        print(f"\n⏱ Inference time: {elapsed:.0f}ms | Model: {result.get('model_version', '?')}")


if __name__ == "__main__":
    main()
