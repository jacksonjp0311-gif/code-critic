"""
analyze.py — Standalone CLI for code analysis.

Usage:
    python analyze.py --file example.py
    python analyze.py --code "def foo(): pass"
    python analyze.py --file example.py --telemetry '{"num_edits": 5}'
    python analyze.py --file example.py --json

This is the main entry point for both standalone use and the AGNT plugin.
The AGNT plugin's Python wrapper calls this CLI via subprocess.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Project root resolution (for imports when run from plugin context)           #
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from code_features import extract_features, FEATURE_DIM
from code_feedback_model import load_model, run_inference
from feedback_generator import generate_feedback


# --------------------------------------------------------------------------- #
# Model path resolution                                                       #
# --------------------------------------------------------------------------- #
def _find_model() -> str:
    """Search for the model file in common locations."""
    candidates = [
        SCRIPT_DIR / "code_feedback_model.pt",
        SCRIPT_DIR / "models" / "code_feedback_model.pt",
        SCRIPT_DIR / "plugin" / "code_feedback_model.pt",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # If not found, return the default path (will error with a clear message)
    return str(SCRIPT_DIR / "code_feedback_model.pt")


# --------------------------------------------------------------------------- #
# Main analysis function                                                      #
# --------------------------------------------------------------------------- #
def analyze_code(
    code: str,
    file_path: str = None,
    telemetry: dict = None,
    model_path: str = None,
) -> dict:
    """
    Analyze a code snippet and return structured feedback.

    This is the primary function called by the AGNT plugin wrapper.

    Args:
        code: Python source code string
        file_path: optional file path for context
        telemetry: optional dict with git/edit metadata
        model_path: optional path to model weights file

    Returns:
        dict with quality_score, issues, suggestions, feedback_text, etc.
    """
    if model_path is None:
        model_path = _find_model()

    if not os.path.exists(model_path):
        return {
            "error": f"Model not found at {model_path}. Run 'python train.py' first.",
            "quality_score": 0.0,
            "issues": [],
            "suggestions": [],
            "positive_notes": [],
            "confidence": 0.0,
            "feedback_text": "⚠️ Model not trained yet. Run training first.",
        }

    # Load model (cached after first call via a simple module-level cache)
    if not hasattr(analyze_code, "_model_cache"):
        analyze_code._model_cache = {}
    if model_path not in analyze_code._model_cache:
        analyze_code._model_cache[model_path] = load_model(model_path)
    model = analyze_code._model_cache[model_path]

    # Extract features
    features = extract_features(code, file_path=file_path, telemetry=telemetry)

    # Run inference
    inference_output = run_inference(model, features)

    # Generate feedback
    feedback = generate_feedback(inference_output, code=code, file_path=file_path)

    # Add metadata
    feedback["model_path"] = model_path
    feedback["feature_dim"] = FEATURE_DIM
    feedback["code_length"] = len(code)
    feedback["file_path"] = file_path

    return feedback


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Code Feedback Neural Net — Analyze Python code quality"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Path to a Python file to analyze")
    group.add_argument("--code", type=str, help="Code string to analyze directly")
    parser.add_argument("--telemetry", type=str, default=None,
                        help="JSON string of telemetry data")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model weights file")
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON instead of formatted text")
    args = parser.parse_args()

    # Get code
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

    # Parse telemetry
    telemetry = None
    if args.telemetry:
        try:
            telemetry = json.loads(args.telemetry)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid telemetry JSON: {e}", file=sys.stderr)
            sys.exit(1)

    # Analyze
    start = time.time()
    result = analyze_code(code, file_path=file_path, telemetry=telemetry,
                          model_path=args.model)
    elapsed = (time.time() - start) * 1000  # ms

    if "error" in result and result["error"]:
        print(f"⚠️ {result['error']}", file=sys.stderr)

    if args.json:
        # JSON output for machine consumption
        output = {
            "quality_score": result.get("quality_score", 0),
            "quality_label": result.get("quality_label", "Unknown"),
            "quality_emoji": result.get("quality_emoji", ""),
            "confidence": result.get("confidence", 0),
            "issues": result.get("issues", []),
            "suggestions": result.get("suggestions", []),
            "positive_notes": result.get("positive_notes", []),
            "inference_time_ms": round(elapsed, 1),
        }
        print(json.dumps(output, indent=2))
    else:
        # Human-readable output
        print(result.get("feedback_text", "No feedback generated."))
        print(f"\n⏱ Inference time: {elapsed:.0f}ms")


if __name__ == "__main__":
    main()
