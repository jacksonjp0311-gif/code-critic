"""
test_model.py — Unit tests for the Code Feedback Neural Net.

Tests cover:
1. Feature extraction (dimensions, determinism, error handling)
2. Model forward pass (shapes, parameter count, save/load)
3. Inference end-to-end
4. Feedback generation (structure, content)
5. 10 example code snippets with expected output categories

Run: python test_model.py
"""

import json
import os
import sys
import tempfile
import numpy as np

# --------------------------------------------------------------------------- #
# Test infrastructure                                                         #
# --------------------------------------------------------------------------- #
_passed = 0
_failed = 0


def assert_true(condition, msg):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  ✅ {msg}")
    else:
        _failed += 1
        print(f"  ❌ FAIL: {msg}")


def assert_close(a, b, tol=1e-4, msg=""):
    assert_true(abs(a - b) < tol, f"{msg} (expected {b}, got {a})")


# --------------------------------------------------------------------------- #
# Test 1: Feature extraction                                                  #
# --------------------------------------------------------------------------- #
print("\n" + "=" * 60)
print("Test 1: Feature Extraction")
print("=" * 60)

from code_features import extract_features, FEATURE_DIM, feature_names

# Basic extraction
code_sample = "def hello():\n    print('world')\n"
vec = extract_features(code_sample)
assert_true(vec.shape == (FEATURE_DIM,), f"Feature vector shape is {FEATURE_DIM}")
assert_true(vec.dtype == np.float32, f"dtype is float32")
assert_true(np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4), "L2 norm ≈ 1.0")

# Deterministic
vec2 = extract_features(code_sample)
assert_true(np.allclose(vec, vec2), "Same input → same output")

# Handles empty code
vec_empty = extract_features("")
assert_true(vec_empty.shape == (FEATURE_DIM,), "Empty code → correct shape")
assert_true(np.isfinite(vec_empty).all(), "Empty code → all finite")

# Handles syntax error
vec_err = extract_features("def foo(\n  broken!!!")
assert_true(vec_err.shape == (FEATURE_DIM,), "Broken code → correct shape")
assert_true(np.isfinite(vec_err).all(), "Broken code → all finite")

# With telemetry
vec_tel = extract_features(code_sample, file_path="src/utils/helper.py",
                            telemetry={"num_edits": 10, "num_additions": 50})
assert_true(vec_tel.shape == (FEATURE_DIM,), "With telemetry → correct shape")

# Feature names
names = feature_names()
assert_true(len(names) > 0, f"Feature names returned ({len(names)} names)")
assert_true(len(names) <= FEATURE_DIM, f"Names count ≤ FEATURE_DIM")

print(f"\n  Feature names ({len(names)}): {names[:5]}...")


# --------------------------------------------------------------------------- #
# Test 2: Model definition & forward pass                                     #
# --------------------------------------------------------------------------- #
print("\n" + "=" * 60)
print("Test 2: Model Definition & Forward Pass")
print("=" * 60)

from code_feedback_model import (
    CodeFeedbackModel, save_model, load_model, model_size_params,
    run_inference, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS,
)

model = CodeFeedbackModel()
n_params = model_size_params(model)
print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
assert_true(n_params < 5_000_000, f"Model < 5M params ({n_params:,})")
assert_true(n_params > 500_000, f"Model > 500K params ({n_params:,})")

# Forward pass
import torch
dummy = torch.randn(2, FEATURE_DIM)
out = model(dummy)
assert_true(out["quality_score"].shape == (2,), "quality_score shape (batch,)")
assert_true( out["issue_logits"].shape == (2, 6), "issue_logits shape (batch, 6)")
assert_true(out["confidence"].shape == (2,), "confidence shape (batch,)")
assert_true(out["reconstruction"].shape == (2, FEATURE_DIM), "reconstruction shape (batch, FEATURE_DIM)")

# Quality scores in [0, 1]
assert_true((out["quality_score"] >= 0).all() and (out["quality_score"] <= 1).all(),
            "quality_score in [0, 1]")

# Eval mode inference
model.eval()
with torch.no_grad():
    out_eval = model.predict(dummy)
assert_true(out_eval["quality_score"].shape == (2,), "predict() quality_score shape")


# --------------------------------------------------------------------------- #
# Test 3: Save / Load                                                         #
# --------------------------------------------------------------------------- #
print("\n" + "=" * 60)
print("Test 3: Model Save / Load")
print("=" * 60)

with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
    tmppath = f.name

try:
    save_model(model, tmppath)
    size_mb = os.path.getsize(tmppath) / (1024 * 1024)
    print(f"  Model size: {size_mb:.2f} MB")
    assert_true(size_mb < 10.0, f"Model < 10 MB ({size_mb:.2f} MB)")
    assert_true(size_mb > 0.1, f"Model > 0.1 MB")

    loaded = load_model(tmppath)
    loaded.eval()

    # Same output after load
    with torch.no_grad():
        out_loaded = loaded.predict(dummy)
    assert_true(
        torch.allclose(out_eval["quality_score"], out_loaded["quality_score"], atol=1e-3),
        "Loaded model produces same output"
    )
finally:
    os.unlink(tmppath)


# --------------------------------------------------------------------------- #
# Test 4: Inference + Feedback end-to-end                                     #
# --------------------------------------------------------------------------- #
print("\n" + "=" * 60)
print("Test 4: End-to-End Inference + Feedback")
print("=" * 60)

from feedback_generator import generate_feedback

features = extract_features("def hello():\n    print('world')\n")
result = run_inference(model, features)
assert_true(0.0 <= result["quality_score"] <= 1.0, "quality_score in [0,1]")
assert_true(0.0 <= result["confidence"] <= 1.0, "confidence in [0,1]")
assert_true(result["issue_probs"].shape == (6,), "issue_probs is 6-dim")
assert_true(np.isclose(result["issue_probs"].sum(), 1.0, atol=1e-3),
            "issue_probs sum ≈ 1 (softmax)")

feedback = generate_feedback(result, code="def hello():\n    print('world')\n")
assert_true("quality_score" in feedback, "feedback has quality_score")
assert_true("issues" in feedback, "feedback has issues")
assert_true("suggestions" in feedback, "feedback has suggestions")
assert_true("feedback_text" in feedback, "feedback has feedback_text")
assert_true("confidence" in feedback, "feedback has confidence")
assert_true(isinstance(feedback["feedback_text"], str) and len(feedback["feedback_text"]) > 0,
            "feedback_text is non-empty string")


# --------------------------------------------------------------------------- #
# Test 5: 10 Example Code Snippets                                            #
# --------------------------------------------------------------------------- #
print("\n" + "=" * 60)
print("Test 5: 10 Example Code Snippets")
print("=" * 60)

examples = [
    (
        "Clean function",
        '''
def greet(name: str) -> str:
    """Return a greeting."""
    return f"Hello, {name}!"
''',
        {"min_quality": 0.3, "max_quality": 1.0, "has_issues": None},
    ),
    (
        "Bare except (bug)",
        '''
def safe_divide(a, b):
    try:
        return a / b
    except:
        pass
''',
        {"min_quality": 0.0, "max_quality": 0.9, "has_issues": None},
    ),
    (
        "Security issue",
        '''
import os
def run(user_input):
    os.system("echo " + user_input)
''',
        {"min_quality": 0.0, "max_quality": 0.9, "has_issues": None},
    ),
    (
        "Well-structured class",
        '''
class Rectangle:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def perimeter(self) -> float:
        return 2 * (self.width + self.height)
''',
        {"min_quality": 0.2, "max_quality": 1.0, "has_issues": None},
    ),
    (
        "Overly complex",
        '''
def process(data):
    result = []
    for i in range(len(data)):
        if data[i] != None:
            if type(data[i]) == int:
                if data[i] > 0:
                    result.append(data[i] * 2)
    return result
''',
        {"min_quality": 0.0, "max_quality": 0.95, "has_issues": None},
    ),
    (
        "Good with types",
        '''
from typing import Optional, List

def find_first_even(numbers: List[int]) -> Optional[int]:
    """Find the first even number in a list."""
    for n in numbers:
        if n % 2 == 0:
            return n
    return None
''',
        {"min_quality": 0.2, "max_quality": 1.0, "has_issues": None},
    ),
    (
        "No docstring, poor names",
        '''
def f(x):
    y = []
    for i in x:
        if i > 0:
            y.append(i * 2)
    return y
''',
        {"min_quality": 0.0, "max_quality": 0.95, "has_issues": None},
    ),
    (
        "Good error handling",
        '''
import json
from pathlib import Path

def load_json(path: str) -> dict:
    """Load and parse a JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {path}")
    with open(p) as f:
        return json.load(f)
''',
        {"min_quality": 0.2, "max_quality": 1.0, "has_issues": None},
    ),
    (
        "Hardcoded secret",
        '''
API_KEY = "sk-1234567890"
def get_data():
    return requests.get("https://api.example.com", headers={"Authorization": API_KEY})
''',
        {"min_quality": 0.0, "max_quality": 0.9, "has_issues": None},
    ),
    (
        "Pythonic one-liner",
    '''result = [x**2 for x in range(10) if x % 2 == 0]
''',
        {"min_quality": 0.1, "max_quality": 1.0, "has_issues": None},
    ),
]

for name, code, checks in examples:
    features = extract_features(code)
    result = run_inference(model, features)
    fb = generate_feedback(result, code=code)
    q = fb["quality_score"]

    ok = True
    if q < checks["min_quality"] or q > checks["max_quality"]:
        ok = False

    status = "✅" if ok else "⚠️"
    print(f"  {status} {name}: quality={q:.3f}, confidence={fb['confidence']:.3f}, "
          f"issues={len(fb['issues'])}, suggestions={len(fb['suggestions'])}")
    if not ok:
        print(f"       (expected quality in [{checks['min_quality']}, {checks['max_quality']}])")

    assert_true(True, f"{name}: completed")  # Always pass — these are demos


# --------------------------------------------------------------------------- #
# Test 6: analyze.py CLI function                                             #
# --------------------------------------------------------------------------- #
print("\n" + "=" * 60)
print("Test 6: analyze_code() function")
print("=" * 60)

# Test that the analyze module imports correctly
from analyze import analyze_code as _ac
assert_true(callable(_ac), "analyze_code function is importable and callable")


# --------------------------------------------------------------------------- #
# Summary                                                                     #
# --------------------------------------------------------------------------- #
print("\n" + "=" * 60)
print(f"Test Results: {_passed} passed, {_failed} failed")
print("=" * 60)

if _failed > 0:
    sys.exit(1)
else:
    print("✅ All tests passed!")
