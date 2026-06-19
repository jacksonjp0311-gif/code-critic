"""
code_features.py — Lightweight code feature extractor.

Extracts structural, lexical, and complexity features from Python source code
using only stdlib (ast, tokenize) plus optional radon for cyclomatic complexity.
All features are returned as fixed-size float32 vectors for the neural net.

Design decisions:
- Pure Python, no heavy deps — keeps the plugin lightweight.
- Graceful degradation: if radon is unavailable, complexity features are zeroed.
- Fixed output dimension (FEATURE_DIM = 128) regardless of input size.
"""

import ast
import io
import math
import sys
import tokenize
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #
FEATURE_DIM = 128  # fixed-size output vector

# AST node types we track for structural features
_AST_NODE_TYPES = [
    "FunctionDef", "AsyncFunctionDef", "ClassDef", "Return", "Delete",
    "Assign", "AugAssign", "AnnAssign", "For", "AsyncFor", "While",
    "If", "With", "AsyncWith", "Raise", "Try", "Assert", "Import",
    "ImportFrom", "Global", "Nonlocal", "Expr", "Pass", "Break",
    "Continue", "BoolOp", "BinOp", "UnaryOp", "Lambda", "IfExp",
    "Dict", "Set", "ListComp", "SetComp", "DictComp", "GeneratorExp",
    "Await", "Yield", "YieldFrom", "Compare", "Call", "FormattedValue",
    "JoinedStr", "Constant", "Attribute", "Subscript", "Starred",
    "Name", "List", "Tuple", "Slice", " comprehension",
]

# Token types we track
_TOKEN_CATEGORIES = [
    "NAME", "NUMBER", "STRING", "OP", "NEWLINE", "INDENT", "DEDENT",
    "COMMENT", "NL", "ENCODING", "ENDMARKER",
]

# Try importing radon; gracefully degrade if unavailable
try:
    from radon.complexity import cc_visit
    from radon.metrics import mi_visit
    from radon.raw import analyze as raw_analyze
    _RADON_AVAILABLE = True
except ImportError:
    _RADON_AVAILABLE = False


# --------------------------------------------------------------------------- #
# AST feature extraction                                                      #
# --------------------------------------------------------------------------- #
def _extract_ast_features(code: str) -> Dict[str, float]:
    """Walk the AST and extract structural counts + derived metrics."""
    features: Dict[str, float] = {}

    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Return zeros on parse failure — model learns "broken code" signal
        for nt in _AST_NODE_TYPES:
            features[f"ast_count_{nt}"] = 0.0
        features["ast_total_nodes"] = 0.0
        features["ast_max_depth"] = 0.0
        features["ast_branching_factor"] = 0.0
        features["ast_num_functions"] = 0.0
        features["ast_num_classes"] = 0.0
        features["ast_num_imports"] = 0.0
        features["ast_num_returns"] = 0.0
        features["ast_num_loops"] = 0.0
        features["ast_num_try_except"] = 0.0
        features["ast_num_with"] = 0.0
        features["ast_syntax_error"] = 1.0
        return features

    # Count node types
    node_counts = Counter()
    total_nodes = 0

    for node in ast.walk(tree):
        node_counts[type(node).__name__] += 1
        total_nodes += 1

    for nt in _AST_NODE_TYPES:
        features[f"ast_count_{nt}"] = float(node_counts.get(nt, 0))

    features["ast_total_nodes"] = float(total_nodes)

    # Tree depth via iterative traversal
    max_depth = 0
    stack: List[Tuple[ast.AST, int]] = [(tree, 0)]
    while stack:
        node, depth = stack.pop()
        if depth > max_depth:
            max_depth = depth
        for child in ast.iter_child_nodes(node):
            stack.append((child, depth + 1))
    features["ast_max_depth"] = float(max_depth)

    # Branching factor: avg children per non-leaf node
    child_counts = []
    for node in ast.walk(tree):
        children = list(ast.iter_child_nodes(node))
        if children:
            child_counts.append(len(children))
    features["ast_branching_factor"] = (
        float(np.mean(child_counts)) if child_counts else 0.0
    )

    # High-level aggregates
    features["ast_num_functions"] = float(
        node_counts.get("FunctionDef", 0) + node_counts.get("AsyncFunctionDef", 0)
    )
    features["ast_num_classes"] = float(node_counts.get("ClassDef", 0))
    features["ast_num_imports"] = float(
        node_counts.get("Import", 0) + node_counts.get("ImportFrom", 0)
    )
    features["ast_num_returns"] = float(node_counts.get("Return", 0))
    features["ast_num_loops"] = float(
        node_counts.get("For", 0) + node_counts.get("While", 0)
    )
    features["ast_num_try_except"] = float(node_counts.get("Try", 0))
    features["ast_num_with"] = float(
        node_counts.get("With", 0) + node_counts.get("AsyncWith", 0)
    )
    features["ast_syntax_error"] = 0.0

    return features


# --------------------------------------------------------------------------- #
# Token feature extraction                                                    #
# --------------------------------------------------------------------------- #
def _extract_token_features(code: str) -> Dict[str, float]:
    """Tokenize and extract lexical features."""
    features: Dict[str, float] = {}

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError):
        for tc in _TOKEN_CATEGORIES:
            features[f"tok_count_{tc}"] = 0.0
        features["tok_total_tokens"] = 0.0
        features["tok_unique_names"] = 0.0
        features["tok_name_diversity"] = 0.0
        features["tok_comment_ratio"] = 0.0
        features["tok_string_ratio"] = 0.0
        features["tok_max_line_length"] = 0.0
        features["tok_avg_line_length"] = 0.0
        features["tok_indent_depth_max"] = 0.0
        features["tok_parse_error"] = 1.0
        return features

    tok_counts = Counter()
    names = Counter()
    total = 0
    max_line_len = 0
    line_lens = []
    max_indent = 0

    for tok in tokens:
        tok_counts[tok.type] += 1
        total += 1
        if tok.type == tokenize.NAME:
            names[tok.string] += 1

    # Map token type numbers to category names
    name_to_cat = {}
    for attr in dir(tokenize):
        val = getattr(tokenize, attr)
        if isinstance(val, int):
            name_to_cat[val] = attr

    for tc in _TOKEN_CATEGORIES:
        features[f"tok_count_{tc}"] = 0.0

    for type_num, count in tok_counts.items():
        cat = name_to_cat.get(type_num, "")
        if cat in _TOKEN_CATEGORIES:
            features[f"tok_count_{cat}"] = float(count)

    features["tok_total_tokens"] = float(total)
    features["tok_unique_names"] = float(len(names))
    features["tok_name_diversity"] = (
        float(len(names) / total) if total > 0 else 0.0
    )

    comment_count = tok_counts.get(tokenize.COMMENT, 0)
    string_count = tok_counts.get(tokenize.STRING, 0)
    features["tok_comment_ratio"] = (
        float(comment_count / total) if total > 0 else 0.0
    )
    features["tok_string_ratio"] = (
        float(string_count / total) if total > 0 else 0.0
    )

    # Line length stats
    lines = code.splitlines()
    if lines:
        lens = [len(l) for l in lines]
        features["tok_max_line_length"] = float(max(lens))
        features["tok_avg_line_length"] = float(np.mean(lens))
    else:
        features["tok_max_line_length"] = 0.0
        features["tok_avg_line_length"] = 0.0

    # Max indent depth
    indent = 0
    for tok in tokens:
        if tok.type == tokenize.INDENT:
            indent += 1
            max_indent = max(max_indent, indent)
        elif tok.type == tokenize.DEDENT:
            indent = max(0, indent - 1)
    features["tok_indent_depth_max"] = float(max_indent)
    features["tok_parse_error"] = 0.0

    return features


# --------------------------------------------------------------------------- #
# Complexity features (radon)                                                  #
# --------------------------------------------------------------------------- #
def _extract_complexity_features(code: str) -> Dict[str, float]:
    """Cyclomatic + maintainability + raw metrics via radon (if available)."""
    defaults = {
        "cc_total": 0.0, "cc_avg": 0.0, "cc_max": 0.0, "cc_min": 0.0,
        "cc_num_functions": 0.0, "mi_score": 0.0,
        "raw_loc": 0.0, "raw_lloc": 0.0, "raw_sloc": 0.0,
        "raw_comments": 0.0, "raw_multi": 0.0, "raw_blank": 0.0,
        "halstead_volume": 0.0, "halstead_difficulty": 0.0,
        "halstead_effort": 0.0,
    }

    if not _RADON_AVAILABLE:
        return defaults

    try:
        # Cyclomatic complexity
        cc_blocks = cc_visit(code)
        if cc_blocks:
            cc_vals = [b.complexity for b in cc_blocks]
            defaults["cc_total"] = float(sum(cc_vals))
            defaults["cc_avg"] = float(np.mean(cc_vals))
            defaults["cc_max"] = float(max(cc_vals))
            defaults["cc_min"] = float(min(cc_vals))
            defaults["cc_num_functions"] = float(len(cc_blocks))
    except Exception:
        pass

    try:
        mi = mi_visit(code, multi=True)
        defaults["mi_score"] = float(mi) if mi else 0.0
    except Exception:
        pass

    try:
        raw = raw_analyze(code)
        defaults["raw_loc"] = float(raw.loc)
        defaults["raw_lloc"] = float(raw.lloc)
        defaults["raw_sloc"] = float(raw.sloc)
        defaults["raw_comments"] = float(raw.comments)
        defaults["raw_multi"] = float(raw.multi)
        defaults["raw_blank"] = float(raw.blank)
    except Exception:
        pass

    return defaults


# --------------------------------------------------------------------------- #
# Diff / telemetry features                                                   #
# --------------------------------------------------------------------------- #
def _extract_telemetry_features(telemetry: Optional[Dict]) -> Dict[str, float]:
    """Extract features from optional telemetry dict (git diffs, edit history)."""
    feats = {
        "tel_num_edits": 0.0, "tel_num_additions": 0.0,
        "tel_num_deletions": 0.0, "tel_num_files_changed": 0.0,
        "tel_has_git_diff": 0.0, "tel_has_edit_history": 0.0,
        "tel_file_age_days": 0.0, "tel_num_authors": 0.0,
    }
    if not telemetry:
        return feats

    if "num_edits" in telemetry:
        feats["tel_num_edits"] = float(telemetry["num_edits"])
    if "num_additions" in telemetry:
        feats["tel_num_additions"] = float(telemetry["num_additions"])
    if "num_deletions" in telemetry:
        feats["tel_num_deletions"] = float(telemetry["num_deletions"])
    if "num_files_changed" in telemetry:
        feats["tel_num_files_changed"] = float(telemetry["num_files_changed"])
    if "git_diff" in telemetry and telemetry["git_diff"]:
        feats["tel_has_git_diff"] = 1.0
    if "edit_history" in telemetry and telemetry["edit_history"]:
        feats["tel_has_edit_history"] = 1.0
    if "file_age_days" in telemetry:
        feats["tel_file_age_days"] = float(telemetry["file_age_days"])
    if "num_authors" in telemetry:
        feats["tel_num_authors"] = float(telemetry["num_authors"])

    return feats


# --------------------------------------------------------------------------- #
# File-path features                                                          #
# --------------------------------------------------------------------------- #
def _extract_path_features(file_path: Optional[str]) -> Dict[str, float]:
    """Simple features derived from the file path."""
    feats = {
        "path_is_test": 0.0, "path_is_init": 0.0,
        "path_is_config": 0.0, "path_depth": 0.0,
        "path_has_hyphen": 0.0,
    }
    if not file_path:
        return feats

    lower = file_path.lower()
    feats["path_is_test"] = 1.0 if "test" in lower else 0.0
    feats["path_is_init"] = 1.0 if "__init__" in lower else 0.0
    feats["path_is_config"] = 1.0 if any(
        k in lower for k in ["config", "settings", "conf"]
    ) else 0.0
    feats["path_depth"] = float(len(file_path.split("/")))
    feats["path_has_hyphen"] = 1.0 if "-" in file_path else 0.0
    return feats


# --------------------------------------------------------------------------- #
# Main extraction → fixed-dim vector                                          #
# --------------------------------------------------------------------------- #
def extract_features(
    code: str,
    file_path: Optional[str] = None,
    telemetry: Optional[Dict] = None,
) -> np.ndarray:
    """
    Extract all features and pack into a fixed-size float32 vector.

    Returns:
        np.ndarray of shape (FEATURE_DIM,), dtype float32
    """
    all_feats: Dict[str, float] = {}
    all_feats.update(_extract_ast_features(code))
    all_feats.update(_extract_token_features(code))
    all_feats.update(_extract_complexity_features(code))
    all_feats.update(_extract_telemetry_features(telemetry))
    all_feats.update(_extract_path_features(file_path))

    # Sort keys for deterministic ordering
    keys = sorted(all_feats.keys())
    vec = np.array([all_feats[k] for k in keys], dtype=np.float32)

    # Pad or truncate to FEATURE_DIM
    if len(vec) < FEATURE_DIM:
        vec = np.pad(vec, (0, FEATURE_DIM - len(vec)), constant_values=0.0)
    elif len(vec) > FEATURE_DIM:
        vec = vec[:FEATURE_DIM]

    # Log-scale large counts to prevent domination by magnitude
    vec = np.sign(vec) * np.log1p(np.abs(vec))

    # L2 normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec


def feature_names() -> List[str]:
    """Return the ordered list of feature names (before padding/truncation)."""
    # Re-derive by extracting from dummy code
    dummy = "x = 1"
    all_feats: Dict[str, float] = {}
    all_feats.update(_extract_ast_features(dummy))
    all_feats.update(_extract_token_features(dummy))
    all_feats.update(_extract_complexity_features(dummy))
    all_feats.update(_extract_telemetry_features(None))
    all_feats.update(_extract_path_features(None))
    return sorted(all_feats.keys())


if __name__ == "__main__":
    # Quick self-test
    sample = '''
def fibonacci(n):
    """Return the n-th Fibonacci number."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


class Calculator:
    def __init__(self):
        self.result = 0

    def add(self, x, y):
        self.result = x + y
        return self.result
'''
    vec = extract_features(sample, file_path="math/fibonacci.py")
    print(f"Feature vector shape: {vec.shape}")
    print(f"Feature vector dtype: {vec.dtype}")
    print(f"Non-zero elements: {np.count_nonzero(vec)}")
    print(f"L2 norm: {np.linalg.norm(vec):.4f}")
    print(f"Feature names ({len(feature_names())}): {feature_names()[:10]}...")
