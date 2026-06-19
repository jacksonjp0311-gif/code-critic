"""
feedback_generator.py — Converts model outputs to rich, actionable feedback.

Takes the raw model inference output and produces:
- Overall quality score with human-readable label
- Categorized issues with severity and descriptions
- Ranked refactoring suggestions
- Positive notes when code is strong
- Confidence score on the feedback itself

Uses template-based generation with logic to select the right templates
based on model output thresholds. No LLM needed — fast and deterministic.
"""

from typing import Dict, List, Optional, Any
import numpy as np

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #
ISSUE_LABELS = [
    "bugs",
    "style",
    "performance",
    "security",
    "maintainability",
    "pythonic",
]

# Thresholds for issue severity
SEVERITY_THRESHOLDS = {
    "critical": 0.8,
    "high": 0.6,
    "medium": 0.4,
    "low": 0.2,
    "info": 0.0,
}

QUALITY_LABELS = [
    (0.9, "Excellent", "🌟"),
    (0.75, "Good", "✅"),
    (0.6, "Acceptable", "👍"),
    (0.45, "Needs Improvement", "⚠️"),
    (0.3, "Poor", "❌"),
    (0.0, "Critical Issues", "🚨"),
]

# --------------------------------------------------------------------------- #
# Issue templates per category                                                #
# --------------------------------------------------------------------------- #
ISSUE_TEMPLATES = {
    "bugs": {
        "high": [
            "Potential bug detected — review error handling paths.",
            "Possible edge case not handled (empty input, None values).",
            "Risk of unhandled exception in complex branches.",
        ],
        "medium": [
            "Consider adding input validation to prevent runtime errors.",
            "Some branches may not cover all edge cases.",
            "Check for potential off-by-one errors in loops.",
        ],
        "low": [
            "Minor: consider defensive checks on external inputs.",
        ],
    },
    "style": {
        "high": [
            "Significant style inconsistencies detected — consider running a linter (flake8/black).",
            "Naming conventions are inconsistent (mix of snake_case and camelCase).",
        ],
        "medium": [
            "Some PEP 8 style violations detected (line length, spacing).",
            "Consider improving variable names for clarity.",
            "Docstrings are missing on public functions/classes.",
        ],
        "low": [
            "Minor style nits: consider consistent quote usage.",
        ],
    },
    "performance": {
        "high": [
            "High cyclomatic complexity detected — consider refactoring into smaller functions.",
            "Potential O(n²) pattern — look for nested loops over the same data.",
        ],
        "medium": [
            "Consider using list/dict comprehensions for simple loops.",
            "Repeated computations could be cached or memoized.",
            "Generator expressions may be more memory-efficient than list building.",
        ],
        "low": [
            "Minor: consider using built-in functions instead of manual loops.",
        ],
    },
    "security": {
        "high": [
            "⚠️ Potential security concern — review any use of eval/exec/input.",
            "Hardcoded credentials or secrets detected.",
        ],
        "medium": [
            "Consider sanitizing user inputs before processing.",
            "Review file path handling for path traversal risks.",
        ],
        "low": [
            "Minor: consider using ast.literal_eval instead of eval.",
        ],
    },
    "maintainability": {
        "high": [
            "Code is difficult to maintain — high complexity and low modularity.",
            "Functions are too long — extract helper functions.",
        ],
        "medium": [
            "Consider breaking large functions into smaller, testable units.",
            "Add type hints to function signatures for better IDE support.",
            "Module has many responsibilities — consider splitting.",
        ],
        "low": [
            "Minor: add module-level docstring for documentation.",
        ],
    },
    "pythonic": {
        "high": [
            "Code is not very Pythonic — consider idiomatic alternatives.",
        ],
        "medium": [
            "Use 'if __name__ == \"__main__\"' guard for script entry points.",
            "Consider using 'enumerate()' instead of range(len(...)).",
            "Use context managers (with statement) for resource handling.",
            "Prefer 'is None' / 'is not None' over '== None'.",
        ],
        "low": [
            "Minor: consider using f-strings for string formatting.",
        ],
    },
}

POSITIVE_TEMPLATES = [
    "Clean, well-structured code. 👍",
    "Good use of Python idioms.",
    "Well-documented with clear docstrings.",
    "Appropriate function decomposition.",
    "Good error handling patterns.",
    "Type hints improve readability.",
    "Consistent naming conventions.",
    "Good use of standard library.",
]


# --------------------------------------------------------------------------- #
# Feedback generation                                                         #
# --------------------------------------------------------------------------- #
def _get_severity(prob: float) -> str:
    """Map a probability to a severity label."""
    for severity, threshold in SEVERITY_THRESHOLDS.items():
        if prob >= threshold:
            return severity
    return "info"


def _get_quality_label(score: float):
    """Map a quality score to (label, emoji)."""
    for threshold, label, emoji in QUALITY_LABELS:
        if score >= threshold:
            return label, emoji
    return "Unknown", "❓"


def generate_feedback(
    inference_output: Dict[str, np.ndarray],
    code: str = "",
    file_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convert model inference output to structured feedback.

    Args:
        inference_output: dict from run_inference() with keys:
            quality_score (float), issue_probs (ndarray),
            confidence (float), reconstruction (ndarray)
        code: original code string (for context-aware messages)
        file_path: optional file path

    Returns:
        Structured feedback dict with:
            quality_score, quality_label, quality_emoji,
            issues (list of categorized issues),
            suggestions (ranked list),
            positive_notes (list),
            confidence,
            feedback_text (human-readable summary)
    """
    quality_score = float(inference_output["quality_score"])
    issue_probs = inference_output["issue_probs"]
    confidence = float(inference_output["confidence"])

    quality_label, quality_emoji = _get_quality_label(quality_score)

    # --- Issues ---
    issues = []
    for i, label in enumerate(ISSUE_LABELS):
        prob = float(issue_probs[i])
        severity = _get_severity(prob)
        if prob > 0.15:  # only report non-trivial issues
            templates = ISSUE_TEMPLATES.get(label, {})
            # Pick template based on severity
            sev_key = severity if severity in ("high", "medium", "low") else "low"
            msgs = templates.get(sev_key, [])
            description = msgs[0] if msgs else f"Potential {label} concern."
            issues.append({
                "category": label,
                "severity": severity,
                "probability": round(prob, 3),
                "description": description,
            })

    # Sort by probability descending
    issues.sort(key=lambda x: x["probability"], reverse=True)

    # --- Suggestions (ranked) ---
    suggestions = []
    for issue in issues:
        suggestions.append({
            "priority": issue["severity"],
            "category": issue["category"],
            "suggestion": issue["description"],
            "confidence": round(issue["probability"] * confidence, 3),
        })
    # Sort by confidence
    suggestions.sort(key=lambda x: x["confidence"], reverse=True)

    # --- Positive notes ---
    positive_notes = []
    if quality_score >= 0.75:
        positive_notes.append(POSITIVE_TEMPLATES[0])
        if quality_score >= 0.85:
            positive_notes.append(POSITIVE_TEMPLATES[1])
    # Check for specific positive signals in code
    if code:
        if '"""' in code or "'''" in code:
            positive_notes.append(POSITIVE_TEMPLATES[2])
        if "def " in code and code.count("def ") <= 5:
            positive_notes.append(POSITIVE_TEMPLATES[3])
        if "try:" in code and "except" in code:
            positive_notes.append(POSITIVE_TEMPLATES[4])
        if " -> " in code or "typing" in code:
            positive_notes.append(POSITIVE_TEMPLATES[5])
        if "import " in code and code.count("\nimport ") + code.count("\nfrom ") <= 5:
            positive_notes.append(POSITIVE_TEMPLATES[7])

    # Deduplicate
    positive_notes = list(dict.fromkeys(positive_notes))

    # --- Build feedback text ---
    lines = []
    lines.append(f"{quality_emoji} Code Quality: {quality_label} ({quality_score:.1%})")
    lines.append(f"   Model Confidence: {confidence:.1%}")
    lines.append("")

    if issues:
        lines.append("📋 Issues Found:")
        for idx, issue in enumerate(issues, 1):
            sev_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}.get(
                issue["severity"], "⚪"
            )
            lines.append(
                f"   {idx}. {sev_emoji} [{issue['severity'].upper()}] "
                f"{issue['category']}: {issue['description']}"
            )
        lines.append("")

    if suggestions:
        lines.append("💡 Top Suggestions:")
        for idx, sug in enumerate(suggestions[:5], 1):
            lines.append(f"   {idx}. [{sug['category']}] {sug['suggestion']}")
        lines.append("")

    if positive_notes:
        lines.append("✨ Strengths:")
        for note in positive_notes[:3]:
            lines.append(f"   • {note}")
        lines.append("")

    feedback_text = "\n".join(lines)

    return {
        "quality_score": round(quality_score, 4),
        "quality_label": quality_label,
        "quality_emoji": quality_emoji,
        "issues": issues,
        "suggestions": suggestions,
        "positive_notes": positive_notes,
        "confidence": round(confidence, 4),
        "feedback_text": feedback_text,
    }


# --------------------------------------------------------------------------- #
# Self-test                                                                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Simulate inference output
    fake_output = {
        "quality_score": 0.72,
        "issue_probs": np.array([0.15, 0.55, 0.30, 0.05, 0.42, 0.25]),
        "confidence": 0.81,
        "reconstruction": np.zeros(128),
    }
    result = generate_feedback(fake_output, code='def hello():\n    """Say hello."""\n    print("hi")\n')
    print(result["feedback_text"])
    print(f"\nQuality: {result['quality_score']} ({result['quality_label']})")
    print(f"Issues: {len(result['issues'])}")
    print(f"Suggestions: {len(result['suggestions'])}")
    print(f"Positive notes: {len(result['positive_notes'])}")
    print("\n✅ feedback_generator self-test passed!")
