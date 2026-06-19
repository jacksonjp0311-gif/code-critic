# Example AGNT Workflow: Code Generation → Neural Feedback → Auto-Iterate

## Workflow Overview

```
┌─────────────────────┐
│  Code Generation     │  Agent generates or modifies Python code
│  Agent / Tool        │
└──────────┬──────────┘
           │ code (string)
           ▼
┌─────────────────────┐
│  Code Feedback       │  analyze-code plugin tool
│  Neural Net          │  (coding-telemetry-feedback-net)
│  ┌───────────────┐  │
│  │ Feature Extract│  │  AST + tokenize + radon + telemetry
│  │ Transformer    │  │  4-layer encoder, 192 hidden
│  │ Multi-task     │  │  quality + 6 issues + confidence
│  │ Feedback Gen   │  │  templates → natural language
│  └───────────────┘  │
└──────────┬──────────┘
           │ quality_score, issues, suggestions, feedback_text
           ▼
┌─────────────────────┐
│  Decision Node       │  If quality_score >= 0.75 → Done
│  (quality >= 0.75?)  │  If quality_score <  0.75 → Iterate
└────┬─────────┬──────┘
     │         │
   ✅ Done    │ feedback_text + original code
              ▼
┌─────────────────────┐
│  Refactoring Agent   │  Agent reads feedback, improves code
│  (with feedback      │  Passes improved code back to
│   context)           │  Code Feedback Neural Net
└──────────┬──────────┘
           │ improved code
           ▼
     (loop back to
      Code Feedback
      Neural Net)
```

## Setting Up the Workflow in AGNT

### Step 1: Create a new workflow
Go to AGNT Workflows → Create New → Name it "Code Review Loop"

### Step 2: Add nodes

1. **Input Node** (trigger)
   - Add a text input parameter named `code` (the code to review)
   - Add an optional text input parameter named `file_path`

2. **Code Feedback Neural Net** (action)
   - Search for "Code Feedback Neural Net" in the node picker
   - Connect the `code` input to the Input Node's code output
   - Connect `file_path` if available

3. **Condition Node** (control)
   - Condition: `quality_score >= 0.75`
   - True branch → Done (output the feedback)
   - False branch → Continue to iteration

4. **LLM/Agent Node** (action)
   - Prompt: "Here is code and feedback from an automated review. Improve the code based on the suggestions:\n\nCode: {{code}}\n\nFeedback: {{feedback_text}}"
   - Connect to the False branch of the Condition Node

5. **Code Feedback Neural Net** (action) — second pass
   - Feed the improved code back into the analyzer
   - Connect output to Done

### Step 3: Wire it up

```
[Input: code, file_path]
        │
        ▼
[Code Feedback Neural Net] ──→ quality_score, feedback_text, suggestions
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
         [Code Feedback Neural Net] ──→ final feedback
              │
              ▼
            [Done]
```

## Standalone Usage (without workflow)

You can also use the tool directly from chat:

```
Analyze this code:

```python
def process(data):
    result = []
    for i in range(len(data)):
        if data[i] != None:
            result.append(data[i] * 2)
    return result
```
```

Or from the command line:

```bash
# Quick analysis
python analyze.py --code "def foo(): pass"

# Full file analysis
python analyze.py --file src/my_module.py

# With git telemetry
python analyze.py --file src/my_module.py --telemetry '{"num_edits": 15, "num_authors": 3}'

# JSON output for scripting
python analyze.py --file src/my_module.py --json
```

## Fine-Tuning on Your Codebase

To make the model learn your project's patterns:

```bash
# Self-supervised pretraining on your repo
python train.py --mode selfsupervised --data-dir ./my_project --epochs 20

# Fine-tune the pretrained model
python train.py --mode finetune --data-dir ./my_project --resume code_feedback_model.pt --epochs 10
```

Then copy the resulting `code_feedback_model.pt` to the plugin directory and rebuild.
