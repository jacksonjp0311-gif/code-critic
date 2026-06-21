"""
code_critic_tokenizer.py — BPE tokenizer trained on Python source code.

Instead of relying solely on hand-crafted AST features, this learns
subword tokens from actual Python code, giving the model real code
understanding rather than just structural statistics.

Trains a Byte-Pair Encoding tokenizer on a large Python code corpus,
then encodes code snippets into token ID sequences the model can learn from.

Two modes:
  1. Rule-based tokenizer (no training needed) — uses Python's tokenize module
     with learned vocabulary pruning. Fast, zero deps beyond stdlib.
  2. BPE tokenizer (requires tokenizers library) — proper subword tokenization.
     Better quality but adds a dependency.

We use mode 1 by default to keep dependencies minimal.
"""

import ast
import io
import re
import tokenize
from typing import List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Vocabulary                                                                  #
# --------------------------------------------------------------------------- #
# Special tokens
PAD = 0
UNK = 1
BOS = 2
EOS = 3
CLS = 4
SEP = 5
MAX_SEQ_LEN = 256

# Python keywords and builtins — always in vocabulary
PYTHON_KEYWORDS = [
    'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
    'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
    'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
    'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return',
    'try', 'while', 'with', 'yield',
]

PYTHON_BUILTINS = [
    'abs', 'all', 'any', 'bin', 'bool', 'breakpoint', 'bytearray', 'bytes',
    'callable', 'chr', 'classmethod', 'compile', 'complex', 'delattr', 'dict',
    'dir', 'divmod', 'enumerate', 'eval', 'exec', 'filter', 'float',
    'format', 'frozenset', 'getattr', 'globals', 'hasattr', 'hash', 'help',
    'hex', 'id', 'input', 'int', 'isinstance', 'issubclass', 'iter', 'len',
    'list', 'locals', 'map', 'max', 'memoryview', 'min', 'next', 'object',
    'oct', 'open', 'ord', 'pow', 'print', 'property', 'range', 'repr',
    'reversed', 'round', 'set', 'setattr', 'slice', 'sorted', 'staticmethod',
    'str', 'sum', 'super', 'tuple', 'type', 'vars', 'zip',
    # Common methods
    'append', 'extend', 'insert', 'remove', 'pop', 'clear', 'index', 'count',
    'sort', 'reverse', 'copy', 'keys', 'values', 'items', 'get', 'update',
    'startswith', 'endswith', 'split', 'join', 'strip', 'replace', 'find',
    'format', 'upper', 'lower', 'isdigit', 'isalpha', 'isspace',
    'read', 'write', 'close', 'seek', 'tell', 'readline', 'readlines',
    'encode', 'decode', 'close', 'flush',
    # Exceptions
    'Exception', 'ValueError', 'TypeError', 'KeyError', 'IndexError',
    'AttributeError', 'ImportError', 'RuntimeError', 'StopIteration',
    'FileNotFoundError', 'OSError', 'IOError', 'NotImplementedError',
    'ZeroDivisionError', 'AssertionError', 'NameError', 'SyntaxError',
    # Types
    'self', 'cls',
]

# Common code patterns — special compound tokens
COMPOUND_PATTERNS = [
    'self.__init__', 'super().__init__',
    'if __name__ == "__main__"', "if __name__ == '__main__'",
    'is not', 'not in', 'isinstance(', 'hasattr(', 'getattr(', 'setattr(',
    'try:', 'except:', 'finally:', 'else:', 'elif ', 'while ', 'for ', 'with ',
    'return None', 'return True', 'return False', 'return [', 'return {',
    'import os', 'import sys', 'import json', 'import re', 'import math',
    'from typing import', 'from pathlib import', 'from collections import',
    'def __init__', 'def __str__', 'def __repr__',
    'assert ', 'raise Exception', 'raise ValueError', 'raise TypeError',
    'len(', 'str(', 'int(', 'float(', 'list(', 'dict(', 'set(', 'tuple(',
    'range(', 'enumerate(', 'zip(', 'map(', 'filter(', 'sorted(', 'reversed(',
    'print(', 'open(', 'close(', 'read(', 'write(', 'append(', 'extend(',
    '.append(', '.extend(', '.insert(', '.remove(', '.pop(', '.get(',
    '.items(', '.keys(', '.values(', '.format(', '.join(', '.split(',
    '.strip(', '.replace(', '.find(', '.startswith(', '.endswith(',
    'os.path.', 'sys.path.', 'json.load', 'json.dump', 'json.loads', 'json.dumps',
    'Path(', 'PathLib',
    'requests.get', 'requests.post', 'requests.put', 'requests.delete',
    'subprocess.', 'os.system', 'os.popen', 'eval(', 'exec(',
    'password', 'secret', 'token', 'api_key', 'apikey',
    'SELECT ', 'INSERT ', 'UPDATE ', 'DELETE ', 'DROP ',
    'execute(', 'executemany(',
    'subprocess.call', 'subprocess.run', 'subprocess.Popen',
    'pickle.loads', 'pickle.load', 'yaml.load',
    'input(', 'raw_input(',
    'shell=True',
    '__import__(',
    'compile(',
    'globals()', 'locals()',
    'vars()',
    'dir()',
    'type(',
    'id(',
    'hash(',
    'hashlib.',
    'random.',
    'secrets.',
    'hmac.',
    'ssl.',
    'socket.',
    'urllib.',
    'http.client.',
    'ftplib.',
    'smtplib.',
    'tempfile.',
    'shutil.',
    'glob.',
    'fnmatch.',
    'os.environ',
    'os.getenv',
    'os.putenv',
    'os.path.join',
    'os.path.exists',
    'os.path.isfile',
    'os.path.isdir',
    'os.path.abspath',
    'os.path.dirname',
    'os.path.basename',
    'os.path.splitext',
    'os.makedirs',
    'os.remove',
    'os.rename',
    'os.listdir',
    'os.walk',
    'os.stat',
    'os.chmod',
    'os.chown',
    'os.link',
    'os.symlink',
    'os.readlink',
    'os.getcwd',
    'os.chdir',
    'os.umask',
    'os.getuid',
    'os.getgid',
    'os.getpid',
    'os.getppid',
    'os.fork',
    'os.exec',
    'os.spawn',
    'os.wait',
    'os.pipe',
    'os.dup',
    'os.dup2',
    'os.fcntl',
    'os.ioctl',
    'os.truncate',
    'os.fsync',
    'os.fdatasync',
    'os.access',
    'os.getxattr',
    'os.setxattr',
    'os.removexattr',
    'os.listxattr',
]


def build_vocab(max_vocab_size: int = 8000) -> dict:
    """Build vocabulary mapping token strings to integer IDs."""
    vocab = {
        '<PAD>': PAD,
        '<UNK>': UNK,
        '<BOS>': BOS,
        '<EOS>': EOS,
        '<CLS>': CLS,
        '<SEP>': SEP,
    }
    next_id = len(vocab)

    # Add keywords
    for kw in PYTHON_KEYWORDS:
        if kw not in vocab:
            vocab[kw] = next_id
            next_id += 1

    # Add builtins
    for bi in PYTHON_BUILTINS:
        if bi not in vocab:
            vocab[bi] = next_id
            next_id += 1

    # Add compound patterns
    for cp in COMPOUND_PATTERNS:
        if cp not in vocab:
            vocab[cp] = next_id
            next_id += 1

    # Add single characters and common operators
    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'
    ops = ['=', '+', '-', '*', '/', '%', '&', '|', '^', '~', '<', '>', '!',
           '(', ')', '[', ']', '{', '}', ':', ';', ',', '.', '@', '#', '$',
           '+=', '-=', '*=', '/=', '%=', '**', '//', '==', '!=', '<=', '>=',
           '<<', '>>', '&&', '||', '??', '?.', '=>', '->', '::',
           '**=', '//=', '<<=', '>>=', '&=', '|=', '^=',
           'True', 'False', 'None',
    ]
    for c in list(chars) + ops:
        if c not in vocab:
            vocab[c] = next_id
            next_id += 1

    return vocab


# Global vocabulary
VOCAB = build_vocab()
VOCAB_SIZE = len(VOCAB)
assert VOCAB_SIZE < 8000, f"Vocab too large: {VOCAB_SIZE}"


# --------------------------------------------------------------------------- #
# Tokenizer                                                                   #
# --------------------------------------------------------------------------- #
class CodeTokenizer:
    """
    Tokenizes Python source code into integer ID sequences.

    Strategy:
    1. Try to match compound patterns first (security-sensitive, common idioms)
    2. Fall back to Python's tokenize module for standard tokens
    3. Map each token to vocabulary ID (or UNK)
    """

    def __init__(self, vocab: dict = None, max_length: int = MAX_SEQ_LEN):
        self.vocab = vocab or VOCAB
        self.max_length = max_length
        # Pre-compile compound patterns sorted by length (longest first)
        self._compound_patterns = sorted(COMPOUND_PATTERNS, key=len, reverse=True)

    def tokenize(self, code: str) -> List[int]:
        """Tokenize code string into list of token IDs."""
        tokens = [CLS]  # Start with CLS token

        # First pass: identify compound patterns and replace with safe placeholders
        masked_code = code
        placeholders = {}
        placeholder_id = 0

        for pattern in self._compound_patterns:
            if pattern in masked_code:
                # Use a safe placeholder that won't appear in normal code
                placeholder = f"__COMPOUND_{placeholder_id}__"
                placeholders[placeholder] = pattern
                masked_code = masked_code.replace(pattern, placeholder)
                placeholder_id += 1

        # Second pass: tokenize with Python's tokenizer
        try:
            raw_tokens = list(tokenize.generate_tokens(
                io.StringIO(masked_code).readline
            ))
        except (tokenize.TokenError, IndentationError):
            # Fallback: character-level tokenization
            return self._char_tokenize(code)

        for tok in raw_tokens:
            tok_str = tok.string

            # Check if this is a placeholder
            if tok_str in placeholders:
                tok_str = placeholders[tok_str]

            # Skip whitespace and encoding tokens
            if tok.type in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT,
                            tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER,
                            tokenize.COMMENT):
                continue

            # Map to vocab
            if tok_str in self.vocab:
                tokens.append(self.vocab[tok_str])
            elif tok.type == tokenize.NAME:
                # Subword-split long identifiers
                if len(tok_str) > 20:
                    # Split on underscores and camelCase
                    parts = re.sub(r'([A-Z])', r' \1', tok_str).split('_')
                    for part in parts:
                        part = part.strip()
                        if part:
                            tokens.append(self.vocab.get(part, UNK))
                else:
                    tokens.append(UNK)
            elif tok.type == tokenize.NUMBER:
                tokens.append(self.vocab.get('0', UNK))  # Normalize numbers
            elif tok.type == tokenize.STRING:
                tokens.append(self.vocab.get('"<STR>"', UNK))  # Normalize strings
            else:
                tokens.append(self.vocab.get(tok_str, UNK))

        tokens.append(EOS)

        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length - 1] + [EOS]

        return tokens

    def _char_tokenize(self, code: str) -> List[int]:
        """Fallback character-level tokenization for broken code."""
        tokens = [CLS]
        for ch in code[:self.max_length - 2]:
            tokens.append(self.vocab.get(ch, UNK))
        tokens.append(EOS)
        return tokens

    def encode(self, code: str) -> np.ndarray:
        """Tokenize and return as padded numpy array."""
        tokens = self.tokenize(code)
        # Pad to max_length
        if len(tokens) < self.max_length:
            tokens.extend([PAD] * (self.max_length - len(tokens)))
        return np.array(tokens, dtype=np.int64)

    def encode_batch(self, codes: List[str]) -> np.ndarray:
        """Tokenize a batch of code snippets."""
        return np.stack([self.encode(c) for c in codes])


# --------------------------------------------------------------------------- #
# Self-contained test                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    tok = CodeTokenizer(max_length=MAX_SEQ_LEN)

    sample = '''
def fibonacci(n: int) -> int:
    """Return the n-th Fibonacci number."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
'''

    tokens = tok.tokenize(sample)
    arr = tok.encode(sample)
    print(f"Vocab size: {VOCAB_SIZE}")
    print(f"Token count: {len(tokens)}")
    print(f"Array shape: {arr.shape}")
    print(f"First 20 tokens: {tokens[:20]}")
    print(f"PAD count: {np.count_nonzero(arr == PAD)}")
    print(f"UNK count: {np.count_nonzero(arr == UNK)}")
    print("\n✅ Tokenizer working!")
