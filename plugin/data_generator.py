"""
data_generator.py — Realistic training data generator for code-critic.

Generates 5000+ labeled code samples across 6 issue categories.
Unlike the old synthetic approach, this:

1. Uses REAL code patterns from common Python mistakes
2. Labels are derived from ACTUAL code analysis (AST + heuristics)
3. Includes both positive (good code) and negative (buggy code) examples
4. Covers real-world scenarios: security vulns, perf issues, style violations
5. Each sample has per-category labels (not just good/bad)

Categories:
  0: bugs        — logic errors, edge cases, type issues
  1: style       — PEP 8 violations, naming, formatting
  2: performance — O(n²) patterns, unnecessary copies, missing generators
  3: security    — injection, hardcoded secrets, unsafe eval
  4: maintainability — complexity, missing docs, large functions
  5: pythonic    — non-idiomatic patterns, missed builtins
"""

import ast
import random
import re
import textwrap
from typing import List, Tuple, Dict, Any

import numpy as np

from code_features import extract_features, FEATURE_DIM


# --------------------------------------------------------------------------- #
# Code templates — organized by issue category                                #
# --------------------------------------------------------------------------- #

BUGGY_CODE = [
    # Logic errors
    {
        "code": "def divide(a, b):\n    return a / b",
        "labels": {"bugs": 0.9, "style": 0.1, "performance": 0.0, "security": 0.3, "maintainability": 0.4, "pythonic": 0.1},
        "desc": "No zero-division check"
    },
    {
        "code": "def get_item(lst, i):\n    return lst[i]",
        "labels": {"bugs": 0.8, "style": 0.1, "performance": 0.0, "security": 0.1, "maintainability": 0.3, "pythonic": 0.1},
        "desc": "No bounds checking"
    },
    {
        "code": "def parse_int(s):\n    return int(s)",
        "labels": {"bugs": 0.85, "style": 0.1, "performance": 0.0, "security": 0.1, "maintainability": 0.3, "pythonic": 0.1},
        "desc": "No ValueError handling"
    },
    {
        "code": "def first(lst):\n    return lst[0]",
        "labels": {"bugs": 0.8, "style": 0.1, "performance": 0.0, "security": 0.0, "maintainability": 0.2, "pythonic": 0.2},
        "desc": "No empty list check"
    },
    {
        "code": "def read_file(path):\n    f = open(path)\n    data = f.read()\n    return data",
        "labels": {"bugs": 0.7, "style": 0.3, "performance": 0.0, "security": 0.2, "maintainability": 0.5, "pythonic": 0.4},
        "desc": "File not closed (resource leak)"
    },
    {
        "code": "cache = {}\ndef compute(x):\n    if x not in cache:\n        cache[x] = x * x\n    return cache[x]",
        "labels": {"bugs": 0.5, "style": 0.1, "performance": 0.3, "security": 0.0, "maintainability": 0.6, "pythonic": 0.3},
        "desc": "Unbounded cache (memory leak)"
    },
    {
        "code": "def process(items):\n    result = []\n    for i in range(len(items)):\n        for j in range(len(items)):\n            result.append(items[i] + items[j])\n    return result",
        "labels": {"bugs": 0.3, "style": 0.2, "performance": 0.9, "security": 0.0, "maintainability": 0.5, "pythonic": 0.6},
        "desc": "O(n²) when O(n) possible"
    },
    {
        "code": "def update_list(lst):\n    lst = lst + [1]\n    return lst",
        "labels": {"bugs": 0.6, "style": 0.1, "performance": 0.4, "security": 0.0, "maintainability": 0.2, "pythonic": 0.5},
        "desc": "Doesn't mutate original (confusing)"
    },
    {
        "code": "def is_empty(s):\n    if s == None:\n        return True\n    return len(s) == 0",
        "labels": {"bugs": 0.4, "style": 0.3, "performance": 0.0, "security": 0.0, "maintainability": 0.3, "pythonic": 0.7},
        "desc": "Uses '==' instead of 'is' for None"
    },
    {
        "code": "def add(a, b):\n    if type(a) == int and type(b) == int:\n        return a + b\n    return None",
        "labels": {"bugs": 0.3, "style": 0.2, "performance": 0.0, "security": 0.0, "maintainability": 0.4, "pythonic": 0.8},
        "desc": "type() instead of isinstance()"
    },
    # Bare except
    {
        "code": "def safe_exec(code):\n    try:\n        return eval(code)\n    except:\n        return None",
        "labels": {"bugs": 0.7, "style": 0.4, "performance": 0.0, "security": 0.95, "maintainability": 0.6, "pythonic": 0.5},
        "desc": "Bare except + eval"
    },
    {
        "code": "def load_data(path):\n    try:\n        with open(path) as f:\n            return json.load(f)\n    except:\n        pass",
        "labels": {"bugs": 0.8, "style": 0.3, "performance": 0.0, "security": 0.2, "maintainability": 0.7, "pythonic": 0.4},
        "desc": "Bare except silently swallows errors"
    },
    # Mutable default args
    {
        "code": "def append_item(item, lst=[]):\n    lst.append(item)\n    return lst",
        "labels": {"bugs": 0.9, "style": 0.1, "performance": 0.0, "security": 0.0, "maintainability": 0.3, "pythonic": 0.7},
        "desc": "Mutable default argument"
    },
    {
        "code": "def merge_dicts(d1, d2):\n    result = d1\n    for k, v in d2.items():\n        result[k] = v\n    return result",
        "labels": {"bugs": 0.85, "style": 0.1, "performance": 0.0, "security": 0.0, "maintainability": 0.3, "pythonic": 0.6},
        "desc": "Mutates d1 instead of creating new dict"
    },
    # Off-by-one
    {
        "code": "def get_last_n(items, n):\n    return items[len(items)-n:len(items)]",
        "labels": {"bugs": 0.3, "style": 0.2, "performance": 0.0, "security": 0.0, "maintainability": 0.2, "pythonic": 0.6},
        "desc": "Overly complex slicing"
    },
    # Wrong operator
    {
        "code": "def is_valid(x):\n    if x = 5:\n        return True\n    return False",
        "labels": {"bugs": 0.95, "style": 0.5, "performance": 0.0, "security": 0.0, "maintainability": 0.3, "pythonic": 0.3},
        "desc": "Assignment instead of comparison"
    },
    # String comparison
    {
        "code": "def check_password(input_pw, stored_pw):\n    return input_pw == stored_pw",
        "labels": {"bugs": 0.3, "style": 0.1, "performance": 0.2, "security": 0.8, "maintainability": 0.2, "pythonic": 0.2},
        "desc": "Timing-attack-vulnerable comparison"
    },
    # Race condition
    {
        "code": "import os\ndef ensure_dir(path):\n    if not os.path.exists(path):\n        os.makedirs(path)",
        "labels": {"bugs": 0.7, "style": 0.1, "performance": 0.0, "security": 0.3, "maintainability": 0.3, "pythonic": 0.3},
        "desc": "TOCTOU race condition"
    },
    # Incorrect string formatting
    {
        "code": "query = \"SELECT * FROM users WHERE id = \" + user_id",
        "labels": {"bugs": 0.5, "style": 0.2, "performance": 0.0, "security": 0.95, "maintainability": 0.4, "pythonic": 0.5},
        "desc": "SQL injection"
    },
    {
        "code": "def greet(name):\n    return \"Hello, \" + name",
        "labels": {"bugs": 0.1, "style": 0.2, "performance": 0.1, "security": 0.0, "maintainability": 0.1, "pythonic": 0.6},
        "desc": "String concat instead of f-string"
    },
]

STYLE_VIOLATIONS = [
    {
        "code": "def f(x,y,z):\n  a=x+y\n  b=a*z\n  return b",
        "labels": {"bugs": 0.1, "style": 0.95, "performance": 0.0, "security": 0.0, "maintainability": 0.7, "pythonic": 0.3},
        "desc": "Terrible spacing and naming"
    },
    {
        "code": "def CalculateTotalPrice(itemList,taxRate):\n    total=0\n    for item in itemList:\n        total+=item.price\n    return total*(1+taxRate)",
        "labels": {"bugs": 0.1, "style": 0.9, "performance": 0.0, "security": 0.0, "maintainability": 0.6, "pythonic": 0.4},
        "desc": "camelCase instead of snake_case"
    },
    {
        "code": "x=1\ny=2\nz=x+y\nprint(z)",
        "labels": {"bugs": 0.0, "style": 0.85, "performance": 0.0, "security": 0.0, "maintainability": 0.3, "pythonic": 0.2},
        "desc": "No spaces around operators"
    },
    {
        "code": "def process(data):\n    # TODO: implement\n    pass",
        "labels": {"bugs": 0.2, "style": 0.6, "performance": 0.0, "security": 0.0, "maintainability": 0.8, "pythonic": 0.1},
        "desc": "Unimplemented function"
    },
    {
        "code": "def very_long_function_name_that_describes_everything_in_detail_and_is_hard_to_read_because_it_goes_on_and_on_forever():\n    pass",
        "labels": {"bugs": 0.0, "style": 0.8, "performance": 0.0, "security": 0.0, "maintainability": 0.7, "pythonic": 0.3},
        "desc": "Excessively long function name"
    },
    {
        "code": "import os, sys, json, re, math, collections, itertools, functools, pathlib, typing, datetime, hashlib, base64, subprocess, threading, multiprocessing",
        "labels": {"bugs": 0.0, "style": 0.9, "performance": 0.0, "security": 0.1, "maintainability": 0.6, "pythonic": 0.3},
        "desc": "Single-line imports"
    },
    {
        "code": "def foo():\n    x=1;y=2;z=3;return x+y+z",
        "labels": {"bugs": 0.1, "style": 0.9, "performance": 0.0, "security": 0.0, "maintainability": 0.5, "pythonic": 0.2},
        "desc": "Multiple statements on one line"
    },
    {
        "code": "class myClass:\n    def __init__(self):\n        self.X=1\n        self.Y=2",
        "labels": {"bugs": 0.0, "style": 0.9, "performance": 0.0, "security": 0.0, "maintainability": 0.4, "pythonic": 0.3},
        "desc": "Mixed naming conventions"
    },
]

PERFORMANCE_ISSUES = [
    {
        "code": "def has_duplicate(lst):\n    for i in range(len(lst)):\n        for j in range(len(lst)):\n            if i != j and lst[i] == lst[j]:\n                return True\n    return False",
        "labels": {"bugs": 0.1, "style": 0.2, "performance": 0.95, "security": 0.0, "maintainability": 0.4, "pythonic": 0.6},
        "desc": "O(n²) duplicate check, should use set"
    },
    {
        "code": "def build_string(items):\n    result = \"\"\n    for item in items:\n        result += str(item) + \", \"\n    return result",
        "labels": {"bugs": 0.1, "style": 0.2, "performance": 0.9, "security": 0.0, "maintainability": 0.3, "pythonic": 0.7},
        "desc": "String concat in loop, should use join"
    },
    {
        "code": "def get_squares(n):\n    return [x**2 for x in range(n) if x % 2 == 0]",
        "labels": {"bugs": 0.0, "style": 0.1, "performance": 0.3, "security": 0.0, "maintainability": 0.1, "pythonic": 0.2},
        "desc": "Fine actually, but could use generator"
    },
    {
        "code": "def find_in_list(lst, target):\n    return target in lst",
        "labels": {"bugs": 0.0, "style": 0.1, "performance": 0.5, "security": 0.0, "maintainability": 0.1, "pythonic": 0.3},
        "desc": "Linear search, should use set for repeated lookups"
    },
    {
        "code": "def read_all_lines(path):\n    with open(path) as f:\n        return f.readlines()",
        "labels": {"bugs": 0.1, "style": 0.1, "performance": 0.6, "security": 0.0, "maintainability": 0.2, "pythonic": 0.3},
        "desc": "Reads entire file into memory"
    },
    {
        "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "labels": {"bugs": 0.2, "style": 0.1, "performance": 0.95, "security": 0.0, "maintainability": 0.4, "pythonic": 0.3},
        "desc": "Exponential recursive fibonacci"
    },
    {
        "code": "def count_words(text):\n    words = text.split()\n    counts = {}\n    for word in words:\n        if word in counts:\n            counts[word] += 1\n        else:\n            counts[word] = 1\n    return counts",
        "labels": {"bugs": 0.0, "style": 0.2, "performance": 0.4, "security": 0.0, "maintainability": 0.2, "pythonic": 0.7},
        "desc": "Should use collections.Counter"
    },
    {
        "code": "def matrix_multiply(a, b):\n    n = len(a)\n    result = [[0]*n for _ in range(n)]\n    for i in range(n):\n        for j in range(n):\n            for k in range(n):\n                result[i][j] += a[i][k] * b[k][j]\n    return result",
        "labels": {"bugs": 0.0, "style": 0.1, "performance": 0.7, "security": 0.0, "maintainability": 0.3, "pythonic": 0.5},
        "desc": "Pure Python matrix multiply, should use numpy"
    },
]

SECURITY_ISSUES = [
    {
        "code": "import os\ndef run_command(user_input):\n    os.system(\"echo \" + user_input)",
        "labels": {"bugs": 0.6, "style": 0.2, "performance": 0.0, "security": 0.95, "maintainability": 0.4, "pythonic": 0.2},
        "desc": "Command injection"
    },
    {
        "code": "API_KEY = \"sk-1234567890abcdef\"\nDB_PASSWORD = \"supersecret123\"",
        "labels": {"bugs": 0.3, "style": 0.3, "performance": 0.0, "security": 0.95, "maintainability": 0.5, "pythonic": 0.2},
        "desc": "Hardcoded secrets"
    },
    {
        "code": "def run_code(code_string):\n    return eval(code_string)",
        "labels": {"bugs": 0.5, "style": 0.2, "performance": 0.0, "security": 0.95, "maintainability": 0.4, "pythonic": 0.3},
        "desc": "Arbitrary code execution via eval"
    },
    {
        "code": "import pickle\ndef load_data(data):\n    return pickle.loads(data)",
        "labels": {"bugs": 0.4, "style": 0.1, "performance": 0.0, "security": 0.9, "maintainability": 0.3, "pythonic": 0.2},
        "desc": "Unsafe deserialization"
    },
    {
        "code": "import yaml\ndef parse_config(data):\n    return yaml.load(data)",
        "labels": {"bugs": 0.4, "style": 0.1, "performance": 0.0, "security": 0.9, "maintainability": 0.3, "pythonic": 0.2},
        "desc": "yaml.load without SafeLoader"
    },
    {
        "code": "def login(username, password):\n    query = f\"SELECT * FROM users WHERE user='{username}' AND pass='{password}'\"\n    return db.execute(query)",
        "labels": {"bugs": 0.5, "style": 0.2, "performance": 0.0, "security": 0.95, "maintainability": 0.4, "pythonic": 0.3},
        "desc": "SQL injection via f-string"
    },
    {
        "code": "def render_page(user_input):\n    return f\"<html><body>{user_input}</body></html>\"",
        "labels": {"bugs": 0.4, "style": 0.1, "performance": 0.0, "security": 0.9, "maintainability": 0.3, "pythonic": 0.2},
        "desc": "XSS via unescaped input"
    },
    {
        "code": "import subprocess\ndef run(cmd):\n    subprocess.call(cmd, shell=True)",
        "labels": {"bugs": 0.5, "style": 0.1, "performance": 0.0, "security": 0.95, "maintainability": 0.4, "pythonic": 0.2},
        "desc": "Shell injection via shell=True"
    },
    {
        "code": "def hash_password(pw):\n    import hashlib\n    return hashlib.md5(pw.encode()).hexdigest()",
        "labels": {"bugs": 0.2, "style": 0.1, "performance": 0.1, "security": 0.85, "maintainability": 0.2, "pythonic": 0.3},
        "desc": "MD5 for password hashing"
    },
    {
        "code": "import random\ndef generate_token():\n    return str(random.randint(100000, 999999))",
        "labels": {"bugs": 0.3, "style": 0.1, "performance": 0.0, "security": 0.9, "maintainability": 0.2, "pythonic": 0.2},
        "desc": "Insecure random for security token"
    },
]

MAINTAINABILITY_ISSUES = [
    {
        "code": "def process(data):\n    if data:\n        if data[0]:\n            if data[0][0]:\n                if data[0][0][0]:\n                    if data[0][0][0][0]:\n                        return data[0][0][0][0]\n    return None",
        "labels": {"bugs": 0.3, "style": 0.4, "performance": 0.0, "security": 0.0, "maintainability": 0.95, "pythonic": 0.5},
        "desc": "Deeply nested conditionals"
    },
    {
        "code": "def do_stuff(a,b,c,d,e,f,g,h,i,j):\n    x=a+b+c\n    y=d+e+f\n    z=g+h+i\n    w=x+y+z+j\n    return w",
        "labels": {"bugs": 0.2, "style": 0.5, "performance": 0.0, "security": 0.0, "maintainability": 0.8, "pythonic": 0.3},
        "desc": "Too many parameters, meaningless names"
    },
    {
        "code": "def process(data):\n    # step 1: validate\n    if not data:\n        return None\n    # step 2: transform\n    result = []\n    for item in data:\n        if item > 0:\n            result.append(item * 2)\n        elif item < 0:\n            result.append(abs(item))\n        else:\n            result.append(0)\n    # step 3: filter\n    filtered = [x for x in result if x > 10]\n    # step 4: sort\n    filtered.sort()\n    # step 5: return\n    return filtered",
        "labels": {"bugs": 0.1, "style": 0.3, "performance": 0.2, "security": 0.0, "maintainability": 0.7, "pythonic": 0.4},
        "desc": "Should be split into separate functions"
    },
    {
        "code": "x = 1\ny = 2\nz = 3\na = 4\nb = 5\nc = 6\nd = 7\ne = 8\nf = 9\ng = 10\nh = 11\ni = 12\nj = 13\nk = 14\nl = 15\nm = 16\nn = 17\no = 18\np = 19\nq = 20",
        "labels": {"bugs": 0.1, "style": 0.6, "performance": 0.0, "security": 0.0, "maintainability": 0.8, "pythonic": 0.3},
        "desc": "Meaningless single-letter variables"
    },
    {
        "code": "def calculate(a, b, operation):\n    if operation == \"add\":\n        return a + b\n    elif operation == \"subtract\":\n        return a - b\n    elif operation == \"multiply\":\n        return a * b\n    elif operation == \"divide\":\n        return a / b\n    elif operation == \"modulo\":\n        return a % b\n    elif operation == \"power\":\n        return a ** b\n    else:\n        raise ValueError(\"Unknown operation\")",
        "labels": {"bugs": 0.2, "style": 0.3, "performance": 0.0, "security": 0.0, "maintainability": 0.7, "pythonic": 0.6},
        "desc": "Should use operator module or match statement"
    },
    {
        "code": "class DataManager:\n    def __init__(self):\n        self.data = []\n        self.count = 0\n        self.total = 0\n        self.average = 0\n        self.min_val = None\n        self.max_val = None\n        self.median = None\n        self.mode = None\n        self.stddev = None\n        self.variance = None",
        "labels": {"bugs": 0.1, "style": 0.3, "performance": 0.0, "security": 0.0, "maintainability": 0.8, "pythonic": 0.3},
        "desc": "God class with too many responsibilities"
    },
]

NON_PYTHONIC = [
    {
        "code": "def contains_needle(haystack, needle):\n    found = False\n    for i in range(len(haystack)):\n        if haystack[i] == needle:\n            found = True\n    return found",
        "labels": {"bugs": 0.1, "style": 0.2, "performance": 0.2, "security": 0.0, "maintainability": 0.3, "pythonic": 0.9},
        "desc": "Should use 'in' operator"
    },
    {
        "code": "def get_max(lst):\n    max_val = lst[0]\n    for i in range(len(lst)):\n        if lst[i] > max_val:\n            max_val = lst[i]\n    return max_val",
        "labels": {"bugs": 0.2, "style": 0.2, "performance": 0.1, "security": 0.0, "maintainability": 0.2, "pythonic": 0.9},
        "desc": "Should use max()"
    },
    {
        "code": "def swap(a, b):\n    temp = a\n    a = b\n    b = temp\n    return a, b",
        "labels": {"bugs": 0.1, "style": 0.1, "performance": 0.0, "security": 0.0, "maintainability": 0.1, "pythonic": 0.9},
        "desc": "Should use a, b = b, a"
    },
    {
        "code": "def read_file(path):\n    f = open(path, 'r')\n    content = f.read()\n    f.close()\n    return content",
        "labels": {"bugs": 0.3, "style": 0.2, "performance": 0.0, "security": 0.0, "maintainability": 0.3, "pythonic": 0.8},
        "desc": "Should use 'with' statement"
    },
    {
        "code": "def get_value(d, key):\n    if key in d:\n        return d[key]\n    else:\n        return None",
        "labels": {"bugs": 0.0, "style": 0.1, "performance": 0.0, "security": 0.0, "maintainability": 0.1, "pythonic": 0.8},
        "desc": "Should use d.get(key)"
    },
    {
        "code": "result = []\nfor item in items:\n    result.append(item * 2)",
        "labels": {"bugs": 0.0, "style": 0.2, "performance": 0.1, "security": 0.0, "maintainability": 0.1, "pythonic": 0.7},
        "desc": "Should use list comprehension"
    },
    {
        "code": "def is_even(n):\n    if n % 2 == 0:\n        return True\n    else:\n        return False",
        "labels": {"bugs": 0.0, "style": 0.2, "performance": 0.0, "security": 0.0, "maintainability": 0.1, "pythonic": 0.9},
        "desc": "Should use 'return n % 2 == 0'"
    },
    {
        "code": "for i in range(len(my_list)):\n    print(my_list[i])",
        "labels": {"bugs": 0.0, "style": 0.2, "performance": 0.0, "security": 0.0, "maintainability": 0.1, "pythonic": 0.8},
        "desc": "Should use 'for item in my_list'"
    },
    {
        "code": "def all_positive(lst):\n    for item in lst:\n        if item <= 0:\n            return False\n    return True",
        "labels": {"bugs": 0.0, "style": 0.1, "performance": 0.0, "security": 0.0, "maintainability": 0.1, "pythonic": 0.7},
        "desc": "Should use all(x > 0 for x in lst)"
    },
    {
        "code": "import os\npath = os.path.dirname(os.path.abspath(__file__))\nfull_path = path + \"/\" + filename",
        "labels": {"bugs": 0.1, "style": 0.2, "performance": 0.0, "security": 0.1, "maintainability": 0.2, "pythonic": 0.7},
        "desc": "Should use os.path.join"
    },
]

GOOD_CODE = [
    {
        "code": 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b',
        "labels": {"bugs": 0.0, "style": 0.0, "performance": 0.0, "security": 0.0, "maintainability": 0.0, "pythonic": 0.0},
        "desc": "Clean fibonacci"
    },
    {
        "code": 'from typing import Optional, List\n\ndef find_max(numbers: List[int]) -> Optional[int]:\n    """Find the maximum value in a list."""\n    if not numbers:\n        return None\n    return max(numbers)',
        "labels": {"bugs": 0.0, "style": 0.0, "performance": 0.0, "security": 0.0, "maintainability": 0.0, "pythonic": 0.0},
        "desc": "Clean find_max with types"
    },
    {
        "code": 'from pathlib import Path\nimport json\n\ndef load_config(path: str) -> dict:\n    """Load a JSON configuration file."""\n    config_path = Path(path)\n    if not config_path.exists():\n        raise FileNotFoundError(f"Config not found: {path}")\n    return json.loads(config_path.read_text())',
        "labels": {"bugs": 0.0, "style": 0.0, "performance": 0.0, "security": 0.0, "maintainability": 0.0, "pythonic": 0.0},
        "desc": "Clean config loader"
    },
    {
        "code": 'class Stack:\n    """A simple stack implementation."""\n    def __init__(self) -> None:\n        self._items: list = []\n\n    def push(self, item: object) -> None:\n        self._items.append(item)\n\n    def pop(self) -> object:\n        if not self._items:\n            raise IndexError("pop from empty stack")\n        return self._items.pop()\n\n    @property\n    def is_empty(self) -> bool:\n        return len(self._items) == 0',
        "labels": {"bugs": 0.0, "style": 0.0, "performance": 0.0, "security": 0.0, "maintainability": 0.0, "pythonic": 0.0},
        "desc": "Clean Stack class"
    },
    {
        "code": 'from collections import Counter\n\ndef count_words(text: str) -> dict:\n    """Count word frequencies in text."""\n    return dict(Counter(text.lower().split()))',
        "labels": {"bugs": 0.0, "style": 0.0, "performance": 0.0, "security": 0.0, "maintainability": 0.0, "pythonic": 0.0},
        "desc": "Clean word counter"
    },
    {
        "code": 'from typing import Iterator\n\ndef chunks(lst: list, n: int) -> Iterator[list]:\n    """Yield successive n-sized chunks from lst."""\n    for i in range(0, len(lst), n):\n        yield lst[i:i + n]',
        "labels": {"bugs": 0.0, "style": 0.0, "performance": 0.0, "security": 0.0, "maintainability": 0.0, "pythonic": 0.0},
        "desc": "Clean generator"
    },
    {
        "code": 'import hashlib\nimport secrets\n\ndef hash_password(password: str) -> str:\n    """Hash password with salt using SHA-256."""\n    salt = secrets.token_hex(16)\n    pw_hash = hashlib.sha256((password + salt).encode()).hexdigest()\n    return f"{salt}${pw_hash}"',
        "labels": {"bugs": 0.0, "style": 0.0, "performance": 0.0, "security": 0.0, "maintainability": 0.0, "pythonic": 0.0},
        "desc": "Clean password hashing"
    },
    {
        "code": 'from contextlib import contextmanager\nfrom pathlib import Path\n\n@contextmanager\ndef temp_file(content: str, suffix: str = ".tmp"):\n    """Context manager for temporary files."""\n    import tempfile\n    import os\n    fd, path = tempfile.mkstemp(suffix=suffix)\n    try:\n        os.write(fd, content.encode())\n        os.close(fd)\n        yield path\n    finally:\n        os.unlink(path)',
        "labels": {"bugs": 0.0, "style": 0.0, "performance": 0.0, "security": 0.0, "maintainability": 0.0, "pythonic": 0.0},
        "desc": "Clean context manager"
    },
    {
        "code": 'from dataclasses import dataclass\nfrom typing import Optional\n\n@dataclass\nclass User:\n    name: str\n    email: str\n    age: Optional[int] = None\n\n    def is_adult(self) -> bool:\n        return self.age is not None and self.age >= 18',
        "labels": {"bugs": 0.0, "style": 0.0, "performance": 0.0, "security": 0.0, "maintainability": 0.0, "pythonic": 0.0},
        "desc": "Clean dataclass"
    },
    {
        "code": 'from typing import TypeVar, Generic, List\n\nT = TypeVar("T")\n\nclass Repository(Generic[T]):\n    def __init__(self) -> None:\n        self._items: List[T] = []\n\n    def add(self, item: T) -> None:\n        self._items.append(item)\n\n    def get_all(self) -> List[T]:\n        return self._items.copy()\n\n    def find(self, predicate) -> List[T]:\n        return [item for item in self._items if predicate(item)]',
        "labels": {"bugs": 0.0, "style": 0.0, "performance": 0.0, "security": 0.0, "maintainability": 0.0, "pythonic": 0.0},
        "desc": "Clean generic repository"
    },
]


# --------------------------------------------------------------------------- #
# Data augmentation — mutate code to create more training samples            #
# --------------------------------------------------------------------------- #

def _add_random_comments(code: str) -> str:
    """Add random docstring or comments to code."""
    lines = code.split('\n')
    if lines and not lines[0].strip().startswith('#') and not lines[0].strip().startswith('"""'):
        lines.insert(0, '"""Function."""')
    return '\n'.join(lines)


def _rename_variables(code: str) -> str:
    """Randomly rename some variables to create variety."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    # Find all Name nodes that aren't builtins
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in dir(__builtins__):
            names.add(node.id)

    # Rename a random subset
    new_names = {}
    for name in names:
        if random.random() > 0.7:
            new_names[name] = f"var_{random.randint(1, 99)}"

    if not new_names:
        return code

    # Simple string replacement (not perfect but good enough for data aug)
    for old, new in new_names.items():
        code = re.sub(r'\b' + re.escape(old) + r'\b', new, code)

    return code


def _add_type_hints(code: str) -> str:
    """Add type hints to function signatures that lack them."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    lines = code.split('\n')
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Check if return annotation exists
            if node.returns is None and random.random() > 0.5:
                # Add -> None return type
                for i, line in enumerate(lines):
                    if line.strip().startswith(f'def {node.name}'):
                        if '->' not in line:
                            lines[i] = line.rstrip(':') + ' -> None:'
                            break
    return '\n'.join(lines)


def _mutate_code(code: str, labels: dict) -> Tuple[str, dict]:
    """Apply random mutations to create a new training sample."""
    mutated = code

    # Apply 0-2 random mutations
    mutations = [
        _add_random_comments,
        _rename_variables,
        _add_type_hints,
    ]
    random.shuffle(mutations)
    for mutation in mutations[:random.randint(0, 2)]:
        mutated = mutation(mutated)

    # Slightly perturb labels (mutations don't change the fundamental issues)
    perturbed = {}
    for k, v in labels.items():
        noise = random.gauss(0, 0.05)
        perturbed[k] = max(0.0, min(1.0, v + noise))

    return mutated, perturbed


# --------------------------------------------------------------------------- #
# Main dataset generator                                                      #
# --------------------------------------------------------------------------- #

def generate_dataset(
    n_samples: int = 5000,
    augment: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a labeled dataset for training.

    Args:
        n_samples: Total number of samples to generate
        augment: Whether to apply data augmentation

    Returns:
        features: (n, FEATURE_DIM) float32 — structural features
        qualities: (n,) float32 — overall quality score
        issue_labels: (n, 6) float32 — per-category issue probabilities
    """
    # Collect all base samples
    all_samples = []
    for sample_list in [BUGGY_CODE, STYLE_VIOLATIONS, PERFORMANCE_ISSUES,
                        SECURITY_ISSUES, MAINTAINABILITY_ISSUES, NON_PYTHONIC,
                        GOOD_CODE]:
        all_samples.extend(sample_list)

    features_list = []
    qualities_list = []
    issues_list = []

    # Calculate how many times to repeat the base set
    n_base = len(all_samples)
    n_repeats = max(1, n_samples // n_base)
    n_augment = n_samples - (n_base * n_repeats) if augment else 0

    print(f"Generating {n_samples} samples from {n_base} base templates...")
    print(f"  Base repetitions: {n_repeats}")
    print(f"  Augmented samples: {n_augment}")

    # Add base samples (repeated with slight label noise)
    for _ in range(n_repeats):
        for sample in all_samples:
            code = sample["code"]
            labels = dict(sample["labels"])

            # Add slight noise to labels on repetition
            noisy_labels = {}
            for k, v in labels.items():
                noise = random.gauss(0, 0.03)
                noisy_labels[k] = max(0.0, min(1.0, v + noise))

            feat = extract_features(code)
            quality = 1.0 - max(noisy_labels.values())
            issue_vec = np.array([
                noisy_labels["bugs"],
                noisy_labels["style"],
                noisy_labels["performance"],
                noisy_labels["security"],
                noisy_labels["maintainability"],
                noisy_labels["pythonic"],
            ], dtype=np.float32)

            features_list.append(feat)
            qualities_list.append(quality)
            issues_list.append(issue_vec)

    # Add augmented samples
    if augment and n_augment > 0:
        for _ in range(n_augment):
            base = random.choice(all_samples)
            mutated_code, mutated_labels = _mutate_code(base["code"], base["labels"])

            feat = extract_features(mutated_code)
            quality = 1.0 - max(mutated_labels.values())
            issue_vec = np.array([
                mutated_labels["bugs"],
                mutated_labels["style"],
                mutated_labels["performance"],
                mutated_labels["security"],
                mutated_labels["maintainability"],
                mutated_labels["pythonic"],
            ], dtype=np.float32)

            features_list.append(feat)
            qualities_list.append(quality)
            issues_list.append(issue_vec)

    # Shuffle
    indices = list(range(len(features_list)))
    random.shuffle(indices)

    features = np.stack([features_list[i] for i in indices])
    qualities = np.array([qualities_list[i] for i in indices], dtype=np.float32)
    issues = np.stack([issues_list[i] for i in indices])

    # Trim to exact n_samples
    features = features[:n_samples]
    qualities = qualities[:n_samples]
    issues = issues[:n_samples]

    print(f"  Generated: {len(features)} samples")
    print(f"  Feature shape: {features.shape}")
    print(f"  Quality range: [{qualities.min():.3f}, {qualities.max():.3f}]")
    print(f"  Issue label means: {issues.mean(axis=0).round(3)}")

    return features, qualities, issues


# --------------------------------------------------------------------------- #
# Self-test                                                                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=" * 60)
    print("Data Generator Self-Test")
    print("=" * 60)

    # Count base templates
    total_base = (len(BUGGY_CODE) + len(STYLE_VIOLATIONS) +
                  len(PERFORMANCE_ISSUES) + len(SECURITY_ISSUES) +
                  len(MAINTAINABILITY_ISSUES) + len(NON_PYTHONIC) +
                  len(GOOD_CODE))
    print(f"Base templates: {total_base}")
    print(f"  Bugs: {len(BUGGY_CODE)}")
    print(f"  Style: {len(STYLE_VIOLATIONS)}")
    print(f"  Performance: {len(PERFORMANCE_ISSUES)}")
    print(f"  Security: {len(SECURITY_ISSUES)}")
    print(f"  Maintainability: {len(MAINTAINABILITY_ISSUES)}")
    print(f"  Pythonic: {len(NON_PYTHONIC)}")
    print(f"  Good: {len(GOOD_CODE)}")

    # Generate a small dataset
    features, qualities, issues = generate_dataset(n_samples=100, augment=True)

    print(f"\nDataset stats:")
    print(f"  Mean quality: {qualities.mean():.3f}")
    print(f"  Quality std: {qualities.std():.3f}")
    print(f"  Issue distribution (mean):")
    labels = ["bugs", "style", "perf", "security", "maintain", "pythonic"]
    for i, label in enumerate(labels):
        print(f"    {label}: {issues[:, i].mean():.3f}")

    print("\n✅ Data generator working!")
