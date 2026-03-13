import re
import os
import json
import hashlib
from typing import Dict, List, Optional, Any

def generate_filename_from_code(code: str, language: str) -> str:
    """Generate a sensible filename from code content."""
    # Language to extension mapping
    ext_map = {
        "python": ".py",
        "bash": ".sh",
        "sh": ".sh",
        "shell": ".sh",
        "javascript": ".js",
        "js": ".js",
        "typescript": ".ts",
        "ts": ".ts",
        "rust": ".rs",
        "go": ".go",
        "java": ".java",
        "c": ".c",
        "cpp": ".cpp",
        "c++": ".cpp",
        "ruby": ".rb",
        "php": ".php",
        "html": ".html",
        "css": ".css",
        "json": ".json",
        "yaml": ".yaml",
        "yml": ".yml",
        "sql": ".sql",
        "markdown": ".md",
        "md": ".md",
    }
    
    ext = ext_map.get(language.lower(), ".txt")
    
    # For shell scripts, check shebang
    if language.lower() in ("bash", "sh", "shell"):
        shebang_match = re.search(r"^#!.*/(bash|sh|zsh|fish)", code, re.MULTILINE)
        if shebang_match:
            return f"script{ext}"
    
    # Try to find function or class name
    func_match = re.search(r"^\s*(?:def|func|fn)\s+([a-zA-Z_][a-zA-Z0-9_]*)", code, re.MULTILINE)
    if func_match:
        name = func_match.group(1)
        if name not in ("main", "init", "__init__", "start", "run", "test"):
            return f"{name}{ext}"
    
    # Match class definitions
    class_match = re.search(r"^\s*class\s+([A-Z][a-zA-Z0-9_]*)", code, re.MULTILINE)
    if class_match:
        return f"{class_match.group(1).lower()}{ext}"
    
    # Match main function as fallback
    if re.search(r"^\s*(?:def|func|fn)\s+main\s*\(", code, re.MULTILINE):
        return f"main{ext}"
    
    # Default filename based on language
    if language.lower() in ("bash", "sh", "shell"):
        return f"script{ext}"
    
    return f"generated{ext}"

def get_cache_key(messages: List[Dict], model: str) -> str:
    """Hash for current session state."""
    normalized = []
    for m in messages:
        n = {"role": m["role"], "content": (m.get("content") or "").strip()}
        if "tool_calls" in m:
            n["tool_calls"] = m["tool_calls"]
        normalized.append(n)
    raw = f"{model}:{json.dumps(normalized, sort_keys=True)}"
    return hashlib.sha256(raw.encode()).hexdigest()
