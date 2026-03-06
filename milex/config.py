"""Configuration management for MILEX CLI."""
import json
import os
import sys
from pathlib import Path
from typing import Optional

CONFIG_DIR = Path.home() / ".milex"
CONFIG_FILE = CONFIG_DIR / "config.json"
HISTORY_FILE = CONFIG_DIR / "history.json"

DEFAULT_CONFIG = {
    "model": "qwen2.5:1.5b",
    "roles": {
        "primary": "qwen2.5:1.5b",
        "coder": "qwen2.5:1.5b", # Use faster model by default for code tasks
        "planner": "qwen2.5:1.5b",
        "embeddings": "nomic-embed-text:latest",
        "fallback": "qwen2.5:1.5b"
    },
    "ollama_host": "http://localhost:11434",
    "theme": "dark",
    "max_tokens": 2048,
    "num_ctx": 2048,        # Optimized for CPU "Air" performance
    "temperature": 0.7,
    "auto_execute": False,
    "show_thinking": True,
    "stream": True,
    "compact_mode": True,
    # Air Logic — Nitro — CPU performance defaults
    "num_batch": 1024,      # Aggressive prefill
    "num_thread": 8,        # Tuned to physical core count (8) for Speed++
    "num_keep": -1,         # Pin entire system prompt in KV cache (Major Speedup)
    "max_history": 15,      # Shorter history = faster inference
    "cache_size": 32,       # Response cache size (items)
    "repeat_penalty": 1.1,  # Prevent repetitive outputs
    "rag": {
        "enabled": True,
        "index_on_startup": False,
        "chunk_size": 1000,
        "chunk_overlap": 100,
        "exclude_dirs": [".git", "__pycache__", ".venv", "node_modules", "dist", "build"]
    },
    "system_prompt": (
        "You are MILEX, an elite AI coding assistant and computer-control agent.\n\n"
        "CORE CAPABILITIES:\n"
        "- Generate complete, production-ready code in ANY programming language.\n"
        "- Execute shell commands and manage system processes.\n"
        "- Full filesystem access: read, write, list, search, and manage files/directories.\n"
        "- Semantic Search (RAG): Index your projects and search for relevant code/docs semanticially using 'rag_index' and 'rag_search'.\n"
        "- Desktop automation: open web browser, manage clipboard, etc.\n\n"
        "OPERATIONAL RULES:\n"
        "1. ALWAYS return COMPLETE code — never use placeholders or trancations (e.g., no '// ...').\n"
        "2. Use appropriate markdown code blocks for all code and terminal output.\n"
        "3. Follow best practices for the language being used (idiomatic, clean, documented).\n"
        "4. Use tool calls for all system-level operations. Do not just describe them.\n"
        "5. Be concise and technical. Avoid conversational filler.\n"
        "6. If a task is complex, break it down into steps and execute them sequentially.\n"
        "7. ALWAYS confirm before performing potentially destructive operations (delete, overwrite) unless auto-execute is enabled.\n"
    ),
    "compact_system_prompt": (
        "You are MILEX, an AI agent. Execute tasks using tools. Be brief. Complete code only."
    ),
}


def ensure_config_dir():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    ensure_config_dir()
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                cfg = json.load(f)
            # merge with defaults for any missing keys
            return {**DEFAULT_CONFIG, **cfg}
        except json.JSONDecodeError as exc:
            print(
                f"[milex] Warning: config file is corrupt ({exc}). Using defaults.",
                file=sys.stderr,
            )
        except OSError as exc:
            print(
                f"[milex] Warning: could not read config ({exc}). Using defaults.",
                file=sys.stderr,
            )
    return DEFAULT_CONFIG.copy()


def save_config(config: dict):
    ensure_config_dir()
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
    except OSError as exc:
        print(
            f"[milex] Warning: could not save config ({exc}).",
            file=sys.stderr,
        )


def load_history() -> list:
    ensure_config_dir()
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            # History is non-critical; start fresh on corruption
            pass
    return []


def save_history(history: list):
    ensure_config_dir()
    # Keep last 1000 entries
    with open(HISTORY_FILE, "w") as f:
        json.dump(history[-1000:], f, indent=2)
