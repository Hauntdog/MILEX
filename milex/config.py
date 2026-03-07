"""Configuration management for MILEX CLI."""
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Mapping
from collections import ChainMap

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
    "allowed_root": None,
    "plugin_dir": str(CONFIG_DIR / "plugins"),
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
        "- Full filesystem access: read, write, edit, append, move, rename, and manage files/directories.\n"
        "- Semantic Search (RAG): Index your projects and search for relevant code/docs semanticially using 'rag_index' and 'rag_search'.\n"
        "- Desktop automation: open web browser, manage clipboard, etc.\n\n"
        "OPERATIONAL RULES:\n"
        "1. ALWAYS return COMPLETE code — never use placeholders or trancations (e.g., no '// ...').\n"
        "2. Use appropriate markdown code blocks for all code and terminal output.\n"
        "3. Follow best practices for the language being used (idiomatic, clean, documented).\n"
        "4. Use tool calls for all system-level operations. Do not just describe them.\n"
        "5. Be concise and technical. Avoid conversational filler.\n"
        "6. If a task is complex, break it down into steps and execute them sequentially.\n"
        "7. ALWAYS confirm before performing potentially destructive operations (delete, overwrite) unless auto-execute is enabled.\n\n"
        "FILE SAVING BEHAVIOR (CRITICAL):\n"
        "- When the user asks you to create, modify, or generate code, ALWAYS save the result to disk using the 'write_file' or 'edit_file' tool.\n"
        "- Do NOT just display code in chat and wait for the user to save it manually.\n"
        "- For NEW files: use 'write_file' to create them directly.\n"
        "- For EXISTING files: prefer 'edit_file' for targeted changes or 'write_file' to replace the whole file.\n"
        "- If a filename is not specified, infer a reasonable filename from context (e.g., 'main.py', 'server.js', 'style.css').\n"
        "- After saving, briefly confirm what was written and where.\n"
    ),
    "compact_system_prompt": (
        "You are MILEX, an AI agent. Execute tasks using tools. Be brief. Complete code only. "
        "ALWAYS save files directly using write_file or edit_file — never just show code in chat."
    ),
}


class ConfigManager:
    """Manages configuration with ChainMap for layered defaults."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._chain = ChainMap(self._config, DEFAULT_CONFIG)

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with fallback to defaults."""
        return self._chain.get(key, default)

    def get_nested(self, key_path: str, default: Any = None) -> Any:
        """Get nested config value using dot notation (e.g., 'rag.enabled')."""
        keys = key_path.split(".")
        value = self._chain
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value if value is not None else default

    def set(self, key: str, value: Any):
        """Set a config value."""
        self._config[key] = value
        # Rebuild chain after modification
        self._chain = ChainMap(self._config, DEFAULT_CONFIG)

    def update(self, updates: Dict[str, Any]):
        """Update multiple config values."""
        self._config.update(updates)
        self._chain = ChainMap(self._config, DEFAULT_CONFIG)

    def to_dict(self) -> Dict[str, Any]:
        """Return merged config as dict."""
        return dict(self._chain)

    @property
    def raw(self) -> Dict[str, Any]:
        """Return the user config without defaults."""
        return self._config


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


def save_history(history: list, append_only: bool = False):
    ensure_config_dir()
    # Keep last 1000 entries
    trimmed = history[-1000:]

    if append_only:
        # For append-only mode, load existing and extend
        try:
            existing = load_history()
            combined = existing + trimmed
            trimmed = combined[-1000:]
        except (json.JSONDecodeError, OSError):
            pass

    with open(HISTORY_FILE, "w") as f:
        json.dump(trimmed, f, indent=2)
