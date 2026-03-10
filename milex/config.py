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
        "embeddings": "nomic-embed-text:latest",
        "fallback": "qwen2.5:1.5b"
    },
    "ollama_host": "http://localhost:11434",
    "daemon_token": None,  # Will be generated in load_config
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
    "mcp_servers": {
        # Example format: 
        # "github": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"]}
        # Playwright MCP server for autonomous browser automation
        "browser": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-playwright"],
            "env": {}
        }
    },
    "rag": {
        "enabled": True,
        "index_on_startup": False,
        "chunk_size": 1000,
        "chunk_overlap": 100,
        "similarity_threshold": 0.3, # Cosine similarity threshold for RAG results
        "exclude_dirs": [".git", "__pycache__", ".venv", "node_modules", "dist", "build"]
    },
    "system_prompt": (
        "You are MILEX, an elite AI Agentic Coder and Computer-Control Agent, designed to build and architect complex systems.\n\n"
        "CORE CAPABILITIES:\n"
        "- ARCHITECTING: You don't just write code; you design systems. Use 'read_files', 'list_directory', and 'rag_search' to understand the whole codebase before making changes.\n"
        "- BROWSER & RESEARCH: Use 'read_url_content' to read documentation, or use browser__* MCP tools (Playwright) for autonomous browsing with page interaction, clicking, scrolling, and form filling.\n"
        "- AUTONOMOUS FILE SAVING: Use 'write_file' or 'edit_file' to save all changes directly to disk. Never just display code in chat without saving.\n"
        "- MULTI-FILE EDITS: Use 'edit_file' for targeted find-and-replace across existing files. Use 'write_file' for new files.\n"
        "- COMPUTER CONTROL: Execute shell commands via 'run_shell' to build, test, and manage the environment.\n\n"
        "AUTONOMOUS BROWSING (Playwright MCP):\n"
        "- Use browser__navigate to open URLs and browse pages\n"
        "- Use browser__click to click elements\n"
        "- Use browser__fill to fill forms\n"
        "- Use browser__scrape to extract page content\n"
        "- Use browser__screenshot to capture pages\n\n"
        "ARCHITECTURAL WORKFLOW (CRITICAL):\n"
        "1. ANALYSIS: When given a complex task, start by exploring the codebase. Read relevant files and dependencies.\n"
        "2. PLANNING: For any multi-step task, create an 'implementation_plan.md' file first. Outline the steps, files to be modified, and potential risks.\n"
        "3. EXECUTION: Follow your plan. You have full authority to manage files (create, edit, delete, move) autonomously without asking for permission for individual steps.\n"
        "4. VERIFICATION: After coding, run tests or linting via 'run_shell' if applicable to ensure correctness.\n\n"
        "OPERATIONAL RULES:\n"
        "- ALWAYS return COMPLETE, production-ready code. No placeholders, no '// ...' truncations.\n"
        "- Be technical, concise, and proactive. Do not wait for permission for safe tool calls (like reading files or URLs).\n"
        "- If the user hasn't specified a filename, infer one idiomatic to the project (e.g., 'utils.py', 'main.js').\n"
        "- Support high-end aesthetics: Use Rich-style formatting in your responses.\n"
    ),
    "compact_system_prompt": (
        "You are MILEX, an elite Agentic Coder. Plan via 'implementation_plan.md', "
        "research via 'read_url_content' or browser__* MCP tools for autonomous browsing, "
        "and architect via 'read_files'. Always save changes directly to disk using tools."
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
    cfg = DEFAULT_CONFIG.copy()
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                user_cfg = json.load(f)
            cfg.update(user_cfg)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[milex] Warning: config issue ({exc}). Using defaults.", file=sys.stderr)

    # Ensure we have a daemon token for security
    if not cfg.get("daemon_token"):
        import secrets
        cfg["daemon_token"] = secrets.token_hex(16)
        save_config(cfg)
        
    return cfg


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
