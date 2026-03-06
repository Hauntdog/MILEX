"""Computer control tools for MILEX CLI."""
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ─── Module-level security constants & helpers ───────────────────────────────

_ALLOWED_ROOT: Optional[Path] = None  # Set to restrict operations to a root dir
MAX_READ_BYTES = 1 * 1024 * 1024  # 1 MB


def _validate_path(path_str: str, must_exist: bool = False) -> Path:
    """Resolve and validate a path is within the allowed root (if set) or CWD."""
    p = Path(path_str).expanduser().resolve()
    root = (_ALLOWED_ROOT or Path.cwd()).resolve()

    # is_relative_to is Python 3.9+; we're on 3.9+ as per pyproject.toml
    try:
        if not p.is_relative_to(root):
            raise PermissionError(f"Path '{p}' is outside the allowed root '{root}'")
    except (AttributeError, ValueError):
        # Fallback for older versions or slightly different behavior
        if not (str(p) == str(root) or str(p).startswith(str(root) + os.sep)):
            raise PermissionError(f"Path '{p}' is outside the allowed root '{root}'")

    if must_exist and not p.exists():
        raise FileNotFoundError(f"Path not found: {path_str}")
    return p


# ─── Tool Definitions (sent to model) ────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Execute a shell command on the user's computer. "
                "Use this to run scripts, install packages, check system info, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory (optional)",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 30)",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file from the filesystem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file (creates or overwrites).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to write to"},
                    "content": {
                        "type": "string",
                        "description": "Content to write",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories at a path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path (default: current directory)",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "List recursively (default: false)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files by name or content pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Search pattern (glob or regex)",
                    },
                    "path": {
                        "type": "string",
                        "description": "Root directory to search from",
                    },
                    "content_search": {
                        "type": "string",
                        "description": "Optional: search for this text inside files",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_info",
            "description": "Get information about the current system (OS, CPU, memory, etc.).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_directory",
            "description": "Create a directory (and parents if needed).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to create"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_path",
            "description": "Delete a file or directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to delete"},
                    "recursive": {
                        "type": "boolean",
                        "description": "Delete directory recursively",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "copy_path",
            "description": "Copy a file or directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "src": {"type": "string", "description": "Source path"},
                    "dst": {"type": "string", "description": "Destination path"},
                },
                "required": ["src", "dst"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_browser",
            "description": "Open a URL in the default web browser.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to open"}
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clipboard_copy",
            "description": "Copy text to the system clipboard.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to copy"}
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_code",
            "description": (
                "Generate complete, production-ready code for a given task. "
                "Returns the code as a string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Description of what the code should do",
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language (e.g. python, javascript, bash)",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Optional filename to save the generated code",
                    },
                },
                "required": ["task", "language"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_index",
            "description": "Index the current project or a directory for semantic search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to index (default: current directory)",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Search the indexed codebase for relevant snippets using natural language.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (e.g. 'how is authentication handled?')",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


# ─── Tool Executor ────────────────────────────────────────────────────────────


class ToolExecutor:
    """Executes tool calls and returns results."""

    def __init__(self, auto_execute: bool = False, confirm_callback=None):
        self.auto_execute = auto_execute
        self.confirm_callback = confirm_callback  # fn(tool_name, args) -> bool

    def execute(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch to the appropriate tool handler."""
        handlers = {
            "run_shell": self._run_shell,
            "read_file": self._read_file,
            "write_file": self._write_file,
            "list_directory": self._list_directory,
            "search_files": self._search_files,
            "get_system_info": self._get_system_info,
            "create_directory": self._create_directory,
            "delete_path": self._delete_path,
            "copy_path": self._copy_path,
            "open_browser": self._open_browser,
            "clipboard_copy": self._clipboard_copy,
            "generate_code": self._generate_code,
            "rag_index": self._rag_index,
            "rag_search": self._rag_search,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}

        # Dangerous tools need confirmation
        dangerous = {"run_shell", "delete_path", "write_file", "copy_path"}
        if tool_name in dangerous and not self.auto_execute:
            if self.confirm_callback and not self.confirm_callback(tool_name, args):
                return {"status": "cancelled", "message": "User cancelled execution"}

        try:
            return handler(**args)
        except Exception as e:
            return {"error": str(e)}

    def _run_shell(self, command: str, cwd: Optional[str] = None, timeout: int = 30) -> dict:
        try:
            # Try list-based execution first (safer)
            try:
                cmd_list = shlex.split(command)
                result = subprocess.run(
                    cmd_list,
                    shell=False,
                    capture_output=True,
                    text=True,
                    cwd=cwd,
                    timeout=timeout,
                )
            except ValueError:
                # shlex.split can fail on complex shell syntax; fall back to shell=True
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=cwd,
                    timeout=timeout,
                )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "error": f"Command timed out after {timeout} seconds",
                "returncode": -1,
            }

    def _read_file(self, path: str) -> dict:
        try:
            p = _validate_path(path, must_exist=True)
        except (PermissionError, FileNotFoundError) as e:
            return {"error": str(e)}
        if p.stat().st_size > MAX_READ_BYTES:
            return {"error": f"File too large ({p.stat().st_size} bytes, max {MAX_READ_BYTES})"}
        content = p.read_text(errors="replace")
        return {"content": content, "size": p.stat().st_size}

    def _write_file(self, path: str, content: str) -> dict:
        try:
            p = _validate_path(path)
        except PermissionError as e:
            return {"error": str(e)}
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return {"success": True, "path": str(p), "bytes_written": len(content)}

    def _list_directory(self, path: str = ".", recursive: bool = False) -> dict:
        try:
            p = _validate_path(path, must_exist=True)
        except (PermissionError, FileNotFoundError) as e:
            return {"error": str(e)}

        entries = []
        if recursive:
            iterator = p.rglob("*")
        else:
            iterator = p.iterdir()

        for x in sorted(iterator):
            try:
                stats = x.stat()
                rel_path = str(x.relative_to(p)) if recursive else x.name
                entries.append(
                    {
                        "name": rel_path,
                        "type": "directory" if x.is_dir() else "file",
                        "size": stats.st_size if x.is_file() else None,
                        "modified": stats.st_mtime,
                    }
                )
            except (OSError, PermissionError):
                continue

        return {"path": str(p), "entries": entries}

    def _search_files(
        self, pattern: str, path: str = ".", content_search: Optional[str] = None
    ) -> dict:
        import fnmatch

        root = Path(path).expanduser()
        if not root.exists():
            return {"error": f"Path not found: {path}"}
        matches = []
        for f in root.rglob("*"):
            if not f.is_file():
                continue
            if fnmatch.fnmatch(f.name, pattern):
                if content_search:
                    try:
                        if content_search in f.read_text(errors="replace"):
                            matches.append(str(f))
                    except Exception:
                        pass
                else:
                    matches.append(str(f))
        return {"matches": matches, "count": len(matches)}

    def _get_system_info(self) -> dict:
        try:
            import psutil

            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            return {
                "os": platform.system(),
                "os_version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python": sys.version,
                "cpu_count": os.cpu_count(),
                "memory_total_gb": round(mem.total / 1e9, 2),
                "memory_available_gb": round(mem.available / 1e9, 2),
                "disk_total_gb": round(disk.total / 1e9, 2),
                "disk_free_gb": round(disk.free / 1e9, 2),
                "cwd": os.getcwd(),
                "home": str(Path.home()),
            }
        except ImportError:
            return {
                "os": platform.system(),
                "machine": platform.machine(),
                "python": sys.version,
                "cwd": os.getcwd(),
            }

    def _create_directory(self, path: str) -> dict:
        p = Path(path).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return {"success": True, "path": str(p.resolve())}

    def _delete_path(self, path: str, recursive: bool = False) -> dict:
        try:
            p = _validate_path(path, must_exist=True)
        except (PermissionError, FileNotFoundError) as e:
            return {"error": str(e)}
        root = (_ALLOWED_ROOT or Path.cwd()).resolve()
        if p == root:
            return {"error": "Refusing to delete the working root directory"}
        if p.is_dir():
            if recursive:
                shutil.rmtree(p)
            else:
                p.rmdir()
        else:
            p.unlink()
        return {"success": True, "deleted": str(p)}

    def _copy_path(self, src: str, dst: str) -> dict:
        try:
            s = _validate_path(src, must_exist=True)
        except (PermissionError, FileNotFoundError) as e:
            return {"error": str(e)}
        try:
            d = _validate_path(dst)
        except PermissionError as e:
            return {"error": str(e)}
        if s.is_dir():
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
        return {"success": True, "src": str(s), "dst": str(d)}

    def _open_browser(self, url: str) -> dict:
        import webbrowser
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return {"error": f"Refusing to open non-HTTP URL scheme: {parsed.scheme!r}"}
        webbrowser.open(url)
        return {"success": True, "url": url}

    def _clipboard_copy(self, text: str) -> dict:
        try:
            import pyperclip

            pyperclip.copy(text)
            return {"success": True}
        except Exception as e:
            return {"error": str(e)}

    def _generate_code(
        self, task: str, language: str, filename: Optional[str] = None
    ) -> dict:
        # This is handled at the agent level; here we just return a marker
        return {"_generate_code": True, "task": task, "language": language, "filename": filename}

    def _rag_index(self, path: str = ".") -> dict:
        # Handled by agent's RagManager
        return {"_rag_index": True, "path": path}

    def _rag_search(self, query: str, count: int = 5) -> dict:
        # Handled by agent's RagManager
        return {"_rag_search": True, "query": query, "count": count}
