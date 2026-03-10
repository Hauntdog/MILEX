from __future__ import annotations
import asyncio
import importlib.util
import os
import platform
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from milex.ui import AgentUI, ThinkingSpinner
    from milex.rag import RagManager
    from milex.agent import MilexAgent
else:
    # Runtime imports for functionality (not needed for type checking only)
    from milex.ui import ThinkingSpinner, print_warning

# ─── Module-level security constants & helpers ───────────────────────────────

MAX_READ_BYTES = 1 * 1024 * 1024  # 1 MB

# Set of dangerous tool names that require confirmation
DANGEROUS_TOOLS = frozenset({
    "run_shell",
})


def _validate_path(path_str: str, config: dict, must_exist: bool = False) -> Path:
    """Resolve and validate a path is within the allowed root (if set)."""
    p = Path(path_str).expanduser().resolve()
    
    allowed_root_str = config.get("allowed_root")
    if allowed_root_str:
        root = Path(allowed_root_str).resolve()
        try:
            if not p.is_relative_to(root):
                raise PermissionError(f"Path '{p}' is outside the allowed root '{root}'")
        except (AttributeError, ValueError):
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
            "name": "edit_file",
            "description": (
                "Edit an existing file by performing targeted find-and-replace operations. "
                "Use this instead of write_file when you only need to change specific parts "
                "of a file. Each edit specifies old_text to find and new_text to replace it with. "
                "If old_text is empty, new_text is inserted at the beginning of the file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to edit"},
                    "edits": {
                        "type": "array",
                        "description": "List of edit operations",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_text": {
                                    "type": "string",
                                    "description": "Exact text to find (empty string = insert at top)",
                                },
                                "new_text": {
                                    "type": "string",
                                    "description": "Replacement text",
                                },
                            },
                            "required": ["old_text", "new_text"],
                        },
                    },
                },
                "required": ["path", "edits"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_file",
            "description": "Append content to the end of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to file"},
                    "content": {
                        "type": "string",
                        "description": "Content to append",
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
            "name": "move_path",
            "description": "Move or rename a file or directory.",
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
    {
        "type": "function",
        "function": {
            "name": "read_url_content",
            "description": "Fetch content from a URL and return its text/markdown content. Useful for research.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to read from"}
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_files",
            "description": "Read multiple files at once. Efficient for understanding dependencies and architecting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of absolute or relative file paths to read",
                    }
                },
                "required": ["paths"],
            },
        },
    },
]


# ─── Tool Executor ────────────────────────────────────────────────────────────


class ToolExecutor:
    """Executes tool calls and returns results."""

    def __init__(
        self,
        config: dict,
        ui: Optional[AgentUI] = None,
        rag: Optional[RagManager] = None,
        agent: Optional[MilexAgent] = None,
        auto_execute: bool = False,
    ):
        self.config = config
        self.ui = ui
        self.rag = rag
        self.agent = agent
        self.auto_execute = auto_execute
        self.plugins: Dict[str, Callable] = {}
        self._load_plugins()

    def _load_plugins(self):
        """Dynamic plugin discovery."""
        plugin_dir = self.config.get("plugin_dir")
        if not plugin_dir: return
        
        path = Path(plugin_dir).expanduser()
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            return

        for py_file in path.glob("*.py"):
            try:
                name = py_file.stem
                spec = importlib.util.spec_from_file_location(name, str(py_file))
                if not spec or not spec.loader: continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                if hasattr(module, "TOOL_DEFINITION") and hasattr(module, "handler"):
                    # Add to definitions (used by model)
                    TOOL_DEFINITIONS.append(module.TOOL_DEFINITION)
                    # Register handler
                    self.plugins[module.TOOL_DEFINITION["function"]["name"]] = module.handler
            except Exception as e:
                print_warning(f"Plugin load error ({py_file.name}): {e}")

    async def execute_async(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool. Handles both internal and MCP tools with telemetry."""
        from .telemetry import telemetry
        import time
        start_time = time.time()
        success = True
        error = None
        
        try:
            # Handle MCP Tools (prefixed with server_name__)
            if "__" in tool_name and self.agent and hasattr(self.agent, "mcp"):
                parts = tool_name.split("__", 1)
                server_name, raw_tool_name = parts[0], parts[1]
                
                # MCP calls are always async
                try:
                    mcp_res = await self.agent.mcp.call_tool(server_name, raw_tool_name, args)
                    # Convert MCP result to dict format
                    result = {"content": [str(c) for c in mcp_res.content], "isError": mcp_res.isError}
                    if mcp_res.isError: success = False
                    return result
                except Exception as e:
                    error = str(e)
                    success = False
                    return {"error": f"MCP Error: {error}"}

            # Internal tools are executed in a thread to keep the event loop free
            result = await asyncio.to_thread(self.execute, tool_name, args)
            if "error" in result: 
                success = False
                error = result["error"]
            return result
        finally:
            # Record execution telemetry
            telemetry.record(tool_name, time.time() - start_time, success, error)

    async def get_all_tools(self) -> List[Dict[str, Any]]:
        """Fetch all available tools (internal + plugins + MCP)."""
        tools = TOOL_DEFINITIONS.copy()
        
        # Add MCP tools if agent and MCP are available
        if self.agent and hasattr(self.agent, "mcp"):
            mcp_tools = await self.agent.mcp.get_all_tools()
            tools.extend(mcp_tools)
            
        return tools

    def execute(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch to the appropriate tool handler."""
        handlers = {
            "run_shell": self._run_shell,
            "read_file": self._read_file,
            "write_file": self._write_file,
            "edit_file": self._edit_file,
            "append_file": self._append_file,
            "list_directory": self._list_directory,
            "search_files": self._search_files,
            "get_system_info": self._get_system_info,
            "create_directory": self._create_directory,
            "delete_path": self._delete_path,
            "copy_path": self._copy_path,
            "move_path": self._move_path,
            "open_browser": self._open_browser,
            "clipboard_copy": self._clipboard_copy,
            "generate_code": self._generate_code,
            "rag_index": self._rag_index,
            "rag_search": self._rag_search,
            "read_url_content": self._read_url_content,
            "read_files": self._read_files,
        }

        # Merge with plugins
        handlers.update(self.plugins)

        handler = handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}

        # Dangerous tools need confirmation
        # Plugins can be dangerous, so they default to dangerous
        # Only truly destructive ops need confirmation.
        # File writes are auto-approved (like Gemini CLI) so the bot
        # can save files on its own without prompting every time.
        is_plugin = tool_name in self.plugins

        if (tool_name in DANGEROUS_TOOLS or is_plugin) and not self.auto_execute:
            if self.ui and not self.ui.confirm_tool(tool_name, args):
                return {"status": "cancelled", "message": "User cancelled execution"}

        try:
            if is_plugin:
                # Plugin API: handler(args_dict, *, config, ui, executor)
                return handler(args, config=self.config, ui=self.ui, executor=self)
            else:
                return handler(**args)
        except Exception as e:
            return {"error": str(e)}

    def _run_shell(
        self, command: str, cwd: Optional[str] = None, timeout: int = 30
    ) -> dict:
        try:
            if cwd:
                _validate_path(cwd, self.config, must_exist=True)

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
            p = _validate_path(path, self.config, must_exist=True)
        except (PermissionError, FileNotFoundError) as e:
            return {"error": str(e)}
        if p.stat().st_size > MAX_READ_BYTES:
            return {
                "error": f"File too large ({p.stat().st_size} bytes, max {MAX_READ_BYTES})"
            }
        content = p.read_text(errors="replace")
        return {"content": content, "size": p.stat().st_size}

    def _write_file(self, path: str, content: str) -> dict:
        try:
            p = _validate_path(path, self.config)
        except PermissionError as e:
            return {"error": str(e)}
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return {"success": True, "path": str(p), "bytes_written": len(content)}

    def _append_file(self, path: str, content: str) -> dict:
        try:
            p = _validate_path(path, self.config)
        except PermissionError as e:
            return {"error": str(e)}
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            f.write(content)
        return {"success": True, "path": str(p), "bytes_appended": len(content)}

    def _edit_file(self, path: str, edits: list) -> dict:
        """Apply targeted find-and-replace edits to a file."""
        try:
            p = _validate_path(path, self.config, must_exist=True)
        except (PermissionError, FileNotFoundError) as e:
            return {"error": str(e)}

        content = p.read_text(errors="replace")
        applied = 0
        failed = []

        for i, edit in enumerate(edits):
            old_text = edit.get("old_text", "")
            new_text = edit.get("new_text", "")

            if not old_text:
                # Empty old_text → insert at beginning
                content = new_text + content
                applied += 1
            elif old_text in content:
                content = content.replace(old_text, new_text, 1)
                applied += 1
            else:
                failed.append({
                    "edit_index": i,
                    "old_text_preview": old_text[:80],
                    "reason": "old_text not found in file",
                })

        p.write_text(content)
        result = {
            "success": True,
            "path": str(p),
            "edits_applied": applied,
            "edits_failed": len(failed),
        }
        if failed:
            result["failures"] = failed
        return result

    def _list_directory(self, path: str = ".", recursive: bool = False) -> dict:
        try:
            p = _validate_path(path, self.config, must_exist=True)
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

        try:
            root = _validate_path(path, self.config, must_exist=True)
        except (PermissionError, FileNotFoundError) as e:
            return {"error": str(e)}

        matches = []
        with ThinkingSpinner(f"Searching for '{pattern}'..."):
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
        try:
            p = _validate_path(path, self.config)
        except PermissionError as e:
            return {"error": str(e)}
        p.mkdir(parents=True, exist_ok=True)
        return {"success": True, "path": str(p.resolve())}

    def _delete_path(self, path: str, recursive: bool = False) -> dict:
        try:
            p = _validate_path(path, self.config, must_exist=True)
        except (PermissionError, FileNotFoundError) as e:
            return {"error": str(e)}
        
        allowed_root_str = self.config.get("allowed_root")
        root = Path(allowed_root_str or Path.cwd()).resolve()
        
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
            s = _validate_path(src, self.config, must_exist=True)
            d = _validate_path(dst, self.config)
        except (PermissionError, FileNotFoundError) as e:
            return {"error": str(e)}
        if s.is_dir():
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
        return {"success": True, "src": str(s), "dst": str(d)}

    def _move_path(self, src: str, dst: str) -> dict:
        try:
            s = _validate_path(src, self.config, must_exist=True)
            d = _validate_path(dst, self.config)
        except (PermissionError, FileNotFoundError) as e:
            return {"error": str(e)}

        # Ensure parent directory of destination exists
        d.parent.mkdir(parents=True, exist_ok=True)

        shutil.move(s, d)
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
        if not self.agent:
            return {"error": "Code generation requires a running agent instance"}
        return self.agent.generate_code_internal(task, language, filename)

    def _rag_index(self, path: str = ".") -> dict:
        if not self.rag:
            return {"error": "RAG is disabled"}
        self.rag.index_directory(path)
        return {"success": True, "path": path, "chunks": len(self.rag.chunks)}

    def _rag_search(self, query: str, count: int = 5) -> dict:
        if not self.rag:
            return {"error": "RAG is disabled"}
        with ThinkingSpinner(f"Searching knowledge base for '{query}'..."):
            results = self.rag.search(query, top_k=count)
        return {"results": results, "count": len(results)}

    def _read_url_content(self, url: str) -> dict:
        try:
            import requests
            from bs4 import BeautifulSoup
            from markdownify import markdownify as md

            headers = {"User-Agent": "Mozilla/5.0 (MILEX Bot)"}
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Basic markdown conversion
            markdown = md(str(soup), heading_style="ATX", bullets="-")
            
            # Clean up whitespace
            markdown = "\n".join(line for line in markdown.splitlines() if line.strip())
            
            return {
                "url": url,
                "title": soup.title.string if soup.title else "No Title",
                "content": markdown[:50000],  # Limit to 50k chars
                "truncated": len(markdown) > 50000
            }
        except Exception as e:
            return {"error": f"Failed to read URL: {str(e)}"}

    def _read_files(self, paths: List[str]) -> dict:
        results = {}
        for path in paths:
            res = self._read_file(path)
            results[path] = res
        return {"files": results}
