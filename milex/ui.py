"""Rich-based UI components for MILEX CLI."""
import sys
from typing import Optional

from rich import box
from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
import json

# ─── Theme ───────────────────────────────────────────────────────────────────

MILEX_THEME = Theme(
    {
        "milex.brand": "bold cyan",
        "milex.user": "bold green",
        "milex.ai": "bold blue",
        "milex.tool": "bold yellow",
        "milex.error": "bold red",
        "milex.success": "bold green",
        "milex.warning": "bold yellow",
        "milex.muted": "dim white",
        "milex.code": "bright_white on grey11",
        "milex.header": "bold bright_cyan",
    }
)

console = Console(theme=MILEX_THEME, highlight=True)


# ─── Banner ───────────────────────────────────────────────────────────────────


def print_banner():
    banner = Text()
    banner.append(
        """
    ███╗   ███╗██╗██╗     ███████╗██╗  ██╗
    ████╗ ████║██║██║     ██╔════╝╚██╗██╔╝
    ██╔████╔██║██║██║     █████╗   ╚███╔╝ 
    ██║╚██╔╝██║██║██║     ██╔══╝   ██╔██╗ 
    ██║ ╚═╝ ██║██║███████╗███████╗██╔╝ ██╗
    ╚═╝     ╚═╝╚═╝╚══════╝╚══════╝╚═╝  ╚═╝
""",
        style="bold cyan",
    )
    subtitle = Text("  AI-Powered CLI  ·  Code Generation  ·  Computer Control\n", style="dim cyan")

    panel = Panel(
        Align.center(Text.assemble(banner, subtitle)),
        border_style="cyan",
        padding=(0, 2),
    )
    console.print(panel)


def print_welcome(model: str, host: str):
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim cyan", justify="right")
    table.add_column(style="white")
    table.add_row("Model", f"[bold cyan]{model}[/]")
    table.add_row("Host", f"[dim]{host}[/]")
    table.add_row("Type", "[dim]'help' for commands  ·  'exit' to quit[/]")
    console.print(Panel(table, border_style="dim cyan", title="[dim cyan]● connected[/]", title_align="right"))


# ─── UI Interfaces ───────────────────────────────────────────────────────────


class AgentUI:
    """Abstract base class for AI agent UI handlers."""

    def print_ai_message(self, text: str, model: str = "MILEX") -> None:
        raise NotImplementedError

    def print_tool_call(self, tool_name: str, args: dict) -> None:
        raise NotImplementedError

    def print_tool_result(self, tool_name: str, result: dict, success: bool = True) -> None:
        raise NotImplementedError

    def print_error(self, message: str) -> None:
        raise NotImplementedError

    def print_success(self, message: str) -> None:
        raise NotImplementedError

    def print_warning(self, message: str) -> None:
        raise NotImplementedError

    def print_info(self, message: str) -> None:
        raise NotImplementedError

    def confirm_tool(self, tool_name: str, args: dict) -> bool:
        raise NotImplementedError

    def create_stream_renderer(self, model: str = "MILEX") -> "StreamRenderer":
        raise NotImplementedError

    def create_thinking_spinner(self, message: str = "Thinking...") -> "ThinkingSpinner":
        raise NotImplementedError

    def print_code_block(self, code: str, language: str = "python", filename: Optional[str] = None) -> None:
        raise NotImplementedError

    def ask_save_file(self, code: str, language: str) -> Optional[str]:
        raise NotImplementedError

    def ask_run_command(self, filename: str) -> bool:
        raise NotImplementedError


class RichUI(AgentUI):
    """Rich-based implementation of the Agent UI with simplified delegation."""

    def __init__(self, console_obj: Optional[Console] = None):
        self.console: Console = console_obj or console

    def print_ai_message(self, text: str, model: str = "MILEX") -> None:
        print_ai_message(text, model)

    def print_tool_call(self, tool_name: str, args: dict) -> None:
        print_tool_call(tool_name, args)

    def print_tool_result(self, tool_name: str, result: dict, success: bool = True) -> None:
        print_tool_result(tool_name, result, success)

    def print_error(self, message: str) -> None:
        print_error(message)

    def print_success(self, message: str) -> None:
        print_success(message)

    def print_warning(self, message: str) -> None:
        print_warning(message)

    def print_info(self, message: str) -> None:
        print_info(message)

    def print_code_block(self, code: str, language: str = "python", filename: Optional[str] = None) -> None:
        print_code_block(code, language, filename)

    def confirm_tool(self, tool_name: str, args: dict) -> bool:
        return confirm_tool_execution(tool_name, args)

    def create_stream_renderer(self, model: str = "MILEX") -> "StreamRenderer":
        return StreamRenderer(model=model)

    def create_thinking_spinner(self, message: str = "Thinking...") -> "ThinkingSpinner":
        return ThinkingSpinner(message=message)

    def ask_save_file(self, code: str, language: str) -> Optional[str]:
        self.console.print(
            f"\n[dim cyan]💾 Save this [bold]{language}[/] code block to a file? "
            f"(Enter filename or press Enter to skip)[/]"
        )
        try:
            filename = Prompt.ask("[dim cyan]Filename[/]", default="")
            return filename.strip() if filename.strip() else None
        except (KeyboardInterrupt, EOFError):
            return None

    def ask_run_command(self, filename: str) -> bool:
        try:
            return Confirm.ask(f"\n[dim cyan]▶ Run [bold]{filename}[/]?[/]", default=False)
        except (KeyboardInterrupt, EOFError):
            return False


# ─── Message Rendering ────────────────────────────────────────────────────────


def print_user_message(text: str):
    console.print()
    console.print(
        Panel(
            text,
            title="[milex.user]You[/]",
            title_align="left",
            border_style="green",
            padding=(0, 1),
        )
    )


def print_ai_message(text: str, model: str = "MILEX"):
    """Render AI response with markdown and syntax highlighting."""
    console.print()
    # Parse and render markdown with code block highlighting
    rendered = _render_markdown_with_syntax(text)
    console.print(
        Panel(
            rendered,
            title=f"[milex.ai]✦ {model}[/]",
            title_align="left",
            border_style="blue",
            padding=(0, 1),
        )
    )


def _render_markdown_with_syntax(text: str):
    """Return a renderable with syntax-highlighted code blocks."""
    return Markdown(text, code_theme="monokai", inline_code_theme="monokai")


def print_tool_call(tool_name: str, args: dict):
    """Display a tool call being executed."""
    args_str = json.dumps(args, indent=2)
    console.print()
    console.print(
        Panel(
            Syntax(args_str, "json", theme="monokai", line_numbers=False),
            title=f"[milex.tool]⚙ Tool: {tool_name}[/]",
            title_align="left",
            border_style="yellow",
            padding=(0, 1),
        )
    )


def print_tool_result(tool_name: str, result: dict, success: bool = True):
    """Display tool execution result. File operations get a compact summary."""
    # Compact one-liner for file write/edit/append operations
    if success and tool_name in ("write_file", "edit_file", "append_file"):
        path = result.get("path", "?")
        if tool_name == "write_file":
            size = result.get("bytes_written", "?")
            console.print(f"\n[bold green]✍ Wrote[/] [dim]{size} bytes →[/] [cyan]{path}[/]")
        elif tool_name == "edit_file":
            applied = result.get("edits_applied", 0)
            failed = result.get("edits_failed", 0)
            status = f"{applied} edit{'s' if applied != 1 else ''} applied"
            if failed:
                status += f", [red]{failed} failed[/]"
            console.print(f"\n[bold green]✏ Edited[/] [cyan]{path}[/] [dim]({status})[/]")
        elif tool_name == "append_file":
            size = result.get("bytes_appended", "?")
            console.print(f"\n[bold green]✍ Appended[/] [dim]{size} bytes →[/] [cyan]{path}[/]")
        return

    result_str = json.dumps(result, indent=2)
    style = "green" if success else "red"
    icon = "✓" if success else "✗"
    console.print(
        Panel(
            Syntax(result_str, "json", theme="monokai", line_numbers=False),
            title=f"[{'milex.success' if success else 'milex.error'}]{icon} Result: {tool_name}[/]",
            title_align="left",
            border_style=style,
            padding=(0, 1),
        )
    )


def print_code_block(code: str, language: str = "python", filename: Optional[str] = None):
    """Display a syntax-highlighted code block."""
    title = f"[milex.code]{filename}[/]" if filename else f"[dim]{language}[/]"
    console.print()
    console.print(
        Panel(
            Syntax(code, language, theme="monokai", line_numbers=True, word_wrap=True),
            title=title,
            border_style="bright_black",
            padding=(0, 0),
        )
    )


def print_error(message: str):
    console.print(f"\n[milex.error]✗ Error:[/] {message}")


def print_success(message: str):
    console.print(f"\n[milex.success]✓[/] {message}")


def print_warning(message: str):
    console.print(f"\n[milex.warning]⚠[/] {message}")


def print_info(message: str):
    console.print(f"\n[dim cyan]ℹ[/] [dim]{message}[/]")


def print_rule(title: str = ""):
    console.print(Rule(title, style="dim cyan"))


# ─── Streaming Output ────────────────────────────────────────────────────────


class StreamRenderer:
    """Renders streaming AI response with live updates."""

    def __init__(self, model: str = "MILEX"):
        self.model = model
        self.buffer = ""
        self._live: Optional[Live] = None
        self._is_terminal = console.is_terminal

    def __enter__(self):
        self._is_terminal = console.is_terminal
        if self._is_terminal:
            # We delay console.print() and Live creation until the first chunk
            pass
        else:
            # Simple header if not a terminal (daemon mode)
            # We still delay this until we have actual text
            pass
        return self

    def _start_display(self):
        """Initialize the display when the first real chunk arrives."""
        if self._is_terminal:
            console.print()
            self._live = Live(
                self._make_panel("▌"),
                console=console,
                refresh_per_second=15,
                vertical_overflow="visible",
            )
            self._live.__enter__()
        else:
            console.print(f"\n[dim cyan]✦ {self.model}[/]\n", end="")
        return self

    def update(self, chunk: str):
        if not self.buffer and chunk.strip():
            self._start_display()

        self.buffer += chunk
        if self._is_terminal and self._live:
            self._live.update(self._make_panel(self.buffer + "▌"))
        elif not self._is_terminal and chunk:
            console.print(chunk, end="", markup=False, highlight=False)

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        if self._is_terminal and self._live is not None:
            try:
                self._live.update(self._make_panel(self.buffer))
                self._live.__exit__(exc_type, exc_val, exc_tb)
            except Exception:
                pass
            self._live = None
        elif not self._is_terminal and self.buffer:
            console.print()

    def _make_panel(self, text: str):
        return Panel(
            Markdown(text, code_theme="monokai", inline_code_theme="monokai"),
            title=f"[milex.ai]✦ {self.model}[/]",
            title_align="left",
            border_style="cyan",
            padding=(0, 1),
        )

    def get_text(self) -> str:
        return self.buffer
class ThinkingSpinner:
    def __init__(self, message: str = "Thinking..."):
        self.message = message
        self._progress = Progress(
            SpinnerColumn(spinner_name="dots", style="cyan"),
            TextColumn(f"[dim cyan]{message}[/]"),
            console=console,
            transient=True,
        )
        self._task = None
        self._is_started = False

    def start(self):
        if console.is_terminal:
            try:
                self._progress.start()
                self._task = self._progress.add_task(self.message)
                self._is_started = True
            except Exception:
                # Fallback if Rich Progress fails (e.g., non-interactive terminal)
                console.print(f"[dim cyan]● {self.message}[/]", end=" ", flush=True)
                self._is_started = False # Indicate that Rich Progress didn't start
        else:
            # Headless mode: just a simple indicator
            console.print(f"[dim cyan]● {self.message}[/]", end=" ", flush=True)
            self._is_started = False # Indicate that Rich Progress didn't start
        return self

    def stop(self, exc_type=None, exc_val=None, exc_tb=None):
        if self._is_started and self._task is not None:
            try:
                self._progress.stop()
            except Exception:
                pass
            self._task = None
        elif not console.is_terminal:
            console.print("[dim cyan]done[/]", flush=True)

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        self.stop(exc_type, exc_val, exc_tb)

    def update(self, message: str):
        self.message = message
        if self._task is not None and self._progress is not None:
            self._progress.update(self._task, description=f"[dim cyan]{message}[/]")
        elif not console.is_terminal:
            console.print(f"\n[dim cyan]● {message}[/]", end=" ", flush=True)


# ─── Confirmation Prompt ─────────────────────────────────────────────────────


def confirm_tool_execution(tool_name: str, args: dict) -> bool:
    """Ask user to confirm a tool execution."""
    console.print()
    console.print(
        Panel(
            f"[yellow]Tool:[/] [bold]{tool_name}[/]\n"
            f"[yellow]Args:[/]\n{json.dumps(args, indent=2)}",
            title="[yellow]⚠ Confirm Execution[/]",
            border_style="yellow",
        )
    )
    return Confirm.ask("[yellow]Execute this action?[/]", default=False)


# ─── Help Panel ──────────────────────────────────────────────────────────────


def print_help():
    table = Table(
        title="MILEX Commands",
        box=box.ROUNDED,
        border_style="cyan",
        title_style="bold cyan",
        show_header=True,
        header_style="bold white",
    )
    table.add_column("Command", style="bold cyan", min_width=20)
    table.add_column("Description", style="white")

    commands = [
        ("/help", "Show this help message"),
        ("/clear", "Clear the conversation history"),
        ("/model <name>", "Switch Ollama model"),
        ("/models", "List available Ollama models"),
        ("/config", "Show current configuration"),
        ("/set <key> <value>", "Set a configuration value"),
        ("/auto [on|off]", "Toggle auto-execute mode (skip confirmations)"),
        ("/history", "Show conversation history"),
        ("/save <file>", "Save conversation to file"),
        ("/code <lang> <task>", "Generate code for a task"),
        ("/run <command>", "Run a shell command directly"),
        ("/sysinfo", "Show system information"),
        ("/exit", "Exit MILEX (also: quit, bye)"),
    ]

    for cmd, desc in commands:
        table.add_row(cmd, desc)

    console.print()
    console.print(table)
    console.print()
    console.print(
        Panel(
            "[dim]Just type naturally! Ask MILEX to generate code, explain concepts,\n"
            "run commands, manage files, or control your computer.[/]",
            border_style="dim cyan",
            title="[dim cyan]Tips[/]",
        )
    )


# ─── Models Table ────────────────────────────────────────────────────────────


def print_models_table(models: list):
    table = Table(
        title="Available Ollama Models",
        box=box.ROUNDED,
        border_style="cyan",
        title_style="bold cyan",
    )
    table.add_column("Name", style="bold cyan")
    table.add_column("Size", style="white", justify="right")
    table.add_column("Modified", style="dim white")

    for m in models:
        # Ollama SDK returns pydantic model objects; fall back to dict access
        if hasattr(m, "model"):
            name = m.model or "unknown"
            size = getattr(m, "size", 0) or 0
            mod = getattr(m, "modified_at", None)
            modified = str(mod)[:10] if mod else ""
        else:
            name = m.get("name", m.get("model", "unknown"))
            size = m.get("size", 0) or 0
            mod = m.get("modified_at", "")
            modified = str(mod)[:10] if mod else ""
        size_str = f"{size / 1e9:.1f} GB" if size > 1e9 else f"{size / 1e6:.0f} MB"
        table.add_row(name, size_str, modified)

    console.print()
    console.print(table)


# ─── Config Table ────────────────────────────────────────────────────────────


def print_config_table(config: dict):
    table = Table(
        title="Current Configuration",
        box=box.ROUNDED,
        border_style="cyan",
        title_style="bold cyan",
    )
    table.add_column("Key", style="bold cyan")
    table.add_column("Value", style="white")

    for k, v in config.items():
        if k == "system_prompt":
            v = str(v)[:80] + "..." if len(str(v)) > 80 else str(v)
        table.add_row(k, str(v))

    console.print()
    console.print(table)
