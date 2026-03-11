"""Main CLI entry point for MILEX."""
import asyncio
import atexit
import json
import os
import re
import signal
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Load environment variables from .env if it exists
load_dotenv()

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style as PTStyle

# Custom prompt_toolkit style
PT_STYLE = PTStyle.from_dict({
    "": "ansigreen",
    "prompt": "ansigreen bold",
})
from rich.prompt import Confirm

from .agent import MilexAgent
from .config import (
    CONFIG_DIR,
    HISTORY_FILE,
    load_config,
    load_history,
    save_config,
    save_history,
)
from .ui import (
    RichUI,
    console,
    print_banner,
    print_code_block,
    print_config_table,
    print_error,
    print_help,
    print_info,
    print_models_table,
    print_rule,
    print_success,
    print_user_message,
    print_warning,
    print_welcome,
)

app = typer.Typer(
    name="milex",
    help="MILEX - AI-powered CLI using Ollama local models",
    add_completion=True,
    rich_markup_mode="rich",
    invoke_without_command=True,
    no_args_is_help=False,
)

# ─── Daemon / IPC constants ────────────────────────────────────────────────────

CONFIG_DIR.mkdir(parents=True, exist_ok=True)

PID_FILE   = CONFIG_DIR / "milex.pid"
SOCK_FILE  = CONFIG_DIR / "milex.sock"
DAEMON_LOG = CONFIG_DIR / "milex-daemon.log"

# ─── PID helpers ──────────────────────────────────────────────────────────────


def _write_pid(pid: int) -> None:
    PID_FILE.write_text(str(pid))


def _read_pid() -> Optional[int]:
    try:
        return int(PID_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _daemon_running() -> bool:
    pid = _read_pid()
    return pid is not None and _pid_alive(pid) and SOCK_FILE.exists()


# ─── Socket Helpers (Async) ───────────────────────────────────────────────────

DELIM = b"\n"
MAX_MSG = 4 * 1024 * 1024


async def _send_msg(writer: asyncio.StreamWriter, obj: dict) -> None:
    data = json.dumps(obj).encode() + DELIM
    writer.write(data)
    await writer.drain()


async def _recv_msg(reader: asyncio.StreamReader) -> Optional[dict]:
    try:
        line = await reader.readline()
        if not line:
            return None
        return json.loads(line)
    except (asyncio.IncompleteReadError, json.JSONDecodeError):
        return None


class DaemonServer:
    """Unix-socket server that wraps a single MilexAgent for persistent sessions."""

    def __init__(self, agent: MilexAgent) -> None:
        self.agent = agent
        self._stop_event = asyncio.Event()
        self._lock = asyncio.Lock()
        self.token = agent.config.get("daemon_token")

    async def _handle_command(self, writer: asyncio.StreamWriter, msg: dict) -> bool:
        async with self._lock:
            mtype   = msg.get("type", "chat")
            content = msg.get("content", "")

            if mtype == "ping":
                await _send_msg(writer, {"type": "pong"})
                return True

            if mtype == "disconnect":
                return False

            if mtype in ("chat", "stream_chat"):
                # Capture output from agent and forward as chunks
                from .ui import console as ui_console
                
                class _WriterProxy:
                    def write(self_, data): # noqa
                        if data:
                            asyncio.create_task(_send_msg(writer, {"type": "chunk", "content": data}))
                    def flush(self_): pass

                old_file = ui_console.file
                ui_console.file = _WriterProxy() # type: ignore
                try:
                    if mtype == "stream_chat":
                        await self.agent.stream_chat(content)
                    else:
                        await self.agent.chat(content)
                finally:
                    ui_console.file = old_file
                await _send_msg(writer, {"type": "done"})
                return True

            if mtype == "slash":
                import io
                from .ui import console as ui_console
                buf = io.StringIO()
                old_file = ui_console.file
                ui_console.file = buf # type: ignore
                try:
                    keep = await handle_slash_command(content, self.agent)
                finally:
                    ui_console.file = old_file
                await _send_msg(writer, {"type": "slash_result", "content": buf.getvalue(), "keep": keep})
                return True

            if mtype == "config_get":
                await _send_msg(writer, {"type": "config", "content": self.agent.config})
                return True

            if mtype == "stop_server":
                await _send_msg(writer, {"type": "bye"})
                self._stop_event.set()
                return False

            return True

    async def _client_connected(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            # Step 1: Authentication Handshake
            if self.token:
                msg = await _recv_msg(reader)
                if not msg or msg.get("type") != "auth" or msg.get("token") != self.token:
                    await _send_msg(writer, {"type": "error", "content": "Authentication failed"})
                    return
                await _send_msg(writer, {"type": "auth_ok"})

            while not self._stop_event.is_set():
                msg = await _recv_msg(reader)
                if msg is None: break
                if not await self._handle_command(writer, msg):
                    break
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except: pass

    async def serve_forever(self):
        if SOCK_FILE.exists():
            SOCK_FILE.unlink()
            
        server = await asyncio.start_unix_server(self._client_connected, path=str(SOCK_FILE))
        
        _write_pid(os.getpid())
        atexit.register(lambda: PID_FILE.unlink(missing_ok=True))
        atexit.register(lambda: SOCK_FILE.unlink(missing_ok=True))

        # Handle signals for graceful shutdown
        def _on_signal():
            self._stop_event.set()
        
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, _on_signal)
            except NotImplementedError:
                # Signal handlers not supported on Windows (not used here but for robustness)
                pass

        async with server:
            # Task to keep agent warm
            self.agent.start_background_tasks()
            await self._stop_event.wait()
            await self.agent.shutdown()


class DaemonClient:
    """Async Unix socket client."""

    def __init__(self):
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None

    async def connect(self, config: dict) -> bool:
        try:
            self._reader, self._writer = await asyncio.open_unix_connection(str(SOCK_FILE))
            
            # Authentication Handshake
            token = config.get("daemon_token")
            if token:
                await _send_msg(self._writer, {"type": "auth", "token": token})
                resp = await _recv_msg(self._reader)
                if not resp or resp.get("type") != "auth_ok":
                    return False
            
            return True
        except (FileNotFoundError, ConnectionRefusedError, OSError):
            return False

    async def close(self):
        if self._writer:
            try:
                await _send_msg(self._writer, {"type": "disconnect"})
                self._writer.close()
                await self._writer.wait_closed()
            except OSError: pass
            self._writer = None

    async def ping(self) -> bool:
        if not self._writer or not self._reader: return False
        await _send_msg(self._writer, {"type": "ping"})
        resp = await _recv_msg(self._reader)
        return resp.get("type") == "pong" if resp else False

    async def stream_chat(self, text: str):
        if not self._writer or not self._reader: return
        await _send_msg(self._writer, {"type": "stream_chat", "content": text})
        while True:
            msg = await _recv_msg(self._reader)
            if not msg: break
            mtype = msg.get("type")
            if mtype == "chunk":
                console.print(msg["content"], end="", markup=False, highlight=False)
            elif mtype == "done":
                console.print()
                break
            elif mtype == "error":
                print_error(msg.get("content", "Unknown error"))
                break

    async def slash(self, cmd: str) -> bool:
        if not self._writer or not self._reader: return False
        await _send_msg(self._writer, {"type": "slash", "content": cmd})
        msg = await _recv_msg(self._reader)
        if not msg: return False
        if msg.get("content"):
            console.print(msg["content"], end="")
        return bool(msg.get("keep", True))

    async def get_config(self) -> dict:
        if not self._writer or not self._reader: return {}
        await _send_msg(self._writer, {"type": "config_get"})
        msg = await _recv_msg(self._reader)
        return msg.get("content", {}) if msg else {}


# ─── Unix double-fork daemonize ───────────────────────────────────────────────


def daemonize(log_path: Path = DAEMON_LOG) -> None:
    """Fork the current process into the background."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid = os.fork()
    if pid > 0: sys.exit(0)
    os.setsid()
    pid = os.fork()
    if pid > 0: sys.exit(0)
    sys.stdout.flush()
    sys.stderr.flush()
    with open(os.devnull, "r") as devnull:
        os.dup2(devnull.fileno(), sys.stdin.fileno())
    log_fd = open(log_path, "a", buffering=1)
    os.dup2(log_fd.fileno(), sys.stdout.fileno())
    os.dup2(log_fd.fileno(), sys.stderr.fileno())


# ─── Command Registry ───────────────────────────────────────────────────────────

from dataclasses import dataclass
from typing import Callable, Awaitable, Optional


@dataclass
class CommandHandler:
    """Represents a slash command handler."""
    handler: Callable[["MilexAgent", list], Awaitable[bool]]
    min_args: int = 0
    description: str = ""


class CommandRegistry:
    """Registry for slash commands - eliminates large if-elif chains."""

    def __init__(self):
        self._commands: Dict[str, CommandHandler] = {}

    def register(self, name: str, handler: Callable, min_args: int = 0, description: str = ""):
        """Register a command handler."""
        self._commands[name] = CommandHandler(handler, min_args, description)

    def get_handler(self, command: str) -> Optional[CommandHandler]:
        """Get handler for a command."""
        return self._commands.get(command)

    def list_commands(self) -> List[str]:
        """List all registered commands."""
        return list(self._commands.keys())


# Create global registry
command_registry = CommandRegistry()


# ─── Prompt Toolkit Setup ─────────────────────────────────────────────────────

# Commands are now registered via command_registry - generate list from registry
def _get_slash_commands() -> List[str]:
    """Get list of slash commands from registry."""
    return ["/" + cmd.lstrip("/") for cmd in command_registry.list_commands()]


SLASH_COMMANDS = _get_slash_commands()

command_completer = WordCompleter(SLASH_COMMANDS, pattern=re.compile(r"[/\w]+"))


# ─── Command Registry ───────────────────────────────────────────────────────────

from dataclasses import dataclass
from typing import Callable, Awaitable, Optional


@dataclass
class CommandHandler:
    """Represents a slash command handler."""
    handler: Callable[["MilexAgent", list], Awaitable[bool]]
    min_args: int = 0
    description: str = ""


class CommandRegistry:
    """Registry for slash commands - eliminates large if-elif chains."""

    def __init__(self):
        self._commands: Dict[str, CommandHandler] = {}

    def register(self, name: str, handler: Callable, min_args: int = 0, description: str = ""):
        """Register a command handler."""
        self._commands[name] = CommandHandler(handler, min_args, description)

    def get_handler(self, command: str) -> Optional[CommandHandler]:
        """Get handler for a command."""
        return self._commands.get(command)

    def list_commands(self) -> List[str]:
        """List all registered commands."""
        return list(self._commands.keys())


# Create global registry
command_registry = CommandRegistry()


def get_toolbar_text(cfg: dict) -> str:
    model = cfg.get("model", "?")
    auto  = "AUTO" if cfg.get("auto_execute") else "CONFIRM"
    return (
        f' <b>MILEX</b> │ model: <ansicyan>{model}</ansicyan> │ '
        f'exec: <ansigreen>{auto}</ansigreen> │ '
        f'<ansibrightblack>/help for commands</ansibrightblack>'
    )


def get_toolbar(agent: MilexAgent):
    """Return bottom toolbar text (agent-local mode)."""
    return HTML(get_toolbar_text(agent.config))


# ─── Command Handlers ─────────────────────────────────────────────────────────


async def _cmd_exit(cmd: str, agent: MilexAgent) -> bool:
    """Handle /exit and /quit commands."""
    return False


async def _cmd_help(cmd: str, agent: MilexAgent) -> bool:
    """Handle /help command."""
    print_help()
    return True


async def _cmd_clear(cmd: str, agent: MilexAgent) -> bool:
    """Handle /clear command."""
    agent.clear_conversation()
    return True


async def _cmd_models(cmd: str, agent: MilexAgent) -> bool:
    """Handle /models command."""
    models = await agent.get_available_models()
    if models:
        print_models_table(models)
    else:
        print_warning("No models found. Make sure Ollama is running.")
    return True


async def _cmd_model(cmd: str, agent: MilexAgent) -> bool:
    """Handle /model command."""
    parts = cmd.strip().split(maxsplit=1)
    if len(parts) < 2:
        print_info(f"Current model: [bold cyan]{agent.config['model']}[/]")
    else:
        agent.switch_model(parts[1])
    return True


async def _cmd_config(cmd: str, agent: MilexAgent) -> bool:
    """Handle /config command."""
    print_config_table(agent.config)
    return True


async def _cmd_set(cmd: str, agent: MilexAgent) -> bool:
    """Handle /set command."""
    parts = cmd.strip().split(maxsplit=2)
    if len(parts) < 3:
        print_error("Usage: /set <key> <value>")
        return True
    key, value = parts[1], parts[2]
    if value.lower() in ("true", "yes"):
        typed = True
    elif value.lower() in ("false", "no"):
        typed = False
    else:
        try:
            typed = int(value)
        except ValueError:
            try:
                typed = float(value)
            except ValueError:
                typed = value
    agent.config[key] = typed
    save_config(agent.config)
    print_success(f"Set [cyan]{key}[/] = [white]{typed}[/]")
    return True


async def _cmd_auto(cmd: str, agent: MilexAgent) -> bool:
    """Handle /auto command."""
    parts = cmd.strip().split(maxsplit=1)
    val = parts[1].lower() in ("on", "true", "1", "yes") if len(parts) >= 2 else not agent.config.get("auto_execute")
    agent.config["auto_execute"] = val
    agent.executor.auto_execute = val
    save_config(agent.config)
    print_info(f"Auto-execute: {'[green]ON[/]' if val else '[red]OFF[/]'}")
    return True


async def _cmd_history(cmd: str, agent: MilexAgent) -> bool:
    """Handle /history command."""
    history = load_history()
    if not history:
        print_info("No history found.")
    else:
        from rich.table import Table
        table = Table(title="Recent History")
        table.add_column("#", width=4)
        table.add_column("Input")
        for i, entry in enumerate(history[-15:], 1):
            table.add_row(str(i), entry.get("input", "")[:80])
        console.print(table)
    return True


async def _cmd_save(cmd: str, agent: MilexAgent) -> bool:
    """Handle /save command."""
    parts = cmd.strip().split(maxsplit=1)
    if len(parts) < 2:
        print_error("Usage: /save <filename>")
        return True
    try:
        Path(parts[1]).write_text(json.dumps({"conversation": agent.conversation}, indent=2))
        print_success(f"Saved to {parts[1]}")
    except Exception as e:
        print_error(f"Save failed: {e}")
    return True


async def _cmd_code(cmd: str, agent: MilexAgent) -> bool:
    """Handle /code command."""
    parts = cmd.strip().split(maxsplit=2)
    if len(parts) < 3:
        print_error("Usage: /code <language> <task description>")
        return True
    lang, rest = parts[1], parts[2]
    filename = None
    save_match = re.search(r"--save\s+(\S+)", rest)
    if save_match:
        filename = save_match.group(1)
        rest = rest.replace(save_match.group(0), "").strip()
    await agent.generate_code_internal(rest, lang, filename)
    return True


async def _cmd_run(cmd: str, agent: MilexAgent) -> bool:
    """Handle /run command."""
    parts = cmd.strip().split(maxsplit=1)
    if len(parts) < 2:
        print_error("Usage: /run <shell command>")
        return True
    res = await agent.executor.execute_async("run_shell", {"command": cmd[5:].strip()})
    if res.get("stdout"):
        console.print(res["stdout"])
    if res.get("stderr"):
        console.print(f"[red]{res['stderr']}[/]")
    return True


async def _cmd_research(cmd: str, agent: MilexAgent) -> bool:
    """Handle /research command."""
    parts = cmd.strip().split(maxsplit=1)
    if len(parts) < 2:
        print_error("Usage: /research <topic>")
        return True
        
    topic = parts[1].strip()
    safe_topic = re.sub(r'[^a-zA-Z0-9_\-]', '_', topic).strip('_')
    if not safe_topic:
        safe_topic = "topic"
    filename = f"{safe_topic}_research.txt"
    
    print_info(f"Starting comprehensive research on: [bold]{topic}[/]. Report will be saved to [cyan]{filename}[/]")
    
    prompt = (
        f"Execute a comprehensive research task on the following topic: '{topic}'.\n"
        "Step 1: Use the `search_web` tool to gather as much detailed information as possible about this topic. Use it multiple times if you need to gather broad context.\n"
        "Step 2: Consolidate the research into a highly detailed, well-organized text report.\n"
        f"Step 3: Crucially, you MUST use the `write_file` tool to save the entire report directly into the file '{filename}'. Do not ask for permission, just use the tool.\n"
        "Step 4: Once saved automatically, give me a brief conversational summary of what you learned."
    )
    # Forward the constructed prompt into the existing agent streaming flow
    await agent.stream_chat(prompt)
    return True


async def _cmd_sysinfo(cmd: str, agent: MilexAgent) -> bool:
    """Handle /sysinfo command."""
    res = await agent.executor.execute_async("get_system_info", {})
    from rich.table import Table
    table = Table(title="System Info")
    for k, v in res.items():
        table.add_row(k, str(v))
    console.print(table)
    return True


async def _cmd_sandbox(cmd: str, agent: MilexAgent) -> bool:
    """Handle /sandbox command."""
    parts = cmd.strip().split(maxsplit=1)
    if len(parts) < 2:
        root = agent.config.get("allowed_root")
        print_info(f"Sandbox Root: [bold green]{root or 'DISABLED'}[/]")
    else:
        val = parts[1]
        if val.lower() in ("off", "none", "disable"):
            agent.config["allowed_root"] = None
        else:
            p = Path(val).resolve()
            if p.exists():
                agent.config["allowed_root"] = str(p)
            else:
                print_error(f"Path invalid: {p}")
    return True


async def _cmd_telemetry(cmd: str, agent: MilexAgent) -> bool:
    """Handle /telemetry command."""
    from .telemetry import telemetry
    from rich.table import Table
    stats = telemetry.get_stats()
    if not stats:
        print_info("No telemetry data yet.")
        return True
    
    table = Table(title="Tool Telemetry")
    table.add_column("Tool")
    table.add_column("Hits", justify="right")
    table.add_column("Errors", justify="right", style="red")
    table.add_column("Avg Latency", justify="right", style="cyan")
    table.add_column("Max Latency", justify="right", style="magenta")
    
    for name, s in stats.items():
        table.add_row(
            name,
            str(s["count"]),
            str(s["errors"]),
            f"{s['avg_ms']}ms",
            f"{s['max_ms']}ms"
        )
    console.print(table)
    return True


# Register all commands
command_registry.register("/exit", _cmd_exit, 0, "Exit MILEX")
command_registry.register("/quit", _cmd_exit, 0, "Exit MILEX (alias)")
command_registry.register("/help", _cmd_help, 0, "Show help message")
command_registry.register("/clear", _cmd_clear, 0, "Clear conversation history")
command_registry.register("/models", _cmd_models, 0, "List available Ollama models")
command_registry.register("/model", _cmd_model, 0, "Switch Ollama model")
command_registry.register("/config", _cmd_config, 0, "Show current configuration")
command_registry.register("/set", _cmd_set, 2, "Set configuration value")
command_registry.register("/auto", _cmd_auto, 0, "Toggle auto-execute mode")
command_registry.register("/history", _cmd_history, 0, "Show conversation history")
command_registry.register("/save", _cmd_save, 1, "Save conversation to file")
command_registry.register("/code", _cmd_code, 2, "Generate code for a task")
command_registry.register("/research", _cmd_research, 1, "Collect info on a topic and save to a txt file")
command_registry.register("/run", _cmd_run, 1, "Run a shell command")
command_registry.register("/sysinfo", _cmd_sysinfo, 0, "Show system information")
command_registry.register("/sandbox", _cmd_sandbox, 0, "Set sandbox root directory")
command_registry.register("/telemetry", _cmd_telemetry, 0, "Show tool performance stats")


async def handle_slash_command(cmd: str, agent: MilexAgent) -> bool:
    """Handle slash commands using command registry."""
    parts = cmd.strip().split(maxsplit=1)
    if not parts:
        return True
    command = parts[0].lower()

    handler = command_registry.get_handler(command)
    if handler:
        return await handler.handler(cmd, agent)

    print_error(f"Unknown command: {command}")
    return True


# ─── REPL Loops ───────────────────────────────────────────────────────────────


async def run_interactive_daemon(client: DaemonClient, cfg: dict):
    """REPL that forwards to daemon."""
    if sys.stdin.isatty():
        session = PromptSession(
            history=FileHistory(str(CONFIG_DIR / "prompt_history")),
            auto_suggest=AutoSuggestFromHistory(),
            completer=command_completer,
            style=PT_STYLE,
        )
    else:
        session = None

    history_log = load_history()

    while True:
        try:
            if session:
                user_input = await session.prompt_async(
                    HTML("\n<prompt>❯</prompt> <ansicyan>You</ansicyan> › "),
                    bottom_toolbar=lambda: HTML(get_toolbar_text(cfg)),
                )
            else:
                user_input = await asyncio.to_thread(input, "\n❯ You › ")
        except (KeyboardInterrupt, EOFError):
            break

        user_input = user_input.strip()
        if not user_input: continue
        if user_input.lower() in ("exit", "quit", "bye"): break

        if user_input.startswith("/"):
            if not await client.slash(user_input): break
            cfg.update(await client.get_config())
            continue

        history_log.append({"input": user_input})
        save_history(history_log)

        print_user_message(user_input)
        await client.stream_chat(user_input)

    await client.close()
    console.print("\n[dim cyan]Goodbye! MILEX session ended.[/]\n")


async def run_interactive(agent: MilexAgent):
    """REPL for standalone mode."""
    if sys.stdin.isatty():
        session = PromptSession(
            history=FileHistory(str(CONFIG_DIR / "prompt_history")),
            auto_suggest=AutoSuggestFromHistory(),
            completer=command_completer,
            style=PT_STYLE,
        )
    else:
        session = None

    agent.start_background_tasks()
    history_log = load_history()

    while True:
        try:
            if session:
                user_input = await session.prompt_async(
                    HTML("\n<prompt>❯</prompt> <ansicyan>You</ansicyan> › "),
                    bottom_toolbar=lambda: HTML(get_toolbar_text(agent.config)),
                )
            else:
                user_input = await asyncio.to_thread(input, "\n❯ You › ")
        except (KeyboardInterrupt, EOFError):
            break

        user_input = user_input.strip()
        if not user_input: continue
        if user_input.lower() in ("exit", "quit", "bye"): break

        if user_input.startswith("/"):
            if not await handle_slash_command(user_input, agent): break
            continue

        history_log.append({"input": user_input})
        save_history(history_log)

        print_user_message(user_input)
        await agent.stream_chat(user_input)

    await agent.shutdown()
    console.print("\n[dim cyan]Goodbye! MILEX session ended.[/]\n")


# ─── CLI Commands ──────────────────────────────────────────────────────────────


@app.callback(invoke_without_command=True)
def main_entry(
    ctx: typer.Context,
    prompt: Optional[str] = typer.Argument(None),
    model: Optional[str]  = typer.Option(None, "--model", "-m"),
    auto:  bool           = typer.Option(False, "--auto", "-a"),
    background: bool      = typer.Option(False, "--background", "-b"),
    no_daemon: bool       = typer.Option(False, "--no-daemon", "-n"),
):
    """MILEX - AI-powered CLI using Ollama."""
    if ctx.invoked_subcommand: return
    asyncio.run(_async_main(prompt, model, auto, background, no_daemon))


async def _async_main(prompt, model, auto, background, no_daemon):
    if background:
        if _daemon_running():
            print_warning("Daemon already running.")
            return
        print_info(f"Starting daemon… log → {DAEMON_LOG}")
        daemonize(DAEMON_LOG)
        
        cfg = load_config()
        try:
            ui = RichUI()
            agent = MilexAgent(config=cfg, ui=ui)
        except Exception as e:
            with open(str(DAEMON_LOG), "a") as lf:
                lf.write(f"[daemon] Failed to init agent: {e}\n")
            sys.exit(1)
            
        server = DaemonServer(agent)
        await server.serve_forever()
        return

    cfg = load_config()
    if model: cfg["model"] = model
    if auto: cfg["auto_execute"] = True

    # Try connection
    if not no_daemon and _daemon_running():
        client = DaemonClient()
        if await client.connect(cfg) and await client.ping():
            if prompt:
                print_user_message(prompt)
                await client.stream_chat(prompt)
                await client.close()
            else:
                daemon_cfg = await client.get_config()
                cfg.update(daemon_cfg)
                print_banner()
                print_welcome(cfg.get("model", "?"), cfg.get("ollama_host", "?"))
                print_info("[dim]Connected to background MILEX daemon[/] ✓")
                await run_interactive_daemon(client, cfg)
            return

    # Fallback to standalone
    print_banner()
    try:
        ui = RichUI()
        agent = MilexAgent(config=cfg, ui=ui)
    except Exception as e:
        print_error(f"Failed to initialize agent: {e}")
        raise typer.Exit(1)
        
    if prompt:
        print_user_message(prompt)
        await agent.stream_chat(prompt)
        await agent.shutdown()
    else:
        print_welcome(cfg.get("model", "?"), cfg.get("ollama_host", "?"))
        await run_interactive(agent)


@app.command(name="start")
def start_daemon_cmd(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Ollama model for the daemon"),
    auto:  bool          = typer.Option(False, "--auto", "-a", help="Auto-execute tools"),
):
    """Start MILEX as a persistent background daemon."""
    asyncio.run(_async_main(None, model, auto, True, False))


@app.command(name="stop")
def stop_daemon_cmd():
    """Stop the background MILEX daemon."""
    async def _do_stop():
        cfg = load_config()
        client = DaemonClient()
        if await client.connect(cfg):
            if client._writer:
                await _send_msg(client._writer, {"type": "stop_server"})
                print_success("Sent stop command to daemon.")
                await client.close()
        elif PID_FILE.exists():
            pid = _read_pid()
            if pid:
                try:
                    os.kill(pid, signal.SIGTERM)
                    print_success(f"Killed PID {pid}")
                except Exception:
                    pass
            PID_FILE.unlink(missing_ok=True)
            SOCK_FILE.unlink(missing_ok=True)
        else:
            print_error("Daemon not running.")
    asyncio.run(_do_stop())


@app.command(name="status")
def daemon_status_cmd():
    """Show whether the background MILEX daemon is running."""
    pid = _read_pid()
    if pid is None or not _pid_alive(pid):
        print_info("MILEX daemon is [bold red]not running[/].")
    else:
        print_success(
            f"MILEX daemon is [bold green]running[/] (PID {pid}).\n"
            f"  Socket:   [dim]{SOCK_FILE}[/]\n"
            f"  Log file: [dim]{DAEMON_LOG}[/]"
        )


@app.command(name="models")
def list_models_cmd():
    """List available Ollama models."""
    async def _impl():
        cfg = load_config()
        try:
            ui = RichUI()
            agent = MilexAgent(config=cfg, ui=ui)
        except Exception as e:
            print_error(f"Failed to initialize agent: {e}")
            raise typer.Exit(1)
        models = await agent.get_available_models()
        if models:
            print_models_table(models)
        else:
            print_warning("No models found or Ollama is not running.")
    asyncio.run(_impl())


@app.command(name="config")
def show_config_cmd():
    """Show current MILEX configuration."""
    cfg = load_config()
    print_config_table(cfg)


@app.command(name="set")
def set_config_cmd(
    key:   str = typer.Argument(..., help="Config key"),
    value: str = typer.Argument(..., help="Config value"),
):
    """Set a configuration value."""
    cfg = load_config()
    if value.lower() in ("true", "yes"):
        typed: object = True
    elif value.lower() in ("false", "no"):
        typed = False
    else:
        try:
            typed = int(value)
        except ValueError:
            try:
                typed = float(value)
            except ValueError:
                typed = value
    cfg[key] = typed
    save_config(cfg)
    print_success(f"Config updated: [cyan]{key}[/] = [white]{typed}[/]")


@app.command(name="mcp-server")
def mcp_server_cmd():
    """Run MILEX as an MCP Server (stdio)."""
    from .mcp_server import main as mcp_main
    asyncio.run(mcp_main())


@app.command(name="telemetry")
def telemetry_cmd():
    """Show tool performance stats."""
    from .telemetry import telemetry
    from rich.table import Table
    stats = telemetry.get_stats()
    if not stats:
        print_info("No telemetry data yet.")
        return
    
    table = Table(title="Tool Telemetry")
    table.add_column("Tool")
    table.add_column("Hits", justify="right")
    table.add_column("Errors", justify="right", style="red")
    table.add_column("Avg Latency", justify="right", style="cyan")
    
    for name, s in stats.items():
        table.add_row(name, str(s["count"]), str(s["errors"]), f"{s['avg_ms']}ms")
    console.print(table)


if __name__ == "__main__":
    app()
