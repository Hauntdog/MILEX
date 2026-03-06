"""Main CLI entry point for MILEX."""
import atexit
import json
import os
import re
import select
import signal
import socket
import sys
import threading
from pathlib import Path
from typing import Optional

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style as PTStyle
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


# ─── Socket server (daemon side) ──────────────────────────────────────────────

DELIM = b"\n"          # message delimiter
MAX_MSG = 4 * 1024 * 1024  # 4 MB


def _send_msg(sock: socket.socket, obj: dict) -> None:
    data = json.dumps(obj).encode() + DELIM
    sock.sendall(data)


def _recv_line(sock: socket.socket, buffer: bytearray) -> Optional[bytes]:
    """Read one newline-terminated line from *sock* using a persistent buffer."""
    while DELIM not in buffer:
        try:
            chunk = sock.recv(4096)
            if not chunk:
                return None
            buffer.extend(chunk)
        except (ConnectionResetError, OSError):
            return None
    
    pos = buffer.find(DELIM)
    line = bytes(buffer[:pos])
    del buffer[:pos + 1]
    return line


class DaemonServer:
    """Unix-socket server that wraps a single MilexAgent for persistent sessions."""

    def __init__(self, agent: MilexAgent) -> None:
        self.agent = agent
        self._stop = threading.Event()

    # ── streaming helper ──────────────────────────────────────────────────────

    def _stream_to_socket(self, conn: socket.socket, user_input: str) -> None:
        """Run agent.stream_chat and forward chunks over *conn*."""
        import io, contextlib
        from .ui import console as ui_console

        # Capture streaming output chunk by chunk via a custom write proxy
        collected = []

        class _Proxy:
            def write(self_, data):   # noqa: N805
                if data:
                    collected.append(data)
                    try:
                        _send_msg(conn, {"type": "chunk", "content": data})
                    except OSError:
                        pass
            def flush(self_):         # noqa: N805
                pass

        old_file = ui_console.file
        ui_console.file = _Proxy()  # type: ignore[assignment]
        try:
            self.agent.stream_chat(user_input)
        finally:
            ui_console.file = old_file
        _send_msg(conn, {"type": "done", "content": ""})

    def _chat_to_socket(self, conn: socket.socket, user_input: str) -> None:
        from .ui import console as ui_console
        import io

        buf = io.StringIO()
        old_file = ui_console.file
        ui_console.file = buf  # type: ignore[assignment]
        try:
            self.agent.chat(user_input)
        finally:
            ui_console.file = old_file
        _send_msg(conn, {"type": "done", "content": buf.getvalue()})

    # ── command dispatcher ────────────────────────────────────────────────────

    def _handle_command(self, conn: socket.socket, msg: dict) -> bool:
        """
        Process one message from the client.
        Returns False if the client requested disconnect.
        """
        mtype   = msg.get("type", "chat")
        content = msg.get("content", "")

        if mtype == "ping":
            _send_msg(conn, {"type": "pong"})
            return True

        if mtype == "disconnect":
            return False

        if mtype in ("chat", "stream_chat"):
            stream = self.agent.config.get("stream", True) or mtype == "stream_chat"
            if stream:
                self._stream_to_socket(conn, content)
            else:
                self._chat_to_socket(conn, content)
            return True

        if mtype == "slash":
            # Run slash command, capture output
            from .ui import console as ui_console
            import io
            buf = io.StringIO()
            old_file = ui_console.file
            ui_console.file = buf  # type: ignore[assignment]
            try:
                _keep = handle_slash_command(content, self.agent)
            finally:
                ui_console.file = old_file
            _send_msg(conn, {"type": "slash_result",
                             "content": buf.getvalue(),
                             "keep": _keep})
            return True

        if mtype == "config_get":
            _send_msg(conn, {"type": "config", "content": self.agent.config})
            return True

        if mtype == "stop_server":
            _send_msg(conn, {"type": "bye"})
            self._stop.set()
            return False

        _send_msg(conn, {"type": "error", "content": f"Unknown message type: {mtype}"})
        return True

    # ── connection worker ─────────────────────────────────────────────────────

    def _handle_conn(self, conn: socket.socket, addr) -> None:
        buffer = bytearray()
        try:
            while not self._stop.is_set():
                raw = _recv_line(conn, buffer)
                if raw is None:
                    break
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    _send_msg(conn, {"type": "error", "content": "Bad JSON"})
                    continue
                if not self._handle_command(conn, msg):
                    break
        finally:
            try:
                conn.close()
            except OSError:
                pass

    # ── main accept loop ──────────────────────────────────────────────────────

    def serve_forever(self) -> None:
        if SOCK_FILE.exists():
            SOCK_FILE.unlink()

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(SOCK_FILE))
        srv.listen(8)
        srv.setblocking(False)

        _write_pid(os.getpid())
        atexit.register(lambda: PID_FILE.unlink(missing_ok=True))
        atexit.register(lambda: SOCK_FILE.unlink(missing_ok=True))

        while not self._stop.is_set():
            ready, _, _ = select.select([srv], [], [], 0.5)
            if not ready:
                continue
            try:
                conn, addr = srv.accept()
                conn.setblocking(True)
            except OSError:
                break
            t = threading.Thread(target=self._handle_conn, args=(conn, addr),
                                 daemon=True)
            t.start()

        srv.close()


# ─── Socket client (foreground side) ─────────────────────────────────────────


class DaemonClient:
    """Thin wrapper around the Unix socket to talk to the daemon."""

    def __init__(self) -> None:
        self._sock: Optional[socket.socket] = None

    def connect(self) -> bool:
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(str(SOCK_FILE))
            self._sock = s
            return True
        except (FileNotFoundError, ConnectionRefusedError, OSError):
            return False

    def close(self) -> None:
        if self._sock:
            try:
                _send_msg(self._sock, {"type": "disconnect"})
            except OSError:
                pass
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _sock_checked(self) -> socket.socket:
        if not self._sock:
            raise RuntimeError("Not connected to daemon")
        return self._sock

    def ping(self) -> bool:
        try:
            _send_msg(self._sock_checked(), {"type": "ping"})
            resp = _recv_line(self._sock_checked())
            if resp is None:
                return False
            return json.loads(resp).get("type") == "pong"
        except (OSError, json.JSONDecodeError):
            return False

    def stream_chat(self, text: str) -> None:
        """Send a chat message and print streamed reply to the terminal."""
        _send_msg(self._sock_checked(), {"type": "stream_chat", "content": text})
        buffer = bytearray()
        while True:
            raw = _recv_line(self._sock_checked(), buffer)
            if raw is None:
                break
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                break
            mtype = msg.get("type")
            content = msg.get("content", "")
            if mtype == "chunk":
                console.print(content, end="", markup=False, highlight=False)
            elif mtype == "done":
                if content:
                    console.print(content, end="")
                console.print()  # newline after full reply
                break
            elif mtype == "error":
                print_error(content)
                break

    def slash(self, cmd: str) -> bool:
        """Run a slash command on the daemon, return keep-running flag."""
        _send_msg(self._sock_checked(), {"type": "slash", "content": cmd})
        buffer = bytearray()
        raw = _recv_line(self._sock_checked(), buffer)
        if raw is None:
            return False
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return True
        if msg.get("content"):
            console.print(msg["content"], end="")
        return bool(msg.get("keep", True))

    def get_config(self) -> dict:
        _send_msg(self._sock_checked(), {"type": "config_get"})
        buffer = bytearray()
        raw = _recv_line(self._sock_checked(), buffer)
        if raw is None:
            return {}
        try:
            msg = json.loads(raw)
            return msg.get("content", {})
        except json.JSONDecodeError:
            return {}


# ─── Unix double-fork daemonize ───────────────────────────────────────────────


def daemonize(log_path: Path = DAEMON_LOG) -> None:
    """Fork the current process into the background (Unix double-fork)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    pid = os.fork()
    if pid > 0:
        sys.exit(0)   # parent exits → shell gets prompt back

    os.setsid()

    pid = os.fork()
    if pid > 0:
        sys.exit(0)

    sys.stdout.flush()
    sys.stderr.flush()
    with open(os.devnull, "r") as devnull:
        os.dup2(devnull.fileno(), sys.stdin.fileno())
    log_fd = open(log_path, "a", buffering=1)
    os.dup2(log_fd.fileno(), sys.stdout.fileno())
    os.dup2(log_fd.fileno(), sys.stderr.fileno())


# ─── Prompt Toolkit Setup ─────────────────────────────────────────────────────

PT_STYLE = PTStyle.from_dict(
    {
        "prompt": "#00ffff bold",
        "bottom-toolbar": "bg:#1a1a2e #888888",
    }
)

SLASH_COMMANDS = [
    "/help",
    "/clear",
    "/model",
    "/models",
    "/config",
    "/set",
    "/auto",
    "/history",
    "/save",
    "/code",
    "/run",
    "/sysinfo",
    "/exit",
]

command_completer = WordCompleter(SLASH_COMMANDS, pattern=re.compile(r"[/\w]+"))


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


def handle_slash_command(cmd: str, agent: MilexAgent) -> bool:
    """
    Handle slash commands.
    Returns True if the main loop should continue, False to exit.
    """
    parts = cmd.strip().split(maxsplit=2)
    command = parts[0].lower()

    if command in ("/exit", "/quit"):
        return False

    elif command == "/help":
        print_help()

    elif command == "/clear":
        agent.clear_conversation()

    elif command == "/models":
        models = agent.get_available_models()
        if models:
            print_models_table(models)
        else:
            print_warning("No models found. Make sure Ollama is running.")

    elif command == "/model":
        if len(parts) < 2:
            print_info(f"Current model: [bold cyan]{agent.config['model']}[/]")
        else:
            agent.switch_model(parts[1])

    elif command == "/config":
        print_config_table(agent.config)

    elif command == "/set":
        if len(parts) < 3:
            print_error("Usage: /set <key> <value>")
        else:
            key, value = parts[1], parts[2]
            if "#" in value:
                value = value.split("#")[0]
            value = value.split("\n")[0].strip()

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

    elif command == "/auto":
        if len(parts) < 2:
            current = agent.config.get("auto_execute", False)
            val = not current
        else:
            val = parts[1].lower() in ("on", "true", "1", "yes")
        agent.config["auto_execute"] = val
        agent.executor.auto_execute = val
        save_config(agent.config)
        state = "[green]ON[/]" if val else "[red]OFF[/]"
        print_info(f"Auto-execute: {state}")

    elif command == "/history":
        history = load_history()
        if not history:
            print_info("No history found.")
        else:
            from rich.table import Table
            from rich import box as rbox
            table = Table(box=rbox.ROUNDED, border_style="dim cyan", title="Session History")
            table.add_column("#",     style="dim",   width=4)
            table.add_column("Input", style="white", max_width=80)
            for i, entry in enumerate(history[-20:], 1):
                table.add_row(str(i), entry.get("input", "")[:80])
            console.print(table)

    elif command == "/save":
        if len(parts) < 2:
            print_error("Usage: /save <filename>")
        else:
            fname = parts[1]
            data  = {"conversation": agent.conversation}
            try:
                Path(fname).write_text(json.dumps(data, indent=2))
                print_success(f"Conversation saved to [bold]{fname}[/]")
            except OSError as exc:
                print_error(f"Could not save file: {exc}")

    elif command == "/code":
        if len(parts) < 3:
            print_error("Usage: /code <language> <task description>")
        else:
            lang = parts[1]
            rest = parts[2]
            filename = None
            save_match = re.search(r"--save\s+(\S+)", rest)
            if save_match:
                filename = save_match.group(1)
                rest = rest.replace(save_match.group(0), "").strip()
            # New internal call
            agent.generate_code_internal(rest, lang, filename)

    elif command == "/run":
        if len(parts) < 2:
            print_error("Usage: /run <shell command>")
        else:
            shell_cmd = cmd[len("/run"):].strip()
            result = agent.executor.execute("run_shell", {"command": shell_cmd})
            if result.get("stdout"):
                console.print(result["stdout"])
            if result.get("stderr"):
                console.print(f"[red]{result['stderr']}[/]")
            if result.get("returncode", 0) != 0:
                print_warning(f"Exit code: {result['returncode']}")

    elif command == "/sysinfo":
        result = agent.executor.execute("get_system_info", {})
        from rich.table import Table
        from rich import box as rbox
        table = Table(box=rbox.ROUNDED, border_style="cyan", title="System Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value",    style="white")
        for k, v in result.items():
            table.add_row(k, str(v))
        console.print(table)

    else:
        print_error(f"Unknown command: {command}. Type /help for available commands.")

    return True


# ─── Interactive loop (connected to daemon) ───────────────────────────────────


def run_interactive_daemon(client: DaemonClient, cfg: dict):
    """REPL that forwards every message to the background daemon."""
    use_pt = sys.stdin.isatty()

    if use_pt:
        session = PromptSession(
            history=FileHistory(str(CONFIG_DIR / "prompt_history")),
            auto_suggest=AutoSuggestFromHistory(),
            completer=command_completer,
            style=PT_STYLE,
            complete_while_typing=True,
        )
    else:
        session = None

    history_log = load_history()

    while True:
        try:
            if session is not None:
                user_input = session.prompt(
                    HTML("\n<ansigreen>❯</ansigreen> <ansicyan>You</ansicyan> › "),
                    bottom_toolbar=lambda: HTML(get_toolbar_text(cfg)),
                    style=PT_STYLE,
                    multiline=False,
                )
            else:
                user_input = input("\n❯ You › ")
        except KeyboardInterrupt:
            print_info("Use /exit or type 'exit' to quit.")
            continue
        except EOFError:
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "bye", "q"):
            break

        if user_input.startswith("/"):
            keep = client.slash(user_input)
            if not keep:
                break
            # Refresh config (e.g. after /model switch)
            cfg.update(client.get_config())
            continue

        history_log.append({"input": user_input})
        save_history(history_log)

        print_user_message(user_input)
        client.stream_chat(user_input)

    client.close()
    console.print("\n[dim cyan]Goodbye! MILEX session ended.[/]\n")


# ─── Interactive loop (standalone / no daemon) ────────────────────────────────


def run_interactive(agent: MilexAgent):
    """Run the interactive REPL locally (no daemon)."""
    use_prompt_toolkit = sys.stdin.isatty()

    if use_prompt_toolkit:
        session = PromptSession(
            history=FileHistory(str(CONFIG_DIR / "prompt_history")),
            auto_suggest=AutoSuggestFromHistory(),
            completer=command_completer,
            style=PT_STYLE,
            complete_while_typing=True,
        )
    else:
        session = None

    history_log  = load_history()
    stream_mode  = agent.config.get("stream", True)

    while True:
        try:
            if session is not None:
                user_input = session.prompt(
                    HTML("\n<ansigreen>❯</ansigreen> <ansicyan>You</ansicyan> › "),
                    bottom_toolbar=lambda: get_toolbar(agent),
                    style=PT_STYLE,
                    multiline=False,
                )
            else:
                user_input = input("\n❯ You › ")
        except KeyboardInterrupt:
            print_info("Use /exit or type 'exit' to quit.")
            continue
        except EOFError:
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "bye", "q"):
            break

        if user_input.startswith("/"):
            if not handle_slash_command(user_input, agent):
                break
            continue

        history_log.append({"input": user_input})
        save_history(history_log)

        print_user_message(user_input)

        if stream_mode:
            agent.stream_chat(user_input)
        else:
            agent.chat(user_input)

    console.print("\n[dim cyan]Goodbye! MILEX session ended.[/]\n")


# ─── CLI Commands ──────────────────────────────────────────────────────────────


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    prompt: Optional[str] = typer.Argument(None, help="One-shot prompt (non-interactive)"),
    model: Optional[str]  = typer.Option(None,  "--model",     "-m", help="Ollama model to use"),
    auto:  bool           = typer.Option(False, "--auto",      "-a", help="Auto-execute tools without confirmation"),
    no_stream: bool       = typer.Option(False, "--no-stream",       help="Disable streaming output"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file"),
    background: bool      = typer.Option(
        False, "--background", "-b",
        help="Start MILEX as a persistent background daemon",
    ),
    no_daemon: bool       = typer.Option(
        False, "--no-daemon", "-n",
        help="Run standalone even if a daemon is already running",
    ),
):
    """
    [bold cyan]MILEX[/] - AI-powered CLI using Ollama local models.

    Generates code, controls your computer, and answers anything.

    When a daemon is running ([bold]milex start[/]), each invocation
    of [bold]milex[/] connects to it automatically — preserving full
    conversation history across terminal sessions.
    """
    if ctx.invoked_subcommand is not None:
        return

    # ── Background / daemon start mode ───────────────────────────────────────
    if background:
        if _daemon_running():
            print_warning(f"MILEX daemon already running (PID {_read_pid()}).")
            raise typer.Exit(0)

        print_info(f"Starting MILEX daemon in background… log → [bold]{DAEMON_LOG}[/]")
        daemonize(DAEMON_LOG)
        # ↓ Only the daemon child reaches this point

        cfg = load_config()
        try:
            ui = RichUI()
            agent = MilexAgent(config=cfg, ui=ui)
        except Exception as e:
            with open(str(DAEMON_LOG), "a") as lf:
                lf.write(f"[daemon] Failed to init agent: {e}\n")
            sys.exit(1)

        server = DaemonServer(agent)
        server.serve_forever()
        sys.exit(0)

    # ── Load config ───────────────────────────────────────────────────────────
    cfg = load_config()
    if model:
        cfg["model"] = model
    if auto:
        cfg["auto_execute"] = True
    if no_stream:
        cfg["stream"] = False

    # ── Try to connect to a running daemon ────────────────────────────────────
    if not no_daemon and not prompt:
        client = DaemonClient()
        if client.connect() and client.ping():
            # Sync any CLI overrides to daemon config
            daemon_cfg = client.get_config()
            cfg.update(daemon_cfg)
            print_banner()
            print_welcome(cfg.get("model", "?"), cfg.get("ollama_host", "?"))
            print_info("[dim]Connected to background MILEX daemon[/] ✓")
            run_interactive_daemon(client, cfg)
            return

    # ── Standalone mode ───────────────────────────────────────────────────────
    print_banner()

    try:
        ui = RichUI()
        agent = MilexAgent(config=cfg, ui=ui)
    except Exception as e:
        print_error(f"Failed to initialize agent: {e}")
        raise typer.Exit(1)

    if prompt:
        print_user_message(prompt)
        if cfg.get("stream", True):
            agent.stream_chat(prompt)
        else:
            agent.chat(prompt)
    else:
        print_welcome(cfg["model"], cfg["ollama_host"])
        run_interactive(agent)


# ─── `milex start` ────────────────────────────────────────────────────────────


@app.command(name="start")
def start_daemon(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Ollama model for the daemon"),
    auto:  bool          = typer.Option(False, "--auto", "-a", help="Auto-execute tools"),
):
    """Start MILEX as a persistent background daemon."""
    if _daemon_running():
        print_warning(f"MILEX daemon already running (PID {_read_pid()}).")
        raise typer.Exit(0)

    print_info(f"Starting MILEX daemon… log → [bold]{DAEMON_LOG}[/]")
    daemonize(DAEMON_LOG)
    # ↓ daemon only

    cfg = load_config()
    if model:
        cfg["model"] = model
    if auto:
        cfg["auto_execute"] = True

    try:
        ui = RichUI()
        agent = MilexAgent(config=cfg, ui=ui)
    except Exception as e:
        with open(str(DAEMON_LOG), "a") as lf:
            lf.write(f"[daemon] Failed to init agent: {e}\n")
        sys.exit(1)

    server = DaemonServer(agent)
    server.serve_forever()
    sys.exit(0)


# ─── `milex stop` ─────────────────────────────────────────────────────────────


@app.command(name="stop")
def stop_daemon():
    """Stop the background MILEX daemon."""
    pid = _read_pid()
    if pid is None or not _pid_alive(pid):
        print_warning("No running MILEX daemon found.")
        raise typer.Exit(0)
    try:
        os.kill(pid, signal.SIGTERM)
        PID_FILE.unlink(missing_ok=True)
        if SOCK_FILE.exists():
            SOCK_FILE.unlink(missing_ok=True)
        print_success(f"MILEX daemon (PID {pid}) stopped.")
    except PermissionError:
        print_error(f"Permission denied when trying to stop PID {pid}.")
        raise typer.Exit(1)


# ─── `milex status` ───────────────────────────────────────────────────────────


@app.command(name="status")
def daemon_status():
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


# ─── `milex models` ───────────────────────────────────────────────────────────


@app.command(name="models")
def list_models():
    """List available Ollama models."""
    cfg = load_config()
    try:
        ui = RichUI()
        agent = MilexAgent(config=cfg, ui=ui)
    except Exception as e:
        print_error(f"Failed to initialize agent: {e}")
        raise typer.Exit(1)
    models = agent.get_available_models()
    if models:
        print_models_table(models)
    else:
        print_warning("No models found or Ollama is not running.")


# ─── `milex config` ───────────────────────────────────────────────────────────


@app.command(name="config")
def show_config():
    """Show current MILEX configuration."""
    cfg = load_config()
    print_config_table(cfg)


# ─── `milex set` ──────────────────────────────────────────────────────────────


@app.command(name="set")
def set_config(
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


if __name__ == "__main__":
    app()
