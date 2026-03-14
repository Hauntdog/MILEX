"""Slash command handlers for MILEX CLI."""
import json
import re
from pathlib import Path
from typing import List, Optional, Any, TYPE_CHECKING
from .ui import (
    console,
    print_error,
    print_help,
    print_info,
    print_models_table,
    print_success,
    print_warning,
    print_config_table,
)
from .config import load_history, save_config, save_history

if TYPE_CHECKING:
    from .agent import MilexAgent

async def cmd_exit(cmd: str, agent: "MilexAgent") -> bool:
    """Handle /exit and /quit commands."""
    return False

async def cmd_help(cmd: str, agent: "MilexAgent") -> bool:
    """Handle /help command."""
    print_help()
    return True

async def cmd_clear(cmd: str, agent: "MilexAgent") -> bool:
    """Handle /clear command."""
    agent.clear_conversation()
    return True

async def cmd_models(cmd: str, agent: "MilexAgent") -> bool:
    """Handle /models command."""
    models = await agent.get_available_models()
    if models:
        print_models_table(models)
    else:
        print_warning("No models found. Make sure Ollama is running.")
    return True

async def cmd_model(cmd: str, agent: "MilexAgent") -> bool:
    """Handle /model command."""
    parts = cmd.strip().split(maxsplit=1)
    if len(parts) < 2:
        print_info(f"Current model: [bold cyan]{agent.config['model']}[/]")
    else:
        agent.switch_model(parts[1])
    return True

async def cmd_config(cmd: str, agent: "MilexAgent") -> bool:
    """Handle /config command."""
    print_config_table(agent.config)
    return True

async def cmd_set(cmd: str, agent: "MilexAgent") -> bool:
    """Handle /set command."""
    parts = cmd.strip().split(maxsplit=2)
    if len(parts) < 3:
        print_error("Usage: /set <key> <value>")
        return True
    key, value = parts[1], parts[2]
    
    # Type conversion
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

async def cmd_auto(cmd: str, agent: "MilexAgent") -> bool:
    """Handle /auto command."""
    parts = cmd.strip().split(maxsplit=1)
    val = parts[1].lower() in ("on", "true", "1", "yes") if len(parts) >= 2 else not agent.config.get("auto_execute")
    agent.config["auto_execute"] = val
    agent.executor.auto_execute = val
    save_config(agent.config)
    print_info(f"Auto-execute: {'[green]ON[/]' if val else '[red]OFF[/]'}")
    return True

async def cmd_history(cmd: str, agent: "MilexAgent") -> bool:
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

async def cmd_save(cmd: str, agent: "MilexAgent") -> bool:
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

async def cmd_code(cmd: str, agent: "MilexAgent") -> bool:
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

async def cmd_run(cmd: str, agent: "MilexAgent") -> bool:
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

async def cmd_research(cmd: str, agent: "MilexAgent") -> bool:
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
        "Step 1: Use the `search_web` tool to gather as much detailed information as possible about this topic.\n"
        "Step 2: Consolidate the research into a highly detailed report.\n"
        f"Step 3: Save the entire report directly into the file '{filename}'.\n"
        "Step 4: Give me a brief conversational summary."
    )
    await agent.stream_chat(prompt)
    return True

async def cmd_sysinfo(cmd: str, agent: "MilexAgent") -> bool:
    """Handle /sysinfo command."""
    res = await agent.executor.execute_async("get_system_info", {})
    from rich.table import Table
    table = Table(title="System Info")
    for k, v in res.items():
        table.add_row(k, str(v))
    console.print(table)
    return True

async def cmd_telemetry(cmd: str, agent: "MilexAgent") -> bool:
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
async def cmd_provider(cmd: str, agent: "MilexAgent") -> bool:
    """Handle /provider command."""
    parts = cmd.strip().split(maxsplit=1)
    if len(parts) < 2:
        print_info(f"Current provider: [bold cyan]{agent.config.get('provider', 'ollama')}[/]")
    else:
        prov = parts[1].lower()
        if prov in ("ollama", "gemini"):
            agent.config["provider"] = prov
            save_config(agent.config)
            print_success(f"Provider set to: [bold cyan]{prov}[/]")
        else:
            print_error("Invalid provider. Use: ollama or gemini")
    return True
