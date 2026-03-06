# MILEX CLI

```
    ███╗   ███╗██╗██╗     ███████╗██╗  ██╗
    ████╗ ████║██║██║     ██╔════╝╚██╗██╔╝
    ██╔████╔██║██║██║     █████╗   ╚███╔╝ 
    ██║╚██╔╝██║██║██║     ██╔══╝   ██╔██╗ 
    ██║ ╚═╝ ██║██║███████╗███████╗██╔╝ ██╗
    ╚═╝     ╚═╝╚═╝╚══════╝╚══════╝╚═╝  ╚═╝
```

**AI-Powered CLI · Code Generation · Computer Control**

MILEX is a powerful command-line interface that leverages **Ollama local models** to generate code, execute shell commands, manage files, and fully control your computer — all with a beautiful, Gemini-inspired terminal UI.

---

## Features

- 🤖 **Local AI** — Uses Ollama models (llama3.2, codellama, mistral, etc.)
- 💻 **Computer Control** — Execute shell commands, manage files, open URLs
- 🧑‍💻 **Code Generation** — Generate complete, production-ready code in any language
- 🎨 **Beautiful UI** — Rich-based interface with syntax highlighting, panels, streaming
- 🔒 **Safe by Default** — Confirmation prompts before dangerous actions
- ⚡ **Streaming** — Real-time streaming output
- 📝 **Persistent History** — Conversation and prompt history
- ⚙️ **Configurable** — Easy model switching and settings

---

## Requirements

- Python 3.9+
- [Ollama](https://ollama.ai) installed and running

---

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install as CLI tool
pip install -e .

# Or run directly
python -m milex.cli
```

---

## Usage

### Interactive Mode (default)
```bash
milex
```

### One-shot Mode
```bash
milex "Write a Python web scraper for HackerNews"
```

### With specific model
```bash
milex --model codellama "Create a REST API in FastAPI"
```

### Auto-execute mode (no confirmations)
```bash
milex --auto
```

---

## Slash Commands

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/clear` | Clear conversation history |
| `/model <name>` | Switch Ollama model |
| `/models` | List available models |
| `/config` | Show configuration |
| `/set <key> <value>` | Update a config value |
| `/auto [on\|off]` | Toggle auto-execute |
| `/code <lang> <task>` | Generate code directly |
| `/run <command>` | Run a shell command |
| `/sysinfo` | Show system information |
| `/save <file>` | Save conversation |
| `/exit` | Exit MILEX |

---

## Available Tools (AI can use these)

| Tool | Description |
|------|-------------|
| `run_shell` | Execute shell commands |
| `read_file` | Read file contents |
| `write_file` | Write/create files |
| `append_file` | Append content to a file |
| `list_directory` | List directory contents |
| `search_files` | Search files by name/content |
| `create_directory` | Create directories |
| `delete_path` | Delete files/directories |
| `copy_path` | Copy files/directories |
| `move_path` | Move or rename files/directories |
| `open_browser` | Open URLs in browser |
| `clipboard_copy` | Copy text to clipboard |
| `get_system_info` | Get system information |
| `generate_code` | Generate complete code |

---

## Configuration

Config is stored at `~/.milex/config.json`:

```json
{
  "model": "llama3.2",
  "ollama_host": "http://localhost:11434",
  "temperature": 0.7,
  "max_tokens": 8192,
  "auto_execute": false,
  "stream": true
}
```

---

## Examples

```
❯ You › Create a Python script that monitors CPU usage and alerts when it exceeds 80%

❯ You › Write a bash script to backup my home directory to /tmp/backup

❯ You › What files are in my Downloads folder?

❯ You › Install the requests library using pip

❯ You › Create a simple Flask web app with a /hello endpoint
```

---

## License

MIT
