import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from .config import CONFIG_DIR

TELEMETRY_FILE = CONFIG_DIR / "telemetry.json"

@dataclass
class ToolExecutionRecord:
    tool_name: str
    duration_ms: float
    success: bool
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class TelemetryManager:
    """Logs and manages tool performance data."""

    def __init__(self):
        self.logs_dir = CONFIG_DIR / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = TELEMETRY_FILE

    def record(self, tool_name: str, duration_sec: float, success: bool, error: Optional[str] = None):
        """Record a tool execution."""
        record = {
            "tool": tool_name,
            "duration_ms": round(duration_sec * 1000, 2),
            "success": success,
            "error": error,
            "timestamp": time.time()
        }
        
        try:
            # Append to a rolling JSON line file for performance
            with open(self.history_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass # Don't crash the agent for telemetry

    def get_stats(self) -> Dict[str, Any]:
        """Summarize tool performance."""
        if not self.history_file.exists():
            return {}

        stats = {}
        try:
            with open(self.history_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    name = data["tool"]
                    if name not in stats:
                        stats[name] = {"count": 0, "errors": 0, "total_ms": 0.0, "max_ms": 0.0}
                    
                    s = stats[name]
                    s["count"] += 1
                    if not data["success"]:
                        s["errors"] += 1
                    s["total_ms"] += data["duration_ms"]
                    s["max_ms"] = max(s["max_ms"], data["duration_ms"])

            # Calculate averages
            for name, s in stats.items():
                s["avg_ms"] = round(s["total_ms"] / s["count"], 2)
        except Exception:
            pass
        return stats

telemetry = TelemetryManager()
