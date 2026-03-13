import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from .config import CONFIG_DIR

TELEMETRY_FILE = CONFIG_DIR / "telemetry.json"

@dataclass
class ToolExecutionRecord:
    tool: str
    duration_ms: float
    success: bool
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp
        }


class TelemetryManager:
    """Logs and manages tool performance data."""

    def __init__(self):
        self.logs_dir = CONFIG_DIR / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = TELEMETRY_FILE

    def record(self, tool_name: str, duration_sec: float, success: bool, error: Optional[str] = None):
        """Record a tool execution."""
        record = ToolExecutionRecord(
            tool=tool_name,
            duration_ms=round(duration_sec * 1000, 2),
            success=success,
            error=error
        )
        
        try:
            # Append to a rolling JSON line file for performance
            with open(self.history_file, "a") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
        except Exception:
            pass # Don't crash the agent for telemetry

    def get_stats(self, limit: int = 1000) -> Dict[str, Any]:
        """Summarize tool performance from the last N records."""
        if not self.history_file.exists():
            return {}

        stats = {}
        try:
            with open(self.history_file, "r") as f:
                all_lines = f.read().splitlines()
                # Use standard list indexing which should be fine
                lines_to_process = all_lines[-limit:] if limit > 0 and len(all_lines) > limit else all_lines
                
                for line in lines_to_process:
                    if not line.strip(): continue
                    try:
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
                    except (json.JSONDecodeError, KeyError):
                        continue

            # Calculate averages
            for name, s in stats.items():
                s["avg_ms"] = round(s["total_ms"] / s["count"], 2)
        except Exception:
            pass
        return stats

    def clear(self):
        """Clear telemetry history."""
        if self.history_file.exists():
            self.history_file.unlink()

telemetry = TelemetryManager()
