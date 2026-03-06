"""Ollama agent with tool-calling support for MILEX CLI."""
import hashlib
import json
import re
import threading
import concurrent.futures
from collections import OrderedDict
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import ollama

from .config import load_config, save_config
from .tools import TOOL_DEFINITIONS, ToolExecutor
from .rag import RagManager
from .ui import (
    AgentUI,
    RichUI,
    StreamRenderer,
    ThinkingSpinner,
    console,
    confirm_tool_execution,
    print_ai_message,
    print_code_block,
    print_error,
    print_info,
    print_success,
    print_tool_call,
    print_tool_result,
    print_warning,
)


class MilexAgent:
    """Main AI agent that interacts with Ollama and executes tools."""

    def __init__(self, config: Optional[dict] = None, ui: Optional[AgentUI] = None):
        self.config = config or load_config()
        self.ui = ui or RichUI()
        self.conversation: List[Dict] = []
        
        # "Air Logic" — RAG Manager
        self.rag = RagManager(self.config) if self.config.get("rag", {}).get("enabled", True) else None

        self.executor = ToolExecutor(
            config=self.config,
            ui=self.ui,
            rag=self.rag,
            agent=self,
            auto_execute=self.config.get("auto_execute", False),
        )
        self._client = ollama.Client(host=self.config["ollama_host"])

        # "Air Logic" — Response cache: avoid re-running identical queries
        self._response_cache: OrderedDict[str, str] = OrderedDict()
        self._cache_max = self.config.get("cache_size", 32)

        # Persistent pool to avoid overhead
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        self._validate_model()

        # "Air Logic" — Keep model warm in memory to eliminate cold-start latency
        self._stop_event = threading.Event()
        self._keepalive_thread = threading.Thread(
            target=self._keep_model_warm_loop, daemon=True
        )
        self._keepalive_thread.start()

    def __del__(self):
        self._stop_event.set()
        if hasattr(self, "_pool"):
            self._pool.shutdown(wait=False)

    def _validate_model(self):
        """Check that the configured model is available; raise early with a clear message."""
        try:
            response = self._client.list()
            available = [m.model for m in response.models]
            model = self.config["model"]
            if model not in available:
                self.ui.print_warning(f"Model '{model}' not found in Ollama.")
                # We don't raise here anymore to allow the app to start and let 
                # user switch models via CLI.
        except Exception as e:
            self.ui.print_warning(f"Cannot connect to Ollama: {e}")

    def _keep_model_warm_loop(self):
        """Send a tiny no-op request periodically to keep unique models loaded."""
        while not self._stop_event.is_set():
            unique_models = set(self.config.get("roles", {}).values())
            if not unique_models:
                unique_models = {self.config.get("model")}

            for model in unique_models:
                if not model: continue
                try:
                    self._client.chat(
                        model=model,
                        messages=[{"role": "user", "content": "ping"}],
                        keep_alive="10m",
                        options={"num_predict": 1},
                    )
                except Exception:
                    pass

            if self._stop_event.wait(300):
                break

    def _get_cache_key(self, messages: List[Dict]) -> str:
        """Generate a hash for the current conversation state."""
        # Normalize messages for more stable caching
        normalized = []
        for m in messages:
            n = {"role": m["role"], "content": (m.get("content") or "").strip()}
            if "tool_calls" in m:
                n["tool_calls"] = m["tool_calls"]
            normalized.append(n)
            
        raw = f"{self.config['model']}:{json.dumps(normalized, sort_keys=True)}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _get_model_for_role(self, role: str) -> str:
        """Get the model name for a specific role (primary, coder, planner)."""
        roles = self.config.get("roles", {})
        return roles.get(role, self.config["model"])

    def _chat_safe(self, model: str, **kwargs):
        """Execute a chat with transparent fallback if the primary model fails."""
        try:
            return self._client.chat(model=model, **kwargs)
        except Exception as e:
            fallback = self._get_model_for_role("fallback")
            if fallback and fallback != model:
                self.ui.print_warning(f"Retrying with fallback '{fallback}'...")
                return self._client.chat(model=fallback, **kwargs)
            raise

    def _get_options(self, override_temp: Optional[float] = None) -> dict:
        """Centralised inference options — tuned for CPU speed."""
        try:
            return {
                "temperature": float(override_temp if override_temp is not None else self.config.get("temperature", 0.7)),
                "num_predict": int(self.config.get("max_tokens", 2048)),
                "num_ctx": int(self.config.get("num_ctx", 4096)),
                "num_thread": int(self.config.get("num_thread", 0)),
                "num_batch": int(self.config.get("num_batch", 1024)),
                "num_keep": int(self.config.get("num_keep", 24)),
                "repeat_penalty": float(self.config.get("repeat_penalty", 1.1)),
            }
        except (ValueError, TypeError):
            return {"temperature": 0.7, "num_predict": 2048, "num_ctx": 4096}

    # ── Public API ────────────────────────────────────────────────────────────

    def chat(self, user_input: str) -> str:
        """Send a message and get a response (handles tool calls automatically)."""
        self.conversation.append({"role": "user", "content": user_input})
        self._prune_history()

        max_rounds = 10
        for _round in range(max_rounds):
            messages = self._build_messages()
            cache_key = self._get_cache_key(messages)
            
            if cache_key in self._response_cache:
                response_text = self._response_cache[cache_key]
                self.ui.print_ai_message(response_text, model=self.config["model"])
                self.conversation.append({"role": "assistant", "content": response_text})
                return response_text

            response_text, tool_calls = self._call_model(messages)

            if tool_calls:
                self._process_tool_calls(response_text, tool_calls)
            else:
                if response_text:
                    self.ui.print_ai_message(response_text, model=self.config["model"])
                    self._extract_and_offer_code(response_text)
                    self._add_to_cache(cache_key, response_text)
                    self.conversation.append({"role": "assistant", "content": response_text})
                    return response_text
                break
        return ""

    def stream_chat(self, user_input: str) -> str:
        """Send a message with streaming response."""
        self.conversation.append({"role": "user", "content": user_input})
        self._prune_history()

        max_rounds = 10
        for _round in range(max_rounds):
            messages = self._build_messages()
            cache_key = self._get_cache_key(messages)

            if cache_key in self._response_cache:
                response_text = self._response_cache[cache_key]
                self.ui.print_ai_message(response_text, model=self.config["model"])
                self.conversation.append({"role": "assistant", "content": response_text})
                return response_text

            last_response = self._unified_stream(messages, cache_key)
            
            # If the last turn resulted in tool results, loop back
            if self.conversation and self.conversation[-1].get("role") == "tool":
                continue
            else:
                return last_response
        return ""

    def _unified_stream(self, messages: List[Dict], cache_key: str) -> str:
        """Stream response and handle tool calls/text in one go."""
        full_text = ""
        tool_calls = []
        model = self._get_model_for_role("primary")
        
        try:
            with self.ui.create_thinking_spinner() as spinner:
                stream = self._chat_safe(
                    model=model,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    stream=True,
                    keep_alive="10m",
                    options=self._get_options(),
                )
                
                with self.ui.create_stream_renderer(model=model) as renderer:
                    for chunk in stream:
                        if spinner:
                            spinner.__exit__(None, None, None)
                            spinner = None # type: ignore
                        
                        msg = chunk.get("message", {})
                        if msg.get("tool_calls"):
                            tool_calls.extend(msg["tool_calls"])
                        
                        delta = msg.get("content", "")
                        if delta:
                            full_text += delta
                            renderer.update(delta)

            if not tool_calls and full_text:
                tool_calls = self._parse_inline_tool_calls(full_text)

            if tool_calls:
                self._process_tool_calls(full_text, tool_calls)
                return full_text
            
            if full_text:
                self.conversation.append({"role": "assistant", "content": full_text})
                self._add_to_cache(cache_key, full_text)
                self._extract_and_offer_code(full_text)
            
            return full_text
        except Exception as e:
            self.ui.print_error(f"Streaming error: {e}")
            return ""

    def _process_tool_calls(self, response_text: Optional[str], tool_calls: List[Any]):
        """Execute tool calls in parallel and update conversation."""
        clean_calls = []
        for tc in tool_calls:
            if hasattr(tc, "model_dump"):
                clean_calls.append(tc.model_dump())
            elif isinstance(tc, dict):
                clean_calls.append(tc)
            else:
                clean_calls.append(str(tc))

        self.conversation.append({
            "role": "assistant",
            "content": response_text or "",
            "tool_calls": clean_calls,
        })
        
        future_to_name = {}
        for call_obj in tool_calls:
            name, args = self._extract_tool_call(call_obj)
            self.ui.print_tool_call(name, args)
            future = self._pool.submit(self.executor.execute, name, args)
            future_to_name[future] = name

        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                result = future.result()
                success = "error" not in result and result.get("status") != "cancelled"
                self.ui.print_tool_result(name, result, success=success)
                self.conversation.append({"role": "tool", "content": json.dumps(result)})
            except Exception as exc:
                self.ui.print_error(f"Tool {name} failed: {exc}")

    def generate_code_internal(self, task: str, language: str, filename: Optional[str] = None) -> dict:
        """Internal worker for the generate_code tool."""
        model = self._get_model_for_role("coder")
        prompt = (
            f"Generate complete {language} code for:\n{task}\n"
            "Return ONLY the code block, no explanation."
        )
        
        # Simple context: last few messages
        history = [m for m in self.conversation if m.get("role") in ("user", "assistant")][-4:]
        messages = [{"role": "system", "content": f"Expert {language} coder."}]
        for m in history:
            messages.append({"role": m["role"], "content": f"Context: {m['content']}"})
        messages.append({"role": "user", "content": prompt})

        try:
            with self.ui.create_thinking_spinner(f"Generating {language}..."):
                response = self._client.chat(model=model, messages=messages, options={"temperature": 0.1})
            
            code = response.message.content.strip()
            code = re.sub(r"^```[\w]*\n?", "", code)
            code = re.sub(r"\n?```$", "", code).strip()

            if not code: return {"error": "No code generated"}

            self.ui.print_code_block(code, language=language, filename=filename)
            
            # Use UI to ask for saving/running if not auto-executing
            final_filename = filename
            if not final_filename:
                final_filename = self.ui.ask_save_file(code, language)
            
            if final_filename:
                write_res = self.executor.execute("write_file", {"path": final_filename, "content": code})
                if "success" in write_res:
                    self.ui.print_success(f"Saved to {final_filename}")
                    if language.lower() in ("python", "bash", "sh") and self.ui.ask_run_command(final_filename):
                        cmd = f"python3 {final_filename}" if language.lower() == "python" else f"bash {final_filename}"
                        run_res = self.executor.execute("run_shell", {"command": cmd})
                        return {"success": True, "filename": final_filename, "run_result": run_res}
                else:
                    return {"error": f"Save failed: {write_res.get('error')}"}

            return {"success": True, "filename": final_filename}
        except Exception as e:
            return {"error": str(e)}

    def _prune_history(self):
        """Sliding window history pruning."""
        max_history = self.config.get("max_history", 25)
        if len(self.conversation) > max_history:
            # Keep system prompt and last N messages
            self.conversation = [self.conversation[0]] + self.conversation[-(max_history-1):]

    def _add_to_cache(self, key: str, value: str):
        if key in self._response_cache:
            del self._response_cache[key]
        self._response_cache[key] = value
        if len(self._response_cache) > self._cache_max:
            self._response_cache.popitem(last=False)

    def _build_messages(self) -> List[Dict]:
        pk = "compact_system_prompt" if self.config.get("compact_mode") else "system_prompt"
        messages = [{"role": "system", "content": self.config.get(pk, self.config["system_prompt"])}]
        messages.extend(self.conversation)
        return messages

    def _call_model(self, messages: List[Dict]) -> Tuple[str, List]:
        model = self._get_model_for_role("primary")
        try:
            with self.ui.create_thinking_spinner():
                response = self._chat_safe(
                    model=model, messages=messages, tools=TOOL_DEFINITIONS, options=self._get_options()
                )
        except Exception as e:
            self.ui.print_error(f"Execution error: {e}")
            return "", []

        msg = response.message
        text = msg.content or ""
        tool_calls = msg.tool_calls or []

        if not tool_calls and text:
            tool_calls = self._parse_inline_tool_calls(text)
            if tool_calls:
                text = re.sub(r"```(?:json)?\s*\{.*?\}\s*```", "", text, flags=re.DOTALL).strip()
        return text, tool_calls

    def _parse_inline_tool_calls(self, text: str) -> List[Dict]:
        """Robustly find JSON tool calls in text."""
        tool_calls = []
        # Look for JSON-like structures that have a "name" key
        pattern = re.compile(r'(\{(?:[^{}]|(?R))*\})', re.DOTALL)
        # Python's re doesn't support recursion (?R), so we use a simpler approach
        # or just find all { ... } and try to parse them.
        for match in re.findall(r'\{[^{}]*?"name"\s*:\s*".*?"[^{}]*?\}', text, re.DOTALL):
            try:
                obj = json.loads(match)
                if "name" in obj:
                    tool_calls.append({
                        "function": {
                            "name": obj["name"],
                            "arguments": obj.get("arguments", obj.get("args", {})),
                        }
                    })
            except: continue
        return tool_calls

    def _extract_and_offer_code(self, text: str):
        blocks = re.findall(r"```(\w+)?\n(.*?)```", text, re.DOTALL)
        for lang, code in blocks:
            lang = lang.strip() if lang else "txt"
            if len(code) < 30: continue
            
            filename = self.ui.ask_save_file(code, lang)
            if filename:
                self.executor.execute("write_file", {"path": filename, "content": code})
                self.ui.print_success(f"Saved to {filename}")

    def _extract_tool_call(self, tc) -> Tuple[str, dict]:
        fn = tc.function if hasattr(tc, "function") else tc.get("function", {})
        name = fn.name if hasattr(fn, "name") else fn.get("name")
        args = fn.arguments if hasattr(fn, "arguments") else fn.get("arguments", {})
        if isinstance(args, str):
            try: args = json.loads(args)
            except: args = {}
        return name, (args or {})

    # Delegates
    def get_available_models(self):
        try: return self._client.list().models
        except: return []

    def switch_model(self, model: str):
        self.config["model"] = model
        save_config(self.config)
        self.ui.print_success(f"Switched to {model}")

    def clear_conversation(self):
        self.conversation.clear()
        self.ui.print_info("History cleared.")
