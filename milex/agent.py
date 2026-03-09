import asyncio
import hashlib
import json
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import ollama

from .config import load_config, save_config
from .tools import TOOL_DEFINITIONS, ToolExecutor
from .rag import RagManager
from .mcp_client import MCPClientManager
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


@dataclass
class ToolCall:
    """Represents a parsed tool call from the model."""
    name: str
    arguments: Dict[str, Any]
    raw: Optional[Dict] = None


@dataclass
class Message:
    """Represents a conversation message."""
    role: str
    content: str
    tool_calls: Optional[List[ToolCall]] = None


class MilexAgent:
    """Main AI agent that interacts with Ollama via Async API and executes tools."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        ui: Optional[AgentUI] = None,
    ):
        self.config: Dict[str, Any] = config or load_config()
        self.ui: AgentUI = ui or RichUI()
        self.conversation: List[Dict[str, Any]] = []

        # RAG Manager
        self.rag: Optional[RagManager] = (
            RagManager(self.config) if self.config.get("rag", {}).get("enabled", True) else None
        )

        # MCP Manager
        self.mcp = MCPClientManager(self.config)

        self.executor = ToolExecutor(
            config=self.config,
            ui=self.ui,
            rag=self.rag,
            agent=self,
            auto_execute=self.config.get("auto_execute", False),
        )
        self._client = ollama.AsyncClient(host=self.config["ollama_host"])

        # Optimized response cache using LRU-style OrderedDict
        self._response_cache: OrderedDict[str, str] = OrderedDict()
        self._cache_max: int = self.config.get("cache_size", 32)

        # Keepalive control
        self._keepalive_task: Optional[asyncio.Task] = None

        # Cache for model roles to avoid repeated dict lookups
        self._role_model_cache: Dict[str, str] = {}
        self._roles_dirty: bool = True

    def start_background_tasks(self):
        """Must be called within an event loop."""
        if not self._keepalive_task:
            self._keepalive_task = asyncio.create_task(self._keep_model_warm_loop())
            # Start model validation in background
            asyncio.create_task(self._validate_model())
            # Connect to MCP servers
            asyncio.create_task(self.mcp.connect_all())

    async def shutdown(self):
        """Cleanup."""
        if self._keepalive_task:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass

    async def _validate_model(self):
        """Async validation."""
        try:
            response = await self._client.list()
            available = [m.model for m in response.models]
            model = self.config["model"]
            if model not in available:
                self.ui.print_warning(f"Model '{model}' not found in Ollama.")
        except Exception as e:
            self.ui.print_warning(f"Cannot connect to Ollama: {e}")

    async def _keep_model_warm_loop(self):
        """Send a tiny no-op request periodically."""
        while True:
            unique_models = set(self.config.get("roles", {}).values())
            if not unique_models:
                unique_models = {self.config.get("model")}

            for model in unique_models:
                if not model: continue
                try:
                    await self._client.chat(
                        model=model,
                        messages=[{"role": "user", "content": "ping"}],
                        keep_alive="10m",
                        options={"num_predict": 1},
                    )
                except Exception:
                    pass
            await asyncio.sleep(300)

    def _get_cache_key(self, messages: List[Dict]) -> str:
        """Hash for current session state."""
        normalized = []
        for m in messages:
            n = {"role": m["role"], "content": (m.get("content") or "").strip()}
            if "tool_calls" in m:
                n["tool_calls"] = m["tool_calls"]
            normalized.append(n)
        raw = f"{self.config['model']}:{json.dumps(normalized, sort_keys=True)}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _get_model_for_role(self, role: str) -> str:
        """Get model name for a role with caching."""
        if self._roles_dirty:
            self._role_model_cache = dict(self.config.get("roles", {}))
            self._roles_dirty = False
        return self._role_model_cache.get(role, self.config["model"])

    def invalidate_role_cache(self):
        """Invalidate the role model cache after config changes."""
        self._roles_dirty = True

    async def _chat_safe(self, model: str, **kwargs):
        """Async chat with fallback retry."""
        try:
            return await self._client.chat(model=model, **kwargs)
        except Exception as e:
            fallback = self._get_model_for_role("fallback")
            if fallback and fallback != model:
                self.ui.print_warning(f"Retrying with fallback '{fallback}'...")
                return await self._client.chat(model=fallback, **kwargs)
            raise

    def _get_options(self, override_temp: Optional[float] = None) -> Dict[str, Any]:
        """Get model options with proper type coercion and defaults."""
        defaults = {
            "temperature": 0.7,
            "num_predict": 2048,
            "num_ctx": 4096,
        }
        try:
            temp = override_temp if override_temp is not None else self.config.get("temperature", 0.7)
            return {
                "temperature": float(temp),
                "num_predict": int(self.config.get("max_tokens", 2048)),
                "num_ctx": int(self.config.get("num_ctx", 4096)),
                "num_thread": int(self.config.get("num_thread", 0)),
                "num_batch": int(self.config.get("num_batch", 1024)),
                "num_keep": int(self.config.get("num_keep", 24)),
                "repeat_penalty": float(self.config.get("repeat_penalty", 1.1)),
            }
        except (ValueError, TypeError):
            return defaults

    # ── Public API (Async) ───────────────────────────────────────────────────

    async def chat(self, user_input: str) -> str:
        """Send message, handle tools, get response."""
        self.conversation.append({"role": "user", "content": user_input})
        self._prune_history()

        max_rounds = self.config.get("max_tool_rounds", 10)
        for _round in range(max_rounds):
            messages = self._build_messages()
            cache_key = self._get_cache_key(messages)

            if cache_key in self._response_cache:
                response_text = self._response_cache[cache_key]
                # Move to end (LRU update)
                self._response_cache.move_to_end(cache_key)
                self.ui.print_ai_message(response_text, model=self.config["model"])
                self.conversation.append({"role": "assistant", "content": response_text})
                return response_text

            response_text, tool_calls = await self._call_model(messages)

            if tool_calls:
                await self._process_tool_calls(response_text, tool_calls)
            else:
                if response_text:
                    self.ui.print_ai_message(response_text, model=self.config["model"])
                    self._extract_and_offer_code(response_text)
                    self._add_to_cache(cache_key, response_text)
                    self.conversation.append({"role": "assistant", "content": response_text})
                    return response_text
                break
        return ""

    async def stream_chat(self, user_input: str) -> str:
        """Streaming async response."""
        self.conversation.append({"role": "user", "content": user_input})
        self._prune_history()

        max_rounds = self.config.get("max_tool_rounds", 10)
        for _round in range(max_rounds):
            messages = self._build_messages()
            cache_key = self._get_cache_key(messages)

            if cache_key in self._response_cache:
                response_text = self._response_cache[cache_key]
                # Move to end (LRU update)
                self._response_cache.move_to_end(cache_key)
                self.ui.print_ai_message(response_text, model=self.config["model"])
                self.conversation.append({"role": "assistant", "content": response_text})
                return response_text

            last_response = await self._unified_stream(messages, cache_key)

            # Continue loop if tools were executed
            if self.conversation and self.conversation[-1].get("role") == "tool":
                continue
            else:
                return last_response
        return ""

    async def _unified_stream(self, messages: List[Dict], cache_key: str) -> str:
        full_text = ""
        tool_calls = []
        model = self._get_model_for_role("primary")
        
        try:
            with self.ui.create_thinking_spinner() as spinner:
                stream = await self._chat_safe(
                    model=model,
                    messages=messages,
                    tools=await self.executor.get_all_tools(),
                    stream=True,
                    keep_alive="10m",
                    options=self._get_options(),
                )
                
                with self.ui.create_stream_renderer(model=model) as renderer:
                    async for chunk in stream:
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
                await self._process_tool_calls(full_text, tool_calls)
                return full_text
            
            if full_text:
                self.conversation.append({"role": "assistant", "content": full_text})
                self._add_to_cache(cache_key, full_text)
                self._extract_and_offer_code(full_text)
            
            return full_text
        except Exception as e:
            self.ui.print_error(f"Streaming error: {e}")
            return ""

    async def _process_tool_calls(self, response_text: Optional[str], tool_calls: List[Any]):
        """Execute tools in parallel using asyncio.gather."""
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
        
        tasks = []
        for call_obj in tool_calls:
            name, args = self._extract_tool_call(call_obj)
            self.ui.print_tool_call(name, args)
            tasks.append(self._execute_and_wrap(name, args))

        results = await asyncio.gather(*tasks)
        for name, result in results:
            success = "error" not in result and result.get("status") != "cancelled"
            self.ui.print_tool_result(name, result, success=success)
            self.conversation.append({"role": "tool", "content": json.dumps(result)})

    async def _execute_and_wrap(self, name, args):
        """Small wrapper for tool execution in gather."""
        try:
            # We assume executor.execute is either async or wrapped in thread
            result = await self.executor.execute_async(name, args)
            return name, result
        except Exception as exc:
            return name, {"error": str(exc)}

    async def generate_code_internal(self, task: str, language: str, filename: Optional[str] = None) -> dict:
        """Coder specialist task."""
        model = self._get_model_for_role("coder")
        prompt = (
            f"Generate complete {language} code for:\n{task}\n"
            "Return ONLY the code block, no explanation."
        )
        
        history = [m for m in self.conversation if m.get("role") in ("user", "assistant")][-4:]
        messages = [{"role": "system", "content": f"Expert {language} coder."}]
        for m in history:
            messages.append({"role": m["role"], "content": f"Context: {m['content']}"})
        messages.append({"role": "user", "content": prompt})

        try:
            with self.ui.create_thinking_spinner(f"Generating {language}..."):
                response = await self._client.chat(model=model, messages=messages, options={"temperature": 0.1})
            
            code = response.message.content.strip()
            code = re.sub(r"^```[\w]*\n?", "", code)
            code = re.sub(r"\n?```$", "", code).strip()

            if not code: return {"error": "No code generated"}

            self.ui.print_code_block(code, language=language, filename=filename)
            
            final_filename = filename
            if not final_filename:
                final_filename = self.ui.ask_save_file(code, language)
            
            if final_filename:
                write_res = await self.executor.execute_async("write_file", {"path": final_filename, "content": code})
                if "success" in write_res:
                    self.ui.print_success(f"Saved to {final_filename}")
                    if language.lower() in ("python", "bash", "sh") and self.ui.ask_run_command(final_filename):
                        cmd = f"python3 {final_filename}" if language.lower() == "python" else f"bash {final_filename}"
                        run_res = await self.executor.execute_async("run_shell", {"command": cmd})
                        return {"success": True, "filename": final_filename, "run_result": run_res}
                else:
                    return {"error": f"Save failed: {write_res.get('error')}"}

            return {"success": True, "filename": final_filename}
        except Exception as e:
            return {"error": str(e)}

    def _prune_history(self):
        max_history = self.config.get("max_history", 25)
        if len(self.conversation) > max_history:
            self.conversation = [self.conversation[0]] + self.conversation[-(max_history-1):]

    def _add_to_cache(self, key: str, value: str):
        """Add response to cache with LRU eviction."""
        if key in self._response_cache:
            self._response_cache.move_to_end(key)
        self._response_cache[key] = value
        if len(self._response_cache) > self._cache_max:
            self._response_cache.popitem(last=False)

    def _build_messages(self) -> List[Dict]:
        pk = "compact_system_prompt" if self.config.get("compact_mode") else "system_prompt"
        messages = [{"role": "system", "content": self.config.get(pk, self.config["system_prompt"])}]
        messages.extend(self.conversation)
        return messages

    async def _call_model(self, messages: List[Dict]) -> Tuple[str, List]:
        model = self._get_model_for_role("primary")
        try:
            with self.ui.create_thinking_spinner():
                response = await self._chat_safe(
                    model=model, 
                    messages=messages, 
                    tools=await self.executor.get_all_tools(), 
                    options=self._get_options()
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
        tool_calls = []
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
        """No-op: the model now saves files proactively via write_file/edit_file tool calls.
        
        Previously this method would extract code blocks from the response and
        prompt the user to save them. With the Gemini-CLI-style auto-save behavior,
        the model is instructed to use tools directly, so this is no longer needed.
        """
        pass

    def _extract_tool_call(self, tc) -> Tuple[str, dict]:
        fn = tc.function if hasattr(tc, "function") else tc.get("function", {})
        name = fn.name if hasattr(fn, "name") else fn.get("name")
        args = fn.arguments if hasattr(fn, "arguments") else fn.get("arguments", {})
        if isinstance(args, str):
            try: args = json.loads(args)
            except: args = {}
        return name, (args or {})

    async def get_available_models(self):
        try: 
            resp = await self._client.list()
            return resp.models
        except: return []

    def switch_model(self, model: str):
        self.config["model"] = model
        self.invalidate_role_cache()
        save_config(self.config)
        self.ui.print_success(f"Switched to {model}")

    def clear_conversation(self):
        self.conversation.clear()
        self.ui.print_info("History cleared.")
