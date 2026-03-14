import asyncio
import hashlib
import json
import logging
import os
import re

# Silence noisy background logs that break terminal spinners
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import ollama

try:
    import google.generativeai as genai
    import warnings
    # Suppress the deprecation warning from the old Gemini SDK
    warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")
except ImportError:
    genai = None

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
from .utils import generate_filename_from_code, get_cache_key


@dataclass
class ToolCall:
    """Represents a parsed tool call from the model."""
    name: str
    arguments: Dict[str, Any]
    id: Optional[str] = None
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
        self.keep_alive = self.config.get("keep_alive", "30m")

        # Optimized response cache using LRU-style OrderedDict
        self._response_cache: OrderedDict[str, str] = OrderedDict()
        self._cache_max: int = self.config.get("cache_size", 32)

        # Keepalive control
        self._keepalive_task: Optional[asyncio.Task] = None

        # Loop detection
        self._last_tool_calls: List[str] = []
        self._max_repeat_threshold: int = 3

        # Cache for model roles to avoid repeated dict lookups
        self._role_model_cache: Dict[str, str] = {}
        self._roles_dirty: bool = True

        # Multi-provider clients
        self._gemini_client = None

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
        model = self.config.get("model")
        if not model: return
        
        provider = self._get_provider(model)
        if provider != "ollama":
            return
            
        try:
            response = await self._client.list()
            available = [m.model for m in response.models]
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
                provider = self._get_provider(model)
                if provider != "ollama":
                    continue
                try:
                    await self._client.chat(
                        model=model,
                        messages=[{"role": "user", "content": "ping"}],
                        keep_alive=self.keep_alive,
                        options={"num_predict": 1},
                    )
                except Exception:
                    pass
            await asyncio.sleep(300)

    def _get_cache_key(self, messages: List[Dict]) -> str:
        return get_cache_key(messages, self.config["model"])

    def _get_model_for_role(self, role: str) -> str:
        """Get model name for a role with caching."""
        if self._roles_dirty:
            self._role_model_cache = self.config.get("roles", {})
            self._roles_dirty = False
        return self._role_model_cache.get(role, self.config["model"])

    def invalidate_role_cache(self):
        """Invalidate the role model cache after config changes."""
        self._roles_dirty = True

    def _get_provider(self, model_name: str) -> str:
        """Infer provider from model name, falling back to global config."""
        name = model_name.lower()
        if name.startswith("gemini-"):
            return "gemini"
            
        # If it looks like an Ollama model (has a colon or starts with known Ollama families), 
        # use Ollama regardless of global provider. This enables hybrid usage.
        if ":" in name or any(name.startswith(p) for p in ["llama", "qwen", "mistral", "phi", "codellama", "deepseek"]):
            return "ollama"

        # Fallback to global config
        return self.config.get("provider", "ollama")

    def _clean_gemini_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Strip fields that Gemini's SDK/API doesn't support in tool definitions."""
        if not isinstance(schema, dict):
            return schema
            
        # Allowed keys for a Gemini Schema object per official API docs
        # Note: 'format' is supported for string types (e.g. 'date-time')
        ALLOWED_KEYS = {"type", "format", "description", "nullable", "enum", "properties", "required", "items"}
        
        cleaned = {}
        for k, v in schema.items():
            if k not in ALLOWED_KEYS:
                continue
            
            if k == "properties" and isinstance(v, dict):
                cleaned[k] = {pk: self._clean_gemini_schema(pv) for pk, pv in v.items()}
            elif k == "items" and isinstance(v, dict):
                cleaned[k] = self._clean_gemini_schema(v)
            elif k == "required" and isinstance(v, list):
                if v: # Only add if non-empty
                    cleaned[k] = v
            elif isinstance(v, list):
                cleaned[k] = [self._clean_gemini_schema(i) if isinstance(i, dict) else i for i in v]
            else:
                cleaned[k] = v
        
        # Ensure 'type' is present - Gemini is very strict
        if "type" not in cleaned:
            if "properties" in cleaned:
                cleaned["type"] = "object"
            elif "items" in cleaned:
                cleaned["type"] = "array"
            else:
                cleaned["type"] = "string" # Safest default
        
        # Handle list-based types (e.g. ["string", "null"])
        if isinstance(cleaned.get("type"), list):
            types = cleaned["type"]
            if "null" in types:
                cleaned["nullable"] = True
                remaining = [t for t in types if t != "null"]
                cleaned["type"] = remaining[0] if remaining else "string"
            else:
                cleaned["type"] = types[0]

        return cleaned

    async def _get_client(self, provider: str):
        """Get or initialize the appropriate provider client."""
        if provider == "gemini":
            if not self._gemini_client:
                if not genai:
                    raise ImportError("Google GenerativeAI SDK not installed. Run 'pip install google-generativeai'")
                key = self.config.get("gemini_key")
                if not key:
                    raise ValueError("Gemini API key missing. Set 'gemini_key' in config.")
                genai.configure(api_key=key)
                self._gemini_client = genai
            return self._gemini_client
            
        return self._client

    async def _chat_safe(self, model: str, **kwargs):
        """Async chat for Ollama with fallback retry and 400 error handling."""
        try:
            return await self._client.chat(model=model, **kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            if "400" in error_msg or "bad request" in error_msg:
                self.ui.print_warning("Request failed (400), likely context overflow. Retrying with pruned history...")
                original_history = self.conversation.copy()
                if len(self.conversation) > 5:
                    self.conversation = [self.conversation[0]] + self.conversation[-5:]
                
                minimal_options = {"temperature": 0.5, "num_ctx": 2048}
                clean_kwargs = kwargs.copy()
                clean_kwargs["messages"] = self._build_messages()
                if "options" in clean_kwargs:
                    clean_kwargs["options"] = {**clean_kwargs.get("options", {}), **minimal_options}
                else:
                    clean_kwargs["options"] = minimal_options
                    
                try:
                    return await self._client.chat(model=model, **clean_kwargs)
                except Exception as retry_error:
                    self.conversation = original_history
                    self.ui.print_error(f"Retry failed: {retry_error}")
            
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
            "num_ctx": 2048,  # Match config for CPU optimization
        }
        try:
            temp = override_temp if override_temp is not None else self.config.get("temperature", 0.7)
            options = {
                "temperature": float(temp),
                "num_predict": int(self.config.get("max_tokens", 2048)),
                "num_ctx": int(self.config.get("num_ctx", 4096)),
                "repeat_penalty": float(self.config.get("repeat_penalty", 1.1)),
            }
            
            # CPU performance tuning: num_batch
            # Lower batch size on CPU can improve Time To First Token (TTFT)
            num_batch = self.config.get("num_batch", 512)
            if num_batch is not None and int(num_batch) > 0:
                options["num_batch"] = int(num_batch)

            # CPU performance tuning: num_thread
            # Ideally matches the physical core count
            num_thread = self.config.get("num_thread", os.cpu_count() or 4)
            if num_thread is not None and int(num_thread) > 0:
                options["num_thread"] = int(num_thread)
            
            # num_keep: -1 means keep all tokens in KV cache (major speedup)
            # Only skip if explicitly set to 0 (disabled)
            num_keep = self.config.get("num_keep", -1)
            if num_keep is not None and int(num_keep) != 0:
                options["num_keep"] = int(num_keep)
            
            return options
        except (ValueError, TypeError):
            return defaults

    def _detect_streaming_loop(self, text: str, min_len: int = 40) -> bool:
        """Heuristic to detect if the model has started repeating a significant chunk of text."""
        if len(text) < min_len * 2:
            return False
            
        # Check for immediate repetition of chunks (e.g. "abcabc")
        for window in range(min_len, min(len(text) // 2, 256)):
            if text[-window:] == text[-2*window:-window]:
                return True
                
        # Check for phrase-level repetition (more expensive)
        return False

    # ── Public API (Async) ───────────────────────────────────────────────────

    async def chat(self, user_input: str) -> str:
        """Send message, handle tools, get response."""
        self.conversation.append({"role": "user", "content": user_input})
        self._prune_history()
        self._last_tool_calls.clear()

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
        self._last_tool_calls.clear()

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
            spinner_msg = "Thinking..."
            if self.conversation and self.conversation[-1].get("role") == "tool":
                spinner_msg = "Synthesizing response..."
                
            spinner = self.ui.create_thinking_spinner(spinner_msg)
            spinner.start()
            try:
                tool_calls_dict = {}
                with self.ui.create_stream_renderer(model=model) as renderer:
                    # Detect provider for specific streaming logic
                    provider = self._get_provider(model)
                    
                    if provider == "ollama":
                        stream = await self._chat_safe(
                            model=model,
                            messages=messages,
                            tools=await self.executor.get_all_tools(),
                            stream=True,
                            keep_alive=self.keep_alive,
                            options=self._get_options(),
                        )
                        async for chunk in stream:
                            msg = chunk.get("message", {})
                            if msg.get("tool_calls"):
                                for i, tc in enumerate(msg["tool_calls"]):
                                    tc_id = getattr(tc, "id", None) or tc.get("id") or str(i)
                                    tool_calls_dict[tc_id] = tc
                            delta = msg.get("content", "")
                            if delta:
                                full_text += delta
                                renderer.update(delta)
                                if self._detect_streaming_loop(full_text):
                                    self.ui.print_warning("Streaming loop detected. Breaking.")
                                    break
                    
                    elif provider == "gemini":
                        # Directly use the member to assist type inference/linting if needed
                        await self._get_client("gemini")
                        g_client = self._gemini_client
                        if not g_client:
                            raise ValueError("Gemini client not initialized")
                            
                        tools_g = await self.executor.get_all_tools()
                        gemini_tools = []
                        if tools_g:
                            functions = []
                            for t in tools_g:
                                f_decl = {
                                    "name": t["function"]["name"],
                                    "description": t["function"]["description"],
                                    "parameters": self._clean_gemini_schema(t["function"]["parameters"])
                                }
                                functions.append(f_decl)
                            gemini_tools = [{"function_declarations": functions}]
                        
                        sys_instr = next((m["content"] for m in messages if m["role"] == "system"), None)
                        model_g = g_client.GenerativeModel(
                            model_name=model,
                            tools=gemini_tools if gemini_tools else None,
                            system_instruction=sys_instr
                        )
                        
                        # Convert history
                        history = []
                        for m in messages:
                            if m["role"] == "system": continue
                            role = "user" if m["role"] in ("user", "tool") else "model"
                            content = m.get("content") or ""
                            history.append({"role": role, "parts": [content]})
                        
                        current_msg = history.pop() if history else {"role": "user", "parts": [""]}
                        chat = model_g.start_chat(history=history if history else None)
                        
                        response_stream = await chat.send_message_async(current_msg["parts"][0], stream=True)
                        async for chunk in response_stream:
                            # Handle text part safely
                            chunk_text = ""
                            try:
                                chunk_text = chunk.text
                            except (ValueError, IndexError):
                                pass
                                
                            if chunk_text:
                                full_text += chunk_text
                                renderer.update(chunk_text)
                                if self._detect_streaming_loop(full_text):
                                    self.ui.print_warning("Streaming loop detected. Breaking.")
                                    break
                            
                            # Check for tool calls
                            if chunk.candidates and chunk.candidates[0].content.parts:
                                for part in chunk.candidates[0].content.parts:
                                    if part.function_call:
                                        call = part.function_call
                                        tc_id = f"call_{hashlib.md5(call.name.encode()).hexdigest()[:8]}"
                                        name = call.name
                                        args = dict(call.args)
                                        tool_calls_dict[tc_id] = {
                                            "id": tc_id,
                                            "function": {"name": name, "arguments": json.dumps(args)}
                                        }
                
                # Close renderer and proceed
                if spinner:
                    spinner.stop()
                    spinner = None
                
                tool_calls = list(tool_calls_dict.values())
            finally:
                if spinner:
                    spinner.stop()

            if not tool_calls and full_text:
                tool_calls = self._parse_inline_tool_calls(full_text)

            if tool_calls:
                await self._process_tool_calls(full_text, tool_calls)
                # After tool calls, we might want a synthesizing response spinner
                # but it will be handled by the next iteration of the round loop in chat/stream_chat
                return full_text
            
            if full_text:
                if not any(m.get("role") == "assistant" and m.get("content") == full_text for m in self.conversation):
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
        unique_call_hashes = []

        for tc in tool_calls:
            name, args, call_id = self._extract_tool_call(tc)
            call_hash = f"{name}:{json.dumps(args, sort_keys=True)}"
            unique_call_hashes.append(call_hash)
            
            call_dict = {
                "function": {
                    "name": name,
                    "arguments": args,
                }
            }
            if call_id:
                call_dict["id"] = call_id
            clean_calls.append(call_dict)

        # Detect loop
        current_batch_hash = "|".join(sorted(unique_call_hashes))
        
        # We check the last N tool call batches for repetitions
        # If the same batch appears multiple times in a row, it's a likely loop
        if len(self._last_tool_calls) > 0 and self._last_tool_calls.count(current_batch_hash) >= self._max_repeat_threshold:
            self.ui.print_error(f"Detected repetitive tool call pattern: {current_batch_hash[:50]}...")
            self.ui.print_warning("MILEX is stopping to prevent an infinite loop.")
            
            loop_msg = "I've detected an infinite loop in my tool calls. I'll stop here to prevent further issues. Please try rephrasing your request."
            self.conversation.append({
                "role": "assistant",
                "content": loop_msg
            })
            return

        self._last_tool_calls.append(current_batch_hash)

        self.conversation.append({
            "role": "assistant",
            "content": response_text or "",
            "tool_calls": clean_calls,
        })
        
        tasks = []
        for tc in tool_calls:
            name, args, call_id = self._extract_tool_call(tc)
            self.ui.print_tool_call(name, args)
            tasks.append(self._execute_and_wrap(name, args, call_id))

        results = await asyncio.gather(*tasks)
        for name, result, call_id in results:
            success = "error" not in result and result.get("status") != "cancelled"
            self.ui.print_tool_result(name, result, success=success)
            
            tool_msg = {"role": "tool", "content": json.dumps(result)}
            if call_id:
                tool_msg["tool_call_id"] = call_id
            self.conversation.append(tool_msg)

    async def _execute_and_wrap(self, name, args, call_id=None):
        """Small wrapper for tool execution in gather."""
        try:
            result = await self.executor.execute_async(name, args)
            return name, result, call_id
        except Exception as exc:
            return name, {"error": str(exc)}, call_id

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
                # Auto-generate filename from code content
                final_filename = generate_filename_from_code(code, language)
                self.ui.print_info(f"Auto-generated filename: {final_filename}")
            
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
        """Keep the conversation history within bounds, but preserve the first system message and recent context."""
        max_history = self.config.get("max_history", 25)
        
        # If we are in the middle of a complex tool-use round, we might want to temporarily exceed max_history
        # to ensure the model has all the tool results it needs to finalize.
        # But for simplicity, we just look at the total count here.
        
        if len(self.conversation) > max_history:
            # Keep index 0 (usually system prompt or first user message)
            # and the most recent (max_history - 1) messages
            # We try to keep things in pairs (user/assistant or tool_call/tool_result)
            preserved = [self.conversation[0]]
            recent = self.conversation[-(max_history-1):]
            
            # If the first message in 'recent' is a tool result, we should probably keep 
            # the preceding assistant message (which has the tool call)
            if recent and recent[0].get("role") == "tool":
                # Find the corresponding tool call message if possible
                pass 
                
            self.conversation = preserved + recent

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
        provider = self._get_provider(model)
        
        try:
            spinner_msg = "Thinking..."
            if self.conversation and self.conversation[-1].get("role") == "tool":
                spinner_msg = "Synthesizing response..."
                
            with self.ui.create_thinking_spinner(spinner_msg):
                if provider == "ollama":
                    response = await self._chat_safe(
                        model=model, 
                        messages=messages, 
                        tools=await self.executor.get_all_tools(), 
                        options=self._get_options()
                    )
                    msg = response.message
                    text = msg.content or ""
                    tool_calls = msg.tool_calls or []
                
                elif provider == "gemini":
                    await self._get_client("gemini")
                    g_client = self._gemini_client
                    if not g_client:
                        raise ValueError("Gemini client not initialized")
                        
                    tools_g = await self.executor.get_all_tools()
                    gemini_tools = []
                    if tools_g:
                        functions = []
                        for t in tools_g:
                            f_decl = {
                                "name": t["function"]["name"],
                                "description": t["function"]["description"],
                                "parameters": self._clean_gemini_schema(t["function"]["parameters"])
                            }
                            functions.append(f_decl)
                        gemini_tools = [{"function_declarations": functions}]
                    
                    model_g = g_client.GenerativeModel(
                        model_name=model,
                        tools=gemini_tools if gemini_tools else None
                    )
                    
                    # Manual conversion of messages to Gemini format
                    history = []
                    for m in messages:
                        if m["role"] == "system": continue
                        role = "user" if m["role"] in ("user", "tool") else "model"
                        content = m.get("content") or ""
                        # Note: tool results in Gemini are complex, for now we simplify to text
                        history.append({"role": role, "parts": [content]})
                    
                    # Last message is the current one
                    current_msg = history.pop() if history else {"role": "user", "parts": [""]}
                    
                    # System instruction
                    sys_instr = next((m["content"] for m in messages if m["role"] == "system"), None)
                    if sys_instr:
                        model_g = g_client.GenerativeModel(
                            model_name=model,
                            tools=gemini_tools if gemini_tools else None,
                            system_instruction=sys_instr
                        )
                    
                    chat = model_g.start_chat(history=history if history else None)
                    resp_g = await chat.send_message_async(current_msg["parts"][0])
                    
                    try:
                        text = resp_g.text
                    except (ValueError, IndexError):
                        text = ""
                        
                    tool_calls = []
                    if resp_g.candidates and resp_g.candidates[0].content.parts:
                        for part in resp_g.candidates[0].content.parts:
                            if part.function_call:
                                call = part.function_call
                                tool_calls.append({
                                    "id": f"call_{hashlib.md5(call.name.encode()).hexdigest()[:8]}",
                                    "function": {"name": call.name, "arguments": json.dumps(dict(call.args))}
                                })

        except Exception as e:
            self.ui.print_error(f"Execution error ({provider}/{model}): {e}")
            return "", []

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
        """Extract code blocks from the response and auto-save them if a filename is mentioned."""
        blocks = text.split("```")
        for i in range(1, len(blocks), 2):
            block = blocks[i]
            if "\n" not in block:
                continue
                
            header, code = block.split("\n", 1)
            code = code.strip()
            if not code:
                continue
                
            filename = None
            
            # 1. Check header for filename (e.g. ```python test.py)
            parts = header.strip().split()
            if len(parts) > 1:
                filename = parts[-1]
                
            # 2. Check preceding text for a filename right before the block
            if not filename:
                pre = blocks[i-1].strip()
                # Matches patterns like: "save the file in the test.py" or "code for `test.py`:"
                match = re.search(r'(?:file|in|to|as)\s+(?:the\s+)?\*?`?([a-zA-Z0-9_/\-]+\.[a-zA-Z0-9]+)`?\*?:?$', pre, re.IGNORECASE)
                if match:
                    filename = match.group(1)
                    
            # 3. Check first line comment of the code
            if not filename:
                first_line = code.split('\n')[0].strip()
                match = re.search(r'^[#/*<!-]+\s*(?:file:\s*)?([a-zA-Z0-9_/\-]+\.[a-zA-Z0-9]+)\s*$', first_line, re.IGNORECASE)
                if match:
                    filename = match.group(1)
                    
            if filename:
                # Auto save the file
                res = self.executor.execute("write_file", {"path": filename, "content": code})
                if res.get("success"):
                    self.ui.print_success(f"Auto-saved code to {filename}")
                else:
                    self.ui.print_warning(f"Failed to auto-save {filename}: {res.get('error')}")

    def _extract_tool_call(self, tc) -> Tuple[str, dict, Optional[str]]:
        """Extract tool call components with ID support."""
        if hasattr(tc, "function"):
            fn = tc.function
            name = getattr(fn, "name", None)
            args = getattr(fn, "arguments", {})
            call_id = getattr(tc, "id", None)
        elif isinstance(tc, dict):
            # Support both function-nested and flat formats
            fn = tc.get("function", tc)
            name = fn.get("name")
            args = fn.get("arguments", fn.get("args", {}))
            call_id = tc.get("id")
        else:
            name = str(tc)
            args = {}
            call_id = None
            
        if isinstance(args, str):
            try: args = json.loads(args)
            except: args = {}
            
        return (name or ""), (args or {}), call_id

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
