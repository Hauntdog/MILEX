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

    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.conversation: List[Dict] = []
        self.executor = ToolExecutor(
            auto_execute=self.config.get("auto_execute", False),
            confirm_callback=self._confirm_tool,
        )
        self._client = ollama.Client(host=self.config["ollama_host"])

        # "Air Logic" — Response cache: avoid re-running identical queries
        self._response_cache: OrderedDict[str, str] = OrderedDict()
        self._cache_max = self.config.get("cache_size", 32)

        # "Air Logic" — RAG Manager
        self.rag = RagManager(self.config) if self.config.get("rag", {}).get("enabled", True) else None

        self._validate_model()

        # "Air Logic" — Keep model warm in memory to eliminate cold-start latency
        self._stop_event = threading.Event()
        self._keepalive_thread = threading.Thread(
            target=self._keep_model_warm_loop, daemon=True
        )
        self._keepalive_thread.start()

    def _validate_model(self):
        """Check that the configured model is available; raise early with a clear message."""
        try:
            response = self._client.list()
            available = [m.model for m in response.models]
            model = self.config["model"]
            if model not in available:
                raise ValueError(
                    f"Model '{model}' not found.\n"
                    f"Available models: {', '.join(available) or 'none'}\n"
                    f"Run: ollama pull {model}  — or use: milex set model <name>"
                )
        except ValueError:
            raise
        except Exception:
            # Ollama may not be running; let the chat methods surface that error naturally
            pass

    def _keep_model_warm_loop(self):
        """Send a tiny no-op request periodically to keep unique models loaded."""
        while not self._stop_event.is_set():
            # Warm all unique models used in roles (primary, coder, etc.)
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

            # Wait 5 minutes before next ping
            if self._stop_event.wait(300):
                break

    def _get_cache_key(self, messages: List[Dict]) -> str:
        """Generate a hash for the current conversation state."""
        def default_serializer(obj):
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            return str(obj)

        raw = f"{self.config['model']}:{json.dumps(messages, default=default_serializer)}"
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
            # If the error is that the model is missing or a general connection issue, try fallback
            if fallback and fallback != model:
                print_warning(f"Primary model '{model}' failed or missing. Retrying with fallback '{fallback}'...")
                return self._client.chat(model=fallback, **kwargs)
            raise

    def _get_options(self, override_temp: Optional[float] = None) -> dict:
        """Centralised inference options — tuned for CPU speed."""
        try:
            return {
                "temperature": float(override_temp if override_temp is not None else self.config.get("temperature", 0.7)),
                "num_predict": int(self.config.get("max_tokens", 2048)),
                "num_ctx": int(self.config.get("num_ctx", 4096)),
                # "Air Logic" — CPU-optimised parameters:
                "num_thread": int(self.config.get("num_thread", 0)),
                "num_batch": int(self.config.get("num_batch", 1024)),
                "num_keep": int(self.config.get("num_keep", 24)),
                "repeat_penalty": float(self.config.get("repeat_penalty", 1.1)),
            }
        except (ValueError, TypeError) as e:
            print_warning(f"Config type error: {e}. Using safe defaults.")
            return {
                "temperature": 0.7,
                "num_predict": 2048,
                "num_ctx": 4096,
                "num_thread": 0,
                "num_batch": 1024,
                "num_keep": 24,
                "repeat_penalty": 1.1,
            }

    # ── Public API ────────────────────────────────────────────────────────────

    def chat(self, user_input: str) -> str:
        """Send a message and get a response (handles tool calls automatically)."""
        self.conversation.append({"role": "user", "content": user_input})
        self._prune_history()

        max_rounds = 10
        for _round in range(max_rounds):
            messages = self._build_messages()
            cache_key = self._get_cache_key(messages)
            
            # Check cache
            if cache_key in self._response_cache:
                response_text = self._response_cache[cache_key]
                self._display_response(response_text)
                self.conversation.append({"role": "assistant", "content": response_text})
                return response_text

            response_text, tool_calls = self._call_model(messages)

            if tool_calls:
                # Process tools and loop back
                self._process_tool_calls(response_text, tool_calls)
            else:
                # No more tool calls → final answer
                if response_text:
                    self._display_response(response_text)
                    self._extract_and_display_code(response_text)
                    # Cache successful final response
                    self._add_to_cache(cache_key, response_text)
                    self.conversation.append(
                        {"role": "assistant", "content": response_text}
                    )
                    return response_text
                break
        return ""

    def stream_chat(self, user_input: str) -> str:
        """Send a message with streaming response.
        
        Optimized to avoid redundant calls and use the first inference for streaming if possible.
        """
        self.conversation.append({"role": "user", "content": user_input})
        self._prune_history()

        max_rounds = 10
        for _round in range(max_rounds):
            messages = self._build_messages()
            cache_key = self._get_cache_key(messages)

            # Check cache (even for streaming)
            if cache_key in self._response_cache:
                response_text = self._response_cache[cache_key]
                print_ai_message(response_text, model=self.config["model"])
                self.conversation.append({"role": "assistant", "content": response_text})
                return response_text

            # Unified streaming turn: handles text streaming and tool detection in one go
            # We use a while loop here instead of recursion in _unified_stream
            last_response = self._unified_stream(messages, cache_key)
            
            # If the last turn resulted in tool calls, the conversation now has 'tool' roles.
            # We loop back to let the model respond to the tool results.
            # We check the last message in conversation to see if it's a tool response.
            if self.conversation and self.conversation[-1].get("role") == "tool":
                continue
            else:
                return last_response
        return ""

    def _unified_stream(self, messages: List[Dict], cache_key: str) -> str:
        """Stream response and handle tool calls/text in one go."""
        full_text = ""
        tool_calls = []
        
        # Decide which model to use. If history has lots of code or 'coder' was requested.
        # For simplicity: check if we are in a 'generate_code' turn.
        # But this method is general. We'll stick to 'primary' for now unless we 
        # add more sophisticated detection.
        model = self._get_model_for_role("primary")
        
        spinner = None
        try:
            spinner = ThinkingSpinner("Thinking...")
            spinner.__enter__()
            
            # Using safe chat wraps the call with a fallback attempt if needed
            stream = self._chat_safe(
                model=model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                stream=True,
                keep_alive="10m",
                options=self._get_options(),
            )
            
            with StreamRenderer(model=model) as renderer:
                # The first iteration of the stream blocks during prefill.
                # We want the spinner to stay visible until then.
                for chunk in stream:
                    if spinner:
                        try:
                            spinner.__exit__(None, None, None)
                        except Exception:
                            pass
                        spinner = None # type: ignore
                    
                    msg = chunk.get("message", {})
                    
                    # 1. Native tool calls
                    if msg.get("tool_calls"):
                        tool_calls.extend(msg["tool_calls"])
                    
                    # 2. Content delta
                    delta = msg.get("content", "")
                    if delta:
                        full_text += delta
                        renderer.update(delta)

            # 3. Inline tool calls (fallback for older models/mismatched responses)
            if not tool_calls and full_text:
                tool_calls = self._parse_inline_tool_calls(full_text)

            if tool_calls:
                # Execute tools and update conversation history
                self._process_tool_calls(full_text, tool_calls)
                # We return the collected text so far (if any), 
                # but the loop in stream_chat will trigger the next round.
                return full_text
            
            if full_text:
                self.conversation.append({"role": "assistant", "content": full_text})
                self._add_to_cache(cache_key, full_text)
                self._extract_and_display_code(full_text)
            
            return full_text
        except Exception as e:
            print_error(f"Streaming error: {e}")
            return ""
        finally:
            if spinner:
                try:
                    spinner.__exit__(None, None, None)
                except Exception:
                    pass

    def _process_tool_calls(self, response_text: Optional[str], tool_calls: List[Any]):
        """Execute a batch of tool calls and update conversation."""
        # Clean tool calls for JSON serializability in history
        clean_calls = []
        for tc in tool_calls:
            if hasattr(tc, "model_dump"):
                clean_calls.append(tc.model_dump())
            elif isinstance(tc, dict):
                clean_calls.append(tc)
            else:
                clean_calls.append(str(tc))

        # Record the assistant turn with tool invocations
        self.conversation.append(
            {
                "role": "assistant",
                "content": response_text or "",
                "tool_calls": clean_calls,
            }
        )
        # "Air Logic": Execute non-destructive tools in parallel to save time
        # Destructive/Expensive tools still run sequentially if safety is an issue,
        # but here we use a thread pool for general speedup.
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(tool_calls), 4)) as executor:
            future_to_call = {}
            for call_obj in tool_calls:
                name, args = self._extract_tool_call(call_obj)
                print_tool_call(name, args)
                
                if name == "generate_code":
                    # Generate code is heavy, we'll run it in the main loop or as a single future
                    future = executor.submit(self._handle_generate_code, args)
                elif name == "rag_index":
                    future = executor.submit(self._handle_rag_index, args)
                elif name == "rag_search":
                    future = executor.submit(self._handle_rag_search, args)
                else:
                    future = executor.submit(self.executor.execute, name, args)
                future_to_call[future] = name

            for future in concurrent.futures.as_completed(future_to_call):
                name = future_to_call[future]
                try:
                    result = future.result()
                    # If this was a RAG search result, we can inject it into context if needed, 
                    # but here we just record it as a tool result.
                    success = "error" not in result and result.get("status") != "cancelled"
                    print_tool_result(name, result, success=success)
                    self.conversation.append(
                        {"role": "tool", "content": json.dumps(result)}
                    )
                except Exception as exc:
                    print_error(f"Tool {name} generated an exception: {exc}")

    def _prune_history(self):
        """Prune conversation if it gets too long (Sliding Window).
        Ensures the system prompt and initial context are preserved.
        """
        max_history = self.config.get("max_history", 25)
        if len(self.conversation) > max_history:
            # Keep system prompt (index 0) and the last N-1 messages.
            # This ensures we don't lose the initial instructions or very recent context.
            self.conversation = [self.conversation[0]] + self.conversation[-(max_history-1):]

    def _add_to_cache(self, key: str, value: str):
        """Add a response to the cache with LRU eviction."""
        if key in self._response_cache:
            del self._response_cache[key]
        self._response_cache[key] = value
        if len(self._response_cache) > self._cache_max:
            self._response_cache.popitem(last=False)

    def get_available_models(self) -> list:
        """Return list of models available in Ollama."""
        try:
            response = self._client.list()
            return response.models if hasattr(response, "models") else response.get("models", [])
        except Exception as e:
            print_error(f"Cannot connect to Ollama: {e}")
            return []

    def switch_model(self, model: str):
        self.config["model"] = model
        save_config(self.config)
        print_success(f"Switched to model: [bold cyan]{model}[/]")

    def clear_conversation(self):
        self.conversation.clear()
        print_info("Conversation history cleared.")

    def set_auto_execute(self, value: bool):
        self.config["auto_execute"] = value
        self.executor.auto_execute = value
        save_config(self.config)
        state = "[green]ON[/]" if value else "[red]OFF[/]"
        print_info(f"Auto-execute: {state}")

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_tool_call(tool_call) -> Tuple[str, dict]:
        """Normalise a tool call (pydantic object or dict) to (name, args)."""
        if hasattr(tool_call, "function"):
            fn = tool_call.function
            name = fn.name if hasattr(fn, "name") else fn["name"]
            args = fn.arguments if hasattr(fn, "arguments") else fn.get("arguments", {})
        else:
            fn = tool_call["function"]
            name = fn["name"]
            args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError) as e:
                print_warning(f"Failed to parse tool arguments: {e}")
                args = {}
        return name, (args or {})

    def _build_messages(self) -> List[Dict]:
        prompt_key = "compact_system_prompt" if self.config.get("compact_mode") else "system_prompt"
        messages = [{"role": "system", "content": self.config.get(prompt_key, self.config["system_prompt"])}]
        messages.extend(self.conversation)
        return messages

    def _call_model(self, messages: Optional[List[Dict]] = None) -> Tuple[str, List]:
        """Call the model and return (text, tool_calls)."""
        messages = messages or self._build_messages()
        model = self._get_model_for_role("primary")
        try:
            with ThinkingSpinner("Thinking..."):
                try:
                    response = self._chat_safe(
                        model=model,
                        messages=messages,
                        tools=TOOL_DEFINITIONS,
                        keep_alive="10m",
                        options=self._get_options(),
                    )
                except ollama.ResponseError as e:
                    # If model doesn't support tools, retry without them
                    if "does not support tools" in str(e).lower():
                        response = self._chat_safe(
                            model=model,
                            messages=messages,
                            keep_alive="10m",
                            options=self._get_options(),
                        )
                    else:
                        raise
        except Exception as e:
            print_error(f"Execution error: {e}")
            return "", []

        msg = response.message if hasattr(response, "message") else response.get("message", {})
        if hasattr(msg, "content"):
            text = msg.content or ""
            tool_calls = msg.tool_calls or []
        else:
            text = msg.get("content", "")
            tool_calls = msg.get("tool_calls", []) or []

        # Fallback: parse JSON tool calls embedded in text if model doesn't
        # natively support tools
        if not tool_calls and text:
            tool_calls = self._parse_inline_tool_calls(text)
            if tool_calls:
                # Strip tool call blocks from text
                text = re.sub(r"```(?:json)?\s*\{.*?\}\s*```", "", text, flags=re.DOTALL).strip()

        return text, tool_calls

    def _stream_response(self, messages: Optional[List[Dict]] = None) -> str:
        """Stream a response from the model."""
        messages = messages or self._build_messages()
        try:
            stream = self._client.chat(
                model=self.config["model"],
                messages=messages,
                stream=True,
                keep_alive="10m",
                options=self._get_options(),
            )
            with StreamRenderer(model=self.config["model"]) as renderer:
                for chunk in stream:
                    msg = chunk.get("message", {})
                    delta = msg.get("content", "")
                    if delta:
                        renderer.update(delta)
            return renderer.get_text()
        except Exception as e:
            print_error(f"Streaming error: {e}")
            return ""

    def _parse_inline_tool_calls(self, text: str) -> List[Dict]:
        """Try to parse tool calls embedded as JSON in the model response."""
        tool_calls = []
        # Match ```json {...} ``` or just {...} with "name" and "arguments"
        pattern = re.compile(
            r'```(?:json)?\s*(\{.*?"name"\s*:\s*"(\w+)".*?\})\s*```',
            re.DOTALL,
        )
        for match in pattern.finditer(text):
            try:
                obj = json.loads(match.group(1))
                if "name" in obj:
                    tool_calls.append(
                        {
                            "function": {
                                "name": obj["name"],
                                "arguments": obj.get("arguments", obj.get("args", {})),
                            }
                        }
                    )
            except Exception:
                pass
        return tool_calls

    def _handle_generate_code(self, args: dict) -> dict:
        """Generate code by calling the model specifically for code generation."""
        task = args.get("task", "")
        language = args.get("language", "python")
        filename = args.get("filename")
        
        # Use 'coder' role for this specialized task
        model = self._get_model_for_role("coder")

        prompt = (
            f"Generate complete, production-ready {language} code for the following task:\n\n"
            f"{task}\n\n"
            f"Requirements:\n"
            f"- Return COMPLETE code — no placeholders, no '...' truncations\n"
            f"- Include all imports, error handling, and comments\n"
            f"- Use best practices and idiomatic {language}\n"
            f"- Return ONLY the code block, no extra explanation"
        )

        # Include some context from the current conversation if it exists
        context_messages = []
        if self.conversation:
            # Add last 5-10 messages for context, excluding tool calls
            recent = [m for m in self.conversation if m.get("role") in ("user", "assistant")][-6:]
            for m in recent:
                context_messages.append({"role": m["role"], "content": f"Context: {m['content']}"})

        generation_messages = [
            {
                "role": "system",
                "content": (
                    f"You are an expert {language} programmer. "
                    "Return ONLY a single complete code block with no explanation. "
                    "The code must be immediately runnable."
                ),
            }
        ]
        generation_messages.extend(context_messages)
        generation_messages.append({"role": "user", "content": prompt})

        try:
            with ThinkingSpinner(f"Generating {language} code..."):
                response = self._client.chat(
                    model=self.config["model"],
                    messages=generation_messages,
                    options={
                        "temperature": 0.1,
                        "num_predict": self.config.get("max_tokens", 2048),
                        "num_ctx": self.config.get("num_ctx", 4096),
                    },
                )

            msg = response.message if hasattr(response, "message") else response.get("message", {})
            code = (msg.content or "") if hasattr(msg, "content") else msg.get("content", "")

            # Strip markdown fences if present
            code = re.sub(r"^```[\w]*\n?", "", code.strip())
            code = re.sub(r"\n?```\s*$", "", code.strip())
            code = code.strip()

            if not code:
                return {"error": "Model returned empty code"}

            # Display the code with syntax highlighting
            print_code_block(code, language=language, filename=filename)
            print_success(
                f"Generated [bold cyan]{len(code.splitlines())} lines[/] of {language} code"
            )

            # Auto-save if filename provided
            if filename:
                result = self.executor.execute("write_file", {"path": filename, "content": code})
                if "error" not in result:
                    print_success(f"Saved to [bold]{filename}[/]")
                else:
                    print_error(f"Failed to save: {result['error']}")
            else:
                # Offer to save
                from rich.prompt import Prompt
                console.print("\n[dim cyan]💾 Save to file? (Enter filename or press Enter to skip)[/]")
                try:
                    save_as = Prompt.ask("[dim cyan]Filename[/]", default="")
                    if save_as.strip():
                        result = self.executor.execute(
                            "write_file", {"path": save_as.strip(), "content": code}
                        )
                        if "error" not in result:
                            print_success(f"Saved to [bold]{save_as}[/]")
                        filename = save_as.strip()
                except (KeyboardInterrupt, EOFError):
                    pass

            # Offer to run (if bash/sh/python)
            if filename and language.lower() in ("python", "py", "bash", "sh"):
                from rich.prompt import Confirm
                try:
                    if Confirm.ask(f"\n[dim cyan]▶ Run [bold]{filename}[/]?[/]", default=False):
                        runner = "python3" if language.lower() in ("python", "py") else "bash"
                        result = self.executor.execute(
                            "run_shell",
                            {"command": f"{runner} {filename}"},
                        )
                        if result.get("stdout"):
                            console.print(f"\n[green]Output:[/]\n{result['stdout']}")
                        if result.get("stderr"):
                            console.print(f"\n[red]Stderr:[/]\n{result['stderr']}")
                except (KeyboardInterrupt, EOFError):
                    pass

            return {
                "success": True,
                "language": language,
                "lines": len(code.splitlines()),
                "filename": filename,
            }
        except Exception as e:
            return {"error": str(e)}

    def _handle_rag_index(self, args: dict) -> dict:
        """Handler for rag_index tool."""
        if not self.rag:
            return {"error": "RAG is disabled in config"}
        path = args.get("path", ".")
        self.rag.index_directory(path)
        return {"success": True, "path": path, "chunks": len(self.rag.chunks)}

    def _handle_rag_search(self, args: dict) -> dict:
        """Handler for rag_search tool."""
        if not self.rag:
            return {"error": "RAG is disabled in config"}
        query = args.get("query", "")
        count = args.get("count", 5)
        results = self.rag.search(query, top_k=count)
        return {"results": results, "count": len(results)}

    def _display_response(self, text: str):
        """Display AI response (non-streaming)."""
        print_ai_message(text, model=self.config["model"])

    def _extract_and_display_code(self, text: str):
        """Find code blocks in AI response and offer to save them."""
        # Match ```lang\ncode\n``` patterns
        pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
        blocks = pattern.findall(text)

        if not blocks:
            return

        # Ask to save if there are code blocks
        for lang, code in blocks:
            lang = lang.strip() if lang else "txt"
            code = code.strip()
            if len(code) < 20:
                continue  # Skip trivial blocks

            from rich.prompt import Prompt
            console.print(
                f"\n[dim cyan]💾 Save this [bold]{lang}[/] code block to a file? "
                f"(Enter filename or press Enter to skip)[/]"
            )
            try:
                filename = Prompt.ask("[dim cyan]Filename[/]", default="")
                if filename.strip():
                    result = self.executor.execute(
                        "write_file", {"path": filename.strip(), "content": code}
                    )
                    if "error" not in result:
                        print_success(f"Code saved to [bold]{filename}[/]")
                    else:
                        print_error(result["error"])
            except (KeyboardInterrupt, EOFError):
                pass

    def _confirm_tool(self, tool_name: str, args: dict) -> bool:
        return confirm_tool_execution(tool_name, args)
