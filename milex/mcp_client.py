"""MCP (Model Context Protocol) Client for MILEX."""
import asyncio
import json
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .ui import console, print_info, print_warning, print_error, ThinkingSpinner

class MCPClientManager:
    """Manages multiple MCP server connections.
    
    Each server runs as a separate subprocess and provides tools, 
    resources, and prompts to the agent.
    """

    def __init__(self, config: dict):
        self.config = config
        self.servers = config.get("mcp_servers", {})
        self.sessions: Dict[str, ClientSession] = {}
        self._tool_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._exit_stacks = []
        self._tasks = []

    async def connect_all(self):
        """Connect to all configured MCP servers."""
        if not self.servers:
            return

        for name, spec in self.servers.items():
            asyncio.create_task(self.connect_to_server(name, spec))

    async def connect_to_server(self, name: str, spec: dict):
        """Connect to a single MCP server via stdio."""
        cmd = spec.get("command")
        args = spec.get("args", [])
        env = {**os.environ, **spec.get("env", {})}

        if not cmd:
            print_warning(f"MCP Server '{name}' missing command.")
            return

        try:
            params = StdioServerParameters(command=cmd, args=args, env=env)
            
            # Using stdio_client from MCP SDK
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.sessions[name] = session
                    print_info(f"Connected to MCP Server: [bold cyan]{name}[/]")
                    
                    # Clear cache on new connection
                    self._tool_cache.pop(name, None)
                    
                    # Keep session alive and process notifications
                    while True:
                        await asyncio.sleep(1)
        except Exception as e:
            print_error(f"Failed to connect to MCP Server '{name}': {e}")
            self.sessions.pop(name, None)

    async def get_all_tools(self) -> List[Dict[str, Any]]:
        """Fetch tool definitions from all connected MCP servers with caching."""
        all_tools = []
        for name, session in self.sessions.items():
            # Return cached tools if available
            if name in self._tool_cache:
                all_tools.extend(self._tool_cache[name])
                continue
                
            try:
                tools_result = await session.list_tools()
                server_tools = []
                for tool in tools_result.tools:
                    # Convert MCP tools to OpenAI/Ollama tool format
                    server_tools.append({
                        "type": "function",
                        "function": {
                            "name": f"{name}__{tool.name}",
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                        "mcp_server": name,
                        "raw_name": tool.name
                    })
                self._tool_cache[name] = server_tools
                all_tools.extend(server_tools)
            except Exception as e:
                print_warning(f"Could not list tools from MCP server '{name}': {e}")
        return all_tools

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> Any:
        """Call a tool on a specific MCP server."""
        session = self.sessions.get(server_name)
        if not session:
            return {"error": f"MCP Server '{server_name}' not connected."}

        try:
            result = await session.call_tool(tool_name, arguments)
            return result
        except Exception as e:
            return {"error": f"MCP Tool call failed: {str(e)}"}

    async def shutdown(self):
        """Disconnect all servers."""
        for name in list(self.sessions.keys()):
            # Cleaning up would involve closing the async context stacks
            # if we managed them manually. sessions.close() etc.
            pass
