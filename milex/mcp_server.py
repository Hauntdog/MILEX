"""MCP (Model Context Protocol) Server for MILEX.

Exposes MILEX's internal tools (shell, filesystem, rag, browser) to 
other MCP clients.
"""
import asyncio
import sys
from typing import Any, Dict, List, Optional, Union

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource

from .agent import MilexAgent
from .tools import TOOL_DEFINITIONS, ToolExecutor

class MILEXMCPServer:
    """Wraps MILEX tools into an MCP Server."""

    def __init__(self, name: str = "milex-server"):
        self.server = Server(name)
        self.agent = MilexAgent()
        self.executor = self.agent.executor
        
        self._setup_handlers()

    def _setup_handlers(self):
        """Register MCP handlers for tools and resources."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available MILEX tools."""
            mcp_tools = []
            for td in TOOL_DEFINITIONS:
                fn = td["function"]
                mcp_tools.append(Tool(
                    name=fn["name"],
                    description=fn["description"],
                    inputSchema=fn["parameters"]
                ))
            return mcp_tools

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[Union[TextContent, ImageContent, EmbeddedResource]]:
            """Execute a tool call."""
            try:
                # MILEX tools are usually executed via the ToolExecutor
                result = await self.executor.execute_async(name, arguments)
                
                # Format result for MCP
                content = []
                if "error" in result:
                    content.append(TextContent(type="text", text=f"Error: {result['error']}"))
                else:
                    # Generic JSON stringification for now
                    import json
                    content.append(TextContent(type="text", text=json.dumps(result, indent=2)))
                
                return content
            except Exception as e:
                return [TextContent(type="text", text=f"Server Error: {str(e)}")]

    async def run(self):
        """Run the server using stdio transport."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="milex",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )

async def main():
    """Entry point for mcp server."""
    server = MILEXMCPServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())
