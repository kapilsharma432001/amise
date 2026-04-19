## These are concrete tools, the actual "hands" that reach out into the world (without tool layer, the system is blind to the outside world -- it can only answer from documents already in the database)
## This is the foundation that the Model Context Protocol (MCP) is built on.
"""
Production tool implementation that registers with the ToolRegistry.

Each tool follows the same lifecycle:
    1. Define schema (inputs + description)
    2. Implement execute() with real API call
    3. Parse response into normalized ToolResult
    4. Handle errors gracefully (networking, parsing, rate limits)

Tools Implemented:
    - WebSearchTool: Search the web via DuckDuckGo (no API key required)
    - StockQuoteTool: Fetch stock data via Yahoo Finance (no API key required)
"""

import os
from typing import Optional

import httpx
import structlog

from src.mcp_servers.registry import (
    BaseTool,
    ToolParameter,
    ToolRegistry,
    ToolResult,
    ToolSchema,
)

logger = structlog.get_logger(__name__)
DEFAULT_HEADERS = {
    "User-Agent": "AMISE/1.0 (Market Intelligence Engine)",
}
HTTP_TIMEOUT = httpx.Timeout(
    connect=5.0,    # Max time to establish TCP connection
    read=15.0,      # Max time to receive response body
    write=5.0,      # Max time to send request body
    pool=5.0,       # Max time to acquire a connection from pool
)