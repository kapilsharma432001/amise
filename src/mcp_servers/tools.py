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



### WEB SEARCH TOOL
class WebSearchTool(BaseTool):

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="web_search",
            description=(
                    "Search the web for current information, news, company data, "
                    "or any real-time knowledge. Use this when the user's question "
                    "requires information beyond your training data."
                ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="The search query string (e.g., 'Tesla Q3 2024 earnings')",
                ),
                ToolParameter(
                    name = "max_results",
                    type = "integer",
                    description = "Maximum number of results to return (1-10)",
                    required = False,
                    default = 5,
                ),
            ],
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute a web search and return normalized results.

        Flow:
            1. Validate inputs
            2. Call DuckDuckGo API
            3. Parse and normalize response
            4. Return structured ToolResult
        """

        query = kwargs.get("query")
        max_results = kwargs.get("max_results", 5)

        if not query or not query.strip():
            return ToolResult(
                success=False,
                error="Parameter 'query' is required and cannot be empty.",
            )

        # Clamp max_results to a sane range
        max_results = max(1, min(max_results, 10))

        try:
            async with httpx.AsyncClient(
                headers=DEFAULT_HEADERS, timeout=HTTP_TIMEOUT
            ) as client:
                response = await client.get(
                    "https://api.duckduckgo.com/",
                    params={
                        "q": query,
                        "format": "json",
                        "no_html": 1,
                        "skip_disambig": 1,
                    },
                )
                response.raise_for_status()
                data = response.json()
            
            # Parse results from DuckDuckGo's response structure
            results = self._parse_ddg_response(data, max_results)

            # If DuckDuckGo returned nothing useful (it often does for
            # specific queries), return a helpful message, not an error.
            if not results:
                return ToolResult(
                    success=True,
                    data={
                        "query": query,
                        "results": [],
                        "summary": f"No instant results for '{query}'. "
                                   f"Try a more general query.",
                    },
                )
            
            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "result_count": len(results),
                    "results": results,
                },
            )
        
        except httpx.TimeoutException:
            return ToolResult(
                success=False,
                error=f"Web search timed out for query: '{query}'",
            )
        except httpx.HTTPStatusError as exc:
            return ToolResult(
                success=False,
                error=f"HTTP {exc.response.status_code} from search API",
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"Web search failed: {type(exc).__name__}: {exc}",
            )
    
    @staticmethod
    def _parse_ddg_response(data: dict, max_results: int) -> list[dict]:
        """
        Normalize DuckDuckGo's response into a clean list.

        DuckDuckGo returns data in several fields:
        - AbstractText: A summary (often from Wikipedia)
        - RelatedTopics: A list of related results
        - Results: Direct answer results

        We merge all of these into a uniform list of
        {"title": ..., "snippet": ..., "url": ...} dicts.
        This normalization means the agent sees the same format
        regardless of which search provider we use tomorrow.
        """
        results = []

        # 1. Abstract (usually a Wikipedia summary)
        if data.get("AbstractText"):
            results.append({
                "title": data.get("Heading", "Summary"),
                "snippet": data["AbstractText"],
                "url": data.get("AbstractURL", ""),
            })
        
        # 2. Related Topics (main search results)
        for topic in data.get("RelatedTopics", []):
            if len(results) >= max_results:
                break

            # Some topics are groups (nested), some are direct results
            if "Text" in topic:
                results.append({
                    "title": topic.get("Text", "")[:80],
                    "snippet": topic.get("Text", ""),
                    "url": topic.get("FirstURL", ""),
                })
            elif "Topics" in topic:
                # Nested group — flatten first few
                for sub in topic["Topics"][:2]:
                    if len(results) >= max_results:
                        break
                    results.append({
                        "title": sub.get("Text", "")[:80],
                        "snippet": sub.get("Text", ""),
                        "url": sub.get("FirstURL", ""),
                    })

        return results[:max_results]