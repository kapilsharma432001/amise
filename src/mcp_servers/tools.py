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
    

### STOCK QUOTE TOOL
class StockQuoteTool(BaseTool):
    """
    Fetches stock market data from Yahoo Fianance's public endpoint.
    """
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="stock_quote",
            description=(
                "Get current stock market data for a company including "
                "price, change, volume, market cap, and 52-week range. "
                "Use the stock ticker symbol (e.g., 'TSLA' for Tesla, "
                "'RELIANCE.NS' for Reliance on NSE)."
            ),
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'INFY.NS')",
                ),
            ],
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """
        Fetch stock quote for a given ticker symbol.
        """
        symbol = kwargs.get("symbol")

        if not symbol or not symbol.strip():
            return ToolResult(
                success=False,
                error="Parameter 'symbol' is required (e.g., 'TSLA').",
            )

        symbol = symbol.strip().upper()

        try:
            async with httpx.AsyncClient(
                headers=DEFAULT_HEADERS, timeout=HTTP_TIMEOUT
            ) as client:
                # Yahoo Finance v8 quote endpoint
                response = await client.get(
                    f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                    params={
                        "interval": "1d",
                        "range": "5d",  # 5 days of data
                    },
                )
                response.raise_for_status()
                data = response.json()

            quote = self._parse_yahoo_response(data, symbol)
            if not quote:
                return ToolResult(
                    success=False,
                    error=f"No data found for symbol '{symbol}'. Verify the ticker.",
                )

            return ToolResult(success=True, data=quote)

        except httpx.TimeoutException:
            return ToolResult(
                success=False,
                error=f"Finance API timed out for symbol: '{symbol}'",
            )
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 404:
                return ToolResult(
                    success=False,
                    error=f"Symbol '{symbol}' not found. Check the ticker.",
                )
            return ToolResult(
                success=False,
                error=f"Finance API returned HTTP {status} for '{symbol}'",
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"Stock quote failed: {type(exc).__name__}: {exc}",
            )
    
    @staticmethod
    def _parse_yahoo_response(data: dict, symbol: str) -> Optional[dict]:
        """
        Extract key financial metrics from Yahoo's response.

        Normalizes the deeply nested Yahoo response into a flat,
        readable dict. If any field is missing, we use None rather
        than crashing — financial APIs are notoriously inconsistent
        in what fields they return for different securities.
        """
        try:
            chart = data["chart"]["result"][0]
            meta = chart["meta"]

            current_price = meta.get("regularMarketPrice", 0)
            previous_close = meta.get("chartPreviousClose", 0)

            # Calculate change and percentage
            change = round(current_price - previous_close, 2) if previous_close else None
            change_pct = (
                round((change / previous_close) * 100, 2)
                if previous_close and change is not None
                else None
            )

            return {
                "symbol": symbol,
                "currency": meta.get("currency", "USD"),
                "current_price": current_price,
                "previous_close": previous_close,
                "change": change,
                "change_percent": change_pct,
                "day_high": meta.get("regularMarketDayHigh"),
                "day_low": meta.get("regularMarketDayLow"),
                "volume": meta.get("regularMarketVolume"),
                "exchange": meta.get("exchangeName", "Unknown"),
                "market_state": meta.get("marketState", "Unknown"),
            }
        except (KeyError, IndexError, TypeError):
            return None

def create_default_registry() -> ToolRegistry:
    """
    Factory function that creates a registry with all standard tools.

    Why a factory function?
    - Agents don't need to know which tools exist or how to instantiate them.
    - Adding a new tool = one line here. Zero changes in agent code.
    - In tests, you can create a registry with only mock tools.

    This is the Dependency Injection pattern:
    the agent receives a fully configured registry,
    it doesn't build one itself.
    """
    registry = ToolRegistry()

    registry.register(WebSearchTool())
    registry.register(StockQuoteTool())

    logger.info(
        "default_registry.created",
        tool_count=len(registry.list_tools()),
        tools=[s.name for s in registry.list_tools()],
    )

    return registry

async def _smoke_test():
    """
    End-to-end test: create registry → list tools → invoke each one.

    This validates:
    1. Tools register correctly
    2. Schemas generate valid LLM function definitions
    3. Tool invocation works with real API calls
    4. Error handling works (try an invalid symbol)
    """
    import json

    print("\n" + "=" * 60)
    print("  AMISE Tool Registry — Smoke Test")
    print("=" * 60)

    # Create registry with all tools
    registry = create_default_registry()

    # --- Test 1: List tools (what the LLM sees) ---
    print("\n📋 Registered Tools:")
    tool_defs = registry.get_llm_tool_definitions()
    for td in tool_defs:
        func = td["function"]
        params = list(func["parameters"]["properties"].keys())
        print(f"   • {func['name']}: {func['description'][:60]}...")
        print(f"     Params: {params}")

    # --- Test 2: Web Search ---
    print("\n🔍 Testing Web Search...")
    search_result = await registry.invoke("web_search", query="Python programming language")
    if search_result.success:
        print(f"   ✅ Found {search_result.data['result_count']} results "
              f"({search_result.latency_ms:.0f}ms)")
        for r in search_result.data["results"][:2]:
            print(f"      → {r['title'][:70]}")
    else:
        print(f"   ❌ Search failed: {search_result.error}")

    # --- Test 3: Stock Quote ---
    print("\n📈 Testing Stock Quote (AAPL)...")
    stock_result = await registry.invoke("stock_quote", symbol="AAPL")
    if stock_result.success:
        d = stock_result.data
        direction = "📈" if (d["change"] or 0) >= 0 else "📉"
        print(f"   ✅ {d['symbol']} @ {d['currency']} {d['current_price']} "
              f"{direction} {d['change']} ({d['change_percent']}%) "
              f"({stock_result.latency_ms:.0f}ms)")
    else:
        print(f"   ❌ Stock quote failed: {stock_result.error}")

    # --- Test 4: Error handling (invalid symbol) ---
    print("\n🧪 Testing Error Handling (invalid symbol)...")
    bad_result = await registry.invoke("stock_quote", symbol="ZZZZZZZZZ")
    print(f"   {'✅' if not bad_result.success else '❌'} "
          f"Correctly handled: {bad_result.error}")

    # --- Test 5: Unknown tool ---
    print("\n🧪 Testing Unknown Tool...")
    unknown_result = await registry.invoke("email_sender", to="test@example.com")
    print(f"   {'✅' if not unknown_result.success else '❌'} "
          f"Correctly handled: {unknown_result.error}")

    # --- Show LLM-ready schema ---
    print("\n📄 LLM-Ready Tool Definitions (what gets passed to the model):")
    print(json.dumps(tool_defs, indent=2)[:500] + "...\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(_smoke_test())