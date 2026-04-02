"""
AMISE - Tool registry (MCP Foundation)
===============================
Implements the core patterns behind the Model Context Protocol:
1. Tool Schema Declaration: JSON schema for inputs
2. Tool Discovery: Agents list availabe tools
3. Standarized Invocation: Uniform call interface with safety

This module contains ZERO business logic. It only defines the
contract that all tools must follow. Concrete tools (web search,
finance, etc.) live in separate modules and register themselves here.

Design Patterns:
    - Abstract Base Class (BaseTool): enforces interface contract
    - Registry Pattern: decouple tool registration from tool usage
    - Strategy Pattern: agents pick tools at runtime by name
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)

# TOOL SCHEMA: How a tool describe itself to LLM?
@dataclass
class ToolParameter:
    """
    A single input parameter for a tool.

    This maps directly to a property in JSON Schema, which is the
    format that OpenAI, Anthropic, and MCP all use for tool definitions.
    """

    name: str
    type: str           # "string", "number", "integer", "boolean"
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolSchema:
    """
    Complete description of a tool's interface.

    Why is this a separate class and not just a dict?
    - Validated at registration time, not at call time
    - IDE autocompletion when building tool definitions
    - Single source of truth: schema lives WITH the tool code

    The to_openai_schema() method converts this to the format that
    LLMs expect during function calling. This is the bridge between
    our internal representation and the LLM's expected format.
    """

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)

    def to_openai_schema(self) -> dict:
        """
        Convert to OpenAI function-calling format.

        This same format works with Anthropic (via LiteLLM) and
        is what MCP standardizes. By generating it from our dataclass,
        we maintain a single source of truth.

        Output shape:
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "...",
                "parameters": {
                    "type": "object",
                    "properties": { ... },
                    "required": [ ... ]
                }
            }
        }
        """
        properties = {}
        required_params = []

        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description,
            }
            if param.required:
                required_params.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required_params,
                },
            },
        }

# Tool result: standarized output from any tool
@dataclass
class ToolResult:
    """
    Uniform response from any tool invocation.

    Why not just return raw data?
    - The agent needs to know if the call SUCCEEDED or FAILED
      without parsing provider-specific error formats.
    - Latency tracking enables performance monitoring per tool.
    - The 'tool_name' field lets us log and debug which tool
      produced which result in a multi-tool agent run.

    This is the "response envelope" pattern — metadata wraps the payload.
    """

    success: bool
    data: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    tool_name: str = ""

# Base Tool: Abstacr base class for all AMISE tools
class BaseTool(ABC):
   """
   Any tool (web search, finance, database lookup, email sender) must implement exactly two methods:
    1. get_schema(): declare what you are what inputs you take
    2. execute(): do the actual work and return a ToolResult
   """

   @abstractmethod
   def get_schema(self) -> ToolSchema:
      """Return the tool's schema. Called once at registration."""
      ...
    
   @abstractmethod
   async def execute(self, **kwargs) -> ToolResult:
      """Execute the tool with validated arguements"""
      ...

# 4. Tool Registry: The center hub - where the tools register and agents discover them.
class ToolRegistry:
    """
    Central registry where tools register and agents discover them.

    This IS the MCP server concept, simplified:
    - MCP Server exposes tools → our Registry holds tools
    - MCP Client lists tools  → our list_tools() method
    - MCP Client calls tools  → our invoke() method

    The registry adds three things raw tool calls don't have:
    1. Discovery   — agents can list all available tools dynamically
    2. Safety      — timeout wrapping prevents runaway tool calls
    3. Observability — every invocation is logged with latency
    """

    def __init__(self, default_timeout: float = 30.0):
        """
        Args:
            default_timeout: Max seconds for any tool call.
                            Prevents a hung API from blocking the agent forever.
        """
        self._tools: dict[str, BaseTool] = {}
        self._timeout = default_timeout

    def register(self, tool: BaseTool) -> None:
        """
        Register a tool. Validates schema on registration, not on call.

        Why validate here?
        - If a tool has a malformed schema, we want to know at STARTUP,
          not when a user is waiting for the agent to respond.
        - This is the "fail fast" principle applied to tool registration.
        """
        schema = tool.get_schema()

        if not schema.name or not schema.description:
            raise ValueError(
                f"Tool schema must have a name and description. Got: {schema}"
            )

        if schema.name in self._tools:
            logger.warning(
                "tool_registry.duplicate_registration",
                tool=schema.name,
            )

        self._tools[schema.name] = tool
        logger.info(
            "tool_registry.registered",
            tool=schema.name,
            param_count=len(schema.parameters),
        )

    def list_tools(self) -> list[ToolSchema]:
        """
        Return schemas of all registered tools.

        The agent calls this to know what tools are available,
        then passes these schemas to the LLM for function calling.
        """
        return [tool.get_schema() for tool in self._tools.values()]

    def get_llm_tool_definitions(self) -> list[dict]:
        """
        Return all tool schemas in OpenAI function-calling format.

        This is what you pass directly to litellm.acompletion(tools=...).
        The LLM reads these definitions and decides which tool to call.
        """
        return [
            tool.get_schema().to_openai_schema()
            for tool in self._tools.values()
        ]

    async def invoke(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Invoke a tool by name with the given arguments.

        Three layers of protection:
        1. Unknown tool check    — returns error, doesn't crash
        2. Timeout wrapper       — kills hung tools after N seconds
        3. Exception catch-all   — no tool error crashes the agent

        Why catch ALL exceptions?
        Because tools call external APIs we don't control. A finance API
        might return malformed JSON. A search API might change its response
        format. The AGENT must keep running — it can try another tool or
        tell the user that one tool failed.
        """
        # Layer 1: Check tool exists
        if tool_name not in self._tools:
            return ToolResult(
                success=False,
                error=f"Unknown tool: '{tool_name}'. Available: {list(self._tools.keys())}",
                tool_name=tool_name,
            )

        tool = self._tools[tool_name]
        start = time.perf_counter()

        try:
            # Layer 2: Timeout protection
            result = await asyncio.wait_for(
                tool.execute(**kwargs),
                timeout=self._timeout,
            )
            result.latency_ms = (time.perf_counter() - start) * 1000
            result.tool_name = tool_name

            logger.info(
                "tool_registry.invocation_success",
                tool=tool_name,
                latency_ms=round(result.latency_ms, 2),
            )
            return result

        except asyncio.TimeoutError:
            # Layer 2 triggered: tool took too long
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(
                "tool_registry.invocation_timeout",
                tool=tool_name,
                timeout_s=self._timeout,
            )
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' timed out after {self._timeout}s",
                tool_name=tool_name,
                latency_ms=elapsed,
            )

        except Exception as exc:
            # Layer 3: Catch-all for unexpected errors
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(
                "tool_registry.invocation_error",
                tool=tool_name,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return ToolResult(
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                tool_name=tool_name,
                latency_ms=elapsed,
            )
