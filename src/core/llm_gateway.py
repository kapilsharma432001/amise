"""
AMISE - LLM gateway
A production-grade model gateway that provides a unified, resilient interface to multiple LLM providers. This is the only module
that talks to the LLMs directly, every other module - RAG, tools, agents - calls this gateway.

Key Capabilities:
- LiteLLM gives you one single interface that works for every provider, the response format - response.choices[0].message.content - is also same
regardless of the provider, LiteLLM translates it internally.
- automatic falllback from primary to seconday model
- Exponential backoff 
- Structured logging for every request (latency, tokens, cost)
- Async-first design for high throughput agent workloads

What is LiteLLM?


Open source python library and proxy server that standardizes interactions with over 100 Large Language Models (LLMs) from providers like OpenAI, Antropic, Bedrock, and Gemini into a single API format.
It simplifies switching between models, handles authentication/load balancing, and provides cost tracking.
Think of it like this, if you want to call OpenAI, you write:
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model = "gpt-4o-mini",
    messages = [{"role": "user", "content": "Hello"}]
    )
    
print(response.choices[0].message.content)

 And tomorrow, your team says, “switch to Anthropic”, you rewrite everything:
from anthropic import Anthropic
client = Anthropic()

response = client.message.create(
    model = "claude-3-haiku-20240307",
    max_tokens = 1024,
    messages = [{"role": "User", "content": "Hello"}]
    )
    
print(response.content[0].text)

Notice the problem, different import, different method name, different response structure (response.choices[0].message.content) vs (response.content[0].text). 
If you have used these calls in 50 files across your project, switching providers means rewriting 50 files across your project. This is a provider-coupled code - your business logic is tightly bound to one vendor’s SDK.
What LiteLLM does?
LiteLLM gives you one single interface that works for every provider:
import litellm

# Call OpenAI
response = await litellm.acompletion(
    model = "gpt-4o-mini",
    messages = [{"role": "user", "content": "Hello"}]
    )

# Call Anthropic - SAME code, just change the model string
response = await litellm.autocompletion(
    model = "claude-3-haiku-20240307",
    messages = [{"role": "user", "content": "Hello"}]
    )

# Call Mistral - again, SAME code
response = await litellm.acompletion(
    model = "mistral/mistral-large-latest",
    messages = [{"role": "user", "content": "hello"}]
    )

The response format is also always the same - response.choices[0].message.content - regardless of which provider actually served the request. LiteLLM translates internally.

"""

import os
import time
from typing import Any, Dict, List, Optional

import litellm
import structlog
from dotenv import load_dotenv

# tenacity is a popular, general-purpose library designed to simplify the task of adding retry logic to code
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)

# read .env file form os.environ
load_dotenv()

# Supress LiteLLM's verbose internal logs; we use our own logging
litellm.supress_debug_info = True
litellm.drop_params  = True

# Initialize structured logger for this module
logger = structlog.get_logger(__name__)


# CONFIGURATION: Centralized, immutable settings
class GatewayConfig:
    PRIMARY_MODEL: str = os.getenv("PRIMARY_MODEL", "gpt-4o-mini")
    FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "claude-3-haiku-20240307")
    TIMEOUT: int = int(os.getenv("LLM_REQUEST_TIMEOUT", "30"))  # seconds
    MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))

    # cost guardrails - refuse any single call that would exceed this
    MAX_TOKENS_HARD_LIMIT: int = int(os.getenv("MAX_TOKENS_HARD_LIMIT", "4096"))


# Custom exception for LLM errors - fine grained error taxonomy
class LLMGatewayError(Exception):
    """Base exception for all gateway failures."""
    pass

class LLMProviderError(LLMGatewayError):
    """Raised when all the LLM providers (primary + fallback) have failed."""
    pass

class LLMConfigError(LLMGatewayError):
    """Raised for the configuration issues (missing keys, invalid model names)."""
    pass

# The single entrypoint for all the LLM intersactions in the AMISE system
class LLMGateway:
    """
    The single entrypoint for all the LLM interactions in the AMISE system.

    Architecture Pattern: ** Facade + Strategy **
    - Facade: Hide the complexity of multiple LLM providers behind a single 'generate()' method.
    - Strategy: The retry and fallback logic can be swapped without changing the ccalling code.
    
    Usage:
        gateway = LLMGateway()
        response = await gateway.generate(
            prompt = "Analyze the competitive landascape for EV companies.",
            system_message = "You are a senior market analyst.",
            )
        print(response["content"])
    """

    def __init__(self, config: Optional[GatewayConfig] = None):
        """
        Initialize the gateway with standard configuration.

        Args:
            config (GatewayConfig, optional): Custom configuration for the gateway. If None, defaults will be used. 
            Defaults to environment based config.
            This pattern enables dependency injection for testing.

            Dependency Injection (DI) is a software design pattern that decpuples classes from their depndencies,
            allowing objects to receive their required dependencies from an external source rather than creating them internally.

        """
        self.config = config or GatewayConfig()
        self._validate_environment()

        logger.info(
            "llm_gateway.initialized",
            primary_model = self.config.PRIMARY_MODEL,
            fallback_model = self.config.FALLBACK_MODEL,
            timeout = self.config.TIMEOUT,
        )

        def _validate_environment(self):
            """
            Fail fast: check that required API keys exist at the startup,
            not at the first LLM call which could be minutes later
            """

            required_key = []

            if "gpt" in self.config.PRIMARY_MODEL.lower() or "gpt" in self.config.FALLBACK_MODEL.lower():
                required_key.append("OPENAI_API_KEY")
            if "claude" in self.config.PRIMARY_MODEL.lower() or "claude" in self.config.FALLBACK_MODEL.lower():
                required_key.append("ANTHROPIC_API_KEY")

            missing = [key for key in required_key if not os.getenv(key)]

            if missing:
                raise LLMConfigError(
                    f"Missing required API keys: {missing}.",
                    f"Add them to your .env file."
                )
            


        