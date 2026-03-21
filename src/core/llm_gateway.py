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
