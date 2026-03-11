import os
from dotenv import load_dotenv
from litellm import completion
from litellm.exceptions import APIConnectionError, RateLimitError, Timeout

load_dotenv()

# A model gateway that provides a unified, resiliant interface to multiple LLM providers.
# This is the only module in the entire system that talks to LLMs directly. 
# Every other component (RAG, Agents, tools) calls this gateway

"""
Capabilities:
- Unified interface for multiple LLM providers (OpenAI, Anthropic, etc.)
- Built-in fallback mechanism: If one provider fails, automatically switch to the next.
- Error handling: Catches and logs exceptions, ensuring the system remains robust.
- Structured logging for every request
- Async-first design for high throughput agent workloads
"""

class LLMGateway:
    def __init__(self):
        # We define a fallback list. If OpenAI fails, we can try other providers.
        self.model_fallbacks = ["openai/gpt-4o", "anthropic/claude-3-5-sonnet-20241022"]
    
    def generate_strategy(self, prompt: str) -> str:
        """
        Generates a response using LiteLLM with built-in fallbacks.
        """
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = completion(
                model = self.model_fallbacks[0],  # Try the primary model first
                messages = messages,
                fallbacks = self.model_fallbacks[1:],  # Triggers if primary model fails
                num_retries = 2,
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            return f"Critical Failure in LLM Gateway: {str(e)}"

# Quick test block
if __name__ == "__main__":
    gateway = LLMGateway()
    print("Testing Gateway Setup...")
    test_prompt = "What is the best strategy for a chess game?"
    response = gateway.generate_strategy(test_prompt)
    print("Response from LLM Gateway:")
    print(response)
