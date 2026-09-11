"""
llm_client.py — Provider-agnostic LLM abstraction layer.

Supports Groq, Google Gemini, and OpenAI with auto-detection based on
available API keys. All providers are swappable without changing
the core RAG pipeline.

Environment Variables:
    LLM_PROVIDER      : Force a provider ('groq', 'gemini', or 'openai'). Optional.
    GROQ_API_KEY      : Groq API key.
    GEMINI_API_KEY    : Google Gemini API key.
    OPENAI_API_KEY    : OpenAI API key.
    LLM_MODEL_NAME    : Override the default model name for the active provider.
"""

import os
import sys
from abc import ABC, abstractmethod
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# Load .env file from the project root so API keys are available
load_dotenv()


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1 token per 4 characters."""
    return max(1, len(text) // 4)


class BaseLLMClient(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> dict:
        """
        Generate a completion from the LLM.

        Returns:
            dict: {
                "text": str,
                "usage": {
                    "prompt_tokens": int,
                    "completion_tokens": int,
                    "total_tokens": int
                }
            }
        """
        pass


class GroqClient(BaseLLMClient):
    """Groq LLM client via the OpenAI-compatible SDK endpoint."""

    def __init__(self, model_name: str = None):
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if api_key.startswith("="):
            api_key = api_key.lstrip("=").strip()

        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")

        if not model_name:
            env_model = os.getenv("GROQ_MODEL_NAME") or os.getenv("LLM_MODEL_NAME")
            if env_model and not ("gemini" in env_model.lower() or "gpt" in env_model.lower()):
                model_name = env_model
            else:
                model_name = "groq/compound"

        super().__init__(model_name)

        from openai import OpenAI
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=3))
    def generate(self, prompt: str, system_prompt: str = "") -> dict:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
            )
        except Exception as e:
            if "model_not_found" in str(e) or "does not exist" in str(e):
                # Fallback in case the requested model (e.g. legacy llama-3.3-70b-versatile) was deprecated by Groq
                response = self._client.chat.completions.create(
                    model="groq/compound",
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1024,
                )
            else:
                raise

        text = response.choices[0].message.content or ""

        # Extract token usage from response metadata
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if hasattr(response, "usage") and response.usage:
            usage["prompt_tokens"] = response.usage.prompt_tokens or 0
            usage["completion_tokens"] = response.usage.completion_tokens or 0
            usage["total_tokens"] = response.usage.total_tokens or 0
        else:
            # Fallback estimation
            usage["prompt_tokens"] = estimate_tokens(system_prompt + prompt)
            usage["completion_tokens"] = estimate_tokens(text)
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

        return {"text": text, "usage": usage}


class GeminiClient(BaseLLMClient):
    """Google Gemini LLM client via the google-genai SDK."""

    def __init__(self, model_name: str = None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")

        if not model_name:
            env_model = os.getenv("GEMINI_MODEL_NAME") or os.getenv("LLM_MODEL_NAME")
            if env_model and "gemini" in env_model.lower():
                model_name = env_model
            else:
                model_name = "gemini-3.5-flash-lite"

        super().__init__(model_name)

        from google import genai
        self._client = genai.Client(api_key=api_key)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=3))
    def generate(self, prompt: str, system_prompt: str = "") -> dict:
        from google.genai import types

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt if system_prompt else None,
                    temperature=0.3,
                    max_output_tokens=1024,
                ),
            )
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "NOT_FOUND" in str(e) or "404" in str(e):
                response = self._client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt if system_prompt else None,
                        temperature=0.3,
                        max_output_tokens=1024,
                    ),
                )
            else:
                raise

        text = response.text or ""

        # Extract token usage from response metadata
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            meta = response.usage_metadata
            usage["prompt_tokens"] = getattr(meta, "prompt_token_count", 0) or 0
            usage["completion_tokens"] = getattr(meta, "candidates_token_count", 0) or 0
            usage["total_tokens"] = getattr(meta, "total_token_count", 0) or 0
        else:
            # Fallback estimation
            usage["prompt_tokens"] = estimate_tokens(system_prompt + prompt)
            usage["completion_tokens"] = estimate_tokens(text)
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

        return {"text": text, "usage": usage}


class OpenAIClient(BaseLLMClient):
    """OpenAI LLM client via the openai SDK."""

    def __init__(self, model_name: str = None):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")

        if not model_name:
            env_model = os.getenv("OPENAI_MODEL_NAME") or os.getenv("LLM_MODEL_NAME")
            if env_model and ("gpt" in env_model.lower() or "o1" in env_model.lower() or "o3" in env_model.lower()):
                model_name = env_model
            else:
                model_name = "gpt-4o-mini"

        super().__init__(model_name)

        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=3))
    def generate(self, prompt: str, system_prompt: str = "") -> dict:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )

        text = response.choices[0].message.content or ""

        # Extract token usage from response metadata
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if hasattr(response, "usage") and response.usage:
            usage["prompt_tokens"] = response.usage.prompt_tokens or 0
            usage["completion_tokens"] = response.usage.completion_tokens or 0
            usage["total_tokens"] = response.usage.total_tokens or 0
        else:
            # Fallback estimation
            usage["prompt_tokens"] = estimate_tokens(system_prompt + prompt)
            usage["completion_tokens"] = estimate_tokens(text)
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

        return {"text": text, "usage": usage}


import threading


class FallbackLLMClient(BaseLLMClient):
    """
    Multi-provider LLM client wrapper that executes queries against the primary
    provider (e.g. Groq) and seamlessly routes to fallback providers (e.g. Gemini, OpenAI)
    whenever encountering rate limits, quota issues, or API retries exhaustion.
    """

    def __init__(self, clients: list[BaseLLMClient]):
        if not clients:
            raise ValueError("FallbackLLMClient requires at least one initialized LLM client.")
        self.clients = clients
        super().__init__(model_name=clients[0].model_name)

    def generate(self, prompt: str, system_prompt: str = "") -> dict:
        last_exception = None
        for i, client in enumerate(self.clients):
            client_name = type(client).__name__.replace("Client", "")
            try:
                result = client.generate(prompt=prompt, system_prompt=system_prompt)
                self.model_name = client.model_name
                return result
            except Exception as e:
                last_exception = e
                if i + 1 < len(self.clients):
                    next_client = self.clients[i + 1]
                    next_name = type(next_client).__name__.replace("Client", "")
                    print(f"\n[FALLBACK] {client_name} rate limited. Routing to {next_name}...", flush=True)
                else:
                    raise last_exception


class RoundRobinLLMClient(BaseLLMClient):
    """
    Round-robin load-balancing LLM client that cycles through available providers
    in configurable request chunks (default: 5) while preserving full multi-provider
    fallback safety if an active provider is rate-limited.
    """

    _lock = threading.Lock()
    _request_counter = 0

    def __init__(self, providers: list[BaseLLMClient], chunk_size: int = 5):
        if not providers:
            raise ValueError("RoundRobinLLMClient requires at least one provider client.")
        self.providers = providers
        self.chunk_size = chunk_size
        super().__init__(model_name=providers[0].model_name)

    def generate(self, prompt: str, system_prompt: str = "") -> dict:
        with RoundRobinLLMClient._lock:
            RoundRobinLLMClient._request_counter += 1
            req_num = RoundRobinLLMClient._request_counter
            # Determine which provider to use for this request chunk
            target_idx = ((req_num - 1) // self.chunk_size) % len(self.providers)
            target_provider = self.providers[target_idx]
            provider_name = type(target_provider).__name__.replace("Client", "")

            # Log rotation status
            if (req_num - 1) % self.chunk_size == 0 and req_num > 1:
                print(f"[ROUND-ROBIN] Switching to {provider_name} for request #{req_num}", flush=True)
            else:
                print(f"[ROUND-ROBIN] Using {provider_name} for request #{req_num}", flush=True)

        # Attempt call on target provider, with fallback across remaining providers
        last_exception = None
        for attempt_offset in range(len(self.providers)):
            active_idx = (target_idx + attempt_offset) % len(self.providers)
            active_client = self.providers[active_idx]
            active_name = type(active_client).__name__.replace("Client", "")

            try:
                result = active_client.generate(prompt=prompt, system_prompt=system_prompt)
                self.model_name = active_client.model_name
                return result
            except Exception as e:
                last_exception = e
                if attempt_offset + 1 < len(self.providers):
                    next_idx = (target_idx + attempt_offset + 1) % len(self.providers)
                    next_client = self.providers[next_idx]
                    next_name = type(next_client).__name__.replace("Client", "")
                    print(f"\n[FALLBACK] {active_name} rate limited. Routing to {next_name}...", flush=True)
                else:
                    print(f"\n[ERROR] All available providers ({[type(c).__name__ for c in self.providers]}) failed.", flush=True)
                    raise last_exception


def get_llm_client(model_name: str = None) -> BaseLLMClient:
    """
    Factory function to instantiate LLM clients with round-robin load balancing
    and multi-provider fallback.

    Initializes available clients based on present API keys:
        - GroqClient (GROQ_API_KEY)
        - GeminiClient (GEMINI_API_KEY)
        - OpenAIClient (OPENAI_API_KEY)

    Primary client is determined by LLM_PROVIDER (default: 'groq').
    Returns a RoundRobinLLMClient wrapping all available providers.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()
    chunk_size = int(os.getenv("ROUND_ROBIN_CHUNK_SIZE", "5"))

    available_clients = []

    # 1. Initialize Groq
    if os.getenv("GROQ_API_KEY"):
        try:
            available_clients.append(GroqClient(model_name=model_name if provider == "groq" else None))
        except Exception as e:
            print(f"[WARN] Could not initialize GroqClient: {e}", flush=True)

    # 2. Initialize Gemini
    if os.getenv("GEMINI_API_KEY"):
        try:
            available_clients.append(GeminiClient(model_name=model_name if provider == "gemini" else None))
        except Exception as e:
            print(f"[WARN] Could not initialize GeminiClient: {e}", flush=True)

    # 3. Initialize OpenAI
    if os.getenv("OPENAI_API_KEY"):
        try:
            available_clients.append(OpenAIClient(model_name=model_name if provider == "openai" else None))
        except Exception as e:
            print(f"[WARN] Could not initialize OpenAIClient: {e}", flush=True)

    if not available_clients:
        print("Error: No LLM provider configured or valid API keys found.", flush=True)
        print("Set at least one of: GROQ_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY", flush=True)
        sys.exit(1)

    # Sort so that preferred provider is first in the list
    if provider == "gemini":
        available_clients.sort(key=lambda c: 0 if isinstance(c, GeminiClient) else 1)
    elif provider == "openai":
        available_clients.sort(key=lambda c: 0 if isinstance(c, OpenAIClient) else 1)
    else:  # default 'groq'
        available_clients.sort(key=lambda c: 0 if isinstance(c, GroqClient) else 1)

    if len(available_clients) == 1:
        return available_clients[0]

    return RoundRobinLLMClient(providers=available_clients, chunk_size=chunk_size)
