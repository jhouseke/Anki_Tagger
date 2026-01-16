"""
Unified embedding client supporting OpenRouter and Ollama providers.
"""
import os
from typing import List, Union, Optional
from enum import Enum

try:
    from openrouter import OpenRouter
except ImportError:
    OpenRouter = None

try:
    from ollama import embed, EmbedResponse, Client
except ImportError:
    embed = None
    EmbedResponse = None
    Client = None


class EmbeddingProvider(Enum):
    """Supported embedding providers."""
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"


class EmbeddingClient:
    """
    Unified client for generating embeddings from multiple providers.
    
    Supports:
    - OpenRouter: Access to various embedding models through OpenRouter API
    - Ollama: Local embedding models (e.g., nomic-embed-text)
    """
    
    def __init__(
        self,
        provider: Union[str, EmbeddingProvider] = EmbeddingProvider.OPENROUTER,
        model: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize the embedding client.
        
        Args:
            provider: Provider to use ('openrouter' or 'ollama')
            model: Model name to use (e.g., 'text-embedding-ada-002' for OpenRouter, 
                   'nomic-embed-text' for Ollama)
            api_key: API key for OpenRouter (not needed for Ollama)
            base_url: Base URL for Ollama (defaults to http://localhost:11434)
            max_tokens: Maximum tokens for text truncation (defaults to 2048 for Ollama, None for OpenRouter)
        """
        if isinstance(provider, str):
            provider = EmbeddingProvider(provider.lower())
        
        self.provider = provider
        self.model = model
        # Only get API key from environment if using OpenRouter and not explicitly provided
        if provider == EmbeddingProvider.OPENROUTER:
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        else:
            self.api_key = api_key  # For Ollama, API key is not needed
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Set max_tokens for truncation (conservative default for Ollama)
        if max_tokens is None:
            if provider == EmbeddingProvider.OLLAMA:
                # For Ollama, use conservative 512 tokens as default
                # Many Ollama embedding models (like nomic-embed-text) have 512 token limit
                # At ~6 chars/token, this gives ~3072 characters
                self.max_tokens = 512
            else:
                self.max_tokens = None
        else:
            self.max_tokens = max_tokens
        
        # Initialize provider-specific clients
        if self.provider == EmbeddingProvider.OPENROUTER:
            if OpenRouter is None:
                raise ImportError(
                    "openrouter package is required for OpenRouter provider. "
                    "Install it with: pip install openrouter"
                )
            if not self.api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY environment variable or api_key parameter is required"
                )
            # Don't store client - create fresh instance each time (OpenRouter client can't be reused after context manager exits)
            # self.client = OpenRouter(api_key=self.api_key)
        
        elif self.provider == EmbeddingProvider.OLLAMA:
            if embed is None or Client is None:
                raise ImportError(
                    "ollama package is required for Ollama provider. "
                    "Install it with: pip install ollama"
                )
            # Initialize Ollama client with the configured base URL
            # self.base_url already has the default fallback to http://localhost:11434
            self.ollama_client = Client(host=self.base_url)
    
    def _truncate_text(self, text: str) -> str:
        """
        Truncate text to fit within token limits.
        
        Args:
            text: Input text to truncate
            
        Returns:
            Truncated text that fits within max_tokens limit
        """
        if self.max_tokens is None:
            return text
        
        # Calculate max_chars from max_tokens
        # Conservative estimate: ~4-6 characters per token for most tokenizers
        # Using 6 chars/token to be safe for models with sentencepiece tokenizers
        max_chars = self.max_tokens * 6
        
        if len(text) <= max_chars:
            return text
        
        # Truncate to max_chars, then find last word boundary
        truncated = text[:max_chars]
        
        # Try to truncate at word boundary
        last_space = truncated.rfind(' ')
        if last_space > max_chars * 0.8:  # Only truncate at word if we keep at least 80% of text
            truncated = truncated[:last_space]
        
        return truncated
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        if self.provider == EmbeddingProvider.OPENROUTER:
            # Create fresh OpenRouter client each time (can't be reused after context manager exits)
            with OpenRouter(api_key=self.api_key) as open_router:
                response = open_router.embeddings.generate(
                    input=text,
                    model=self.model
                )
                # Extract embedding from response while still in context manager
                # Response structure: response.data is a list of embedding objects
                # Each object has an 'embedding' attribute
                if hasattr(response, 'data') and response.data:
                    if hasattr(response.data[0], 'embedding'):
                        embedding = response.data[0].embedding
                    elif isinstance(response.data[0], dict) and 'embedding' in response.data[0]:
                        embedding = response.data[0]['embedding']
                    else:
                        raise ValueError(f"Unexpected response format: {response}")
                elif isinstance(response, dict) and 'data' in response:
                    if response['data'] and isinstance(response['data'][0], dict):
                        embedding = response['data'][0]['embedding']
                    else:
                        raise ValueError(f"Unexpected response format: {response}")
                else:
                    raise ValueError(f"Unexpected response format: {response}")
                # Return embedding (extracted while still in context manager)
                return embedding
        
        elif self.provider == EmbeddingProvider.OLLAMA:
            # Truncate text if necessary to fit within context limits
            truncated_text = self._truncate_text(text)
            
            try:
                # Use the configured client (always set with self.base_url)
                response: EmbedResponse = self.ollama_client.embed(
                    model=self.model,
                    input=truncated_text
                )
                if response.embeddings and len(response.embeddings) > 0:
                    return response.embeddings[0]
                else:
                    raise ValueError(f"Unexpected response format: {response}")
            except Exception as e:
                # If we still get an error, try with even more aggressive truncation
                if "context length" in str(e).lower() or "exceeds" in str(e).lower():
                    # Reduce to 50% of max_tokens and retry
                    if self.max_tokens and self.max_tokens > 512:
                        original_max = self.max_tokens
                        self.max_tokens = max(512, self.max_tokens // 2)
                        truncated_text = self._truncate_text(text)
                        self.max_tokens = original_max
                        
                        # Retry with the configured client
                        response: EmbedResponse = self.ollama_client.embed(
                            model=self.model,
                            input=truncated_text
                        )
                        if response.embeddings and len(response.embeddings) > 0:
                            return response.embeddings[0]
                raise
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing).
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of embedding vectors
        """
        if self.provider == EmbeddingProvider.OPENROUTER:
            # Create fresh OpenRouter client each time (can't be reused after context manager exits)
            with OpenRouter(api_key=self.api_key) as open_router:
                response = open_router.embeddings.generate(
                    input=texts,
                    model=self.model
                )
                # Extract embeddings from response while still in context manager
                # Response structure: response.data is a list of embedding objects
                # Each object has an 'embedding' attribute
                embeddings = []
                if hasattr(response, 'data') and response.data:
                    for item in response.data:
                        if hasattr(item, 'embedding'):
                            embeddings.append(item.embedding)
                        elif isinstance(item, dict) and 'embedding' in item:
                            embeddings.append(item['embedding'])
                elif isinstance(response, dict) and 'data' in response:
                    embeddings = [item['embedding'] for item in response['data'] if 'embedding' in item]
                
                if not embeddings:
                    raise ValueError(f"Unexpected response format: {response}")
                # Return embeddings (extracted while still in context manager)
                return embeddings
        
        elif self.provider == EmbeddingProvider.OLLAMA:
            # Use the configured client (always set with self.base_url)
            response: EmbedResponse = self.ollama_client.embed(
                model=self.model,
                input=texts
            )
            if response.embeddings:
                return response.embeddings
            else:
                raise ValueError(f"Unexpected response format: {response}")
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")


# Convenience function for backward compatibility
def get_embedding(
    text: str,
    engine: str = "text-embedding-ada-002",
    provider: Union[str, EmbeddingProvider] = EmbeddingProvider.OPENROUTER,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> List[float]:
    """
    Generate embedding for a single text (backward compatible interface).
    
    Args:
        text: Input text to embed
        engine: Model name (default: 'text-embedding-ada-002')
        provider: Provider to use ('openrouter' or 'ollama')
        api_key: API key for OpenRouter (optional, uses env var if not provided)
        base_url: Base URL for Ollama (optional, defaults to http://localhost:11434)
        
    Returns:
        List of floats representing the embedding vector
    """
    client = EmbeddingClient(
        provider=provider,
        model=engine,
        api_key=api_key,
        base_url=base_url
    )
    return client.get_embedding(text)
