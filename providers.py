"""
OpenAI model provider configuration.
Single provider approach as specified in PRP.
"""

from typing import Optional
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
import openai
from settings import settings

def get_llm_model(model_choice: Optional[str] = None) -> OpenAIModel:
    """Get OpenAI LLM model configuration."""
    model_name = model_choice or settings.llm_model
    
    provider = OpenAIProvider(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key
    )
    return OpenAIModel(model_name, provider=provider)

def get_embedding_client() -> openai.AsyncOpenAI:
    """Get OpenAI embedding client."""
    return openai.AsyncOpenAI(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key
    )

def get_embedding_model() -> str:
    """Get embedding model name."""
    return settings.embedding_model