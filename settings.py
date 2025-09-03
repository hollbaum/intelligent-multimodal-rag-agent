"""
Environment configuration for RAG Knowledge Graph Agent.
Minimal setup focused on PostgreSQL + Neo4j + OpenAI.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, ConfigDict
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Essential settings for RAG Knowledge Graph Agent."""
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # OpenAI LLM Configuration
    llm_provider: str = Field(default="openai", description="LLM provider (openai only)")
    llm_api_key: str = Field(..., description="OpenAI API key for LLM")
    llm_model: str = Field(default="gpt-4o-mini", description="OpenAI model with tool support")
    llm_base_url: str = Field(default="https://api.openai.com/v1", description="OpenAI API base URL")
    
    # OpenAI Embedding Configuration  
    embedding_provider: str = Field(default="openai", description="Embedding provider (openai only)")
    embedding_api_key: str = Field(..., description="OpenAI API key for embeddings")
    embedding_model: str = Field(default="text-embedding-3-small", description="OpenAI embedding model")
    embedding_base_url: str = Field(default="https://api.openai.com/v1", description="OpenAI embedding API")
    
    # Database Configuration
    database_url: str = Field(..., description="PostgreSQL connection with pgvector")
    neo4j_uri: str = Field(..., description="Neo4j bolt connection URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(..., description="Neo4j password")
    
    # Application Settings
    app_env: str = Field(default="development", description="Environment")
    log_level: str = Field(default="INFO", description="Logging level")
    debug: bool = Field(default=False, description="Debug mode")
    max_retries: int = Field(default=3, description="Max retry attempts")
    timeout_seconds: int = Field(default=30, description="Default timeout")
    
    # RAG-specific Configuration
    chunk_size: int = Field(default=800, description="Document chunk size")
    chunk_overlap: int = Field(default=150, description="Chunk overlap size")
    vector_dimension: int = Field(default=1536, description="Vector dimension for embeddings")
    max_search_results: int = Field(default=10, description="Max search results")
    
    @field_validator("llm_api_key", "embedding_api_key", "neo4j_password")
    @classmethod
    def validate_required_keys(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Required API key/password cannot be empty")
        return v

def load_settings() -> Settings:
    """Load settings with error handling."""
    try:
        return Settings()
    except Exception as e:
        error_msg = f"Configuration error: {e}"
        if "api_key" in str(e).lower():
            error_msg += "\nEnsure OpenAI API keys are set in .env file"
        if "database_url" in str(e).lower():
            error_msg += "\nEnsure DATABASE_URL is configured for PostgreSQL"
        if "neo4j" in str(e).lower():
            error_msg += "\nEnsure Neo4j connection details are configured"
        raise ValueError(error_msg) from e

settings = load_settings()