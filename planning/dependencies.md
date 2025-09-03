# RAG Knowledge Graph AI Assistant - Dependencies Configuration

## Overview

This document specifies the minimal, essential configuration for the RAG Knowledge Graph AI Assistant. The configuration follows the **"configure only what's needed"** principle, focusing on the three core requirements from the PRP:

1. **OpenAI API** for LLM and embeddings (single provider approach)
2. **PostgreSQL with pgvector** for semantic vector search
3. **Neo4j with Graphiti** for knowledge graph operations

## Project Structure

```
dependencies/
├── __init__.py
├── settings.py       # Environment configuration with pydantic-settings  
├── providers.py      # OpenAI model provider setup
├── dependencies.py   # Agent dependencies dataclass
├── agent.py         # Agent initialization with dependencies
├── .env.example     # Environment template
└── requirements.txt # Python dependencies
```

## Core Environment Variables

Based on PRP requirements, the agent needs these **essential** environment variables:

### LLM Configuration (OpenAI)
```bash
LLM_PROVIDER=openai
LLM_API_KEY=sk-your-openai-api-key-here
LLM_MODEL=gpt-4o-mini
LLM_BASE_URL=https://api.openai.com/v1
```

### Embedding Configuration (OpenAI)
```bash
EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=sk-your-openai-api-key-here  # Can be same as LLM_API_KEY
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_BASE_URL=https://api.openai.com/v1
```

### Database Configuration
```bash
# PostgreSQL with pgvector
DATABASE_URL=postgresql://raguser:ragpass123@localhost:5432/agentic_rag

# Neo4j for knowledge graph
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

### Application Settings
```bash
APP_ENV=development
LOG_LEVEL=INFO
DEBUG=false
MAX_RETRIES=3
TIMEOUT_SECONDS=30
```

## settings.py Configuration

```python
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
```

## providers.py Model Configuration

```python
"""
OpenAI model provider configuration.
Single provider approach as specified in PRP.
"""

from typing import Optional
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
import openai
from .settings import settings

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
```

## dependencies.py Agent Dependencies

```python
"""
Agent dependencies for RAG Knowledge Graph operations.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging
import asyncpg
from graphiti import Graphiti

logger = logging.getLogger(__name__)

@dataclass
class AgentDependencies:
    """
    Dependencies for RAG Knowledge Graph Agent.
    Handles PostgreSQL + Neo4j connections and runtime context.
    """
    
    # Session Context
    session_id: str
    user_id: Optional[str] = None
    
    # Search Configuration
    search_preferences: Dict[str, Any] = field(default_factory=dict)
    max_retries: int = 3
    timeout: int = 30
    debug: bool = False
    
    # Database Connections (initialized lazily)
    _pg_pool: Optional[asyncpg.Pool] = field(default=None, init=False, repr=False)
    _graphiti_client: Optional[Graphiti] = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        """Initialize default search preferences."""
        if not self.search_preferences:
            self.search_preferences = {
                "use_vector": True,
                "use_graph": True,
                "default_limit": 10,
                "min_score": 0.7
            }
    
    @property
    async def pg_pool(self) -> asyncpg.Pool:
        """Lazy initialization of PostgreSQL connection pool."""
        if self._pg_pool is None:
            from .settings import settings
            self._pg_pool = await asyncpg.create_pool(
                settings.database_url,
                min_size=5,
                max_size=20,
                max_queries=50000,
                max_inactive_connection_lifetime=300.0
            )
            logger.info("PostgreSQL connection pool initialized")
        return self._pg_pool
    
    @property
    async def graphiti_client(self) -> Graphiti:
        """Lazy initialization of Graphiti knowledge graph client."""
        if self._graphiti_client is None:
            from .settings import settings
            from .providers import get_embedding_client, get_embedding_model
            
            self._graphiti_client = Graphiti(
                uri=settings.neo4j_uri,
                user=settings.neo4j_user,
                password=settings.neo4j_password,
                embedder_client=get_embedding_client(),
                embedder_model=get_embedding_model()
            )
            logger.info("Graphiti client initialized")
        return self._graphiti_client
    
    async def cleanup(self):
        """Cleanup database connections."""
        if self._pg_pool:
            await self._pg_pool.close()
            logger.info("PostgreSQL pool closed")
        
        if self._graphiti_client:
            await self._graphiti_client.close()
            logger.info("Graphiti client closed")
    
    @classmethod
    def from_settings(cls, settings, session_id: str, **kwargs):
        """Create dependencies from settings."""
        return cls(
            session_id=session_id,
            max_retries=kwargs.get('max_retries', settings.max_retries),
            timeout=kwargs.get('timeout', settings.timeout_seconds),
            debug=kwargs.get('debug', settings.debug),
            **{k: v for k, v in kwargs.items() 
               if k not in ['max_retries', 'timeout', 'debug']}
        )
```

## agent.py Agent Initialization  

```python
"""
RAG Knowledge Graph AI Assistant - Main Agent
"""

import logging
from typing import Optional
from pydantic_ai import Agent

from .providers import get_llm_model
from .dependencies import AgentDependencies
from .settings import settings

logger = logging.getLogger(__name__)

# System prompt (provided by prompt-engineer subagent)
SYSTEM_PROMPT = """
You are an intelligent RAG assistant with knowledge graph capabilities.
[Full system prompt will be inserted here]
"""

# Initialize the agent
rag_agent = Agent(
    get_llm_model(),
    deps_type=AgentDependencies,
    system_prompt=SYSTEM_PROMPT,
    retries=settings.max_retries
)

# Tool registration will be handled by tool-integrator subagent
# Tools: vector_search, hybrid_search, graph_search, comprehensive_search,
#        list_documents, get_document, get_entity_relationships, get_entity_timeline

async def run_agent(
    prompt: str,
    session_id: str,
    user_id: Optional[str] = None,
    **dependency_overrides
) -> str:
    """
    Run the agent with automatic dependency management.
    """
    deps = AgentDependencies.from_settings(
        settings,
        session_id=session_id,
        user_id=user_id,
        **dependency_overrides
    )
    
    try:
        result = await rag_agent.run(prompt, deps=deps)
        return result.data
    finally:
        await deps.cleanup()
```

## .env.example Environment Template

```bash
# OpenAI LLM Configuration (REQUIRED)
LLM_PROVIDER=openai
LLM_API_KEY=sk-your-openai-api-key-here
LLM_MODEL=gpt-4o-mini
LLM_BASE_URL=https://api.openai.com/v1

# OpenAI Embedding Configuration (REQUIRED)
EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=sk-your-openai-api-key-here
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_BASE_URL=https://api.openai.com/v1

# Database Configuration (REQUIRED)
DATABASE_URL=postgresql://raguser:ragpass123@localhost:5432/agentic_rag
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Application Settings
APP_ENV=development
LOG_LEVEL=INFO
DEBUG=false
MAX_RETRIES=3
TIMEOUT_SECONDS=30

# RAG Configuration
CHUNK_SIZE=800
CHUNK_OVERLAP=150
VECTOR_DIMENSION=1536
MAX_SEARCH_RESULTS=10
```

## requirements.txt Dependencies

```txt
# Core Pydantic AI
pydantic-ai>=0.3.0
pydantic>=2.11.0
pydantic-settings>=2.10.0
python-dotenv>=1.1.0

# OpenAI Provider
openai>=1.90.0

# Database Drivers
asyncpg>=0.30.0  # PostgreSQL with pgvector
neo4j>=5.28.0    # Neo4j driver

# Knowledge Graph
graphiti>=0.1.13
graphiti-core>=0.12.4

# Async utilities
httpx>=0.28.0
aiofiles>=23.0.0

# Development and Testing
pytest>=8.4.0
pytest-asyncio>=1.0.0

# CLI Interface
rich>=14.0.0
click>=8.2.0

# Utilities
numpy>=2.3.0  # For vector operations
```

## Security Configuration

### API Key Management
- Use environment variables exclusively
- Never commit `.env` files to version control
- Validate all API keys on startup
- Support key rotation via environment updates

### Database Security
- Use connection pooling with proper limits
- Parameterized queries for SQL injection prevention
- Connection timeouts and proper cleanup
- SSL/TLS for database connections in production

### Input Validation
- Pydantic models for all tool parameters
- Query sanitization for database operations
- Rate limiting through connection pools
- Content validation for responses

## Connection Pool Configuration

### PostgreSQL Pool
- Min connections: 5
- Max connections: 20
- Max queries per connection: 50,000
- Connection timeout: 30 seconds
- Idle connection lifetime: 5 minutes

### Neo4j Configuration
- Bolt protocol over TCP
- Connection pooling handled by Neo4j driver
- Query timeout: 30 seconds
- Retry logic for transient failures

## Testing Configuration

### TestModel Integration
```python
from pydantic_ai.models.test import TestModel

# Test agent with mock model
test_agent = rag_agent.override(model=TestModel())

# Test dependencies
test_deps = AgentDependencies(
    session_id="test_session",
    debug=True
)
```

### Environment Variables for Testing
```bash
# Test Database URLs
TEST_DATABASE_URL=postgresql://test:test@localhost:5433/test_rag
TEST_NEO4J_URI=bolt://localhost:7688
TEST_NEO4J_PASSWORD=test_password

# Mock API Keys for Testing
TEST_LLM_API_KEY=test-openai-key
TEST_EMBEDDING_API_KEY=test-embedding-key
```

## Error Handling Strategy

### Database Connection Failures
- Automatic retry with exponential backoff
- Connection pool recreation on failure
- Graceful degradation when databases unavailable
- Comprehensive error logging

### API Rate Limiting
- Respect OpenAI rate limits
- Exponential backoff for API errors
- Fallback responses when API unavailable
- Request queuing for high-traffic scenarios

### Tool Error Recovery
- Individual tool failure isolation
- Partial results when some tools fail
- Error context preservation for debugging
- User-friendly error messages

## Production Considerations

### Monitoring Requirements
- Database connection health checks
- API response time monitoring  
- Error rate tracking
- Resource usage metrics

### Scalability Settings
- Connection pool sizing based on load
- Embedding cache for frequently used vectors
- Result caching for common queries
- Horizontal scaling support

### Configuration Validation
- Startup health checks for all services
- Environment variable validation
- Database schema verification
- API connectivity testing