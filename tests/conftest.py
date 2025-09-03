"""
Test configuration and fixtures for RAG Knowledge Graph AI Assistant.
Comprehensive pytest setup with TestModel, FunctionModel, and mocked dependencies.
"""

import pytest
import os
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import asyncpg

from pydantic_ai.models.test import TestModel
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelTextResponse

# Set test environment
os.environ.setdefault("LLM_API_KEY", "test-api-key")
os.environ.setdefault("EMBEDDING_API_KEY", "test-api-key")
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")

# Import application modules after setting environment variables
from agent import rag_agent
from dependencies import AgentDependencies
from settings import settings
from tools import generate_embedding


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_model():
    """Create TestModel for rapid agent validation."""
    return TestModel()


@pytest.fixture
def test_agent(test_model):
    """Create agent with TestModel for testing."""
    return rag_agent.override(model=test_model)


@pytest.fixture
def function_model():
    """Create FunctionModel for custom behavior testing."""
    def create_function():
        call_count = 0
        
        async def test_function(messages, tools):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                # First call - analyze request
                return ModelTextResponse(
                    content="I'll help you search for that information."
                )
            elif call_count == 2:
                # Second call - perform tool call
                return {
                    "vector_search": {
                        "query": "test query",
                        "limit": 10
                    }
                }
            else:
                # Final response
                return ModelTextResponse(
                    content="Based on the search results, here's what I found..."
                )
        
        return test_function
    
    return FunctionModel(create_function())


@pytest.fixture
def function_agent(function_model):
    """Create agent with FunctionModel for behavior testing."""
    return rag_agent.override(model=function_model)


@pytest.fixture
def mock_pg_pool():
    """Mock PostgreSQL connection pool."""
    mock_pool = AsyncMock()
    mock_connection = AsyncMock()
    
    # Mock connection context manager
    mock_connection.__aenter__ = AsyncMock(return_value=mock_connection)
    mock_connection.__aexit__ = AsyncMock(return_value=None)
    
    # Mock pool acquire context manager
    mock_pool.acquire.return_value = mock_connection
    
    # Mock database query results
    mock_connection.fetch.return_value = [
        {
            "chunk_id": "test-chunk-1",
            "content": "Test content about OpenAI",
            "document_id": "test-doc-1",
            "title": "OpenAI Documentation",
            "file_path": "/test/path.md",
            "similarity_score": 0.95
        },
        {
            "chunk_id": "test-chunk-2", 
            "content": "Test content about Microsoft",
            "document_id": "test-doc-2",
            "title": "Microsoft Documentation",
            "file_path": "/test/path2.md",
            "similarity_score": 0.87
        }
    ]
    
    mock_connection.fetchrow.return_value = {
        "id": "test-session-id",
        "user_id": "test-user",
        "metadata": "{}",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "expires_at": None
    }
    
    return mock_pool


@pytest.fixture
def mock_graphiti_client():
    """Mock Graphiti knowledge graph client."""
    mock_client = AsyncMock()
    
    # Mock search results
    mock_fact1 = Mock()
    mock_fact1.uuid = "fact-uuid-1"
    mock_fact1.fact = "OpenAI was founded by Sam Altman and others"
    mock_fact1.created_at = datetime.now(timezone.utc)
    
    mock_fact2 = Mock()
    mock_fact2.uuid = "fact-uuid-2"
    mock_fact2.fact = "Microsoft invested heavily in OpenAI"
    mock_fact2.created_at = datetime.now(timezone.utc)
    
    mock_client.search_facts.return_value = [mock_fact1, mock_fact2]
    
    return mock_client


@pytest.fixture
def test_dependencies(mock_pg_pool, mock_graphiti_client):
    """Create test dependencies with mocked connections."""
    deps = AgentDependencies(
        session_id="test-session-id",
        user_id="test-user",
        debug=True
    )
    
    # Override the lazy properties with mocks
    deps._pg_pool = mock_pg_pool
    deps._graphiti_client = mock_graphiti_client
    
    return deps


@pytest.fixture
def mock_embedding():
    """Mock embedding generation."""
    # Standard 1536-dimensional embedding for OpenAI text-embedding-3-small
    return [0.1] * 1536


@pytest.fixture
async def mock_embedding_client(mock_embedding):
    """Mock OpenAI embedding client."""
    mock_client = AsyncMock()
    
    # Mock embedding response
    mock_response = Mock()
    mock_data = Mock()
    mock_data.embedding = mock_embedding
    mock_response.data = [mock_data]
    
    mock_client.embeddings.create.return_value = mock_response
    
    return mock_client


@pytest.fixture
def sample_vector_results():
    """Sample vector search results."""
    return [
        {
            "chunk_id": "chunk-1",
            "content": "OpenAI is an artificial intelligence research laboratory.",
            "document_id": "doc-1",
            "title": "OpenAI Overview",
            "file_path": "/docs/openai.md",
            "similarity_score": 0.95,
            "search_type": "vector"
        },
        {
            "chunk_id": "chunk-2",
            "content": "Microsoft has partnered with OpenAI for AI development.",
            "document_id": "doc-2", 
            "title": "Microsoft AI Partnership",
            "file_path": "/docs/microsoft.md",
            "similarity_score": 0.88,
            "search_type": "vector"
        }
    ]


@pytest.fixture
def sample_graph_results():
    """Sample graph search results."""
    return [
        {
            "fact_uuid": "fact-1",
            "fact_text": "Sam Altman is the CEO of OpenAI",
            "created_at": "2024-01-01T00:00:00",
            "entities": ["Sam Altman", "OpenAI"],
            "confidence": 0.9,
            "search_type": "graph"
        },
        {
            "fact_uuid": "fact-2",
            "fact_text": "Microsoft invested $10 billion in OpenAI",
            "created_at": "2024-01-02T00:00:00",
            "entities": ["Microsoft", "OpenAI"],
            "confidence": 0.95,
            "search_type": "graph"
        }
    ]


@pytest.fixture
def sample_hybrid_results():
    """Sample hybrid search results."""
    return [
        {
            "chunk_id": "chunk-1",
            "content": "OpenAI develops advanced AI models like GPT-4",
            "document_id": "doc-1",
            "title": "OpenAI Models",
            "file_path": "/docs/models.md",
            "combined_score": 0.92,
            "vector_score": 0.89,
            "text_score": 0.95,
            "search_type": "hybrid"
        }
    ]


@pytest.fixture
def sample_comprehensive_results(sample_vector_results, sample_graph_results):
    """Sample comprehensive search results."""
    return {
        "vector_results": sample_vector_results,
        "graph_results": sample_graph_results,
        "total_results": len(sample_vector_results) + len(sample_graph_results),
        "search_type": "comprehensive"
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "security: Security tests")


# Async test utilities
@pytest.fixture
async def async_mock():
    """Create async mock utility."""
    def create_async_mock(**kwargs):
        mock = AsyncMock(**kwargs)
        return mock
    return create_async_mock


# Mock patch helpers
@pytest.fixture
def mock_generate_embedding(mock_embedding):
    """Patch generate_embedding function."""
    with patch('tools.generate_embedding', new_callable=AsyncMock) as mock:
        mock.return_value = mock_embedding
        yield mock


@pytest.fixture  
def mock_db_operations():
    """Mock database operations."""
    with patch('agent.db_utils.db_pool') as mock_pool:
        yield mock_pool


@pytest.fixture
def mock_graph_operations():
    """Mock graph operations."""
    with patch('agent.graph_utils.graph_client') as mock_client:
        yield mock_client


# Environment setup
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    test_env = {
        "LLM_API_KEY": "test-api-key",
        "EMBEDDING_API_KEY": "test-api-key",
        "DATABASE_URL": "postgresql://test:test@localhost:5432/test",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_PASSWORD": "test-password",
        "APP_ENV": "testing",
        "DEBUG": "true",
        "LOG_LEVEL": "DEBUG"
    }
    
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield
    
    # Restore original environment
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


# Error simulation fixtures
@pytest.fixture
def database_error_mock():
    """Mock database connection errors."""
    return asyncpg.ConnectionDoesNotExistError("Test database connection error")


@pytest.fixture
def api_error_mock():
    """Mock API errors."""
    class APIError(Exception):
        pass
    
    return APIError("Test API connection error")


# Performance testing fixtures  
@pytest.fixture
def performance_timer():
    """Timer for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def elapsed_ms(self):
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time) * 1000
            return 0
    
    return Timer()


# Data validation fixtures
@pytest.fixture
def invalid_parameters():
    """Invalid parameters for testing validation."""
    return {
        "empty_query": "",
        "too_long_query": "x" * 10000,
        "invalid_limit": 0,
        "negative_limit": -1,
        "too_large_limit": 1000,
        "invalid_weight": -0.5,
        "weight_too_large": 1.5
    }


@pytest.fixture
def valid_parameters():
    """Valid parameters for testing."""
    return {
        "query": "test search query",
        "limit": 10,
        "text_weight": 0.3,
        "use_vector": True,
        "use_graph": True
    }