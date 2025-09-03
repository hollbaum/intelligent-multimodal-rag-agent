"""
Test individual tool implementations with parameter validation and error handling.
Validates all 4 essential tools: vector_search, graph_search, hybrid_search, comprehensive_search.
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, Mock
from pydantic import ValidationError

from tools import (
    VectorSearchParams, GraphSearchParams, HybridSearchParams, ComprehensiveSearchParams,
    generate_embedding, vector_search_tool, graph_search_tool, 
    hybrid_search_tool, comprehensive_search_tool
)
from dependencies import AgentDependencies
from agent import rag_agent


class TestParameterValidation:
    """Test Pydantic parameter validation for all tools."""
    
    def test_vector_search_params_valid(self):
        """Test valid vector search parameters."""
        params = VectorSearchParams(query="test query", limit=10)
        assert params.query == "test query"
        assert params.limit == 10
    
    def test_vector_search_params_invalid(self):
        """Test invalid vector search parameters."""
        # Empty query
        with pytest.raises(ValidationError):
            VectorSearchParams(query="", limit=10)
        
        # Invalid limit range
        with pytest.raises(ValidationError):
            VectorSearchParams(query="test", limit=0)
        
        with pytest.raises(ValidationError):
            VectorSearchParams(query="test", limit=100)
    
    def test_graph_search_params_valid(self):
        """Test valid graph search parameters."""
        params = GraphSearchParams(query="test graph query")
        assert params.query == "test graph query"
    
    def test_graph_search_params_invalid(self):
        """Test invalid graph search parameters."""
        with pytest.raises(ValidationError):
            GraphSearchParams(query="")
    
    def test_hybrid_search_params_valid(self):
        """Test valid hybrid search parameters."""
        params = HybridSearchParams(query="test", limit=15, text_weight=0.7)
        assert params.query == "test"
        assert params.limit == 15
        assert params.text_weight == 0.7
    
    def test_hybrid_search_params_invalid(self):
        """Test invalid hybrid search parameters."""
        # Invalid text weight
        with pytest.raises(ValidationError):
            HybridSearchParams(query="test", text_weight=-0.1)
        
        with pytest.raises(ValidationError):
            HybridSearchParams(query="test", text_weight=1.5)
    
    def test_comprehensive_search_params_valid(self):
        """Test valid comprehensive search parameters."""
        params = ComprehensiveSearchParams(
            query="test comprehensive",
            limit=20,
            use_vector=True,
            use_graph=False
        )
        assert params.query == "test comprehensive"
        assert params.limit == 20
        assert params.use_vector is True
        assert params.use_graph is False


class TestEmbeddingGeneration:
    """Test embedding generation functionality."""
    
    @pytest.mark.asyncio
    async def test_generate_embedding_success(self, mock_embedding_client, mock_embedding):
        """Test successful embedding generation."""
        with patch('tools.get_embedding_client', return_value=mock_embedding_client):
            with patch('tools.get_embedding_model', return_value="text-embedding-3-small"):
                embedding = await generate_embedding("test query")
                
                assert isinstance(embedding, list)
                assert len(embedding) == 1536  # OpenAI text-embedding-3-small dimension
                assert all(isinstance(x, float) for x in embedding)
    
    @pytest.mark.asyncio
    async def test_generate_embedding_api_error(self):
        """Test embedding generation with API error."""
        mock_client = AsyncMock()
        mock_client.embeddings.create.side_effect = Exception("API Error")
        
        with patch('tools.get_embedding_client', return_value=mock_client):
            with pytest.raises(Exception, match="API Error"):
                await generate_embedding("test query")


class TestVectorSearchTool:
    """Test vector search tool functionality."""
    
    @pytest.mark.asyncio
    async def test_vector_search_success(self, test_dependencies, mock_embedding, sample_vector_results):
        """Test successful vector search."""
        # Mock embedding generation
        with patch('tools.generate_embedding', return_value=mock_embedding):
            # Mock database connection and results
            mock_connection = AsyncMock()
            mock_connection.fetch.return_value = [
                {
                    "chunk_id": "chunk-1",
                    "content": "Test content",
                    "document_id": "doc-1",
                    "title": "Test Document",
                    "file_path": "/test.md",
                    "similarity_score": 0.95
                }
            ]
            
            test_dependencies._pg_pool.acquire.return_value.__aenter__.return_value = mock_connection
            
            # Create a mock tool function
            from pydantic_ai import RunContext
            
            # Get the actual vector_search tool from the agent
            vector_tool = None
            for tool in rag_agent.tools:
                if tool.name == "vector_search":
                    vector_tool = tool
                    break
            
            assert vector_tool is not None, "vector_search tool not found"
            
            # Create context
            ctx = Mock(spec=RunContext)
            ctx.deps = test_dependencies
            
            # Call the tool function
            result = await vector_tool.function(ctx, "test query", 10)
            
            assert isinstance(result, list)
            assert len(result) > 0
            
            # Validate result structure
            for item in result:
                assert "chunk_id" in item
                assert "content" in item
                assert "document_id" in item
                assert "title" in item
                assert "similarity_score" in item
                assert "search_type" in item
                assert item["search_type"] == "vector"
    
    @pytest.mark.asyncio
    async def test_vector_search_embedding_error(self, test_dependencies):
        """Test vector search with embedding generation error."""
        with patch('tools.generate_embedding', side_effect=Exception("Embedding error")):
            from pydantic_ai import RunContext
            
            vector_tool = None
            for tool in rag_agent.tools:
                if tool.name == "vector_search":
                    vector_tool = tool
                    break
            
            ctx = Mock(spec=RunContext)
            ctx.deps = test_dependencies
            
            result = await vector_tool.function(ctx, "test query", 10)
            
            # Should return empty list on error
            assert result == []
    
    @pytest.mark.asyncio
    async def test_vector_search_database_error(self, test_dependencies, mock_embedding):
        """Test vector search with database error."""
        with patch('tools.generate_embedding', return_value=mock_embedding):
            # Mock database connection error
            mock_connection = AsyncMock()
            mock_connection.fetch.side_effect = Exception("Database error")
            
            test_dependencies._pg_pool.acquire.return_value.__aenter__.return_value = mock_connection
            
            from pydantic_ai import RunContext
            
            vector_tool = None
            for tool in rag_agent.tools:
                if tool.name == "vector_search":
                    vector_tool = tool
                    break
            
            ctx = Mock(spec=RunContext)
            ctx.deps = test_dependencies
            
            result = await vector_tool.function(ctx, "test query", 10)
            
            # Should return empty list on database error
            assert result == []


class TestGraphSearchTool:
    """Test graph search tool functionality."""
    
    @pytest.mark.asyncio
    async def test_graph_search_success(self, test_dependencies):
        """Test successful graph search."""
        # Mock Graphiti search results
        mock_fact1 = Mock()
        mock_fact1.uuid = "fact-1"
        mock_fact1.fact = "Test fact about OpenAI"
        mock_fact1.created_at = None
        
        mock_fact2 = Mock()
        mock_fact2.uuid = "fact-2"
        mock_fact2.fact = "Test fact about Microsoft"
        mock_fact2.created_at = None
        
        test_dependencies._graphiti_client.search_facts.return_value = [mock_fact1, mock_fact2]
        
        from pydantic_ai import RunContext
        
        graph_tool = None
        for tool in rag_agent.tools:
            if tool.name == "graph_search":
                graph_tool = tool
                break
        
        assert graph_tool is not None, "graph_search tool not found"
        
        ctx = Mock(spec=RunContext)
        ctx.deps = test_dependencies
        
        result = await graph_tool.function(ctx, "test graph query")
        
        assert isinstance(result, list)
        assert len(result) == 2
        
        # Validate result structure
        for item in result:
            assert "fact_uuid" in item
            assert "fact_text" in item
            assert "created_at" in item
            assert "entities" in item
            assert "search_type" in item
            assert item["search_type"] == "graph"
    
    @pytest.mark.asyncio
    async def test_graph_search_error(self, test_dependencies):
        """Test graph search with error."""
        test_dependencies._graphiti_client.search_facts.side_effect = Exception("Graph error")
        
        from pydantic_ai import RunContext
        
        graph_tool = None
        for tool in rag_agent.tools:
            if tool.name == "graph_search":
                graph_tool = tool
                break
        
        ctx = Mock(spec=RunContext)
        ctx.deps = test_dependencies
        
        result = await graph_tool.function(ctx, "test query")
        
        # Should return empty list on error
        assert result == []


class TestHybridSearchTool:
    """Test hybrid search tool functionality."""
    
    @pytest.mark.asyncio
    async def test_hybrid_search_success(self, test_dependencies, mock_embedding):
        """Test successful hybrid search."""
        with patch('tools.generate_embedding', return_value=mock_embedding):
            # Mock database connection and results
            mock_connection = AsyncMock()
            mock_connection.fetch.return_value = [
                {
                    "chunk_id": "chunk-1",
                    "content": "Test hybrid content",
                    "document_id": "doc-1",
                    "title": "Test Document",
                    "file_path": "/test.md",
                    "combined_score": 0.88,
                    "vector_score": 0.85,
                    "text_score": 0.91
                }
            ]
            
            test_dependencies._pg_pool.acquire.return_value.__aenter__.return_value = mock_connection
            
            from pydantic_ai import RunContext
            
            hybrid_tool = None
            for tool in rag_agent.tools:
                if tool.name == "hybrid_search":
                    hybrid_tool = tool
                    break
            
            assert hybrid_tool is not None, "hybrid_search tool not found"
            
            ctx = Mock(spec=RunContext)
            ctx.deps = test_dependencies
            
            result = await hybrid_tool.function(ctx, "test query", 10, 0.3)
            
            assert isinstance(result, list)
            assert len(result) > 0
            
            # Validate result structure
            for item in result:
                assert "chunk_id" in item
                assert "content" in item
                assert "combined_score" in item
                assert "vector_score" in item
                assert "text_score" in item
                assert "search_type" in item
                assert item["search_type"] == "hybrid"
    
    @pytest.mark.asyncio
    async def test_hybrid_search_fallback_text_only(self, test_dependencies):
        """Test hybrid search fallback to text-only when embedding fails."""
        with patch('tools.generate_embedding', side_effect=Exception("Embedding error")):
            # Mock database connection for text-only search
            mock_connection = AsyncMock()
            mock_connection.fetch.return_value = [
                {
                    "chunk_id": "chunk-1",
                    "content": "Test text-only content",
                    "document_id": "doc-1",
                    "title": "Test Document",
                    "file_path": "/test.md",
                    "combined_score": 0.75,
                    "vector_score": 0.0,
                    "text_score": 0.75
                }
            ]
            
            test_dependencies._pg_pool.acquire.return_value.__aenter__.return_value = mock_connection
            
            from pydantic_ai import RunContext
            
            hybrid_tool = None
            for tool in rag_agent.tools:
                if tool.name == "hybrid_search":
                    hybrid_tool = tool
                    break
            
            ctx = Mock(spec=RunContext)
            ctx.deps = test_dependencies
            
            result = await hybrid_tool.function(ctx, "test query", 10, 0.3)
            
            assert isinstance(result, list)
            assert len(result) > 0
            
            # Should have vector_score of 0.0 for text-only fallback
            for item in result:
                assert item["vector_score"] == 0.0
                assert item["text_score"] > 0.0


class TestComprehensiveSearchTool:
    """Test comprehensive search tool functionality."""
    
    @pytest.mark.asyncio
    async def test_comprehensive_search_both_methods(self, test_dependencies, mock_embedding):
        """Test comprehensive search with both vector and graph search."""
        # Mock embedding generation
        with patch('tools.generate_embedding', return_value=mock_embedding):
            # Mock vector search results
            mock_connection = AsyncMock()
            mock_connection.fetch.return_value = [
                {
                    "chunk_id": "chunk-1",
                    "content": "Vector result",
                    "document_id": "doc-1",
                    "title": "Test Doc",
                    "file_path": "/test.md",
                    "similarity_score": 0.9
                }
            ]
            
            test_dependencies._pg_pool.acquire.return_value.__aenter__.return_value = mock_connection
            
            # Mock graph search results
            mock_fact = Mock()
            mock_fact.uuid = "fact-1"
            mock_fact.fact = "Graph result fact"
            mock_fact.created_at = None
            
            test_dependencies._graphiti_client.search_facts.return_value = [mock_fact]
            
            from pydantic_ai import RunContext
            
            comprehensive_tool = None
            for tool in rag_agent.tools:
                if tool.name == "comprehensive_search":
                    comprehensive_tool = tool
                    break
            
            assert comprehensive_tool is not None, "comprehensive_search tool not found"
            
            ctx = Mock(spec=RunContext)
            ctx.deps = test_dependencies
            
            result = await comprehensive_tool.function(
                ctx, "test query", 10, use_vector=True, use_graph=True
            )
            
            assert isinstance(result, dict)
            assert "vector_results" in result
            assert "graph_results" in result
            assert "total_results" in result
            assert "search_type" in result
            assert result["search_type"] == "comprehensive"
            
            assert len(result["vector_results"]) > 0
            assert len(result["graph_results"]) > 0
            assert result["total_results"] == len(result["vector_results"]) + len(result["graph_results"])
    
    @pytest.mark.asyncio
    async def test_comprehensive_search_vector_only(self, test_dependencies, mock_embedding):
        """Test comprehensive search with vector search only."""
        with patch('tools.generate_embedding', return_value=mock_embedding):
            # Mock vector search results
            mock_connection = AsyncMock()
            mock_connection.fetch.return_value = [
                {
                    "chunk_id": "chunk-1",
                    "content": "Vector only result",
                    "document_id": "doc-1",
                    "title": "Test Doc",
                    "file_path": "/test.md",
                    "similarity_score": 0.9
                }
            ]
            
            test_dependencies._pg_pool.acquire.return_value.__aenter__.return_value = mock_connection
            
            from pydantic_ai import RunContext
            
            comprehensive_tool = None
            for tool in rag_agent.tools:
                if tool.name == "comprehensive_search":
                    comprehensive_tool = tool
                    break
            
            ctx = Mock(spec=RunContext)
            ctx.deps = test_dependencies
            
            result = await comprehensive_tool.function(
                ctx, "test query", 10, use_vector=True, use_graph=False
            )
            
            assert isinstance(result, dict)
            assert len(result["vector_results"]) > 0
            assert len(result["graph_results"]) == 0
            assert result["total_results"] == len(result["vector_results"])
    
    @pytest.mark.asyncio
    async def test_comprehensive_search_neither_method(self, test_dependencies):
        """Test comprehensive search with neither method enabled."""
        from pydantic_ai import RunContext
        
        comprehensive_tool = None
        for tool in rag_agent.tools:
            if tool.name == "comprehensive_search":
                comprehensive_tool = tool
                break
        
        ctx = Mock(spec=RunContext)
        ctx.deps = test_dependencies
        
        result = await comprehensive_tool.function(
            ctx, "test query", 10, use_vector=False, use_graph=False
        )
        
        assert isinstance(result, dict)
        assert result["vector_results"] == []
        assert result["graph_results"] == []
        assert result["total_results"] == 0
    
    @pytest.mark.asyncio
    async def test_comprehensive_search_partial_failure(self, test_dependencies, mock_embedding):
        """Test comprehensive search with one method failing."""
        with patch('tools.generate_embedding', return_value=mock_embedding):
            # Mock successful vector search
            mock_connection = AsyncMock()
            mock_connection.fetch.return_value = [
                {
                    "chunk_id": "chunk-1",
                    "content": "Vector result",
                    "document_id": "doc-1",
                    "title": "Test Doc",
                    "file_path": "/test.md",
                    "similarity_score": 0.9
                }
            ]
            
            test_dependencies._pg_pool.acquire.return_value.__aenter__.return_value = mock_connection
            
            # Mock failing graph search
            test_dependencies._graphiti_client.search_facts.side_effect = Exception("Graph error")
            
            from pydantic_ai import RunContext
            
            comprehensive_tool = None
            for tool in rag_agent.tools:
                if tool.name == "comprehensive_search":
                    comprehensive_tool = tool
                    break
            
            ctx = Mock(spec=RunContext)
            ctx.deps = test_dependencies
            
            result = await comprehensive_tool.function(
                ctx, "test query", 10, use_vector=True, use_graph=True
            )
            
            assert isinstance(result, dict)
            assert len(result["vector_results"]) > 0  # Vector search succeeded
            assert len(result["graph_results"]) == 0  # Graph search failed
            assert result["total_results"] == len(result["vector_results"])


class TestToolIntegration:
    """Test tool integration with agent."""
    
    def test_all_tools_registered(self):
        """Test all expected tools are registered with agent."""
        expected_tools = ["vector_search", "graph_search", "hybrid_search", "comprehensive_search"]
        registered_tools = [tool.name for tool in rag_agent.tools]
        
        assert set(expected_tools) == set(registered_tools)
    
    def test_tool_function_signatures(self):
        """Test tool functions have correct signatures."""
        for tool in rag_agent.tools:
            assert hasattr(tool, 'function')
            assert callable(tool.function)
            
            # All tools should accept RunContext as first parameter
            import inspect
            sig = inspect.signature(tool.function)
            params = list(sig.parameters.keys())
            assert 'ctx' in params or 'context' in params or len(params) > 0


@pytest.mark.integration
class TestToolsWithMockedDependencies:
    """Integration tests for tools with mocked external dependencies."""
    
    @pytest.mark.asyncio
    async def test_all_tools_with_real_parameters(self, test_dependencies, mock_embedding):
        """Test all tools can be called with realistic parameters."""
        from pydantic_ai import RunContext
        
        # Setup mocks for all tools
        with patch('tools.generate_embedding', return_value=mock_embedding):
            # Mock database results
            mock_connection = AsyncMock()
            mock_connection.fetch.return_value = [
                {
                    "chunk_id": "chunk-1",
                    "content": "Mock result",
                    "document_id": "doc-1",
                    "title": "Test Doc",
                    "file_path": "/test.md",
                    "similarity_score": 0.9,
                    "combined_score": 0.85,
                    "vector_score": 0.8,
                    "text_score": 0.9
                }
            ]
            
            test_dependencies._pg_pool.acquire.return_value.__aenter__.return_value = mock_connection
            
            # Mock graph results
            mock_fact = Mock()
            mock_fact.uuid = "fact-1"
            mock_fact.fact = "Mock graph fact"
            mock_fact.created_at = None
            
            test_dependencies._graphiti_client.search_facts.return_value = [mock_fact]
            
            ctx = Mock(spec=RunContext)
            ctx.deps = test_dependencies
            
            # Test all tools
            for tool in rag_agent.tools:
                if tool.name == "vector_search":
                    result = await tool.function(ctx, "test query", 5)
                    assert isinstance(result, list)
                
                elif tool.name == "graph_search":
                    result = await tool.function(ctx, "test query")
                    assert isinstance(result, list)
                
                elif tool.name == "hybrid_search":
                    result = await tool.function(ctx, "test query", 5, 0.3)
                    assert isinstance(result, list)
                
                elif tool.name == "comprehensive_search":
                    result = await tool.function(ctx, "test query", 5, True, True)
                    assert isinstance(result, dict)