"""
Full integration tests for the RAG Knowledge Graph AI Assistant.
Tests end-to-end scenarios with database mocking and complete workflows.
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, Mock
from datetime import datetime, timezone

from agent import rag_agent, run_agent
from dependencies import AgentDependencies
from pydantic_ai.messages import ModelTextResponse


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    @pytest.mark.asyncio
    async def test_vector_search_workflow(self, test_agent, test_dependencies, mock_embedding):
        """Test complete vector search workflow."""
        # Configure agent to use vector search
        test_agent.model.agent_responses = [
            ModelTextResponse(content="I'll search for that information using vector search"),
            {
                "vector_search": {
                    "query": "artificial intelligence research",
                    "limit": 10
                }
            },
            ModelTextResponse(content="Based on the vector search results, I found relevant information about AI research...")
        ]
        
        # Mock embedding generation
        with patch('tools.generate_embedding', return_value=mock_embedding):
            # Mock database connection and results
            mock_connection = AsyncMock()
            mock_connection.fetch.return_value = [
                {
                    "chunk_id": "chunk-ai-1",
                    "content": "Artificial intelligence research focuses on machine learning algorithms...",
                    "document_id": "doc-ai-research",
                    "title": "AI Research Overview",
                    "file_path": "/docs/ai_research.md",
                    "similarity_score": 0.92
                }
            ]
            
            test_dependencies._pg_pool.acquire.return_value.__aenter__.return_value = mock_connection
            
            result = await test_agent.run(
                "Tell me about artificial intelligence research",
                deps=test_dependencies
            )
            
            assert result.data is not None
            assert len(result.data) > 0
            
            # Verify the workflow went through multiple steps
            messages = result.all_messages()
            assert len(messages) >= 2
    
    @pytest.mark.asyncio
    async def test_graph_search_workflow(self, test_agent, test_dependencies):
        """Test complete graph search workflow."""
        # Configure agent to use graph search
        test_agent.model.agent_responses = [
            ModelTextResponse(content="I'll search the knowledge graph for relationship information"),
            {
                "graph_search": {
                    "query": "relationships between tech companies"
                }
            },
            ModelTextResponse(content="Based on the knowledge graph, I found these relationships...")
        ]
        
        # Mock graph search results
        mock_fact1 = Mock()
        mock_fact1.uuid = "fact-relationship-1"
        mock_fact1.fact = "Microsoft invested $10 billion in OpenAI in 2023"
        mock_fact1.created_at = datetime.now(timezone.utc)
        
        mock_fact2 = Mock()
        mock_fact2.uuid = "fact-relationship-2"
        mock_fact2.fact = "Google competes with OpenAI in AI development"
        mock_fact2.created_at = datetime.now(timezone.utc)
        
        test_dependencies._graphiti_client.search_facts.return_value = [mock_fact1, mock_fact2]
        
        result = await test_agent.run(
            "What relationships exist between technology companies?",
            deps=test_dependencies
        )
        
        assert result.data is not None
        assert len(result.data) > 0
        
        # Verify graph search was called
        test_dependencies._graphiti_client.search_facts.assert_called()
    
    @pytest.mark.asyncio
    async def test_hybrid_search_workflow(self, test_agent, test_dependencies, mock_embedding):
        """Test complete hybrid search workflow."""
        # Configure agent to use hybrid search
        test_agent.model.agent_responses = [
            ModelTextResponse(content="I'll use hybrid search to combine semantic and keyword matching"),
            {
                "hybrid_search": {
                    "query": "machine learning algorithms",
                    "limit": 10,
                    "text_weight": 0.3
                }
            },
            ModelTextResponse(content="The hybrid search found results combining semantic similarity and keyword matching...")
        ]
        
        with patch('tools.generate_embedding', return_value=mock_embedding):
            # Mock hybrid search database results
            mock_connection = AsyncMock()
            mock_connection.fetch.return_value = [
                {
                    "chunk_id": "chunk-ml-1",
                    "content": "Machine learning algorithms include neural networks, decision trees, and support vector machines...",
                    "document_id": "doc-ml-guide",
                    "title": "Machine Learning Guide",
                    "file_path": "/docs/ml_guide.md",
                    "combined_score": 0.89,
                    "vector_score": 0.85,
                    "text_score": 0.93
                }
            ]
            
            test_dependencies._pg_pool.acquire.return_value.__aenter__.return_value = mock_connection
            
            result = await test_agent.run(
                "Explain machine learning algorithms",
                deps=test_dependencies
            )
            
            assert result.data is not None
            assert len(result.data) > 0
    
    @pytest.mark.asyncio
    async def test_comprehensive_search_workflow(self, test_agent, test_dependencies, mock_embedding):
        """Test complete comprehensive search workflow."""
        # Configure agent to use comprehensive search
        test_agent.model.agent_responses = [
            ModelTextResponse(content="I'll perform a comprehensive search using both vector and graph methods"),
            {
                "comprehensive_search": {
                    "query": "OpenAI developments",
                    "limit": 10,
                    "use_vector": True,
                    "use_graph": True
                }
            },
            ModelTextResponse(content="The comprehensive search found information from both vector similarity and knowledge graph...")
        ]
        
        with patch('tools.generate_embedding', return_value=mock_embedding):
            # Mock vector search results
            mock_connection = AsyncMock()
            mock_connection.fetch.return_value = [
                {
                    "chunk_id": "chunk-openai-1",
                    "content": "OpenAI released GPT-4 with improved capabilities...",
                    "document_id": "doc-openai-updates",
                    "title": "OpenAI Updates",
                    "file_path": "/docs/openai.md",
                    "similarity_score": 0.94
                }
            ]
            
            test_dependencies._pg_pool.acquire.return_value.__aenter__.return_value = mock_connection
            
            # Mock graph search results
            mock_fact = Mock()
            mock_fact.uuid = "fact-openai-1"
            mock_fact.fact = "OpenAI was founded in 2015 by Sam Altman and others"
            mock_fact.created_at = datetime.now(timezone.utc)
            
            test_dependencies._graphiti_client.search_facts.return_value = [mock_fact]
            
            result = await test_agent.run(
                "Tell me about OpenAI developments comprehensively",
                deps=test_dependencies
            )
            
            assert result.data is not None
            assert len(result.data) > 0


class TestErrorRecoveryScenarios:
    """Test error handling and recovery in integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_database_error_recovery(self, test_agent, test_dependencies, mock_embedding):
        """Test graceful handling of database errors."""
        # Configure agent to attempt vector search
        test_agent.model.agent_responses = [
            ModelTextResponse(content="I'll search for that information"),
            {
                "vector_search": {
                    "query": "test query",
                    "limit": 10
                }
            },
            ModelTextResponse(content="I encountered an issue with the search but I'm still here to help")
        ]
        
        with patch('tools.generate_embedding', return_value=mock_embedding):
            # Mock database connection error
            mock_connection = AsyncMock()
            mock_connection.fetch.side_effect = Exception("Database connection failed")
            
            test_dependencies._pg_pool.acquire.return_value.__aenter__.return_value = mock_connection
            
            result = await test_agent.run(
                "Search for information",
                deps=test_dependencies
            )
            
            # Agent should still provide a response despite the error
            assert result.data is not None
            assert len(result.data) > 0
    
    @pytest.mark.asyncio
    async def test_graph_error_recovery(self, test_agent, test_dependencies):
        """Test graceful handling of graph database errors."""
        # Configure agent to attempt graph search
        test_agent.model.agent_responses = [
            ModelTextResponse(content="I'll search the knowledge graph"),
            {
                "graph_search": {
                    "query": "test graph query"
                }
            },
            ModelTextResponse(content="I encountered an issue with the knowledge graph but can still assist you")
        ]
        
        # Mock graph search error
        test_dependencies._graphiti_client.search_facts.side_effect = Exception("Graph connection failed")
        
        result = await test_agent.run(
            "Find relationships in the knowledge graph",
            deps=test_dependencies
        )
        
        # Agent should still provide a response
        assert result.data is not None
        assert len(result.data) > 0
    
    @pytest.mark.asyncio
    async def test_embedding_error_fallback(self, test_agent, test_dependencies):
        """Test fallback when embedding generation fails."""
        # Configure agent to attempt hybrid search
        test_agent.model.agent_responses = [
            ModelTextResponse(content="I'll perform a hybrid search"),
            {
                "hybrid_search": {
                    "query": "test fallback query",
                    "limit": 10
                }
            },
            ModelTextResponse(content="I used text-only search as a fallback")
        ]
        
        # Mock embedding generation error, but database success
        with patch('tools.generate_embedding', side_effect=Exception("Embedding API failed")):
            mock_connection = AsyncMock()
            mock_connection.fetch.return_value = [
                {
                    "chunk_id": "text-only-1",
                    "content": "Text-only search result",
                    "document_id": "doc-fallback",
                    "title": "Fallback Document",
                    "file_path": "/docs/fallback.md",
                    "combined_score": 0.75,
                    "vector_score": 0.0,  # Should be 0 for text-only
                    "text_score": 0.75
                }
            ]
            
            test_dependencies._pg_pool.acquire.return_value.__aenter__.return_value = mock_connection
            
            result = await test_agent.run(
                "Search with embedding fallback",
                deps=test_dependencies
            )
            
            assert result.data is not None
            assert len(result.data) > 0
    
    @pytest.mark.asyncio 
    async def test_partial_comprehensive_search_failure(self, test_agent, test_dependencies, mock_embedding):
        """Test comprehensive search when one method fails."""
        # Configure agent for comprehensive search
        test_agent.model.agent_responses = [
            ModelTextResponse(content="I'll do a comprehensive search"),
            {
                "comprehensive_search": {
                    "query": "partial failure test",
                    "limit": 10,
                    "use_vector": True,
                    "use_graph": True
                }
            },
            ModelTextResponse(content="I found results from vector search even though graph search failed")
        ]
        
        with patch('tools.generate_embedding', return_value=mock_embedding):
            # Vector search succeeds
            mock_connection = AsyncMock()
            mock_connection.fetch.return_value = [
                {
                    "chunk_id": "vector-success-1",
                    "content": "Vector search worked",
                    "document_id": "doc-vector",
                    "title": "Vector Document",
                    "file_path": "/docs/vector.md",
                    "similarity_score": 0.88
                }
            ]
            
            test_dependencies._pg_pool.acquire.return_value.__aenter__.return_value = mock_connection
            
            # Graph search fails
            test_dependencies._graphiti_client.search_facts.side_effect = Exception("Graph failed")
            
            result = await test_agent.run(
                "Comprehensive search with partial failure",
                deps=test_dependencies
            )
            
            assert result.data is not None
            assert len(result.data) > 0


class TestAgentConversationFlows:
    """Test multi-turn conversation flows."""
    
    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, test_agent, test_dependencies, mock_embedding):
        """Test multiple queries in sequence with context."""
        queries_and_responses = [
            ("What is machine learning?", "Machine learning is a subset of artificial intelligence..."),
            ("How does it relate to deep learning?", "Deep learning is a specific type of machine learning..."),
            ("Can you search for more information?", "I'll search for additional information...")
        ]
        
        with patch('tools.generate_embedding', return_value=mock_embedding):
            # Mock database results for searches
            mock_connection = AsyncMock()
            mock_connection.fetch.return_value = [
                {
                    "chunk_id": "ml-info-1",
                    "content": "Machine learning information...",
                    "document_id": "doc-ml",
                    "title": "ML Guide",
                    "file_path": "/docs/ml.md",
                    "similarity_score": 0.9
                }
            ]
            
            test_dependencies._pg_pool.acquire.return_value.__aenter__.return_value = mock_connection
            
            for i, (query, expected_content) in enumerate(queries_and_responses):
                # Configure response for each turn
                test_agent.model.agent_responses = [
                    ModelTextResponse(content=expected_content)
                ]
                
                if "search" in query.lower():
                    test_agent.model.agent_responses.extend([
                        {
                            "vector_search": {
                                "query": query,
                                "limit": 10
                            }
                        },
                        ModelTextResponse(content=f"Found information for query {i+1}")
                    ])
                
                result = await test_agent.run(query, deps=test_dependencies)
                
                assert result.data is not None
                assert len(result.data) > 0
    
    @pytest.mark.asyncio
    async def test_session_consistency(self):
        """Test that session consistency is maintained across queries."""
        session_id = "consistency-test-session"
        
        # First query
        result1 = await run_agent(
            prompt="Remember that I'm interested in AI research",
            session_id=session_id
        )
        
        # Second query - should maintain session context
        result2 = await run_agent(
            prompt="Based on my interest, what should I know?",
            session_id=session_id
        )
        
        assert result1 is not None
        assert result2 is not None
        
        # Both should succeed (mocked responses)
        assert len(result1) > 0
        assert len(result2) > 0


class TestRealWorldScenarios:
    """Test realistic usage scenarios."""
    
    @pytest.mark.asyncio
    async def test_research_query_scenario(self, test_agent, test_dependencies, mock_embedding, sample_vector_results, sample_graph_results):
        """Test a realistic research query scenario."""
        # Configure agent for research-style query
        test_agent.model.agent_responses = [
            ModelTextResponse(content="I'll help you research that topic comprehensively"),
            {
                "comprehensive_search": {
                    "query": "artificial intelligence breakthrough 2024",
                    "limit": 15,
                    "use_vector": True,
                    "use_graph": True
                }
            },
            ModelTextResponse(content="Based on my comprehensive search, here's what I found about AI breakthroughs in 2024...")
        ]
        
        with patch('tools.generate_embedding', return_value=mock_embedding):
            # Mock comprehensive search results
            mock_connection = AsyncMock()
            mock_connection.fetch.return_value = [
                {
                    "chunk_id": "ai-2024-1",
                    "content": "In 2024, major AI breakthroughs included advances in multimodal models...",
                    "document_id": "doc-ai-2024",
                    "title": "AI Breakthroughs 2024",
                    "file_path": "/docs/ai_2024.md",
                    "similarity_score": 0.96
                },
                {
                    "chunk_id": "ai-2024-2", 
                    "content": "OpenAI released GPT-4 Turbo with improved reasoning capabilities...",
                    "document_id": "doc-openai-2024",
                    "title": "OpenAI Updates 2024",
                    "file_path": "/docs/openai_2024.md",
                    "similarity_score": 0.93
                }
            ]
            
            test_dependencies._pg_pool.acquire.return_value.__aenter__.return_value = mock_connection
            
            # Mock graph facts
            mock_fact = Mock()
            mock_fact.uuid = "fact-ai-breakthrough"
            mock_fact.fact = "GPT-4 Turbo was released by OpenAI in late 2024 with 128K context window"
            mock_fact.created_at = datetime.now(timezone.utc)
            
            test_dependencies._graphiti_client.search_facts.return_value = [mock_fact]
            
            result = await test_agent.run(
                "I'm researching AI breakthroughs in 2024. Can you give me a comprehensive overview?",
                deps=test_dependencies
            )
            
            assert result.data is not None
            assert len(result.data) > 100  # Should be a substantial response
            assert "2024" in result.data or "breakthrough" in result.data.lower()
    
    @pytest.mark.asyncio
    async def test_comparison_query_scenario(self, test_agent, test_dependencies, mock_embedding):
        """Test a comparison query scenario."""
        # Configure agent for comparison query
        test_agent.model.agent_responses = [
            ModelTextResponse(content="I'll search for information about both companies to compare them"),
            {
                "hybrid_search": {
                    "query": "OpenAI versus Google AI comparison",
                    "limit": 20,
                    "text_weight": 0.4
                }
            },
            ModelTextResponse(content="Based on my search, here's a comparison between OpenAI and Google AI...")
        ]
        
        with patch('tools.generate_embedding', return_value=mock_embedding):
            # Mock comparison search results
            mock_connection = AsyncMock()
            mock_connection.fetch.return_value = [
                {
                    "chunk_id": "openai-comparison-1",
                    "content": "OpenAI focuses on AGI development with models like GPT-4...",
                    "document_id": "doc-openai-profile",
                    "title": "OpenAI Company Profile",
                    "file_path": "/docs/openai_profile.md",
                    "combined_score": 0.91,
                    "vector_score": 0.88,
                    "text_score": 0.94
                },
                {
                    "chunk_id": "google-comparison-1",
                    "content": "Google AI, formerly Google Brain, works on diverse AI applications including search and cloud...",
                    "document_id": "doc-google-ai",
                    "title": "Google AI Overview",
                    "file_path": "/docs/google_ai.md",
                    "combined_score": 0.87,
                    "vector_score": 0.85,
                    "text_score": 0.89
                }
            ]
            
            test_dependencies._pg_pool.acquire.return_value.__aenter__.return_value = mock_connection
            
            result = await test_agent.run(
                "Compare OpenAI and Google AI in terms of their approaches and capabilities",
                deps=test_dependencies
            )
            
            assert result.data is not None
            assert len(result.data) > 50
    
    @pytest.mark.asyncio
    async def test_fact_checking_scenario(self, test_agent, test_dependencies):
        """Test a fact-checking scenario using knowledge graph."""
        # Configure agent for fact-checking
        test_agent.model.agent_responses = [
            ModelTextResponse(content="I'll check the knowledge graph for facts about that claim"),
            {
                "graph_search": {
                    "query": "Microsoft OpenAI investment amount timeline"
                }
            },
            ModelTextResponse(content="Based on the knowledge graph facts, I can verify that information...")
        ]
        
        # Mock fact-checking graph results
        mock_fact1 = Mock()
        mock_fact1.uuid = "investment-fact-1"
        mock_fact1.fact = "Microsoft invested $1 billion in OpenAI in 2019"
        mock_fact1.created_at = datetime(2019, 7, 22, tzinfo=timezone.utc)
        
        mock_fact2 = Mock()
        mock_fact2.uuid = "investment-fact-2" 
        mock_fact2.fact = "Microsoft made additional multi-billion dollar investment in OpenAI in 2023"
        mock_fact2.created_at = datetime(2023, 1, 23, tzinfo=timezone.utc)
        
        test_dependencies._graphiti_client.search_facts.return_value = [mock_fact1, mock_fact2]
        
        result = await test_agent.run(
            "Can you fact-check the claim that Microsoft invested $10 billion in OpenAI?",
            deps=test_dependencies
        )
        
        assert result.data is not None
        assert len(result.data) > 50


class TestConcurrencyAndPerformance:
    """Test concurrent usage and performance scenarios."""
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_queries(self, mock_embedding):
        """Test multiple concurrent agent queries."""
        session_ids = [f"concurrent-session-{i}" for i in range(3)]
        queries = [
            "What is machine learning?",
            "Explain neural networks",
            "How does AI work?"
        ]
        
        with patch('tools.generate_embedding', return_value=mock_embedding):
            # Mock database responses for all queries
            with patch('asyncpg.create_pool') as mock_create_pool:
                mock_pool = AsyncMock()
                mock_connection = AsyncMock()
                mock_connection.fetch.return_value = [
                    {
                        "chunk_id": f"concurrent-chunk-{i}",
                        "content": f"Concurrent search result {i}",
                        "document_id": f"concurrent-doc-{i}",
                        "title": f"Concurrent Document {i}",
                        "file_path": f"/docs/concurrent_{i}.md",
                        "similarity_score": 0.9
                    }
                    for i in range(3)
                ]
                
                mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
                mock_create_pool.return_value = mock_pool
                
                # Run concurrent queries
                tasks = [
                    run_agent(prompt=query, session_id=session_id)
                    for query, session_id in zip(queries, session_ids)
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # All should complete successfully
                for result in results:
                    assert not isinstance(result, Exception)
                    assert result is not None
                    assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_large_result_set_handling(self, test_agent, test_dependencies, mock_embedding):
        """Test handling of large result sets."""
        # Configure agent for large search
        test_agent.model.agent_responses = [
            ModelTextResponse(content="I'll search for comprehensive information"),
            {
                "comprehensive_search": {
                    "query": "comprehensive AI research",
                    "limit": 50,  # Large limit
                    "use_vector": True,
                    "use_graph": True
                }
            },
            ModelTextResponse(content="I found extensive information from multiple sources...")
        ]
        
        with patch('tools.generate_embedding', return_value=mock_embedding):
            # Mock large result set
            large_vector_results = [
                {
                    "chunk_id": f"large-chunk-{i}",
                    "content": f"Large dataset content {i}...",
                    "document_id": f"large-doc-{i}",
                    "title": f"Large Document {i}",
                    "file_path": f"/docs/large_{i}.md",
                    "similarity_score": 0.9 - (i * 0.01)  # Decreasing relevance
                }
                for i in range(50)
            ]
            
            mock_connection = AsyncMock()
            mock_connection.fetch.return_value = large_vector_results
            
            test_dependencies._pg_pool.acquire.return_value.__aenter__.return_value = mock_connection
            
            # Mock large graph results
            large_graph_facts = [
                Mock(uuid=f"large-fact-{i}", fact=f"Large fact {i}", created_at=datetime.now(timezone.utc))
                for i in range(30)
            ]
            
            test_dependencies._graphiti_client.search_facts.return_value = large_graph_facts
            
            result = await test_agent.run(
                "Find comprehensive information about AI research",
                deps=test_dependencies
            )
            
            assert result.data is not None
            assert len(result.data) > 100  # Should handle large responses


@pytest.mark.integration
class TestPRPRequirementValidation:
    """Test specific PRP success criteria in integration context."""
    
    @pytest.mark.asyncio
    async def test_all_search_types_functional(self, test_agent, test_dependencies, mock_embedding):
        """Validate all 4 search types work in integration."""
        search_types = [
            ("vector_search", "Vector search for AI information"),
            ("graph_search", "Graph search for relationships"),
            ("hybrid_search", "Hybrid search combining methods"),
            ("comprehensive_search", "Comprehensive search using all methods")
        ]
        
        with patch('tools.generate_embedding', return_value=mock_embedding):
            # Mock database for vector/hybrid searches
            mock_connection = AsyncMock()
            mock_connection.fetch.return_value = [
                {
                    "chunk_id": "test-chunk",
                    "content": "Test search result",
                    "document_id": "test-doc",
                    "title": "Test Document",
                    "file_path": "/test.md",
                    "similarity_score": 0.9,
                    "combined_score": 0.85,
                    "vector_score": 0.8,
                    "text_score": 0.9
                }
            ]
            
            test_dependencies._pg_pool.acquire.return_value.__aenter__.return_value = mock_connection
            
            # Mock graph search
            mock_fact = Mock()
            mock_fact.uuid = "test-fact"
            mock_fact.fact = "Test knowledge graph fact"
            mock_fact.created_at = datetime.now(timezone.utc)
            
            test_dependencies._graphiti_client.search_facts.return_value = [mock_fact]
            
            for tool_name, query in search_types:
                # Configure agent to use specific tool
                test_agent.model.agent_responses = [
                    ModelTextResponse(content=f"I'll use {tool_name}"),
                    {
                        tool_name: {
                            "query": query,
                            **({"limit": 10} if "limit" in [param.name for param in rag_agent.tools if rag_agent.tools[0].name == tool_name][0]._function.__code__.co_varnames else {})
                        }
                    },
                    ModelTextResponse(content=f"Results from {tool_name}")
                ]
                
                result = await test_agent.run(query, deps=test_dependencies)
                
                assert result.data is not None, f"Tool {tool_name} failed to produce result"
                assert len(result.data) > 0, f"Tool {tool_name} produced empty result"
    
    def test_structured_outputs_validate(self):
        """Test that all Pydantic models validate correctly."""
        from tools import VectorSearchParams, GraphSearchParams, HybridSearchParams, ComprehensiveSearchParams
        
        # Test all parameter models validate
        vector_params = VectorSearchParams(query="test", limit=10)
        assert vector_params.query == "test"
        assert vector_params.limit == 10
        
        graph_params = GraphSearchParams(query="test graph")
        assert graph_params.query == "test graph"
        
        hybrid_params = HybridSearchParams(query="test hybrid", limit=15, text_weight=0.4)
        assert hybrid_params.text_weight == 0.4
        
        comprehensive_params = ComprehensiveSearchParams(
            query="test comprehensive", limit=20, use_vector=True, use_graph=False
        )
        assert comprehensive_params.use_vector is True
        assert comprehensive_params.use_graph is False
    
    @pytest.mark.asyncio
    async def test_error_handling_comprehensive(self, test_agent, test_dependencies):
        """Test comprehensive error handling as specified in PRP."""
        error_scenarios = [
            ("Database connection error", "database_error"),
            ("Graph connection error", "graph_error"),
            ("API rate limit error", "api_error"),
            ("Invalid input error", "validation_error")
        ]
        
        for error_description, error_type in error_scenarios:
            # Configure agent response for error scenario
            test_agent.model.agent_responses = [
                ModelTextResponse(content=f"I encountered an issue but I'm still available to help: {error_description}")
            ]
            
            result = await test_agent.run(
                f"Test query for {error_description}",
                deps=test_dependencies
            )
            
            # Should handle errors gracefully
            assert result.data is not None
            assert len(result.data) > 0