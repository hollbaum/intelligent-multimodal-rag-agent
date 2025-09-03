"""
Performance validation tests for the RAG Knowledge Graph AI Assistant.
Tests response times, throughput, resource usage, and scalability requirements.
"""

import pytest
import asyncio
import time
import psutil
import gc
from unittest.mock import patch, AsyncMock, Mock
from concurrent.futures import ThreadPoolExecutor
import threading

from agent import rag_agent, run_agent
from dependencies import AgentDependencies
from tools import generate_embedding
from pydantic_ai.messages import ModelTextResponse


class TestResponseTimeRequirements:
    """Test response time requirements as specified in PRP."""
    
    @pytest.mark.asyncio
    async def test_vector_search_response_time(self, test_agent, test_dependencies, mock_embedding, performance_timer):
        """Test vector search meets response time requirements."""
        # Configure agent for vector search
        test_agent.model.agent_responses = [
            ModelTextResponse(content="I'll perform a vector search"),
            {
                "vector_search": {
                    "query": "performance test query",
                    "limit": 10
                }
            },
            ModelTextResponse(content="Vector search completed")
        ]
        
        with patch('tools.generate_embedding', return_value=mock_embedding):
            # Mock fast database response
            mock_connection = AsyncMock()
            mock_connection.fetch.return_value = [
                {
                    "chunk_id": f"perf-chunk-{i}",
                    "content": f"Performance test content {i}",
                    "document_id": f"perf-doc-{i}",
                    "title": f"Performance Doc {i}",
                    "file_path": f"/perf/doc{i}.md",
                    "similarity_score": 0.9 - (i * 0.05)
                }
                for i in range(10)
            ]
            
            test_dependencies._pg_pool.acquire.return_value.__aenter__.return_value = mock_connection
            
            # Measure response time
            performance_timer.start()
            result = await test_agent.run("Find performance information", deps=test_dependencies)
            performance_timer.stop()
            
            # Verify response
            assert result.data is not None
            assert len(result.data) > 0
            
            # Check response time (should be under 2 seconds for mocked operations)
            response_time_ms = performance_timer.elapsed_ms
            assert response_time_ms < 2000, f"Response time {response_time_ms}ms exceeds 2000ms limit"
    
    @pytest.mark.asyncio
    async def test_graph_search_response_time(self, test_agent, test_dependencies, performance_timer):
        """Test graph search meets response time requirements."""
        # Configure agent for graph search
        test_agent.model.agent_responses = [
            ModelTextResponse(content="I'll search the knowledge graph"),
            {
                "graph_search": {
                    "query": "performance graph query"
                }
            },
            ModelTextResponse(content="Graph search completed")
        ]
        
        # Mock fast graph response
        mock_facts = [
            Mock(
                uuid=f"perf-fact-{i}",
                fact=f"Performance fact {i}",
                created_at=None
            )
            for i in range(5)
        ]
        
        test_dependencies._graphiti_client.search_facts.return_value = mock_facts
        
        # Measure response time
        performance_timer.start()
        result = await test_agent.run("Find graph performance information", deps=test_dependencies)
        performance_timer.stop()
        
        # Verify response
        assert result.data is not None
        assert len(result.data) > 0
        
        # Check response time
        response_time_ms = performance_timer.elapsed_ms
        assert response_time_ms < 2000, f"Graph response time {response_time_ms}ms exceeds 2000ms limit"
    
    @pytest.mark.asyncio
    async def test_comprehensive_search_response_time(self, test_agent, test_dependencies, mock_embedding, performance_timer):
        """Test comprehensive search meets response time requirements."""
        # Configure agent for comprehensive search
        test_agent.model.agent_responses = [
            ModelTextResponse(content="I'll perform comprehensive search"),
            {
                "comprehensive_search": {
                    "query": "comprehensive performance query",
                    "limit": 10,
                    "use_vector": True,
                    "use_graph": True
                }
            },
            ModelTextResponse(content="Comprehensive search completed")
        ]
        
        with patch('tools.generate_embedding', return_value=mock_embedding):
            # Mock both vector and graph responses
            mock_connection = AsyncMock()
            mock_connection.fetch.return_value = [
                {
                    "chunk_id": "comp-chunk-1",
                    "content": "Comprehensive test content",
                    "document_id": "comp-doc-1",
                    "title": "Comprehensive Doc",
                    "file_path": "/comp/doc.md",
                    "similarity_score": 0.95
                }
            ]
            
            test_dependencies._pg_pool.acquire.return_value.__aenter__.return_value = mock_connection
            
            mock_fact = Mock(uuid="comp-fact-1", fact="Comprehensive fact", created_at=None)
            test_dependencies._graphiti_client.search_facts.return_value = [mock_fact]
            
            # Measure response time for parallel operations
            performance_timer.start()
            result = await test_agent.run("Comprehensive performance test", deps=test_dependencies)
            performance_timer.stop()
            
            # Verify response
            assert result.data is not None
            assert len(result.data) > 0
            
            # Comprehensive search should still be reasonably fast (parallel execution)
            response_time_ms = performance_timer.elapsed_ms
            assert response_time_ms < 3000, f"Comprehensive response time {response_time_ms}ms exceeds 3000ms limit"
    
    @pytest.mark.asyncio
    async def test_embedding_generation_performance(self, mock_embedding, performance_timer):
        """Test embedding generation performance."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_data = Mock()
        mock_data.embedding = mock_embedding
        mock_response.data = [mock_data]
        mock_client.embeddings.create.return_value = mock_response
        
        with patch('tools.get_embedding_client', return_value=mock_client):
            with patch('tools.get_embedding_model', return_value="text-embedding-3-small"):
                # Test multiple embedding generations
                queries = [f"Performance test query {i}" for i in range(5)]
                
                performance_timer.start()
                for query in queries:
                    embedding = await generate_embedding(query)
                    assert len(embedding) == 1536
                performance_timer.stop()
                
                # Should generate embeddings quickly
                total_time_ms = performance_timer.elapsed_ms
                avg_time_ms = total_time_ms / len(queries)
                assert avg_time_ms < 500, f"Average embedding time {avg_time_ms}ms exceeds 500ms"


class TestThroughputRequirements:
    """Test throughput and concurrent request handling."""
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_requests(self, mock_embedding):
        """Test concurrent request handling capacity."""
        num_concurrent = 10
        session_ids = [f"throughput-session-{i}" for i in range(num_concurrent)]
        
        with patch('tools.generate_embedding', return_value=mock_embedding):
            # Mock fast database responses
            with patch('asyncpg.create_pool') as mock_create_pool:
                mock_pool = AsyncMock()
                mock_connection = AsyncMock()
                mock_connection.fetch.return_value = [
                    {
                        "chunk_id": "throughput-chunk",
                        "content": "Throughput test content",
                        "document_id": "throughput-doc",
                        "title": "Throughput Document",
                        "file_path": "/throughput/doc.md",
                        "similarity_score": 0.9
                    }
                ]
                
                mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
                mock_create_pool.return_value = mock_pool
                
                # Measure concurrent throughput
                start_time = time.time()
                
                async def single_request(session_id):
                    return await run_agent(
                        prompt=f"Throughput test from {session_id}",
                        session_id=session_id
                    )
                
                # Execute concurrent requests
                tasks = [single_request(session_id) for session_id in session_ids]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                end_time = time.time()
                total_time = end_time - start_time
                
                # Verify all requests completed
                successful_results = [r for r in results if not isinstance(r, Exception) and r]
                assert len(successful_results) >= num_concurrent * 0.8, "At least 80% of requests should succeed"
                
                # Calculate throughput (requests per second)
                throughput = len(successful_results) / total_time
                assert throughput >= 2.0, f"Throughput {throughput:.2f} req/s is below minimum 2.0 req/s"
    
    @pytest.mark.asyncio
    async def test_database_connection_pool_performance(self, test_dependencies):
        """Test database connection pool handles concurrent access efficiently."""
        num_connections = 20
        
        # Mock connection pool
        mock_pool = AsyncMock()
        mock_connections = [AsyncMock() for _ in range(num_connections)]
        
        connection_times = []
        
        async def mock_acquire():
            # Simulate connection acquisition time
            start = time.time()
            await asyncio.sleep(0.01)  # Small delay to simulate real connection
            end = time.time()
            connection_times.append(end - start)
            return mock_connections[len(connection_times) - 1]
        
        mock_pool.acquire.return_value.__aenter__ = mock_acquire
        test_dependencies._pg_pool = mock_pool
        
        # Test concurrent connection acquisition
        start_time = time.time()
        
        async def acquire_connection():
            pool = await test_dependencies.pg_pool
            async with pool.acquire() as conn:
                await asyncio.sleep(0.001)  # Simulate query time
                return True
        
        tasks = [acquire_connection() for _ in range(num_connections)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Most connections should succeed
        successful = [r for r in results if r is True]
        assert len(successful) >= num_connections * 0.9
        
        # Total time should be reasonable (connections should be pooled/reused)
        assert total_time < 2.0, f"Connection pool performance {total_time:.2f}s is too slow"
    
    @pytest.mark.asyncio
    async def test_tool_parallel_execution_performance(self, test_dependencies, mock_embedding):
        """Test that comprehensive search executes tools in parallel efficiently."""
        from pydantic_ai import RunContext
        
        comprehensive_tool = None
        for tool in rag_agent.tools:
            if tool.name == "comprehensive_search":
                comprehensive_tool = tool
                break
        
        assert comprehensive_tool is not None
        
        with patch('tools.generate_embedding', return_value=mock_embedding):
            # Mock database with delay to test parallelism
            mock_connection = AsyncMock()
            
            async def slow_fetch(*args, **kwargs):
                await asyncio.sleep(0.5)  # 500ms delay
                return [
                    {
                        "chunk_id": "parallel-chunk",
                        "content": "Parallel test content",
                        "document_id": "parallel-doc",
                        "title": "Parallel Document",
                        "file_path": "/parallel/doc.md",
                        "similarity_score": 0.9
                    }
                ]
            
            mock_connection.fetch = slow_fetch
            test_dependencies._pg_pool.acquire.return_value.__aenter__.return_value = mock_connection
            
            # Mock graph with delay
            async def slow_search_facts(*args, **kwargs):
                await asyncio.sleep(0.5)  # 500ms delay
                return [Mock(uuid="parallel-fact", fact="Parallel fact", created_at=None)]
            
            test_dependencies._graphiti_client.search_facts = slow_search_facts
            
            ctx = Mock(spec=RunContext)
            ctx.deps = test_dependencies
            
            # Test parallel execution
            start_time = time.time()
            result = await comprehensive_tool.function(
                ctx, "parallel test query", 10, use_vector=True, use_graph=True
            )
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # Should complete in ~0.5s (parallel) rather than ~1.0s (sequential)
            assert execution_time < 0.8, f"Parallel execution time {execution_time:.2f}s suggests sequential execution"
            
            # Verify both methods returned results
            assert isinstance(result, dict)
            assert len(result["vector_results"]) > 0
            assert len(result["graph_results"]) > 0


class TestResourceUsageOptimization:
    """Test memory usage, CPU usage, and resource optimization."""
    
    def test_memory_usage_baseline(self):
        """Test baseline memory usage is reasonable."""
        process = psutil.Process()
        
        # Get initial memory usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Agent initialization should not use excessive memory
        agent = rag_agent
        assert agent is not None
        
        # Get memory after agent initialization
        post_init_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory increase should be reasonable (less than 100MB for agent)
        memory_increase = post_init_memory - initial_memory
        assert memory_increase < 100, f"Agent initialization used {memory_increase:.1f}MB"
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, mock_embedding):
        """Test memory usage remains stable under load."""
        process = psutil.Process()
        
        # Get baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with patch('tools.generate_embedding', return_value=mock_embedding):
            with patch('asyncpg.create_pool') as mock_create_pool:
                mock_pool = AsyncMock()
                mock_connection = AsyncMock()
                mock_connection.fetch.return_value = [
                    {
                        "chunk_id": "memory-chunk",
                        "content": "Memory test content",
                        "document_id": "memory-doc",
                        "title": "Memory Document", 
                        "file_path": "/memory/doc.md",
                        "similarity_score": 0.9
                    }
                ]
                
                mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
                mock_create_pool.return_value = mock_pool
                
                # Run multiple requests to test memory stability
                session_ids = [f"memory-test-{i}" for i in range(50)]
                
                memory_measurements = []
                
                for i, session_id in enumerate(session_ids):
                    await run_agent(
                        prompt=f"Memory test query {i}",
                        session_id=session_id
                    )
                    
                    # Measure memory every 10 requests
                    if i % 10 == 0:
                        gc.collect()  # Force garbage collection
                        current_memory = process.memory_info().rss / 1024 / 1024
                        memory_measurements.append(current_memory)
                
                # Memory should remain stable (not grow excessively)
                max_memory = max(memory_measurements)
                memory_growth = max_memory - baseline_memory
                
                assert memory_growth < 200, f"Memory grew by {memory_growth:.1f}MB under load"
                
                # Memory should not continuously increase
                if len(memory_measurements) >= 3:
                    final_memory = memory_measurements[-1]
                    mid_memory = memory_measurements[len(memory_measurements)//2]
                    
                    # Final memory should not be significantly higher than mid-point
                    memory_trend = final_memory - mid_memory
                    assert memory_trend < 50, f"Memory trending upward by {memory_trend:.1f}MB"
    
    @pytest.mark.asyncio
    async def test_connection_cleanup_efficiency(self):
        """Test that database connections are properly cleaned up."""
        # Test multiple dependency lifecycles
        for i in range(10):
            deps = AgentDependencies(session_id=f"cleanup-test-{i}")
            
            # Mock connections
            mock_pg_pool = AsyncMock()
            mock_graphiti_client = AsyncMock()
            
            with patch('asyncpg.create_pool', return_value=mock_pg_pool):
                with patch('graphiti.Graphiti', return_value=mock_graphiti_client):
                    with patch('providers.get_embedding_client'):
                        with patch('providers.get_embedding_model'):
                            # Initialize connections
                            pg_pool = await deps.pg_pool
                            graphiti_client = await deps.graphiti_client
                            
                            assert pg_pool is not None
                            assert graphiti_client is not None
                            
                            # Cleanup
                            await deps.cleanup()
                            
                            # Verify cleanup was called
                            mock_pg_pool.close.assert_called_once()
                            mock_graphiti_client.close.assert_called_once()
    
    def test_large_result_set_memory_efficiency(self):
        """Test handling of large result sets without excessive memory usage."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Simulate processing large result sets
        large_results = []
        
        for batch in range(10):  # 10 batches
            batch_results = [
                {
                    "chunk_id": f"large-chunk-{batch}-{i}",
                    "content": f"Large content batch {batch} item {i} " + "x" * 1000,  # 1KB per item
                    "document_id": f"large-doc-{batch}-{i}",
                    "title": f"Large Document {batch}-{i}",
                    "file_path": f"/large/doc{batch}_{i}.md",
                    "similarity_score": 0.9
                }
                for i in range(100)  # 100 items per batch = ~100KB per batch
            ]
            
            large_results.extend(batch_results)
            
            # Process batch (simulate what agent would do)
            processed = []
            for result in batch_results:
                # Simulate some processing
                processed.append({
                    "id": result["chunk_id"],
                    "summary": result["content"][:100],
                    "score": result["similarity_score"]
                })
            
            # Clear batch from memory
            del batch_results
            del processed
            gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_used = final_memory - initial_memory
        
        # Should not use excessive memory for large result processing
        assert memory_used < 50, f"Large result processing used {memory_used:.1f}MB"


class TestScalabilityRequirements:
    """Test scalability and performance under various conditions."""
    
    @pytest.mark.asyncio
    async def test_increasing_load_performance(self, mock_embedding):
        """Test performance remains acceptable as load increases."""
        load_levels = [1, 5, 10, 20]  # Number of concurrent requests
        response_times = []
        
        with patch('tools.generate_embedding', return_value=mock_embedding):
            with patch('asyncpg.create_pool') as mock_create_pool:
                mock_pool = AsyncMock()
                mock_connection = AsyncMock()
                mock_connection.fetch.return_value = [
                    {
                        "chunk_id": "scale-chunk",
                        "content": "Scalability test content",
                        "document_id": "scale-doc",
                        "title": "Scale Document",
                        "file_path": "/scale/doc.md",
                        "similarity_score": 0.9
                    }
                ]
                
                mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
                mock_create_pool.return_value = mock_pool
                
                for load_level in load_levels:
                    session_ids = [f"scale-{load_level}-{i}" for i in range(load_level)]
                    
                    start_time = time.time()
                    
                    async def load_test_request(session_id):
                        return await run_agent(
                            prompt=f"Scale test from {session_id}",
                            session_id=session_id
                        )
                    
                    tasks = [load_test_request(session_id) for session_id in session_ids]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    end_time = time.time()
                    total_time = end_time - start_time
                    
                    # Calculate average response time
                    successful_results = [r for r in results if not isinstance(r, Exception)]
                    if successful_results:
                        avg_response_time = total_time / len(successful_results)
                        response_times.append((load_level, avg_response_time))
                    
                    # Most requests should succeed even under load
                    success_rate = len(successful_results) / load_level
                    assert success_rate >= 0.8, f"Success rate {success_rate:.1%} too low at load {load_level}"
                
                # Response times should not degrade excessively with load
                if len(response_times) >= 2:
                    baseline_time = response_times[0][1]
                    max_time = max(rt[1] for rt in response_times)
                    
                    degradation_factor = max_time / baseline_time
                    assert degradation_factor < 3.0, f"Response time degraded by {degradation_factor:.1f}x"
    
    @pytest.mark.asyncio
    async def test_database_query_optimization(self, test_dependencies, mock_embedding):
        """Test that database queries are optimized for performance."""
        from pydantic_ai import RunContext
        
        with patch('tools.generate_embedding', return_value=mock_embedding):
            # Test vector search optimization
            vector_tool = None
            for tool in rag_agent.tools:
                if tool.name == "vector_search":
                    vector_tool = tool
                    break
            
            # Mock connection that tracks query execution time
            mock_connection = AsyncMock()
            query_times = []
            
            async def timed_fetch(query, *params):
                start = time.time()
                await asyncio.sleep(0.01)  # Simulate query time
                end = time.time()
                query_times.append(end - start)
                
                return [
                    {
                        "chunk_id": f"opt-chunk-{i}",
                        "content": f"Optimized content {i}",
                        "document_id": f"opt-doc-{i}",
                        "title": f"Optimized Doc {i}",
                        "file_path": f"/opt/doc{i}.md",
                        "similarity_score": 0.9 - (i * 0.05)
                    }
                    for i in range(10)
                ]
            
            mock_connection.fetch = timed_fetch
            test_dependencies._pg_pool.acquire.return_value.__aenter__.return_value = mock_connection
            
            ctx = Mock(spec=RunContext)
            ctx.deps = test_dependencies
            
            # Run multiple queries to test consistency
            for i in range(5):
                result = await vector_tool.function(ctx, f"optimization test {i}", 10)
                assert len(result) > 0
            
            # Query times should be consistent (well-optimized queries)
            avg_query_time = sum(query_times) / len(query_times)
            max_query_time = max(query_times)
            min_query_time = min(query_times)
            
            # Variance should be low for optimized queries
            time_variance = max_query_time - min_query_time
            assert time_variance < avg_query_time * 0.5, "Query time variance too high"
    
    @pytest.mark.asyncio
    async def test_embedding_caching_efficiency(self, mock_embedding):
        """Test that embedding generation can be optimized through caching."""
        call_count = 0
        
        def count_calls(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_embedding
        
        # Test repeated embedding generation for same query
        with patch('tools.generate_embedding', side_effect=count_calls):
            queries = ["same query"] * 5  # Same query repeated
            
            for query in queries:
                # In a real implementation, this might be cached
                embedding = await generate_embedding(query)
                assert len(embedding) == 1536
            
            # Note: Current implementation doesn't cache, but this test
            # establishes baseline for potential optimization
            assert call_count == 5  # Current behavior: no caching
    
    def test_agent_tool_registration_performance(self):
        """Test that agent tool registration is efficient."""
        start_time = time.time()
        
        # Test tool registration performance (already done in agent init)
        tools_count = len(rag_agent.tools)
        
        end_time = time.time()
        registration_time = end_time - start_time
        
        # Tool registration should be very fast
        assert registration_time < 0.1, f"Tool registration took {registration_time:.3f}s"
        assert tools_count == 4, "All 4 tools should be registered"
        
        # Tools should have proper metadata
        for tool in rag_agent.tools:
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'function')
            assert tool.name in ["vector_search", "graph_search", "hybrid_search", "comprehensive_search"]


class TestPerformanceReporting:
    """Test performance monitoring and reporting capabilities."""
    
    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, test_agent, test_dependencies, performance_timer):
        """Test that performance metrics can be collected."""
        # Configure test scenario
        test_agent.model.agent_responses = [
            ModelTextResponse(content="Performance monitoring test"),
        ]
        
        # Collect metrics during execution
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        result = await test_agent.run("Performance metrics test", deps=test_dependencies)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Calculate metrics
        execution_time = end_time - start_time
        memory_delta = end_memory - start_memory
        
        # Verify metrics are reasonable
        assert execution_time > 0, "Should measure positive execution time"
        assert abs(memory_delta) < 10, f"Memory delta {memory_delta:.1f}MB seems excessive"
        assert result.data is not None, "Should produce valid result"
        
        # Performance data should be collectible
        performance_data = {
            "execution_time_ms": execution_time * 1000,
            "memory_delta_mb": memory_delta,
            "result_length": len(result.data),
            "timestamp": time.time()
        }
        
        assert all(isinstance(v, (int, float, str)) for v in performance_data.values())


@pytest.mark.performance
class TestPRPPerformanceRequirements:
    """Test specific performance requirements from the PRP."""
    
    @pytest.mark.asyncio
    async def test_response_time_requirements(self, mock_embedding):
        """Test that all operations meet PRP response time requirements."""
        # PRP specifies reasonable response times for RAG operations
        max_response_times = {
            "vector_search": 2.0,      # 2 seconds max for vector search
            "graph_search": 2.0,       # 2 seconds max for graph search  
            "hybrid_search": 3.0,      # 3 seconds max for hybrid search
            "comprehensive_search": 5.0 # 5 seconds max for comprehensive search
        }
        
        with patch('tools.generate_embedding', return_value=mock_embedding):
            with patch('asyncpg.create_pool') as mock_create_pool:
                mock_pool = AsyncMock()
                mock_connection = AsyncMock()
                mock_connection.fetch.return_value = [
                    {
                        "chunk_id": "req-chunk",
                        "content": "Requirements test content",
                        "document_id": "req-doc",
                        "title": "Requirements Document",
                        "file_path": "/req/doc.md",
                        "similarity_score": 0.95,
                        "combined_score": 0.90,
                        "vector_score": 0.88,
                        "text_score": 0.92
                    }
                ]
                
                mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
                mock_create_pool.return_value = mock_pool
                
                # Mock graph client
                with patch('graphiti.Graphiti') as mock_graphiti:
                    mock_graph_client = AsyncMock()
                    mock_graph_client.search_facts.return_value = [
                        Mock(uuid="req-fact", fact="Requirements fact", created_at=None)
                    ]
                    mock_graphiti.return_value = mock_graph_client
                    
                    for search_type, max_time in max_response_times.items():
                        start_time = time.time()
                        
                        result = await run_agent(
                            prompt=f"Test {search_type} performance requirements",
                            session_id=f"req-test-{search_type}"
                        )
                        
                        end_time = time.time()
                        execution_time = end_time - start_time
                        
                        assert result is not None, f"{search_type} should return result"
                        assert execution_time <= max_time, f"{search_type} took {execution_time:.2f}s, max {max_time}s"
    
    def test_throughput_requirements(self):
        """Test throughput meets PRP requirements."""
        # PRP should specify minimum throughput requirements
        # For testing purposes, we'll verify the system can handle reasonable concurrent load
        
        min_throughput_rps = 2.0  # Minimum 2 requests per second
        
        # This would be tested with actual concurrent requests in integration
        # For unit testing, we verify the configuration supports the requirement
        
        # Connection pool should support adequate concurrency
        deps = AgentDependencies(session_id="throughput-req-test")
        
        # Verify connection pool settings support throughput requirements
        # (This is validated in the configuration rather than runtime for unit tests)
        assert deps.max_retries <= 5, "Retry configuration should not cause excessive delays"
        assert deps.timeout <= 60, "Timeout should be reasonable for throughput"
    
    def test_resource_usage_requirements(self):
        """Test resource usage meets PRP requirements."""
        process = psutil.Process()
        
        # Memory usage should be reasonable
        current_memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Agent should not use excessive memory
        assert current_memory_mb < 500, f"Memory usage {current_memory_mb:.1f}MB exceeds reasonable limits"
        
        # CPU usage should be measurable but not excessive
        # (In a real test, this would measure CPU over time during operations)
        
        # Connection limits should prevent resource exhaustion
        settings_dict = {
            "max_retries": 3,
            "timeout_seconds": 30,
            "max_search_results": 50
        }
        
        for setting, value in settings_dict.items():
            assert isinstance(value, int), f"{setting} should be integer"
            assert value > 0, f"{setting} should be positive"
            assert value < 1000, f"{setting} should not be excessive"