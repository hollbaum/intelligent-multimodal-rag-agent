"""
Test dependency injection and database connection management.
Validates AgentDependencies, connection pooling, and resource cleanup.
"""

import pytest
import asyncio
import os
from unittest.mock import patch, AsyncMock, Mock
from datetime import datetime, timezone

from dependencies import AgentDependencies
from settings import settings


class TestAgentDependenciesInitialization:
    """Test AgentDependencies initialization and configuration."""
    
    def test_basic_initialization(self):
        """Test basic dependency initialization."""
        deps = AgentDependencies(
            session_id="test-session",
            user_id="test-user"
        )
        
        assert deps.session_id == "test-session"
        assert deps.user_id == "test-user"
        assert deps.max_retries == 3
        assert deps.timeout == 30
        assert deps.debug is False
        assert isinstance(deps.search_preferences, dict)
    
    def test_default_search_preferences(self):
        """Test default search preferences are properly set."""
        deps = AgentDependencies(session_id="test-session")
        
        expected_defaults = {
            "use_vector": True,
            "use_graph": True,
            "default_limit": 10,
            "min_score": 0.7
        }
        
        for key, expected_value in expected_defaults.items():
            assert key in deps.search_preferences
            assert deps.search_preferences[key] == expected_value
    
    def test_custom_search_preferences(self):
        """Test custom search preferences override defaults."""
        custom_prefs = {
            "use_vector": False,
            "use_graph": True,
            "default_limit": 20,
            "custom_setting": "test"
        }
        
        deps = AgentDependencies(
            session_id="test-session",
            search_preferences=custom_prefs
        )
        
        assert deps.search_preferences == custom_prefs
    
    def test_from_settings_creation(self):
        """Test creating dependencies from settings."""
        deps = AgentDependencies.from_settings(
            settings=settings,
            session_id="settings-test",
            user_id="settings-user",
            debug=True
        )
        
        assert deps.session_id == "settings-test"
        assert deps.user_id == "settings-user"
        assert deps.debug is True
        assert deps.max_retries == settings.max_retries
        assert deps.timeout == settings.timeout_seconds


class TestDatabaseConnectionPool:
    """Test PostgreSQL connection pool management."""
    
    @pytest.mark.asyncio
    async def test_pg_pool_lazy_initialization(self):
        """Test PostgreSQL pool is lazily initialized."""
        deps = AgentDependencies(session_id="test-session")
        
        # Initially no pool should be set
        assert deps._pg_pool is None
        
        # Mock asyncpg.create_pool
        mock_pool = AsyncMock()
        
        with patch('asyncpg.create_pool', return_value=mock_pool) as mock_create_pool:
            pool = await deps.pg_pool
            
            assert pool is mock_pool
            assert deps._pg_pool is mock_pool
            
            # Verify pool configuration
            mock_create_pool.assert_called_once_with(
                settings.database_url,
                min_size=5,
                max_size=20,
                max_queries=50000,
                max_inactive_connection_lifetime=300.0
            )
    
    @pytest.mark.asyncio
    async def test_pg_pool_reuse(self):
        """Test PostgreSQL pool is reused after initialization."""
        deps = AgentDependencies(session_id="test-session")
        
        mock_pool = AsyncMock()
        
        with patch('asyncpg.create_pool', return_value=mock_pool) as mock_create_pool:
            # First call initializes
            pool1 = await deps.pg_pool
            
            # Second call reuses
            pool2 = await deps.pg_pool
            
            assert pool1 is pool2
            assert pool1 is mock_pool
            
            # Should only be called once
            mock_create_pool.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_pg_pool_initialization_error(self):
        """Test handling of pool initialization errors."""
        deps = AgentDependencies(session_id="test-session")
        
        with patch('asyncpg.create_pool', side_effect=Exception("Connection failed")):
            with pytest.raises(Exception, match="Connection failed"):
                await deps.pg_pool


class TestGraphitiClientConnection:
    """Test Graphiti knowledge graph client management."""
    
    @pytest.mark.asyncio
    async def test_graphiti_client_lazy_initialization(self):
        """Test Graphiti client is lazily initialized."""
        deps = AgentDependencies(session_id="test-session")
        
        # Initially no client should be set
        assert deps._graphiti_client is None
        
        # Mock Graphiti and dependencies
        mock_client = AsyncMock()
        mock_embedding_client = AsyncMock()
        
        with patch('graphiti.Graphiti', return_value=mock_client) as mock_graphiti:
            with patch('providers.get_embedding_client', return_value=mock_embedding_client):
                with patch('providers.get_embedding_model', return_value="text-embedding-3-small"):
                    client = await deps.graphiti_client
                    
                    assert client is mock_client
                    assert deps._graphiti_client is mock_client
                    
                    # Verify Graphiti configuration
                    mock_graphiti.assert_called_once_with(
                        uri=settings.neo4j_uri,
                        user=settings.neo4j_user,
                        password=settings.neo4j_password,
                        embedder_client=mock_embedding_client,
                        embedder_model="text-embedding-3-small"
                    )
    
    @pytest.mark.asyncio
    async def test_graphiti_client_reuse(self):
        """Test Graphiti client is reused after initialization."""
        deps = AgentDependencies(session_id="test-session")
        
        mock_client = AsyncMock()
        mock_embedding_client = AsyncMock()
        
        with patch('graphiti.Graphiti', return_value=mock_client) as mock_graphiti:
            with patch('providers.get_embedding_client', return_value=mock_embedding_client):
                with patch('providers.get_embedding_model', return_value="text-embedding-3-small"):
                    # First call initializes
                    client1 = await deps.graphiti_client
                    
                    # Second call reuses
                    client2 = await deps.graphiti_client
                    
                    assert client1 is client2
                    assert client1 is mock_client
                    
                    # Should only be called once
                    mock_graphiti.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_graphiti_client_initialization_error(self):
        """Test handling of Graphiti initialization errors."""
        deps = AgentDependencies(session_id="test-session")
        
        with patch('graphiti.Graphiti', side_effect=Exception("Neo4j connection failed")):
            with patch('providers.get_embedding_client'):
                with patch('providers.get_embedding_model'):
                    with pytest.raises(Exception, match="Neo4j connection failed"):
                        await deps.graphiti_client


class TestDependencyCleanup:
    """Test resource cleanup functionality."""
    
    @pytest.mark.asyncio
    async def test_cleanup_with_initialized_connections(self):
        """Test cleanup when both connections are initialized."""
        deps = AgentDependencies(session_id="test-session")
        
        # Mock initialized connections
        mock_pg_pool = AsyncMock()
        mock_graphiti_client = AsyncMock()
        
        deps._pg_pool = mock_pg_pool
        deps._graphiti_client = mock_graphiti_client
        
        await deps.cleanup()
        
        # Verify cleanup methods were called
        mock_pg_pool.close.assert_called_once()
        mock_graphiti_client.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_with_no_connections(self):
        """Test cleanup when no connections are initialized."""
        deps = AgentDependencies(session_id="test-session")
        
        # Should not raise any errors
        await deps.cleanup()
    
    @pytest.mark.asyncio
    async def test_cleanup_with_partial_connections(self):
        """Test cleanup when only some connections are initialized."""
        deps = AgentDependencies(session_id="test-session")
        
        # Only initialize pg_pool
        mock_pg_pool = AsyncMock()
        deps._pg_pool = mock_pg_pool
        
        await deps.cleanup()
        
        # Only pg_pool cleanup should be called
        mock_pg_pool.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_error_handling(self):
        """Test cleanup handles errors gracefully."""
        deps = AgentDependencies(session_id="test-session")
        
        # Mock connections that raise errors on cleanup
        mock_pg_pool = AsyncMock()
        mock_pg_pool.close.side_effect = Exception("Cleanup error")
        
        mock_graphiti_client = AsyncMock()
        mock_graphiti_client.close.side_effect = Exception("Graph cleanup error")
        
        deps._pg_pool = mock_pg_pool
        deps._graphiti_client = mock_graphiti_client
        
        # Should not raise errors even if cleanup fails
        await deps.cleanup()
        
        # Both cleanup methods should have been attempted
        mock_pg_pool.close.assert_called_once()
        mock_graphiti_client.close.assert_called_once()


class TestDependencyConfiguration:
    """Test dependency configuration and validation."""
    
    def test_dependency_validation_with_settings(self):
        """Test dependencies are properly validated against settings."""
        # Test that dependencies use settings values
        deps = AgentDependencies.from_settings(
            settings=settings,
            session_id="config-test"
        )
        
        assert deps.max_retries == settings.max_retries
        assert deps.timeout == settings.timeout_seconds
        assert deps.debug == settings.debug
    
    def test_dependency_override_behavior(self):
        """Test that overrides work properly."""
        custom_retries = 10
        custom_timeout = 60
        
        deps = AgentDependencies.from_settings(
            settings=settings,
            session_id="override-test",
            max_retries=custom_retries,
            timeout=custom_timeout,
            debug=True
        )
        
        assert deps.max_retries == custom_retries
        assert deps.timeout == custom_timeout
        assert deps.debug is True
    
    def test_session_id_required(self):
        """Test that session_id is required."""
        with pytest.raises(TypeError):
            AgentDependencies()


class TestConnectionContextManager:
    """Test connection management in context manager scenarios."""
    
    @pytest.mark.asyncio
    async def test_connection_lifecycle(self):
        """Test full connection lifecycle."""
        deps = AgentDependencies(session_id="lifecycle-test")
        
        mock_pg_pool = AsyncMock()
        mock_graphiti_client = AsyncMock()
        
        with patch('asyncpg.create_pool', return_value=mock_pg_pool):
            with patch('graphiti.Graphiti', return_value=mock_graphiti_client):
                with patch('providers.get_embedding_client'):
                    with patch('providers.get_embedding_model'):
                        # Initialize connections
                        pg_pool = await deps.pg_pool
                        graphiti_client = await deps.graphiti_client
                        
                        assert pg_pool is mock_pg_pool
                        assert graphiti_client is mock_graphiti_client
                        
                        # Cleanup
                        await deps.cleanup()
                        
                        mock_pg_pool.close.assert_called_once()
                        mock_graphiti_client.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_multiple_dependency_instances_isolation(self):
        """Test that multiple dependency instances are isolated."""
        deps1 = AgentDependencies(session_id="test-1")
        deps2 = AgentDependencies(session_id="test-2")
        
        assert deps1.session_id != deps2.session_id
        assert deps1._pg_pool is None
        assert deps2._pg_pool is None
        
        # Initialize one
        mock_pool1 = AsyncMock()
        with patch('asyncpg.create_pool', return_value=mock_pool1):
            pool1 = await deps1.pg_pool
            
            assert deps1._pg_pool is mock_pool1
            assert deps2._pg_pool is None  # Should remain uninitialized


class TestEnvironmentVariableHandling:
    """Test environment variable handling in dependencies."""
    
    @pytest.mark.asyncio
    async def test_database_url_from_settings(self):
        """Test database URL is correctly passed from settings."""
        deps = AgentDependencies(session_id="env-test")
        
        mock_pool = AsyncMock()
        
        with patch('asyncpg.create_pool', return_value=mock_pool) as mock_create_pool:
            await deps.pg_pool
            
            # Verify the database URL from settings was used
            mock_create_pool.assert_called_once()
            call_args = mock_create_pool.call_args[0]
            assert call_args[0] == settings.database_url
    
    @pytest.mark.asyncio  
    async def test_neo4j_credentials_from_settings(self):
        """Test Neo4j credentials are correctly passed from settings."""
        deps = AgentDependencies(session_id="neo4j-test")
        
        mock_client = AsyncMock()
        mock_embedding_client = AsyncMock()
        
        with patch('graphiti.Graphiti', return_value=mock_client) as mock_graphiti:
            with patch('providers.get_embedding_client', return_value=mock_embedding_client):
                with patch('providers.get_embedding_model', return_value="test-model"):
                    await deps.graphiti_client
                    
                    # Verify Neo4j credentials from settings were used
                    mock_graphiti.assert_called_once()
                    call_kwargs = mock_graphiti.call_args[1]
                    assert call_kwargs['uri'] == settings.neo4j_uri
                    assert call_kwargs['user'] == settings.neo4j_user
                    assert call_kwargs['password'] == settings.neo4j_password


@pytest.mark.integration
class TestDependencyIntegration:
    """Integration tests for dependencies with mocked external services."""
    
    @pytest.mark.asyncio
    async def test_full_dependency_lifecycle(self):
        """Test complete dependency lifecycle in realistic scenario."""
        # Simulate full agent usage pattern
        deps = AgentDependencies.from_settings(
            settings=settings,
            session_id="integration-test",
            user_id="test-user",
            debug=True
        )
        
        mock_pg_pool = AsyncMock()
        mock_graphiti_client = AsyncMock()
        mock_embedding_client = AsyncMock()
        
        with patch('asyncpg.create_pool', return_value=mock_pg_pool):
            with patch('graphiti.Graphiti', return_value=mock_graphiti_client):
                with patch('providers.get_embedding_client', return_value=mock_embedding_client):
                    with patch('providers.get_embedding_model', return_value="text-embedding-3-small"):
                        # Initialize dependencies as agent would
                        pg_pool = await deps.pg_pool
                        graphiti_client = await deps.graphiti_client
                        
                        # Verify initialized
                        assert pg_pool is not None
                        assert graphiti_client is not None
                        
                        # Simulate some usage
                        assert deps.session_id == "integration-test"
                        assert deps.user_id == "test-user"
                        assert deps.debug is True
                        
                        # Cleanup
                        await deps.cleanup()
                        
                        # Verify cleanup was called
                        mock_pg_pool.close.assert_called_once()
                        mock_graphiti_client.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_concurrent_dependency_access(self):
        """Test concurrent access to dependency properties."""
        deps = AgentDependencies(session_id="concurrent-test")
        
        mock_pg_pool = AsyncMock()
        
        with patch('asyncpg.create_pool', return_value=mock_pg_pool):
            # Simulate concurrent access to pg_pool property
            tasks = [deps.pg_pool for _ in range(5)]
            results = await asyncio.gather(*tasks)
            
            # All should return the same pool instance
            for result in results:
                assert result is mock_pg_pool
            
            # Pool should only be created once
            assert deps._pg_pool is mock_pg_pool