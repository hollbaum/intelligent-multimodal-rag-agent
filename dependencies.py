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
            from settings import settings
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
            from settings import settings
            from providers import get_embedding_client, get_embedding_model
            
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