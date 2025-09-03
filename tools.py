"""
RAG Knowledge Graph Assistant - Core Tools
Implements 4 essential tools based on planning specifications.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from pydantic_ai import RunContext
from pydantic import BaseModel, Field

from dependencies import AgentDependencies
from settings import settings
from providers import get_embedding_client, get_embedding_model

logger = logging.getLogger(__name__)

# Parameter validation models
class VectorSearchParams(BaseModel):
    """Parameters for vector search operations."""
    query: str = Field(..., description="Search query", min_length=1)
    limit: int = Field(10, ge=1, le=50, description="Maximum results")

class GraphSearchParams(BaseModel):
    """Parameters for graph search operations."""
    query: str = Field(..., description="Search query", min_length=1)

class HybridSearchParams(BaseModel):
    """Parameters for hybrid search operations."""
    query: str = Field(..., description="Search query", min_length=1)
    limit: int = Field(10, ge=1, le=50, description="Maximum results")
    text_weight: float = Field(0.3, ge=0.0, le=1.0, description="Text similarity weight")

class ComprehensiveSearchParams(BaseModel):
    """Parameters for comprehensive search operations."""
    query: str = Field(..., description="Search query", min_length=1)
    limit: int = Field(10, ge=1, le=50, description="Maximum results per method")
    use_vector: bool = Field(True, description="Include vector search")
    use_graph: bool = Field(True, description="Include graph search")

async def generate_embedding(query: str) -> List[float]:
    """Generate embedding for query using OpenAI."""
    try:
        client = get_embedding_client()
        model = get_embedding_model()
        
        response = await client.embeddings.create(
            input=query,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        raise

def vector_search_tool(agent):
    """Register vector search tool with agent."""
    
    @agent.tool
    async def vector_search(
        ctx: RunContext[AgentDependencies],
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant information using semantic similarity.
        
        This tool performs vector similarity search across document chunks
        to find semantically related content using PostgreSQL pgvector.
        
        Args:
            query: Search query to find similar content
            limit: Maximum number of results to return (1-50, default 10)
        
        Returns:
            List of matching chunks with content, similarity scores, and metadata
        """
        try:
            # Validate parameters
            params = VectorSearchParams(query=query, limit=limit)
            
            # Generate embedding for query
            embedding = await generate_embedding(params.query)
            embedding_str = f"[{','.join(map(str, embedding))}]"
            
            # Get database connection
            pool = await ctx.deps.pg_pool
            
            async with pool.acquire() as conn:
                # Use vector similarity search
                results = await conn.fetch("""
                    SELECT 
                        c.chunk_id,
                        c.content,
                        c.document_id,
                        d.title,
                        d.file_path,
                        1 - (c.embedding <=> $1::vector) as similarity_score
                    FROM document_chunks c
                    JOIN documents d ON c.document_id = d.document_id
                    WHERE c.embedding IS NOT NULL
                    ORDER BY c.embedding <=> $1::vector
                    LIMIT $2
                """, embedding_str, params.limit)
                
                return [
                    {
                        "chunk_id": str(row["chunk_id"]),
                        "content": row["content"],
                        "document_id": str(row["document_id"]),
                        "title": row["title"],
                        "file_path": row["file_path"],
                        "similarity_score": float(row["similarity_score"]),
                        "search_type": "vector"
                    }
                    for row in results
                ]
                
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

def graph_search_tool(agent):
    """Register graph search tool with agent."""
    
    @agent.tool
    async def graph_search(
        ctx: RunContext[AgentDependencies],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Search the knowledge graph for facts and relationships.
        
        This tool queries Neo4j via Graphiti to find specific facts, 
        entity relationships, and temporal information.
        
        Args:
            query: Search query to find facts and relationships
        
        Returns:
            List of facts with UUIDs, temporal data, and source information
        """
        try:
            # Validate parameters
            params = GraphSearchParams(query=query)
            
            # Get Graphiti client
            graphiti = await ctx.deps.graphiti_client
            
            # Perform semantic search on knowledge graph
            facts = await graphiti.search_facts(params.query, limit=10)
            
            return [
                {
                    "fact_uuid": fact.uuid,
                    "fact_text": fact.fact,
                    "created_at": fact.created_at.isoformat() if fact.created_at else None,
                    "entities": [entity.name for entity in getattr(fact, 'entities', [])],
                    "confidence": getattr(fact, 'confidence', None),
                    "search_type": "graph"
                }
                for fact in facts
            ]
            
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []

def hybrid_search_tool(agent):
    """Register hybrid search tool with agent."""
    
    @agent.tool
    async def hybrid_search(
        ctx: RunContext[AgentDependencies],
        query: str,
        limit: int = 10,
        text_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Perform both vector and keyword search for comprehensive results.
        
        This tool combines semantic similarity search with keyword matching
        using PostgreSQL TSVector for best coverage.
        
        Args:
            query: Search query for hybrid search
            limit: Maximum number of results to return (1-50, default 10)
            text_weight: Weight for text similarity vs vector similarity (0.0-1.0, default 0.3)
        
        Returns:
            List of chunks ranked by combined relevance score
        """
        try:
            # Validate parameters
            params = HybridSearchParams(query=query, limit=limit, text_weight=text_weight)
            
            # Generate embedding for vector component
            try:
                embedding = await generate_embedding(params.query)
                embedding_str = f"[{','.join(map(str, embedding))}]"
                vector_available = True
            except Exception as e:
                logger.warning(f"Embedding generation failed, using text-only search: {e}")
                vector_available = False
            
            # Get database connection
            pool = await ctx.deps.pg_pool
            
            async with pool.acquire() as conn:
                if vector_available:
                    # Hybrid search combining vector and text similarity
                    results = await conn.fetch("""
                        SELECT 
                            c.chunk_id,
                            c.content,
                            c.document_id,
                            d.title,
                            d.file_path,
                            (1 - (c.embedding <=> $1::vector)) * (1 - $3) +
                            ts_rank(c.content_tsvector, plainto_tsquery($2)) * $3 as combined_score,
                            1 - (c.embedding <=> $1::vector) as vector_score,
                            ts_rank(c.content_tsvector, plainto_tsquery($2)) as text_score
                        FROM document_chunks c
                        JOIN documents d ON c.document_id = d.document_id
                        WHERE c.embedding IS NOT NULL 
                            AND c.content_tsvector @@ plainto_tsquery($2)
                        ORDER BY combined_score DESC
                        LIMIT $4
                    """, embedding_str, params.query, params.text_weight, params.limit)
                else:
                    # Text-only search fallback
                    results = await conn.fetch("""
                        SELECT 
                            c.chunk_id,
                            c.content,
                            c.document_id,
                            d.title,
                            d.file_path,
                            ts_rank(c.content_tsvector, plainto_tsquery($1)) as combined_score,
                            0.0 as vector_score,
                            ts_rank(c.content_tsvector, plainto_tsquery($1)) as text_score
                        FROM document_chunks c
                        JOIN documents d ON c.document_id = d.document_id
                        WHERE c.content_tsvector @@ plainto_tsquery($1)
                        ORDER BY combined_score DESC
                        LIMIT $2
                    """, params.query, params.limit)
                
                return [
                    {
                        "chunk_id": str(row["chunk_id"]),
                        "content": row["content"],
                        "document_id": str(row["document_id"]),
                        "title": row["title"],
                        "file_path": row["file_path"],
                        "combined_score": float(row["combined_score"]),
                        "vector_score": float(row["vector_score"]),
                        "text_score": float(row["text_score"]),
                        "search_type": "hybrid"
                    }
                    for row in results
                ]
                
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

def comprehensive_search_tool(agent):
    """Register comprehensive search tool with agent."""
    
    @agent.tool
    async def comprehensive_search(
        ctx: RunContext[AgentDependencies],
        query: str,
        limit: int = 10,
        use_vector: bool = True,
        use_graph: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive search using multiple methods in parallel.
        
        This tool runs vector similarity search and knowledge graph search
        concurrently, then combines results for maximum coverage.
        
        Args:
            query: Search query for comprehensive search
            limit: Maximum results per search method (default 10)
            use_vector: Whether to include vector search (default True)
            use_graph: Whether to include graph search (default True)
        
        Returns:
            Dictionary containing vector_results, graph_results, and total_results count
        """
        try:
            # Validate parameters
            params = ComprehensiveSearchParams(
                query=query, limit=limit, use_vector=use_vector, use_graph=use_graph
            )
            
            # Prepare parallel searches
            search_tasks = []
            
            if params.use_vector:
                search_tasks.append(vector_search(ctx, params.query, params.limit))
            
            if params.use_graph:
                search_tasks.append(graph_search(ctx, params.query))
            
            if not search_tasks:
                return {
                    "vector_results": [],
                    "graph_results": [],
                    "total_results": 0,
                    "search_type": "comprehensive"
                }
            
            # Execute searches in parallel
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            vector_results = []
            graph_results = []
            
            # Process results, handling exceptions
            result_index = 0
            
            if params.use_vector:
                if isinstance(results[result_index], Exception):
                    logger.error(f"Vector search failed: {results[result_index]}")
                else:
                    vector_results = results[result_index]
                result_index += 1
                
            if params.use_graph:
                if isinstance(results[result_index], Exception):
                    logger.error(f"Graph search failed: {results[result_index]}")
                else:
                    graph_results = results[result_index]
            
            return {
                "vector_results": vector_results,
                "graph_results": graph_results,
                "total_results": len(vector_results) + len(graph_results),
                "search_type": "comprehensive"
            }
            
        except Exception as e:
            logger.error(f"Comprehensive search failed: {e}")
            return {
                "vector_results": [],
                "graph_results": [],
                "total_results": 0,
                "error": str(e),
                "search_type": "comprehensive"
            }

def register_all_tools(agent):
    """Register all essential tools with the agent."""
    vector_search_tool(agent)
    graph_search_tool(agent)
    hybrid_search_tool(agent)
    comprehensive_search_tool(agent)
    
    logger.info(f"Registered {len(agent.tools)} tools with RAG agent")
    
    # Verify expected tools are registered
    expected_tools = ["vector_search", "graph_search", "hybrid_search", "comprehensive_search"]
    registered_tools = [tool.name for tool in agent.tools]
    
    for tool_name in expected_tools:
        if tool_name not in registered_tools:
            logger.warning(f"Expected tool '{tool_name}' not found in agent tools")