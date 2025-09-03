# RAG Knowledge Graph Assistant - Essential Tool Specifications

This document specifies 4 essential tools for the RAG Knowledge Graph AI Assistant. These tools provide comprehensive search capabilities across vector databases (PostgreSQL pgvector) and knowledge graphs (Neo4j/Graphiti).

## Tool Architecture Overview

The agent uses @agent.tool decorators with RunContext[AgentDependencies] for context-aware operations. All tools include comprehensive error handling and return structured data for the agent to process.

## Essential Tool Set (4 Core Tools)

### 1. Vector Similarity Search Tool

```python
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
```

**Implementation Requirements:**
- Generate embeddings using OpenAI text-embedding-3-small
- Query PostgreSQL pgvector with vector similarity
- Return ChunkResult objects with content, scores, and metadata
- Handle embedding generation failures gracefully
- Log search performance metrics

**Error Handling:**
- Catch embedding API failures and return empty results
- Log search errors without exposing sensitive data
- Handle PostgreSQL connection issues with connection pooling

### 2. Knowledge Graph Search Tool

```python
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
```

**Implementation Requirements:**
- Use Graphiti client for Neo4j knowledge graph queries
- Search across facts, entities, and relationships
- Return GraphSearchResult objects with fact text and temporal data
- Handle graph traversal efficiently with query optimization
- Support entity relationship discovery

**Error Handling:**
- Catch Neo4j connection failures
- Handle query syntax errors gracefully
- Log graph query performance and errors
- Return empty results on failures

### 3. Hybrid Search Tool (Vector + Keyword)

```python
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
```

**Implementation Requirements:**
- Generate embeddings for vector component
- Use PostgreSQL TSVector for full-text search component
- Combine scores using configurable weighting
- Use stored procedures for efficient hybrid ranking
- Return unified results with combined scores

**Error Handling:**
- Handle embedding failures by falling back to text-only search
- Catch PostgreSQL full-text search errors
- Validate text_weight parameter bounds (0.0-1.0)
- Log hybrid search performance metrics

### 4. Comprehensive Search Tool (Parallel Execution)

```python
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
```

**Implementation Requirements:**
- Use asyncio.gather() for parallel execution of vector and graph search
- Handle partial failures gracefully (return successful results)
- Combine results from different search methods
- Return structured results with separate sections for each method
- Track total result counts across all methods

**Error Handling:**
- Use return_exceptions=True in asyncio.gather()
- Check for exceptions in each parallel result
- Log individual search method failures
- Return available results even if some methods fail
- Provide clear error messages for debugging

## Parameter Validation Models

```python
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
```

## Database Integration Patterns

### PostgreSQL pgvector Integration
- Use AsyncPG connection pools for high performance
- Generate embeddings with OpenAI text-embedding-3-small model
- Store vectors as PostgreSQL vector type with proper indexing
- Use stored procedures for efficient similarity search
- Implement connection lifecycle management with proper cleanup

### Neo4j Graphiti Integration
- Use Graphiti client for knowledge graph operations
- Store entities, relationships, and temporal facts
- Support semantic search across graph content
- Handle episode-based content ingestion
- Implement proper graph query optimization

## Agent Dependencies

```python
@dataclass
class AgentDependencies:
    """Dependencies for RAG Knowledge Graph agent."""
    session_id: str
    user_id: Optional[str] = None
    search_preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.search_preferences is None:
            self.search_preferences = {
                "use_vector": True,
                "use_graph": True,
                "default_limit": 10
            }
```

## Error Handling Strategy

### Database Connection Errors
- Implement connection pooling with retry logic
- Handle temporary connection failures gracefully
- Log connection issues for monitoring
- Return meaningful error messages to users

### API Integration Errors
- Handle OpenAI API rate limits and failures
- Implement exponential backoff for retries
- Cache embeddings when possible to reduce API calls
- Fall back to text-only search if embedding generation fails

### Search Result Processing
- Validate result formats before returning to agent
- Handle empty result sets appropriately  
- Filter and sanitize content for safe display
- Track search performance metrics

## Security Considerations

### Input Validation
- Use Pydantic models for all tool parameters
- Sanitize search queries to prevent injection attacks
- Validate parameter bounds (limits, weights, dates)
- Log security-related events for monitoring

### Database Security
- Use parameterized queries to prevent SQL injection
- Implement proper connection string handling
- Limit database user permissions appropriately
- Monitor for unusual query patterns

## Performance Requirements

### Response Times
- Vector search: < 500ms for typical queries
- Graph search: < 1000ms for typical queries
- Hybrid search: < 750ms for typical queries
- Comprehensive search: < 1200ms for parallel execution

### Scalability
- Connection pooling for database connections
- Efficient vector indexing (HNSW) for similarity search
- Query optimization for graph traversal
- Result caching for frequently accessed data

## Testing Strategy

### Unit Testing
- Test each tool with TestModel for rapid validation
- Mock database connections for isolated testing
- Validate parameter parsing and error handling
- Test edge cases (empty results, invalid parameters)

### Integration Testing  
- Test with real PostgreSQL and Neo4j connections
- Validate search result quality and relevance
- Test parallel execution and error handling
- Performance testing with realistic data volumes

### End-to-End Testing
- Test complete search workflows with agent
- Validate conversation context and memory
- Test streaming responses and tool visibility
- Monitor search performance under load

## Configuration Requirements

### Environment Variables
```bash
# LLM Configuration
LLM_PROVIDER=openai
LLM_API_KEY=sk-your-openai-api-key
LLM_MODEL=gpt-4o-mini

# Embedding Configuration  
EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=sk-your-openai-api-key
EMBEDDING_MODEL=text-embedding-3-small

# Database Configuration
DATABASE_URL=postgresql://raguser:ragpass@localhost:5432/agentic_rag
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=neo4jpass123
```

### Default Values
- Vector search limit: 10 results
- Hybrid search text weight: 0.3
- Graph search depth: 2 levels
- Connection pool size: 10 connections
- Query timeout: 30 seconds

## Tool Registration Pattern

```python
def register_tools(agent, deps_type):
    """
    Register all essential tools with the agent.
    
    Args:
        agent: Pydantic AI agent instance
        deps_type: Agent dependencies type (AgentDependencies)
    """
    
    # Tools are registered using @agent.tool decorators
    # in the main agent module with proper RunContext typing
    
    logger.info(f"Registered {len(agent.tools)} tools with RAG agent")
    
    # Expected tools: vector_search, graph_search, hybrid_search, comprehensive_search
    expected_tools = ["vector_search", "graph_search", "hybrid_search", "comprehensive_search"]
    
    for tool_name in expected_tools:
        if tool_name not in [tool.name for tool in agent.tools]:
            logger.warning(f"Expected tool '{tool_name}' not found in agent tools")
```

## Implementation Notes

### Simplicity Principles
- Each tool has a single, clear purpose
- Minimal parameter sets (1-4 parameters per tool)
- Direct implementation without complex abstractions
- Focus on the 4 essential search capabilities only

### Pattern Selection
- Use @agent.tool for all tools (they all need database context)
- Avoid complex tool chaining or dynamic tool generation
- Keep dependency injection simple with dataclass pattern
- Return structured dictionaries for agent consumption

### Quality Checklist
- ✅ All 4 required search integrations specified
- ✅ Comprehensive error handling strategy defined
- ✅ Type hints and parameter validation included
- ✅ Database integration patterns documented
- ✅ Security measures outlined
- ✅ Performance requirements specified
- ✅ Testing strategy comprehensive
- ✅ Configuration properly defined

This specification provides everything needed to implement the essential RAG Knowledge Graph search tools while maintaining simplicity and focus on core functionality.