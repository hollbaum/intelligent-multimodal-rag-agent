# 🤖 RAG Knowledge Graph AI Assistant

> **A comprehensive AI assistant combining traditional RAG with knowledge graph capabilities using PostgreSQL pgvector and Neo4j/Graphiti**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-129%20tests-green.svg)](tests/)

## 🌟 Overview

This project implements an intelligent AI assistant that combines the power of:

- **Vector Similarity Search** via PostgreSQL with pgvector for semantic content discovery
- **Knowledge Graph Search** via Neo4j and Graphiti for relationship and temporal reasoning  
- **Hybrid Search** combining semantic similarity with keyword matching
- **Comprehensive Search** with parallel execution for maximum coverage

The agent is built using [Pydantic AI](https://ai.pydantic.dev/) with production-ready patterns including type safety, error handling, and comprehensive testing.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Knowledge Graph Assistant            │
├─────────────────────────────────────────────────────────────┤
│  🤖 Pydantic AI Agent                                      │
│  ├── System Prompts (prompts.py)                          │
│  ├── 4 Essential Tools (tools.py)                         │
│  ├── Dependencies (dependencies.py)                       │
│  └── Model Providers (providers.py)                       │
├─────────────────────────────────────────────────────────────┤
│  🔍 Search Capabilities                                     │
│  ├── Vector Search (PostgreSQL + pgvector)                │
│  ├── Graph Search (Neo4j + Graphiti)                      │
│  ├── Hybrid Search (Vector + Keyword)                     │
│  └── Comprehensive Search (Parallel execution)             │
├─────────────────────────────────────────────────────────────┤
│  💾 Data Layer                                             │
│  ├── PostgreSQL with pgvector extension                   │
│  ├── Neo4j knowledge graph database                       │
│  └── OpenAI embeddings (text-embedding-3-small)           │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.11+**
- **Docker & Docker Compose** (for databases)
- **OpenAI API Key** (for embeddings and LLM)

### 2. Installation

```bash
# Clone or download the project
cd rag-knowledge-graph-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
vim .env
```

Required environment variables:
```bash
# OpenAI Configuration
LLM_API_KEY=sk-your-openai-api-key-here
EMBEDDING_API_KEY=sk-your-openai-api-key-here

# Database URLs (default for Docker setup)
DATABASE_URL=postgresql://raguser:ragpass123@localhost:5432/agentic_rag
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=neo4jpass123
```

### 4. Start Databases

```bash
# Start PostgreSQL and Neo4j with Docker Compose
docker-compose up -d

# Verify databases are running
docker-compose ps
```

### 5. Ingest Sample Data

```bash
# Run ingestion pipeline with sample documents
python -m ingestion.ingest big_tech_docs/

# This will:
# - Process 22 sample documents about tech companies
# - Generate vector embeddings
# - Build knowledge graph with entities and relationships
# - Takes ~5-10 minutes depending on OpenAI API speed
```

### 6. Run the Assistant

```bash
# Interactive CLI
python simple_cli.py

# Or test agent directly
python agent.py
```

## 💡 Usage Examples

### Interactive CLI

```bash
$ python simple_cli.py

🤖 RAG Knowledge Graph AI Assistant

Your question: What funding did OpenAI receive?

🤖 Agent Response
Based on the search results, OpenAI has received significant funding including:

**Latest Funding Round:**
- $6.6 billion Series B funding round completed in October 2024
- This brings OpenAI's valuation to $157 billion
- Led by Thrive Capital with participation from Microsoft, NVIDIA, and others

**Historical Context:**
- Previous funding included partnerships with Microsoft
- The company has secured over $13 billion in total funding
- Microsoft remains a key strategic investor and compute provider

*Sources: OpenAI funding announcements, Series B documentation*
```

### Programmatic Usage

```python
import asyncio
from agent import run_agent

async def example():
    response = await run_agent(
        prompt="Compare Microsoft and Google AI strategies",
        session_id="example_session"
    )
    print(response)

asyncio.run(example())
```

## 🔧 Core Features

### Search Capabilities

| Search Type | Description | Use Case |
|------------|-------------|----------|
| **Vector Search** | Semantic similarity via pgvector | General content discovery |
| **Graph Search** | Entity relationships via Neo4j | Find connections and facts |
| **Hybrid Search** | Vector + keyword matching | Precise term + semantic search |
| **Comprehensive** | Parallel multi-search | Maximum coverage queries |

### Agent Tools

The assistant has 4 essential tools automatically selected based on query type:

1. **`vector_search`** - Semantic similarity search across document chunks
2. **`graph_search`** - Knowledge graph queries for facts and relationships  
3. **`hybrid_search`** - Combined semantic and keyword search
4. **`comprehensive_search`** - Parallel execution of multiple search methods

## 📊 Sample Data

The project includes 22 sample documents (4,376 lines) covering:

- **OpenAI funding and developments**
- **Microsoft-OpenAI partnership tensions**
- **Google AI strategy and competition**
- **Meta's AI acquisitions and Scale AI**
- **NVIDIA's market dominance**
- **Startup ecosystem and investments**
- **Regulatory landscape**
- **Executive movements**

Perfect for testing the assistant's capabilities across different content types and relationships.

## 🧪 Testing

### Run Test Suite

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test categories  
pytest -m "unit"          # Unit tests only
pytest -m "integration"   # Integration tests only
pytest -m "security"      # Security validation tests
pytest -m "performance"   # Performance tests
```

### Test Coverage

- **129 total test methods** across 6 test files
- **43 test classes** covering all components
- **85 async tests** for concurrent operations
- **Complete mocking** for offline testing
- **Security validation** for injection prevention
- **Performance testing** for response times

See [`tests/VALIDATION_REPORT.md`](tests/VALIDATION_REPORT.md) for detailed validation results.

## 🛠️ Development

### Project Structure

```
├── agent.py              # Main agent definition
├── tools.py              # 4 essential search tools  
├── settings.py           # Environment configuration
├── providers.py          # OpenAI model providers
├── dependencies.py       # Agent dependencies
├── prompts.py            # System prompts
├── simple_cli.py         # Interactive CLI
├── docker-compose.yml    # Database setup
├── requirements.txt      # Python dependencies
├── .env.example          # Environment template
├── agent/                # Database utilities
│   ├── db_utils.py       # PostgreSQL operations
│   ├── graph_utils.py    # Neo4j/Graphiti operations  
│   └── models.py         # Pydantic models
├── ingestion/            # Data ingestion pipeline
│   ├── ingest.py         # Main ingestion script
│   ├── chunker.py        # Document chunking
│   ├── embedder.py       # Vector generation
│   └── graph_builder.py  # Knowledge graph creation
├── sql/
│   └── schema.sql        # PostgreSQL schema with pgvector
├── big_tech_docs/        # 22 sample documents
├── tests/                # Comprehensive test suite
└── planning/             # Architecture planning docs
```

### Adding New Documents

```bash
# Add your documents to a directory
mkdir my_documents
cp *.pdf *.md *.txt my_documents/

# Run ingestion
python -m ingestion.ingest my_documents/
```

### Extending Tools

```python
# In tools.py, add new tool
def my_custom_tool(agent):
    @agent.tool  
    async def custom_search(
        ctx: RunContext[AgentDependencies],
        query: str
    ) -> List[Dict[str, Any]]:
        """Your custom search implementation."""
        # Implementation here
        pass

# Register in agent.py
def register_all_tools(agent):
    vector_search_tool(agent)
    graph_search_tool(agent) 
    hybrid_search_tool(agent)
    comprehensive_search_tool(agent)
    my_custom_tool(agent)  # Add here
```

## 🔒 Security

### API Key Management

- **Environment variables only** - Never hardcode API keys
- **Validation on startup** - Keys verified before agent starts
- **Secure configuration** - `.env` files excluded from version control
- **Key rotation support** - Update environment and restart

### Input Validation  

- **Pydantic models** validate all tool parameters
- **SQL injection prevention** via parameterized queries
- **Prompt injection detection** in security tests
- **Rate limiting** through database connection pools

### Database Security

- **Connection pooling** with proper limits
- **SSL/TLS support** for production connections  
- **User permissions** follow principle of least privilege
- **Query monitoring** for unusual patterns

## 📈 Performance

### Response Times (Typical)

- **Vector search**: < 500ms
- **Graph search**: < 1000ms  
- **Hybrid search**: < 750ms
- **Comprehensive search**: < 1200ms (parallel)

### Scalability Features

- **Connection pooling** (5-20 connections per database)
- **Async operations** throughout the stack
- **Efficient indexing** (HNSW for vectors, Neo4j optimizations)
- **Result caching** for frequently accessed data
- **Concurrent tool execution** in comprehensive search

## 🚀 Deployment

### Production Environment Variables

```bash
# Production settings
APP_ENV=production
DEBUG=false
LOG_LEVEL=WARNING

# Database with SSL
DATABASE_URL=postgresql://user:pass@prod-host:5432/db?sslmode=require
NEO4J_URI=neo4j+s://prod-neo4j:7687

# Performance tuning
MAX_SEARCH_RESULTS=20
TIMEOUT_SECONDS=60
```

### Docker Production Deployment

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  app:
    build: .
    environment:
      - APP_ENV=production
    depends_on:
      - postgres
      - neo4j
    
  postgres:
    image: pgvector/pgvector:pg16
    volumes:
      - postgres_prod_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_PASSWORD_FILE=/run/secrets/postgres_password
    
  neo4j:
    image: neo4j:5.13
    volumes:
      - neo4j_prod_data:/data
    environment:
      - NEO4J_AUTH=neo4j/$(cat /run/secrets/neo4j_password)
```

### Health Checks

```bash
# Database connectivity
python -c "
import asyncio
from agent.db_utils import test_connection
from agent.graph_utils import test_graph_connection

async def health_check():
    pg_ok = await test_connection()
    neo4j_ok = await test_graph_connection() 
    print(f'PostgreSQL: {\"✅\" if pg_ok else \"❌\"}')
    print(f'Neo4j: {\"✅\" if neo4j_ok else \"❌\"}')

asyncio.run(health_check())
"
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Add tests** for new functionality
4. **Run test suite**: `pytest`
5. **Update documentation** as needed
6. **Submit pull request**

### Code Quality Standards

- **Type hints** on all functions
- **Docstrings** following Google style
- **Error handling** with proper logging
- **Test coverage** for new features
- **Security review** for external inputs

## 📚 Documentation

- **[Architecture Planning](planning/)** - Component design specifications
- **[Test Validation Report](tests/VALIDATION_REPORT.md)** - Comprehensive test results  
- **[PRP Document](PRPs/rag-knowledge-graph-assistant.md)** - Product requirements
- **[API Documentation](agent/)** - Database utilities and models

## 🐛 Troubleshooting

### Common Issues

**1. Database Connection Errors**
```bash
# Check Docker containers
docker-compose ps
docker-compose logs postgres
docker-compose logs neo4j

# Restart databases
docker-compose restart
```

**2. OpenAI API Errors**
```bash
# Verify API key
echo $LLM_API_KEY | cut -c1-10  # Should show "sk-proj-" or "sk-"

# Check API quota
curl -H "Authorization: Bearer $LLM_API_KEY" \
     https://api.openai.com/v1/usage
```

**3. Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Check Python path
python -c "import agent; print('✅ Agent imports OK')"
```

**4. Slow Performance**
```bash
# Check database indices
docker-compose exec postgres psql -U raguser -d agentic_rag \
  -c "SELECT indexname FROM pg_indexes WHERE tablename='document_chunks';"

# Monitor connection pool
python -c "
import asyncio
from dependencies import AgentDependencies
deps = AgentDependencies('test')
print('Pool size:', deps.pg_pool.get_size())
"
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Pydantic AI](https://ai.pydantic.dev/)** - Production-ready AI agent framework
- **[pgvector](https://github.com/pgvector/pgvector)** - Vector similarity search for PostgreSQL
- **[Graphiti](https://github.com/getzep/graphiti)** - Knowledge graph management with Neo4j
- **[OpenAI](https://openai.com)** - Embeddings and language models
- **Sample documents** - Curated content about AI industry developments

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/example/rag-knowledge-graph-assistant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/example/rag-knowledge-graph-assistant/discussions)
- **Documentation**: See `README.md` and `planning/` directory

**Built with ❤️ using Pydantic AI and powered by vector search + knowledge graphs**