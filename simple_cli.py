#!/usr/bin/env python3
"""
Simple CLI for RAG Knowledge Graph AI Assistant.
Direct agent interaction without API layer.
"""

import asyncio
import logging
import sys
from datetime import datetime
from uuid import uuid4

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.spinner import Spinner

# Import our agent components
from agent import run_agent
from settings import settings

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGCli:
    """Simple CLI for RAG Knowledge Graph AI Assistant."""
    
    def __init__(self):
        self.console = Console()
        self.session_id = str(uuid4())
        self.user_id = "cli_user"
    
    def print_banner(self):
        """Print welcome banner."""
        banner = """
# 🤖 RAG Knowledge Graph AI Assistant

**Capabilities:**
- Vector similarity search (PostgreSQL pgvector)
- Knowledge graph search (Neo4j + Graphiti)
- Hybrid search (semantic + keyword)
- Comprehensive parallel search

**Commands:**
- Type your questions naturally
- Type 'help' for more commands
- Type 'exit' or Ctrl+C to quit
        """
        self.console.print(Panel(Markdown(banner), title="Welcome", border_style="blue"))
    
    def print_help(self):
        """Print help information."""
        help_text = """
## Available Commands

- **Natural questions**: Ask anything about your data
- **help**: Show this help message
- **clear**: Clear the screen
- **exit** / **quit**: Exit the CLI

## Example Queries

- "What funding did OpenAI receive?"
- "Compare Microsoft and Google AI strategies"  
- "What relationships exist between tech companies?"
- "Tell me about Sam Altman"
- "Show me documents about acquisitions"

## Search Types

The agent automatically chooses the best search methods:
- **Vector search**: For semantic similarity
- **Hybrid search**: For semantic + keyword matching
- **Graph search**: For relationships and facts
- **Comprehensive**: Parallel execution of multiple methods
        """
        self.console.print(Panel(Markdown(help_text), title="Help", border_style="green"))
    
    async def run_query(self, query: str) -> str:
        """Run a query through the agent."""
        try:
            with self.console.status(f"[bold green]Processing query...", spinner="dots"):
                response = await run_agent(
                    prompt=query,
                    session_id=self.session_id,
                    user_id=self.user_id
                )
            return response
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return f"❌ Error processing query: {str(e)}"
    
    def display_response(self, response: str):
        """Display agent response with formatting."""
        self.console.print(Panel(
            Markdown(response), 
            title="🤖 Agent Response", 
            border_style="cyan",
            padding=(1, 1)
        ))
    
    async def interactive_loop(self):
        """Main interactive CLI loop."""
        self.print_banner()
        
        while True:
            try:
                # Get user input
                query = Prompt.ask("\n[bold blue]Your question")
                
                if not query.strip():
                    continue
                
                # Handle commands
                query_lower = query.lower().strip()
                
                if query_lower in ['exit', 'quit', 'q']:
                    self.console.print("\n[green]Goodbye! 👋[/green]")
                    break
                    
                elif query_lower == 'help':
                    self.print_help()
                    continue
                    
                elif query_lower == 'clear':
                    self.console.clear()
                    self.print_banner()
                    continue
                
                # Process query through agent
                response = await self.run_query(query)
                self.display_response(response)
                
            except KeyboardInterrupt:
                self.console.print("\n[green]Goodbye! 👋[/green]")
                break
            except Exception as e:
                logger.error(f"CLI error: {e}")
                self.console.print(f"[red]❌ CLI Error: {str(e)}[/red]")

async def main():
    """Main function."""
    # Check if databases are configured
    try:
        from settings import settings
        if not settings.llm_api_key.startswith('sk-'):
            print("❌ OpenAI API key not configured. Please check your .env file.")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        print("Please ensure .env file is configured properly.")
        sys.exit(1)
    
    # Start CLI
    cli = RAGCli()
    await cli.interactive_loop()

if __name__ == "__main__":
    asyncio.run(main())