"""
RAG Knowledge Graph AI Assistant - Main Agent
"""

import logging
from typing import Optional
from pydantic_ai import Agent

from providers import get_llm_model
from dependencies import AgentDependencies
from settings import settings
from prompts import SYSTEM_PROMPT
from tools import register_all_tools

logger = logging.getLogger(__name__)

# Initialize the agent
rag_agent = Agent(
    get_llm_model(),
    deps_type=AgentDependencies,
    system_prompt=SYSTEM_PROMPT,
    retries=settings.max_retries
)

# Register all tools
register_all_tools(rag_agent)

async def run_agent(
    prompt: str,
    session_id: str,
    user_id: Optional[str] = None,
    **dependency_overrides
) -> str:
    """
    Run the agent with automatic dependency management.
    """
    deps = AgentDependencies.from_settings(
        settings,
        session_id=session_id,
        user_id=user_id,
        **dependency_overrides
    )
    
    try:
        result = await rag_agent.run(prompt, deps=deps)
        return result.data
    finally:
        await deps.cleanup()

if __name__ == "__main__":
    # Basic test to verify agent initialization
    print(f"RAG Agent initialized with {len(rag_agent.tools)} tools:")
    for tool in rag_agent.tools:
        print(f"  - {tool.name}")
    print(f"Model: {rag_agent.model}")
    print("Agent ready for use!")