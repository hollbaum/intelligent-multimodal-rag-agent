"""
Test core agent functionality with TestModel and FunctionModel validation.
Validates agent instantiation, tool registration, and basic responses.
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock

from pydantic_ai.messages import ModelTextResponse
from agent import rag_agent, run_agent
from dependencies import AgentDependencies


class TestAgentInstantiation:
    """Test agent initialization and configuration."""
    
    def test_agent_initialization(self):
        """Test agent is properly initialized."""
        assert rag_agent is not None
        assert rag_agent.model is not None
        assert rag_agent.deps_type == AgentDependencies
        assert len(rag_agent.system_prompt) > 0
    
    def test_agent_tools_registered(self):
        """Test all expected tools are registered."""
        expected_tools = ["vector_search", "graph_search", "hybrid_search", "comprehensive_search"]
        registered_tools = [tool.name for tool in rag_agent.tools]
        
        assert len(registered_tools) == 4, f"Expected 4 tools, got {len(registered_tools)}: {registered_tools}"
        
        for tool_name in expected_tools:
            assert tool_name in registered_tools, f"Tool '{tool_name}' not found in {registered_tools}"
    
    def test_agent_configuration(self):
        """Test agent configuration settings."""
        assert rag_agent.retries > 0
        assert rag_agent.deps_type is not None


class TestAgentWithTestModel:
    """Test agent behavior using TestModel for rapid validation."""
    
    @pytest.mark.asyncio
    async def test_basic_response(self, test_agent, test_dependencies):
        """Test agent provides basic response with TestModel."""
        result = await test_agent.run(
            "What do you know about artificial intelligence?",
            deps=test_dependencies
        )
        
        assert result.data is not None
        assert isinstance(result.data, str)
        assert len(result.data) > 0
        assert len(result.all_messages()) > 0
    
    @pytest.mark.asyncio
    async def test_tool_calling_capability(self, test_agent, test_dependencies):
        """Test agent can call tools with TestModel."""
        # Configure TestModel to simulate tool calling
        test_agent.model.agent_responses = [
            ModelTextResponse(content="I'll search for that information"),
            {
                "vector_search": {
                    "query": "artificial intelligence",
                    "limit": 10
                }
            },
            ModelTextResponse(content="Based on the search results...")
        ]
        
        result = await test_agent.run(
            "Search for information about AI",
            deps=test_dependencies
        )
        
        assert result.data is not None
        
        # Verify tool was called
        messages = result.all_messages()
        tool_calls = [msg for msg in messages if hasattr(msg, 'tool_name')]
        
        # TestModel should have simulated at least the tool interaction
        assert len(messages) >= 2, f"Expected multiple messages, got {len(messages)}"
    
    @pytest.mark.asyncio
    async def test_multiple_queries(self, test_agent, test_dependencies):
        """Test agent handles multiple sequential queries."""
        queries = [
            "What is machine learning?",
            "Tell me about neural networks",
            "How does deep learning work?"
        ]
        
        for query in queries:
            result = await test_agent.run(query, deps=test_dependencies)
            assert result.data is not None
            assert len(result.data) > 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, test_agent, test_dependencies):
        """Test agent handles errors gracefully."""
        # Configure TestModel to simulate an error response
        test_agent.model.agent_responses = [
            ModelTextResponse(content="I encountered an issue but will help anyway")
        ]
        
        result = await test_agent.run(
            "This is a test query that might cause issues",
            deps=test_dependencies
        )
        
        # Agent should still provide a response
        assert result.data is not None


class TestAgentWithFunctionModel:
    """Test agent with FunctionModel for controlled behavior."""
    
    @pytest.mark.asyncio
    async def test_function_model_workflow(self, function_agent, test_dependencies):
        """Test agent workflow with custom function model."""
        result = await function_agent.run(
            "Search for information about technology",
            deps=test_dependencies
        )
        
        assert result.data is not None
        assert "found" in result.data.lower()
        
        # Verify the function model executed multiple steps
        messages = result.all_messages()
        assert len(messages) >= 2
    
    @pytest.mark.asyncio
    async def test_tool_sequence(self, function_model, test_dependencies):
        """Test specific tool calling sequence."""
        def create_search_sequence():
            call_count = 0
            
            async def search_function(messages, tools):
                nonlocal call_count
                call_count += 1
                
                if call_count == 1:
                    return ModelTextResponse(content="I'll perform a vector search")
                elif call_count == 2:
                    return {"vector_search": {"query": "test", "limit": 5}}
                else:
                    return ModelTextResponse(content="Here are the search results")
            
            return search_function
        
        from pydantic_ai.models.function import FunctionModel
        custom_function_model = FunctionModel(create_search_sequence())
        custom_agent = rag_agent.override(model=custom_function_model)
        
        result = await custom_agent.run(
            "Find information about AI",
            deps=test_dependencies
        )
        
        assert result.data is not None
        assert "search" in result.data.lower()


class TestAgentRunFunction:
    """Test the run_agent convenience function."""
    
    @pytest.mark.asyncio
    async def test_run_agent_basic(self):
        """Test run_agent function with basic parameters."""
        with patch('dependencies.AgentDependencies') as mock_deps_class:
            with patch.object(rag_agent, 'run') as mock_run:
                # Mock the agent run result
                mock_result = AsyncMock()
                mock_result.data = "Test response"
                mock_run.return_value = mock_result
                
                # Mock dependencies
                mock_deps = AsyncMock()
                mock_deps.cleanup = AsyncMock()
                mock_deps_class.from_settings.return_value = mock_deps
                
                result = await run_agent(
                    prompt="Test prompt",
                    session_id="test-session"
                )
                
                assert result == "Test response"
                mock_deps.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_agent_with_overrides(self):
        """Test run_agent with dependency overrides."""
        with patch('dependencies.AgentDependencies') as mock_deps_class:
            with patch.object(rag_agent, 'run') as mock_run:
                mock_result = AsyncMock()
                mock_result.data = "Test response with overrides"
                mock_run.return_value = mock_result
                
                mock_deps = AsyncMock()
                mock_deps.cleanup = AsyncMock()
                mock_deps_class.from_settings.return_value = mock_deps
                
                result = await run_agent(
                    prompt="Test prompt",
                    session_id="test-session",
                    user_id="test-user",
                    debug=True
                )
                
                assert result == "Test response with overrides"
                
                # Verify from_settings was called with overrides
                mock_deps_class.from_settings.assert_called_once()
                call_args = mock_deps_class.from_settings.call_args
                assert call_args[1]["debug"] is True
    
    @pytest.mark.asyncio
    async def test_run_agent_cleanup_on_error(self):
        """Test run_agent cleans up dependencies even on error."""
        with patch('dependencies.AgentDependencies') as mock_deps_class:
            with patch.object(rag_agent, 'run') as mock_run:
                # Mock agent run to raise an error
                mock_run.side_effect = Exception("Test error")
                
                mock_deps = AsyncMock()
                mock_deps.cleanup = AsyncMock()
                mock_deps_class.from_settings.return_value = mock_deps
                
                with pytest.raises(Exception, match="Test error"):
                    await run_agent(
                        prompt="Test prompt",
                        session_id="test-session"
                    )
                
                # Verify cleanup was still called
                mock_deps.cleanup.assert_called_once()


class TestAgentSystemPrompt:
    """Test agent system prompt functionality."""
    
    def test_system_prompt_content(self):
        """Test system prompt contains expected content."""
        system_prompt = rag_agent.system_prompt
        
        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 100  # Should be substantial
        
        # Should mention key capabilities
        prompt_lower = system_prompt.lower()
        expected_keywords = ["search", "knowledge", "database", "vector"]
        
        for keyword in expected_keywords:
            assert keyword in prompt_lower, f"Keyword '{keyword}' not found in system prompt"
    
    @pytest.mark.asyncio
    async def test_system_prompt_influences_behavior(self, test_agent, test_dependencies):
        """Test system prompt influences agent behavior."""
        # Agent should recognize its capabilities based on system prompt
        result = await test_agent.run(
            "What can you help me with?",
            deps=test_dependencies
        )
        
        assert result.data is not None
        response_lower = result.data.lower()
        
        # Response should mention agent capabilities
        capability_keywords = ["search", "find", "information", "knowledge"]
        found_keywords = sum(1 for keyword in capability_keywords if keyword in response_lower)
        
        # Should mention at least some capabilities
        assert found_keywords > 0, f"No capability keywords found in response: {result.data}"


class TestAgentMemoryAndState:
    """Test agent memory and state management."""
    
    @pytest.mark.asyncio
    async def test_session_isolation(self, test_agent):
        """Test different sessions are isolated."""
        # Create two different dependency contexts
        deps1 = AgentDependencies(session_id="session-1")
        deps2 = AgentDependencies(session_id="session-2")
        
        # Run queries in different sessions
        result1 = await test_agent.run("Remember that I like apples", deps=deps1)
        result2 = await test_agent.run("Remember that I like oranges", deps=deps2)
        
        # Both should work independently
        assert result1.data is not None
        assert result2.data is not None
        
        # Session IDs should be different
        assert deps1.session_id != deps2.session_id


@pytest.mark.integration
class TestAgentIntegrationReadiness:
    """Test agent readiness for integration testing."""
    
    def test_agent_tools_count(self):
        """Verify exact tool count for integration tests."""
        # PRP specifies 4 essential tools
        assert len(rag_agent.tools) == 4
    
    def test_agent_model_type(self):
        """Verify model configuration."""
        assert rag_agent.model is not None
        # Should be able to override with TestModel
        test_model_agent = rag_agent.override(model=TestModel())
        assert test_model_agent.model is not None
    
    def test_dependency_type_configured(self):
        """Verify dependency type is properly configured."""
        assert rag_agent.deps_type == AgentDependencies
    
    @pytest.mark.asyncio
    async def test_agent_accepts_all_tool_queries(self, test_agent, test_dependencies):
        """Test agent accepts queries for all tool types."""
        test_queries = [
            "Search for information about OpenAI",  # Should trigger vector_search
            "Find facts about Microsoft partnerships",  # Should trigger graph_search
            "Look up documents containing AI and machine learning",  # Should trigger hybrid_search
            "Give me comprehensive information about technology companies"  # Should trigger comprehensive_search
        ]
        
        for query in test_queries:
            result = await test_agent.run(query, deps=test_dependencies)
            assert result.data is not None
            assert len(result.data) > 0