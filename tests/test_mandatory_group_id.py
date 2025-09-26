"""Test that group_id is mandatory for all operations."""

import pytest
from unittest.mock import Mock, patch

from knowledge_graph import KnowledgeGraphClient


class TestMandatoryGroupId:
    """Test that group_id is mandatory for all operations."""

    @patch('knowledge_graph.client.LibrarySettings')
    @patch('knowledge_graph.client.Graphiti')
    def test_client_without_group_id_fails_operations(self, mock_graphiti, mock_settings):
        """Test that operations fail when no group_id is provided."""
        # Create a proper mock settings object
        mock_settings_instance = Mock()
        mock_settings_instance.neo4j_uri = "bolt://localhost:7687"
        mock_settings_instance.neo4j_user = "neo4j"
        mock_settings_instance.neo4j_password = "password"
        mock_settings_instance.google_api_key = "test_key"
        mock_settings_instance.google_llm_model = "gemini-2.5-flash"
        mock_settings_instance.google_embedding_model = "embedding-001"
        mock_settings_instance.google_reranker_model = "gemini-2.5-flash-lite"
        mock_settings_instance.small_model = "gemini-2.5-flash"
        mock_settings_instance.max_tokens = 1000
        mock_settings_instance.gemini_openai_base = "https://generativelanguage.googleapis.com/v1beta/openai/"
        mock_settings.return_value = mock_settings_instance
        
        # Create client without group_id
        client = KnowledgeGraphClient()

    @pytest.mark.asyncio
    async def test_add_text_requires_group_id(self):
        """Test that add_text requires group_id."""
        with patch('knowledge_graph.client.LibrarySettings') as mock_settings, \
             patch('knowledge_graph.client.Graphiti') as mock_graphiti:
            
            mock_settings_instance = Mock()
            mock_settings_instance.neo4j_uri = "bolt://localhost:7687"
            mock_settings_instance.neo4j_user = "neo4j"
            mock_settings_instance.neo4j_password = "password"
            mock_settings_instance.google_api_key = "test_key"
            mock_settings_instance.google_llm_model = "gemini-2.5-flash"
            mock_settings_instance.google_embedding_model = "embedding-001"
            mock_settings_instance.google_reranker_model = "gemini-2.5-flash-lite"
            mock_settings_instance.small_model = "gemini-2.5-flash"
            mock_settings_instance.max_tokens = 1000
            mock_settings_instance.gemini_openai_base = "https://generativelanguage.googleapis.com/v1beta/openai/"
            mock_settings.return_value = mock_settings_instance
            
            client = KnowledgeGraphClient()
            
            with pytest.raises(ValueError, match="group_id must be provided either in constructor or method call"):
                await client.add_text("test text")

    @pytest.mark.asyncio
    async def test_search_requires_group_id(self):
        """Test that search requires group_id."""
        with patch('knowledge_graph.client.LibrarySettings') as mock_settings, \
             patch('knowledge_graph.client.Graphiti') as mock_graphiti:
            
            mock_settings_instance = Mock()
            mock_settings_instance.neo4j_uri = "bolt://localhost:7687"
            mock_settings_instance.neo4j_user = "neo4j"
            mock_settings_instance.neo4j_password = "password"
            mock_settings_instance.google_api_key = "test_key"
            mock_settings_instance.google_llm_model = "gemini-2.5-flash"
            mock_settings_instance.google_embedding_model = "embedding-001"
            mock_settings_instance.google_reranker_model = "gemini-2.5-flash-lite"
            mock_settings_instance.small_model = "gemini-2.5-flash"
            mock_settings_instance.max_tokens = 1000
            mock_settings_instance.gemini_openai_base = "https://generativelanguage.googleapis.com/v1beta/openai/"
            mock_settings.return_value = mock_settings_instance
            
            client = KnowledgeGraphClient()
            
            with pytest.raises(ValueError, match="group_id must be provided either in constructor or method call"):
                await client.search("test query")

    @pytest.mark.asyncio
    async def test_get_answer_requires_group_id(self):
        """Test that get_answer requires group_id."""
        with patch('knowledge_graph.client.LibrarySettings') as mock_settings, \
             patch('knowledge_graph.client.Graphiti') as mock_graphiti:
            
            mock_settings_instance = Mock()
            mock_settings_instance.neo4j_uri = "bolt://localhost:7687"
            mock_settings_instance.neo4j_user = "neo4j"
            mock_settings_instance.neo4j_password = "password"
            mock_settings_instance.google_api_key = "test_key"
            mock_settings_instance.google_llm_model = "gemini-2.5-flash"
            mock_settings_instance.google_embedding_model = "embedding-001"
            mock_settings_instance.google_reranker_model = "gemini-2.5-flash-lite"
            mock_settings_instance.small_model = "gemini-2.5-flash"
            mock_settings_instance.max_tokens = 1000
            mock_settings_instance.gemini_openai_base = "https://generativelanguage.googleapis.com/v1beta/openai/"
            mock_settings.return_value = mock_settings_instance
            
            client = KnowledgeGraphClient()
            
            with pytest.raises(ValueError, match="group_id must be provided either in constructor or method call"):
                await client.get_answer("test question")

    @pytest.mark.asyncio
    async def test_list_documents_requires_group_id(self):
        """Test that list_documents requires group_id."""
        with patch('knowledge_graph.client.LibrarySettings') as mock_settings, \
             patch('knowledge_graph.client.Graphiti') as mock_graphiti:
            
            mock_settings_instance = Mock()
            mock_settings_instance.neo4j_uri = "bolt://localhost:7687"
            mock_settings_instance.neo4j_user = "neo4j"
            mock_settings_instance.neo4j_password = "password"
            mock_settings_instance.google_api_key = "test_key"
            mock_settings_instance.google_llm_model = "gemini-2.5-flash"
            mock_settings_instance.google_embedding_model = "embedding-001"
            mock_settings_instance.google_reranker_model = "gemini-2.5-flash-lite"
            mock_settings_instance.small_model = "gemini-2.5-flash"
            mock_settings_instance.max_tokens = 1000
            mock_settings_instance.gemini_openai_base = "https://generativelanguage.googleapis.com/v1beta/openai/"
            mock_settings.return_value = mock_settings_instance
            
            client = KnowledgeGraphClient()
            
            with pytest.raises(ValueError, match="group_id must be provided either in constructor or method call"):
                await client.list_documents()

    def test_client_with_group_id_works(self):
        """Test that client with group_id can be created."""
        with patch('knowledge_graph.client.LibrarySettings') as mock_settings, \
             patch('knowledge_graph.client.Graphiti') as mock_graphiti:
            
            mock_settings_instance = Mock()
            mock_settings_instance.neo4j_uri = "bolt://localhost:7687"
            mock_settings_instance.neo4j_user = "neo4j"
            mock_settings_instance.neo4j_password = "password"
            mock_settings_instance.google_api_key = "test_key"
            mock_settings_instance.google_llm_model = "gemini-2.5-flash"
            mock_settings_instance.google_embedding_model = "embedding-001"
            mock_settings_instance.google_reranker_model = "gemini-2.5-flash-lite"
            mock_settings_instance.small_model = "gemini-2.5-flash"
            mock_settings_instance.max_tokens = 1000
            mock_settings_instance.gemini_openai_base = "https://generativelanguage.googleapis.com/v1beta/openai/"
            mock_settings.return_value = mock_settings_instance
            
            # This should work
            client = KnowledgeGraphClient(group_id="test_group")
            assert client._default_group_id == "test_group"


if __name__ == "__main__":
    pytest.main([__file__])
