"""V1 client implementation using TrueBlackboardSystem architecture."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import V1 dependencies, with graceful fallback - make imports lazy
SENTENCE_TRANSFORMERS_AVAILABLE = None
SENTENCE_TRANSFORMER_CLASS = None

def _ensure_sentence_transformers():
    """Lazily import sentence_transformers when needed."""
    global SENTENCE_TRANSFORMERS_AVAILABLE, SENTENCE_TRANSFORMER_CLASS
    if SENTENCE_TRANSFORMERS_AVAILABLE is None:
        try:
            from sentence_transformers import SentenceTransformer
            SENTENCE_TRANSFORMER_CLASS = SentenceTransformer
            SENTENCE_TRANSFORMERS_AVAILABLE = True
        except ImportError:
            SENTENCE_TRANSFORMERS_AVAILABLE = False
            SENTENCE_TRANSFORMER_CLASS = None
    return SENTENCE_TRANSFORMERS_AVAILABLE, SENTENCE_TRANSFORMER_CLASS

# Make V1 dependencies lazy too
V1_DEPENDENCIES_AVAILABLE = None
V1_MODULES = {}

def _ensure_v1_dependencies():
    """Lazily import V1 dependencies when needed."""
    global V1_DEPENDENCIES_AVAILABLE, V1_MODULES
    if V1_DEPENDENCIES_AVAILABLE is None:
        try:
            # Import the V1 system components from the packaged module
            from knowledge_graph.v1 import (
                TrueBlackboardSystem,
                embed_query,
                extract_text_from_uploaded_file,
                new_invoke_ingestion,
                get_ingestion_logger,
                config,
            )
            V1_MODULES.update({
                'TrueBlackboardSystem': TrueBlackboardSystem,
                'embed_query': embed_query,
                'extract_text_from_uploaded_file': extract_text_from_uploaded_file,
                'new_invoke_ingestion': new_invoke_ingestion,
                'get_ingestion_logger': get_ingestion_logger,
                'config': config,
            })
            V1_DEPENDENCIES_AVAILABLE = True
        except ImportError as e:
            print(f"Error importing V1 dependencies: {e}")
            V1_DEPENDENCIES_AVAILABLE = False
            # Create dummy classes for when dependencies aren't available
            class TrueBlackboardSystem:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("V1 dependencies not available. Please ensure knowledge_graph_v1 modules are accessible.")

            def embed_query(*args, **kwargs):
                raise RuntimeError("V1 dependencies not available")

            def extract_text_from_uploaded_file(*args, **kwargs):
                raise RuntimeError("V1 dependencies not available")

            def new_invoke_ingestion(*args, **kwargs):
                raise RuntimeError("V1 dependencies not available")

            def get_ingestion_logger():
                import logging
                return logging.getLogger(__name__)

            class config:
                class llm:
                    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

            V1_MODULES.update({
                'TrueBlackboardSystem': TrueBlackboardSystem,
                'embed_query': embed_query,
                'extract_text_from_uploaded_file': extract_text_from_uploaded_file,
                'new_invoke_ingestion': new_invoke_ingestion,
                'get_ingestion_logger': get_ingestion_logger,
                'config': config,
            })
    return V1_DEPENDENCIES_AVAILABLE

# Initialize logger - lazy
logger = None

def _get_logger():
    """Get the logger, initializing V1 dependencies if needed."""
    global logger
    if logger is None:
        _ensure_v1_dependencies()
        logger = V1_MODULES['get_ingestion_logger']()
    return logger


class KnowledgeGraphClientV1:
    """V1 client implementation using TrueBlackboardSystem multi-agent architecture.
    
    This client provides a different approach to knowledge graph operations using
    a multi-agent blackboard system instead of Graphiti. It supports project-based
    organization and uses specialized agents for query processing.
    """

    def __init__(self, group_id: Optional[str] = None) -> None:
        """Initialize the V1 client with optional group context.
        
        Args:
            group_id: Default group identifier for operations. If not provided,
                      operations will require explicit group_id parameters.
        """
        if not _ensure_v1_dependencies():
            raise RuntimeError(
                "V1 client dependencies not available. Please ensure knowledge_graph_v1 "
                "modules are accessible and all required packages are installed."
            )
        
        available, cls = _ensure_sentence_transformers()
        if not available:
            raise RuntimeError(
                "sentence-transformers package not available. Please install it with: "
                "pip install sentence-transformers"
            )
        
        self.group_id = group_id
        
        # Initialize Neo4j connection
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USER") 
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        if not all([self.neo4j_uri, self.neo4j_user, self.neo4j_password]):
            raise RuntimeError(
                "Missing required Neo4j environment variables: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD"
            )
        
        self.driver = GraphDatabase.driver(
            self.neo4j_uri, 
            auth=(self.neo4j_user, self.neo4j_password)
        )
        
        # Initialize embedding model
        _ensure_v1_dependencies()
        available, cls = _ensure_sentence_transformers()
        self.model = cls(V1_MODULES['config'].llm.embedding_model_name, device='cpu')
        
        # Create projects directory
        self.projects_dir = Path("projects")
        self.projects_dir.mkdir(exist_ok=True)

    def _ensure_group_id(self, group_id: Optional[str] = None) -> str:
        """Ensure a group_id exists, using default if not provided."""
        project = group_id or self.group_id
        if not project:
            raise ValueError("group_id must be provided either in constructor or method call")
        return project


    async def group_id_exists(self, group_id: str) -> bool:
        """Check if a group exists (stored as Project node).
        
        Args:
            group_id: Group identifier to check
            
        Returns:
            True if project exists, False otherwise
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    "MATCH (p:Project {name: $name}) RETURN p",
                    name=group_id
                )
                return result.single() is not None
        except Exception:
            return False

    async def add_text(
        self,
        text: str,
        name: str | None = None,
        source_description: str = "",
        reference_time: Optional[datetime] = None,
        group_id: Optional[str] = None,
    ) -> None:
        """Add text content to the knowledge graph.
        
        Args:
            text: Text content to add
            name: Optional name for the content
            source_description: Description of the source
            reference_time: Timestamp for the content
            group_id: Group identifier (required for V1)
        """
        project_name = self._ensure_group_id(group_id)
        
        # Create project if it doesn't exist
        await self._create_project_if_not_exists(project_name)
        
        text_content = text.strip()
        if not text_content:
            raise ValueError("text must not be empty")

        heading = name or f"Text snippet {datetime.utcnow().isoformat(timespec='seconds')}"
        metadata_prefix: List[str] = []
        if source_description:
            metadata_prefix.append(f"Source: {source_description}")
        if reference_time:
            metadata_prefix.append(f"Reference Time: {reference_time.isoformat()}")
        if metadata_prefix:
            text_content = "\n".join(metadata_prefix) + "\n\n" + text_content

        chunk = self._build_text_chunk(project_name, heading, text_content)

        try:
            V1_MODULES['new_invoke_ingestion'](self.driver, project_name, [chunk])
        except Exception as exc:
            _get_logger().error(f"Error ingesting ad-hoc text for group {project_name}: {exc}")
            raise

    async def ingest_file(
        self, 
        file_path: str | Path, 
        original_filename: Optional[str] = None, 
        group_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Ingest a document file into the knowledge graph.
        
        Args:
            file_path: Path to the document file
            original_filename: Override filename
            group_id: Group identifier (required for V1)
            
        Returns:
            Dict with ingestion status and metadata
        """
        project_name = self._ensure_group_id(group_id)
        
        # Create project if it doesn't exist
        await self._create_project_if_not_exists(project_name)
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        filename = original_filename or path.name
        
        _get_logger().info(f"Starting ingestion of {filename} into group {project_name}")
        
        try:
            # Read file content
            with open(path, 'rb') as f:
                file_content = f.read()
            
            # Determine file type
            file_type = path.suffix.lower()
            if not file_type:
                file_type = 'txt'  # Default to text
            
            # Extract text using V1 preprocessor
            extracted_json = V1_MODULES['extract_text_from_uploaded_file'](
                file_content, filename, file_type
            )

            if isinstance(extracted_json, dict):
                error_message = extracted_json.get("error") or "Unknown extraction error"
                raise ValueError(error_message)

            if not extracted_json:
                raise ValueError(
                    "No textual content could be extracted from the file. "
                    "Ensure the document contains readable text and required dependencies are installed."
                )
            
            # Process through V1 ingestion pipeline
            V1_MODULES['new_invoke_ingestion'](self.driver, project_name, extracted_json)
            
            _get_logger().info(f"Successfully ingested {filename} into group {project_name}")

            return {
                "status": "completed",
                "filename": filename,
                "file_type": file_type,
                "project_name": project_name,
                "chunks_ingested": len(extracted_json)
            }

        except Exception as e:
            _get_logger().error(f"Error ingesting file {filename}: {e}")
            raise IOError(f"Failed to ingest file {file_path}: {e}") from e

    def get_supported_document_types(self) -> List[str]:
        """Get list of supported document file extensions.
        
        Returns:
            List of supported file extensions
        """
        return [".pdf"]

    async def search(self, question: str, group_id: str) -> Dict[str, Any]:
        """Search the knowledge graph using V1 multi-agent system.
        
        Args:
            question: Search query
            group_id: Group identifier to search in
            
        Returns:
            Dict with search results
        """
        project_name = self._ensure_group_id(group_id)
        
        if not await self.group_id_exists(project_name):
            return {}
        
        try:
            # Initialize the blackboard system for this project
            system = V1_MODULES['TrueBlackboardSystem'](self.driver, project_name=project_name)

            # Process the query
            response = system.process_query(question)
            
            # Convert V1 response format to V2-compatible format
            if response.get("success", False):
                return {
                    "answer": response.get("message", ""),
                    "success": True
                }
            else:
                return {
                    "answer": response.get("message", "No results found"),
                    "success": False
                }
                
        except Exception as e:
            _get_logger().error(f"Error searching group {project_name}: {e}")
            return {}

    async def list_documents(self, group_id: str) -> List[Dict[str, Any]]:
        """List all documents in a project.
        
        Args:
            group_id: Group identifier to list documents for
            
        Returns:
            List of document metadata
        """
        project_name = self._ensure_group_id(group_id)
        
        if not await self.group_id_exists(project_name):
            return []
        
        try:
            with self.driver.session() as session:
                # Query for entities in the project
                result = session.run("""
                    MATCH (e:Entity {project_name: $project_name})
                    WHERE e.content IS NOT NULL AND e.type = 'root_node'
                    RETURN DISTINCT e.name AS filename
                """, project_name=project_name)
                
                documents = []
                for record in result:
                    
                    filename = record.get("filename")
                    
                    documents.append({
                        "filename": filename,
                        "file_type": "pdf"
                    })
                
                return documents
                
        except Exception as e:
            _get_logger().error(f"Error listing documents for group {project_name}: {e}")
            return []

    async def get_answer(self, question: str, group_id: str, k: int = 5) -> Dict[str, Any]:
        """Generate an answer using V1 multi-agent system.
        
        Args:
            question: Question to answer
            group_id: Group identifier to search in
            k: Number of context snippets (ignored in V1)
            
        Returns:
            Dict with answer and context
        """
        project_name = self._ensure_group_id(group_id)
        
        if not await self.group_id_exists(project_name):
            return {
                "answer": "Project not found or no information available.",
                "context": []
            }
        
        try:
            # Use the search method which already returns V2-compatible format
            result = await self.search(question, project_name)
            return result
            
        except Exception as e:
            _get_logger().error(f"Error getting answer for group {project_name}: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "context": []
            }

    async def _create_project_if_not_exists(self, group_id: str) -> None:
        """Create a group (Project node) in the database if it doesn't exist."""
        try:
            with self.driver.session() as session:
                session.run("""
                    MERGE (p:Project {name: $project_name})
                    SET p.description = "",
                        p.created_at = datetime(),
                        p.file_count = 0
                """, project_name=group_id)
                
                # Create project directory
                project_dir = self.projects_dir / group_id
                project_dir.mkdir(exist_ok=True)
                
        except Exception as e:
            _get_logger().error(f"Error creating group {group_id}: {e}")
            raise

    def _build_text_chunk(self, project_name: str, heading: str, content: str) -> Dict[str, Any]:
        """Construct a synthetic chunk compatible with the V1 ingestion pipeline."""
        chunk_id = uuid.uuid4().hex
        word_count = len(content.split())

        return {
            "id": chunk_id,
            "heading": heading,
            "heading_level": "H1 (Chapter/Main Title)",
            "level": 1,
            "parent_id": None,
            "parent_heading": project_name,
            "page": 0,
            "font_size": 12,
            "is_bold": True,
            "word_count": word_count,
            "image_count": 0,
            "table_count": 0,
            "content": content,
        }

    def __del__(self):
        """Clean up Neo4j driver on destruction."""
        if hasattr(self, 'driver'):
            self.driver.close()
