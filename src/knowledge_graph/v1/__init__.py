"""V1 multi-agent blackboard system package for knowledge_graph.

This package contains a self-contained copy of the legacy V1 implementation
organized under the main library so it can be distributed via wheel/PyPI.
"""

# Re-export common entry points for convenience
from .models.blackboard import TrueBlackboardSystem, embed_query  # noqa: F401
from .services.preprocessor import extract_text_from_uploaded_file  # noqa: F401
from .services.ingestion import new_invoke_ingestion  # noqa: F401
from .utils.logger_config import get_ingestion_logger  # noqa: F401
from .config.config import config  # noqa: F401


