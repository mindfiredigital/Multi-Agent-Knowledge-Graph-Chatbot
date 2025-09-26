"""
Logger configuration (packaged under knowledge_graph.v1).
"""
import logging

# A library should not configure logging handlers. It should just get a logger.
# The application using the library is responsible for configuring logging.
# We add a NullHandler to prevent "No handler found" warnings if the
# library is used without logging configuration.
logging.getLogger("knowledge_graph.v1").addHandler(logging.NullHandler())


def get_ingestion_logger():
    """Returns a logger for the ingestion service."""
    return logging.getLogger("knowledge_graph.v1.ingestion")


def get_retrieval_logger():
    """Returns a logger for the retrieval service."""
    return logging.getLogger("knowledge_graph.v1.retrieval")


def get_api_logger():
    """Returns a logger for the API."""
    return logging.getLogger("knowledge_graph.v1.api")


def get_frontend_logger():
    """Returns a logger for the frontend."""
    return logging.getLogger("knowledge_graph.v1.frontend")


def get_frontend_retrieval_logger():
    """Returns a logger for the frontend retrieval."""
    return logging.getLogger("knowledge_graph.v1.frontend_retrieval")