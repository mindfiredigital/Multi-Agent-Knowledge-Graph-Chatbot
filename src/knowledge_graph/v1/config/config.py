"""
Configuration management for Multi-Agent RAG system (packaged under knowledge_graph.v1).
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    temperature: float = float(os.getenv("LLM_TEMPERATURE", 0.2))
    timeout: float = float(os.getenv("LLM_TIMEOUT", 60))
    eval_temperature: float = float(os.getenv("LLM_EVAL_TEMPERATURE", 0.0))
    model_name: str = os.getenv("LLM_MODEL_NAME", "gpt-4")
    # ollama_model_name: str = os.getenv("LLM_OLLAMA_MODEL_NAME", "mistral")
    topk: int = int(os.getenv("LLM_TOPK", 10))
    embedding_model_name: str = os.getenv("LLM_EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")
    reranker_model_name: str = os.getenv("LLM_RERANKER_MODEL_NAME", "Qwen/Qwen3-reranker-0.6B")
    rerank_top_k: int = int(os.getenv("LLM_RERANK_TOP_K", 3))
    openai_embedding_model_name: str = os.getenv("OPENAI_EMBEDDING_MODEL_NAME", "text-embedding-ada-002")


@dataclass
class AmazonBedrockConfig:
    region_name: str = os.getenv("BEDROCK_REGION_NAME", "us-west-2")
    llm_model_name: str = os.getenv("BEDROCK_LLM_MODEL_NAME", "openai.gpt-oss-20b-1:0")
    reranker_model_name: str = os.getenv("BEDROCK_RERANKER_MODEL_NAME", "amazon.rerank-v1:0")
    embedding_name: str = os.getenv("BEDROCK_EMBEDDING_NAME", "amazon.titan-embed-text-v2:0")
    temperature: float = float(os.getenv("AWS_TEMPERATURE", 0.0))


@dataclass
class SearchConfig:
    confidence_threshold: float = float(os.getenv("SEARCH_CONFIDENCE_THRESHOLD", 0.2))
    domain_search_k: int = int(os.getenv("SEARCH_DOMAIN_SEARCH_K", 5))
    domain_threshold: float = float(os.getenv("SEARCH_DOMAIN_THRESHOLD", 0.3))
    retrieval_temperature: float = float(os.getenv("RETRIEVAL_TEMPERATURE", 0.5))
    activation_timeout: int = int(os.getenv("SEARCH_ACTIVATION_TIMEOUT", 300))


@dataclass
class DocumentProcessingConfig:
    min_font_size: float = float(os.getenv("DOC_MIN_FONT_SIZE", 12.0))
    heading_font_threshold_large: float = float(os.getenv("DOC_HEADING_FONT_THRESHOLD_LARGE", 20.0))
    heading_font_threshold_medium: float = float(os.getenv("DOC_HEADING_FONT_THRESHOLD_MEDIUM", 17.0))
    heading_font_threshold_small: float = float(os.getenv("DOC_HEADING_FONT_THRESHOLD_SMALL", 14.0))
    heading_font_threshold_merge: float = float(os.getenv("DOC_HEADING_FONT_THRESHOLD_MERGE", 18.0))
    max_heading_words: int = int(os.getenv("DOC_MAX_HEADING_WORDS", 10))



class ConfigManager:
    def __init__(self):
        self.llm = LLMConfig()
        self.search = SearchConfig()
        self.document_processing = DocumentProcessingConfig()
        self.amazon_bedrock = AmazonBedrockConfig()


config = ConfigManager()


def reload_config():
    global config
    config = ConfigManager()
    return config


