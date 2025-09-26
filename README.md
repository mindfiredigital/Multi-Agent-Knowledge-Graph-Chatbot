# Knowledge Graph Library

A reusable library for building and querying knowledge graphs backed by Neo4j, powered by Graphiti and Gemini.
## Installation

### From Wheel File (Recommended)

Download the latest `knowledge_graph-0.1.0-py3-none-any.whl` file from our releases and install:

```bash
pip install knowledge_graph-0.1.0-py3-none-any.whl
```

### From Source (Development)

Using uv (recommended and supported):

```bash
git clone https://github.com/your-org/knowledge-graph.git
cd knowledge-graph
uv sync

# With dev extras (tests, linters)
uv sync --extra dev

# All extras (includes dev dependencies)
uv sync --extra all
```

**Optional Dependencies:**

- `dev`: Adds pytest, black, mypy, flake8, and other development tools
- `all`: Installs all optional dependencies (dev only)

**Using requirements files:**

```bash
# Core runtime dependencies (includes all functionality)
pip install -r requirements.txt

# Development dependencies (core + dev tools)
pip install -r requirements-dev.txt

# All dependencies (core + dev)
pip install -r requirements-all.txt
```

### Pip Package (Coming Soon)

We plan to distribute this package via PyPI for easy installation:

```bash
pip install knowledge-graph  # Coming soon
```

## Quick Start

The library provides two client implementations. **We recommend using V2 (KnowledgeGraphClient) as it is easier to setup and use with modern infrastructure.**

### KnowledgeGraphClientV2 (Recommended - Current/Default)

The modern implementation using Graphiti with Google Gemini. This is the recommended choice for most users:

```python
from knowledge_graph import KnowledgeGraphClient  # or KnowledgeGraphClientV2

# group_id is mandatory - must be provided in constructor or method calls
client = KnowledgeGraphClient(group_id="team_alpha")  # uses env vars
await client.build_indices_and_constraints()

# Add text (namespaced to team_alpha)
await client.add_text("Some content about CNN architecture.")

# Ingest documents (auto-detects file type; also namespaced)
await client.ingest_file("/path/to/document.pdf")
await client.ingest_file("/path/to/document.docx")
await client.ingest_file("/path/to/document.txt")
await client.ingest_file("/path/to/document.md")
await client.ingest_file("/path/to/document.html")

# List all ingested documents in the default namespace
documents = await client.list_documents()
print(f"Found {len(documents)} documents")

# Get supported document types
supported_types = client.get_supported_document_types()
print(f"Supported types: {supported_types}")

# Retrieve-and-answer within the namespace
resp = await client.get_answer("What is convolution?")
print(resp["answer"])           # final answer from LLM
print(resp["context"])          # list of context snippets used

# Or just retrieve without generation within the namespace
result = await client.search("What is convolution?")  # raw retrieval output
print(result)
```

### KnowledgeGraphClientV1 (Legacy - Multi-Agent Architecture)

**⚠️ V1 is a legacy implementation and not recommended for new projects.** It requires complex setup with multiple dependencies and specialized infrastructure.

Alternative implementation using a multi-agent blackboard system:

```python
from knowledge_graph import KnowledgeGraphClientV1

# Initialize with a group_id (not project_name)
client = KnowledgeGraphClientV1(group_id="my_project")

# Build indices (no-op for V1, kept for compatibility)
await client.build_indices_and_constraints()

# Add text content
await client.add_text("Some content about CNN architecture.", group_id="my_project")

# Ingest documents (limited to PDF only)
await client.ingest_file("/path/to/document.pdf", group_id="my_project")

# List documents in the project
documents = await client.list_documents("my_project")
print(f"Found {len(documents)} documents")

# Query using multi-agent system
resp = await client.get_answer("What is convolution?", group_id="my_project")
print(resp["answer"])           # final answer from multi-agent system
print(resp["context"])          # list of context snippets used

# Search without answer generation
result = await client.search("What is convolution?", group_id="my_project")
print(result)
```

## Client Comparison

### 🏆 KnowledgeGraphClientV2 (Recommended - Graphiti-based)

**The modern, production-ready implementation that we strongly recommend for all new projects.**

- **✅ Architecture**: Uses Graphiti framework with Google Gemini for robust, scalable knowledge graphs
- **✅ Namespacing**: Uses `group_id` for data isolation (mandatory parameter)
- **✅ Query Processing**: Direct Graphiti search and retrieval with advanced RAG capabilities
- **✅ Document Processing**: Comprehensive support for PDF, DOCX, Markdown, HTML, and TXT files with intelligent token merging
- **✅ Dependencies**: Simple setup with Google API key and Neo4j - no complex infrastructure needed
- **✅ Setup**: Easy to configure and deploy in production environments
- **✅ Performance**: Optimized for speed and scalability with modern LLM infrastructure
- **✅ Best for**: Production applications, enterprise deployments, and most use cases

### KnowledgeGraphClientV1 (Multi-Agent System)

**Alternative implementation using a multi-agent blackboard architecture.**

- **Architecture**: Multi-agent blackboard system with specialized agents for different tasks
- **Namespacing**: Uses `group_id` for data organization
- **Query Processing**: Agent-based retrieval system with confidence scoring
- **Document Processing**: Supports ONLY PDF files with right formatting using custom preprocessing pipeline 
- **Dependencies**: Requires sentence-transformers, OpenAI API, AWS Bedrock API and Neo4j
- **Setup**: Requires additional infrastructure setup and configuration to run qwen-0.6B and embedding models
- **Performance**: Designed for complex multi-agent workflows
- **Best for**: Applications requiring specialized domain agents and complex reasoning workflows

### Choosing Between V1 and V2

**We recommend V2 (KnowledgeGraphClient) for most use cases** due to its simpler setup and broader document support.

- **V2 Advantages**: Easier setup, more document types, modern architecture
- **V1 Advantages**: Multi-agent capabilities for specialized workflows
- **Data Separation**: V1 and V2 use separate Neo4j schemas and cannot share data
- **Independent Operation**: Each client maintains its own knowledge graph

Namespacing (group_id)
----------------------

Graph namespacing allows you to isolate data and queries per logical group, such as tenants, teams, or environments. **The `group_id` parameter is mandatory** for all operations - you must provide it either in the constructor or in each method call.

- Constructor-level default `group_id` (recommended):

```python
client = KnowledgeGraphClientV2(group_id="tenant_42")
# All operations will use "tenant_42" as the default group_id
await client.add_text("Some content")
await client.search("query")
```

- Method-level overrides (take precedence over constructor default):

```python
client = KnowledgeGraphClientV2(group_id="tenant_42")

# Ingest into another namespace
await client.ingest_file("/path/to/notes.pdf", group_id="tenant_17")

# Add free text into a specific namespace
await client.add_text("Customer Jane loves the new shoes.", group_id="customer_team")

# Search in a different namespace
results = await client.search("Wool Runners", group_id="product_catalog")

# Generate answer constrained to a namespace
resp = await client.get_answer("What categories exist?", group_id="product_catalog")

# List documents within a namespace
docs = await client.list_documents(group_id="product_catalog")
```

**Important**: If you create a client without a `group_id`, all operations will raise a `ValueError` unless you explicitly provide `group_id` in each method call.

This behavior follows Graphiti's group_id namespacing support. See the official docs: `https://help.getzep.com/graphiti/core-concepts/graph-namespacing`.

## Supported Document Types

### V2 Client (KnowledgeGraphClient)

The V2 client supports comprehensive document processing with intelligent chunking and token merging:

- **PDF** (`.pdf`) - Extracts text per page with metadata preservation
- **Microsoft Word** (`.docx`) - Extracts text from paragraphs and tables with formatting
- **Plain Text** (`.txt`, `.text`) - Extracts text with automatic encoding detection
- **Markdown** (`.md`, `.markdown`, `.mdown`, `.mkdn`, `.mkd`) - Extracts text by sections and headers
- **HTML** (`.html`, `.htm`, `.xhtml`) - Extracts text content with optional structure preservation

### V1 Client (KnowledgeGraphClientV1)

**⚠️ Limited support - PDF only**
- **PDF** (`.pdf`) - Basic text extraction (no advanced processing)

**Note**: Spreadsheet formats (CSV, Excel) are intentionally not supported in either version.

Each document type is processed by specialized processors that:
- Extract text content in appropriate chunks
- Preserve document structure where relevant
- Handle encoding and format-specific issues
- Merge chunks intelligently to respect token limits

## Configuration

### Required Environment Variables

Before using the library, set these environment variables:

```bash
# Neo4j Database (Required)
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_neo4j_password"

# Google Gemini API (Required for V2, Optional for V1)
export GOOGLE_API_KEY="your_google_api_key_here"
```

### Optional Environment Variables

```bash
# Google Gemini Model Configuration (V2 only)
export GOOGLE_LLM_MODEL="gemini-2.5-flash"                    # LLM for answer generation
export GOOGLE_EMBEDDING_MODEL="embedding-001"                 # Text embeddings
export GOOGLE_RERANKER_MODEL="gemini-2.5-flash-lite"          # Search result reranking
export GEMINI_OPENAI_BASE="https://generativelanguage.googleapis.com/v1beta/openai/"

# OpenAI API (V1 only - if not using Gemini)
export OPENAI_API_KEY="your_openai_api_key_here"             # Only needed for V1
```

### Getting Started

1. **Set up Neo4j**: Install and start Neo4j database
2. **Get Google API Key**: Visit [Google AI Studio](https://aistudio.google.com/apikey) to create an API key
3. **Install the library**: Use the wheel file installation method above
4. **Set environment variables**: Configure the required environment variables
5. **Initialize client**: Use `KnowledgeGraphClient(group_id="your_namespace")` for V2
6. **Build indices**: Call `await client.build_indices_and_constraints()` once after setup



