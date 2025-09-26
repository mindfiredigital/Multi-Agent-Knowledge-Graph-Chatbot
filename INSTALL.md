# Installation Guide for knowledge-graph

This guide explains how to install and use the knowledge-graph Python package.

## Prerequisites

- Python 3.12 or higher
- Neo4j database (running locally or remotely)
- Google API key for Gemini

## Installation Methods

### 1. Install from Source (Development)

```bash
# Clone the repository
git clone https://github.com/your-org/knowledge-graph.git
cd knowledge-graph

# Using uv (recommended)
uv sync

# Dev extras (tests, linters)
uv sync --extra dev

# Demo extras (Streamlit)
uv sync --extra demo

# All extras
uv sync --extra all
```

### 2. Install with Optional Dependencies

uv is used for dependency management. Use extras via `uv sync --extra ...` as shown above.

### 3. Install from PyPI (Coming Soon)

Published wheels can be installed with uv once available.

## Environment Setup

Create a `.env` file or set environment variables:

```bash
# Required
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"
export GOOGLE_API_KEY="your_google_api_key"

# Optional (with defaults)
export GOOGLE_LLM_MODEL="gemini-2.5-flash"
export GOOGLE_EMBEDDING_MODEL="embedding-001"
export GOOGLE_RERANKER_MODEL="gemini-2.5-flash-lite"
```

## Quick Test

```python
import asyncio
from knowledge_graph import KnowledgeGraphClient

async def test_installation():
    client = KnowledgeGraphClient()
    await client.build_indices_and_constraints()
    print("✅ Installation successful!")

# Run the test
asyncio.run(test_installation())
```

## Package Structure

```
knowledge_graph/
├── __init__.py          # Main package exports
├── client.py            # KnowledgeGraphClient
├── config.py            # Configuration management
├── documents/           # Document processors
│   ├── __init__.py
│   ├── base.py         # Base DocumentProcessor
│   ├── pdf.py          # PDF processor
│   ├── docx.py         # DOCX processor
│   ├── txt.py          # TXT processor
│   ├── markdown.py     # Markdown processor
│   ├── html.py         # HTML processor
│   └── pdf.py          # PDF processor
└── py.typed            # Type hints marker
```

## Building Distribution Packages

```bash
# Install build tools
uv add build

# Build wheel
python -m build --wheel

# Build source distribution
python -m build --sdist

# Both wheel and sdist
python -m build
```

## Publishing to PyPI

```bash
# Install twine
uv add twine

# Upload to PyPI
twine upload dist/*
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'graphiti_core'**
   - Make sure you have Python 3.10+ installed
   - Install dependencies: `uv sync` or `pip install -r requirements.txt`

2. **Neo4j Connection Error**
   - Verify Neo4j is running
   - Check connection details in environment variables

3. **Google API Error**
   - Verify GOOGLE_API_KEY is set correctly
   - Check API quotas and permissions

### Getting Help

- Check the README.md for detailed usage examples
- Open an issue on GitHub for bugs or feature requests
- Review the examples/ directory for usage patterns
