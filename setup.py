#!/usr/bin/env python3
"""Setup script for knowledge-graph package."""

from setuptools import setup, find_packages
import os


# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()


# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [
            line.strip() for line in fh if line.strip() and not line.startswith("#")
        ]


setup(
    name="knowledge-graph",
    version="0.1.0",
    author="Knowledge Graph Team",
    author_email="team@example.com",
    description="Reusable library for building and querying knowledge graphs backed by Neo4j, powered by Graphiti and Gemini",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/knowledge-graph",
    project_urls={
        "Bug Reports": "https://github.com/your-org/knowledge-graph/issues",
        "Source": "https://github.com/your-org/knowledge-graph",
        "Documentation": "https://github.com/your-org/knowledge-graph#readme",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Markup :: HTML",
        "Topic :: Text Processing :: Markup :: Markdown",
    ],
    python_requires=">=3.12",
    install_requires=[
        "graphiti-core[google-genai]>=0.20.4",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "pypdf>=6.0.0",
        "openai>=1.107.3",
        "python-docx>=1.1.0",
        "beautifulsoup4>=4.12.0",
        "markdown>=3.5.0",
        "chardet>=5.2.0",
        "tiktoken>=0.5.0",
        "python-dotenv>=1.0.0",
        "openpyxl>=3.1.0",
        "pandas>=2.0.0",
        # Multi-agent and RAG dependencies
        "neo4j>=5.0.0",
        "sentence-transformers>=3.0.0",
        "PyMuPDF>=1.24.0",
        "langchain-aws>=0.2.3",
        "ragas>=0.1.13",
        "datasets>=2.19.0",
        "langchain-openai>=0.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-asyncio>=0.21.0",
            "httpx>=0.24.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
            "coverage>=7.10.7",
        ],
        "all": [
            "knowledge-graph[dev]",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="knowledge-graph neo4j graphiti gemini llm rag",
)
