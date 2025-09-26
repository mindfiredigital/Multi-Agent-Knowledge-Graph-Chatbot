# TODO

## Improvements
- [ ] Add structured tracing around Graphiti operations (ingestion/query) with optional OpenTelemetry exporters.
- [ ] Add memory-efficient streaming for large document processing to handle files >100MB without memory issues.
- [ ] Implement connection pooling and health checks for Neo4j database connections.
- [ ] Add data validation and sanitization (guardrails) for user inputs to prevent injection attacks and malformed data.
- [ ] Implement rate limiting for API calls to external services to respect usage quotas.
- [ ] Add performance monitoring and metrics collection for ingestion and query operations.
- [ ] Switch to other LLM models when default one is down.
- [ ] Implement caching layer for frequently accessed graph data and embeddings.


## New Features
- [ ] Implement additional document processors for `.csv`, `.pptx`, `.xlsx`, and remote URLs with automatic metadata enrichment.
- [ ] Introduce a background ingestion pipeline (queue/worker) to support large batch uploads with progress tracking.
- [ ] Offer multi-provider LLM support with pluggable clients (Gemini, OpenAI, Azure, Anthropic) and smart fallback handling.
- [ ] Add graph visualization capabilities with export to various formats (PNG, SVG, JSON).
- [ ] Add search with advanced filtering options (date ranges, content types, relevance scores).
- [ ] Implement knowledge graph analytics dashboard showing insights, popular topics, and projects available.
- [ ] Implement knowledge graph export/import functionality for backup and migration purposes.
- [ ] Add support for interactive query refinement with follow-up questions and context narrowing.
