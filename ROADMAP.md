# BookTutor AI → Enterprise Documentation System
## Modal + MCP Integration Roadmap

**Target Use Case**: Complex technical documentation (Juniper Networks-level)
**Architecture**: Modal serverless + MCP (Model Context Protocol) + Claude
**Current State**: CLI-based single-PDF QA system
**End Goal**: Scalable multi-document RAG system with graph-based knowledge retrieval

---

## Executive Summary

This roadmap transforms BookTutor AI from a simple PDF chatbot into an enterprise-grade documentation assistant capable of handling complex technical documentation like Juniper Networks, Cisco, AWS, or similar large-scale technical knowledge bases.

**Timeline**: 9-13 weeks (2-3 months)
**Estimated Cost**: $50-200/month on Modal (production)
**Tech Stack**: Python, LangChain, Modal, MCP, Claude API, FAISS/Qdrant, NetworkX/Neo4j

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Claude / Claude Code                          │
│                     (User Interface)                             │
└──────────────────────┬──────────────────────────────────────────┘
                       │ MCP Protocol
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                MCP Server (FastAPI on Modal)                     │
│  Tools:                                                          │
│  • search_docs(query, product, version, hierarchy_level)        │
│  • get_config_hierarchy(path)                                   │
│  • get_command_syntax(command)                                  │
│  • find_related_topics(topic)                                   │
│  • compare_versions(feature, v1, v2)                            │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│              Modal Functions (Serverless Workers)                │
│                                                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────────┐│
│  │  Doc Ingestion │  │  Smart Chunker │  │  Graph Builder    ││
│  │  • PDF/HTML    │  │  • Hierarchy   │  │  • Relationships  ││
│  │  • Metadata    │  │  • Context     │  │  • Cross-refs     ││
│  │  • Versioning  │  │  • Code blocks │  │  • Dependencies   ││
│  └────────────────┘  └────────────────┘  └───────────────────┘│
│                                                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────────┐│
│  │ Vector Search  │  │  LLM Query     │  │  Cache Layer      ││
│  │ • Semantic     │  │  • Claude API  │  │  • Redis/Modal    ││
│  │ • Hybrid       │  │  • RAG         │  │  • Query results  ││
│  │ • Reranking    │  │  • Synthesis   │  │  • Embeddings     ││
│  └────────────────┘  └────────────────┘  └───────────────────┘│
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Data Storage (Modal Volumes)                   │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐│
│  │Vector DB     │  │ Graph DB     │  │  Document Store        ││
│  │(FAISS/Qdrant)│  │(NetworkX/Neo4j│  │  • Raw docs           ││
│  │• Embeddings  │  │• Hierarchy   │  │  • Processed chunks   ││
│  │• Fast search │  │• Relations   │  │  • Metadata           ││
│  └──────────────┘  └──────────────┘  └────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Foundation - Modal + MCP Integration

**Duration**: 2-3 weeks
**Goal**: Deploy BookTutor on Modal with basic MCP server
**Status**: Not Started

### 1.1 Modal Deployment Setup (Week 1)

**Tasks**:
- [ ] Install Modal CLI and setup account
- [ ] Create `modal_app.py` with basic Modal configuration
- [ ] Define container image with all dependencies
- [ ] Setup Modal Volumes for data persistence
- [ ] Migrate core BookTutor functions to Modal functions
- [ ] Test local Modal deployment (`modal run`)
- [ ] Deploy to Modal cloud (`modal deploy`)

**Deliverables**:
- `modal_app.py` - Main Modal application
- `modal_config.py` - Configuration and environment variables
- Working Modal deployment accessible via CLI

**Code Structure**:
```python
# modal_app.py
import modal

app = modal.App("booktutor-docs")

# Container image
image = (
    modal.Image.debian_slim()
    .pip_install(
        "langchain",
        "langchain-anthropic",
        "langchain-community",
        "docling",
        "faiss-cpu",
        "fastapi",
        "uvicorn",
    )
)

# Volumes
vector_store_volume = modal.Volume.from_name("vector-store", create_if_missing=True)
doc_cache_volume = modal.Volume.from_name("doc-cache", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/data": vector_store_volume},
    timeout=3600,
    memory=4096,
)
def process_pdf(pdf_path: str) -> dict:
    """Process PDF and create vector store"""
    pass

@app.function(
    image=image,
    volumes={"/data": vector_store_volume},
    memory=2048,
)
def query_docs(question: str, context: dict) -> dict:
    """Query processed documents"""
    pass
```

### 1.2 Claude API Integration (Week 1-2)

**Tasks**:
- [ ] Create Anthropic API account
- [ ] Replace local LM Studio with Claude API
- [ ] Update `ChatOpenAI` to `ChatAnthropic`
- [ ] Test with Claude 3.5 Sonnet
- [ ] Implement streaming responses
- [ ] Add error handling and rate limiting
- [ ] Test cost optimization (prompt caching)

**Code Changes**:
```python
# Before: booktutor.py:124-129
llm = ChatOpenAI(
    model="local-model",
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="not-needed",
    temperature=0,
)

# After: modal_app.py
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
    temperature=0,
    max_tokens=4096,
)
```

### 1.3 FastAPI Server (Week 2)

**Tasks**:
- [ ] Create FastAPI application structure
- [ ] Define REST API endpoints
- [ ] Add request/response models with Pydantic
- [ ] Implement authentication (API keys)
- [ ] Add CORS configuration
- [ ] Setup logging and monitoring
- [ ] Deploy FastAPI on Modal with ASGI

**API Endpoints**:
```python
# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="BookTutor Docs API")

class QueryRequest(BaseModel):
    question: str
    doc_id: str
    chat_history: list[tuple[str, str]] = []

class QueryResponse(BaseModel):
    answer: str
    source_chunks: list[dict]
    confidence: float

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the document knowledge base"""
    pass

@app.post("/ingest")
async def ingest_document(pdf_url: str, metadata: dict):
    """Ingest a new document"""
    pass

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

### 1.4 MCP Server Implementation (Week 2-3)

**Tasks**:
- [ ] Install MCP Python SDK (`pip install mcp`)
- [ ] Create MCP server with tool definitions
- [ ] Implement MCP protocol handlers
- [ ] Define initial tools: `search_docs`, `ask_question`
- [ ] Setup MCP server on Modal
- [ ] Create MCP client configuration for Claude Code
- [ ] Test MCP integration with Claude Desktop/Code

**MCP Server**:
```python
# mcp_server.py
from mcp.server import Server
from mcp.types import Tool, TextContent
import modal

app = modal.App("booktutor-mcp")

# MCP Server
mcp_server = Server("booktutor-docs")

@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_docs",
            description="Search technical documentation with semantic and keyword search",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "doc_filter": {"type": "string", "description": "Optional document filter"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="ask_question",
            description="Ask a question about technical documentation and get contextual answer",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "context": {"type": "string"},
                },
                "required": ["question"],
            },
        ),
    ]

@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "search_docs":
        results = await search_documents(arguments["query"], arguments.get("doc_filter"))
        return [TextContent(type="text", text=str(results))]
    elif name == "ask_question":
        answer = await answer_question(arguments["question"], arguments.get("context"))
        return [TextContent(type="text", text=answer)]
```

**MCP Client Config** (for Claude Code):
```json
// ~/.config/claude-code/mcp_servers.json
{
  "booktutor-docs": {
    "command": "modal",
    "args": ["run", "mcp_server.py"],
    "env": {
      "ANTHROPIC_API_KEY": "sk-..."
    }
  }
}
```

### 1.5 Multi-Document Support (Week 3)

**Tasks**:
- [ ] Extend `DoclingBookLoader` to handle multiple files
- [ ] Create document registry/catalog
- [ ] Add document metadata tracking
- [ ] Implement batch processing
- [ ] Add document versioning support
- [ ] Create document management API

**Code Enhancement**:
```python
# document_manager.py
class DocumentManager:
    def __init__(self, volume_path: str):
        self.volume_path = volume_path
        self.catalog = self.load_catalog()

    def ingest_documents(self, file_paths: list[str], metadata: dict = None):
        """Batch ingest multiple documents"""
        for path in file_paths:
            doc_id = self.generate_doc_id(path)
            self.catalog[doc_id] = {
                "path": path,
                "metadata": metadata or {},
                "ingested_at": datetime.now(),
                "status": "processing"
            }
            self.process_document(doc_id, path)

    def get_document(self, doc_id: str) -> dict:
        """Retrieve document info"""
        return self.catalog.get(doc_id)

    def list_documents(self, filters: dict = None) -> list[dict]:
        """List all documents with optional filters"""
        pass
```

### Phase 1 Deliverables

- ✅ Modal deployment working
- ✅ Claude API integrated
- ✅ FastAPI REST API
- ✅ Basic MCP server with 2-3 tools
- ✅ Multi-document ingestion
- ✅ Documentation for deployment

**Success Metrics**:
- Can process 10+ PDFs
- Average query response < 3 seconds
- MCP tools accessible from Claude Code
- Modal deployment cost < $20/month (dev environment)

---

## Phase 2: Enhanced RAG System

**Duration**: 3-4 weeks
**Goal**: Improve retrieval quality with advanced techniques
**Status**: Not Started

### 2.1 Hierarchy-Aware Chunking (Week 4)

**Problem**: Current chunking loses document structure and context.

**Tasks**:
- [ ] Implement document structure parser
- [ ] Detect headings, sections, subsections
- [ ] Preserve hierarchy metadata in chunks
- [ ] Smart splitting (don't break tables, code blocks)
- [ ] Add breadcrumb trails to chunks
- [ ] Implement parent-child chunk relationships

**Implementation**:
```python
# hierarchical_chunker.py
from typing import List
from dataclasses import dataclass

@dataclass
class Chunk:
    content: str
    metadata: dict
    hierarchy_path: list[str]  # ["Chapter 3", "BGP Configuration", "Basic Setup"]
    parent_chunk_id: str = None
    child_chunk_ids: list[str] = None

class HierarchicalChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, document: str, doc_metadata: dict) -> List[Chunk]:
        """
        Split document while preserving hierarchy

        Process:
        1. Parse markdown/HTML structure
        2. Identify heading levels (h1-h6)
        3. Build hierarchy tree
        4. Chunk within sections
        5. Add parent context to each chunk
        """
        hierarchy = self.parse_structure(document)
        chunks = []

        for section in hierarchy:
            section_chunks = self.chunk_section(
                section.content,
                section.hierarchy_path,
                section.metadata
            )
            chunks.extend(section_chunks)

        return chunks

    def parse_structure(self, document: str) -> List[Section]:
        """Parse document into hierarchical sections"""
        # Detect headings: #, ##, ###, etc.
        # Build tree structure
        # Extract metadata from headings
        pass

    def chunk_section(self, content: str, hierarchy_path: list, metadata: dict) -> List[Chunk]:
        """Chunk a section while preserving context"""
        # Split content into chunks
        # Add hierarchy breadcrumb to each chunk
        # Maintain overlap between chunks
        pass
```

**Metadata Enhancement**:
```python
chunk_metadata = {
    "source": "junos-bgp-guide.pdf",
    "hierarchy_path": ["Chapter 3", "BGP Configuration", "Basic Setup"],
    "section_title": "Basic Setup",
    "page_number": 42,
    "doc_type": "configuration_guide",
    "hierarchy_level": 3,  # h3 heading
    "parent_section": "BGP Configuration",
    "has_code_block": True,
    "has_table": False,
}
```

### 2.2 Hybrid Search System (Week 4-5)

**Problem**: Vector search alone misses exact matches (commands, error codes).

**Tasks**:
- [ ] Implement BM25 keyword search
- [ ] Add sparse vector support
- [ ] Combine vector + keyword results
- [ ] Implement reciprocal rank fusion (RRF)
- [ ] Add query intent detection
- [ ] Optimize search weights

**Hybrid Search Architecture**:
```python
# hybrid_search.py
from rank_bm25 import BM25Okapi
from typing import List, Dict

class HybridSearchEngine:
    def __init__(self, vector_store, bm25_index):
        self.vector_store = vector_store  # FAISS
        self.bm25_index = bm25_index
        self.vector_weight = 0.7  # Tune these
        self.keyword_weight = 0.3

    def search(self, query: str, top_k: int = 10, filters: dict = None) -> List[Dict]:
        """
        Hybrid search combining semantic and keyword matching

        Steps:
        1. Detect query intent (code lookup, concept explanation, etc.)
        2. Run vector similarity search
        3. Run BM25 keyword search
        4. Merge results using RRF
        5. Apply filters (product, version, hierarchy)
        6. Return ranked results
        """
        intent = self.detect_intent(query)

        # Adjust weights based on intent
        if intent == "exact_match":  # e.g., "show bgp summary command"
            self.keyword_weight = 0.8
            self.vector_weight = 0.2
        elif intent == "conceptual":  # e.g., "how does BGP work?"
            self.keyword_weight = 0.2
            self.vector_weight = 0.8

        # Vector search
        vector_results = self.vector_store.similarity_search_with_score(
            query, k=top_k * 2
        )

        # Keyword search
        keyword_results = self.bm25_search(query, top_k * 2)

        # Reciprocal Rank Fusion
        merged_results = self.reciprocal_rank_fusion(
            vector_results, keyword_results, top_k
        )

        # Apply filters
        filtered_results = self.apply_filters(merged_results, filters)

        return filtered_results[:top_k]

    def detect_intent(self, query: str) -> str:
        """Classify query intent"""
        # Check for command patterns: "show ...", "set ...", "configure ..."
        # Check for exact term queries: quotes, specific codes
        # Check for conceptual queries: "what is", "how does", "why"
        pass

    def reciprocal_rank_fusion(self, *result_lists, k: int = 60) -> List[Dict]:
        """
        RRF formula: score(doc) = Σ 1/(k + rank_i(doc))
        Combines multiple ranked lists into single ranking
        """
        scores = {}
        for results in result_lists:
            for rank, (doc, score) in enumerate(results, start=1):
                doc_id = doc.metadata['id']
                if doc_id not in scores:
                    scores[doc_id] = {'doc': doc, 'score': 0}
                scores[doc_id]['score'] += 1 / (k + rank)

        sorted_docs = sorted(scores.values(), key=lambda x: x['score'], reverse=True)
        return [item['doc'] for item in sorted_docs]
```

### 2.3 Structured Metadata Extraction (Week 5-6)

**Problem**: Need rich metadata for filtering and context.

**Tasks**:
- [ ] Extract product names and versions
- [ ] Identify command syntax patterns
- [ ] Extract error codes and messages
- [ ] Detect configuration hierarchies
- [ ] Extract related topics and references
- [ ] Build metadata taxonomy

**Metadata Extractor**:
```python
# metadata_extractor.py
import re
from typing import Dict, List

class MetadataExtractor:
    def __init__(self):
        self.command_patterns = [
            r'(show|set|configure|delete|edit)\s+[\w\-]+',
            r'\[edit\s+[\w\-\s]+\]',
        ]
        self.version_patterns = [
            r'Junos OS (\d+\.\d+\w*)',
            r'Release (\d+\.\d+\.\d+)',
        ]
        self.error_patterns = [
            r'Error:\s+.*',
            r'ERROR_\w+',
        ]

    def extract_metadata(self, text: str, doc_metadata: dict) -> Dict:
        """Extract structured metadata from text"""
        return {
            "commands": self.extract_commands(text),
            "versions": self.extract_versions(text),
            "errors": self.extract_errors(text),
            "hierarchy_paths": self.extract_hierarchy_paths(text),
            "related_topics": self.extract_topics(text),
            "has_code_block": self.has_code_block(text),
            "has_table": self.has_table(text),
            "doc_section_type": self.classify_section(text),
        }

    def extract_commands(self, text: str) -> List[str]:
        """Extract CLI commands"""
        commands = []
        for pattern in self.command_patterns:
            commands.extend(re.findall(pattern, text))
        return list(set(commands))

    def extract_hierarchy_paths(self, text: str) -> List[str]:
        """Extract configuration hierarchy paths like [edit protocols bgp]"""
        return re.findall(r'\[edit\s+([\w\s\-]+)\]', text)

    def classify_section(self, text: str) -> str:
        """Classify section type"""
        # Reference, guide, example, troubleshooting, etc.
        if "example" in text.lower()[:100]:
            return "example"
        elif "configure" in text.lower()[:100]:
            return "configuration"
        elif "error" in text.lower() or "troubleshoot" in text.lower():
            return "troubleshooting"
        else:
            return "reference"
```

### 2.4 Version Management System (Week 6)

**Problem**: Documentation changes across product versions.

**Tasks**:
- [ ] Design version tracking schema
- [ ] Add version metadata to all chunks
- [ ] Implement version filtering in search
- [ ] Track deprecated features
- [ ] Build version comparison system
- [ ] Add "version changed" annotations

**Version Schema**:
```python
# version_manager.py
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class VersionInfo:
    product: str
    version: str
    release_date: str
    status: str  # "current", "deprecated", "eol"

@dataclass
class VersionedFeature:
    feature_name: str
    introduced_in: str
    deprecated_in: Optional[str] = None
    removed_in: Optional[str] = None
    replacement: Optional[str] = None
    changes: List[dict] = None

class VersionManager:
    def __init__(self):
        self.versions = {}
        self.features = {}

    def add_version(self, version_info: VersionInfo):
        """Register a product version"""
        self.versions[version_info.version] = version_info

    def track_feature(self, feature: VersionedFeature):
        """Track feature across versions"""
        self.features[feature.feature_name] = feature

    def get_feature_status(self, feature_name: str, version: str) -> str:
        """Check if feature is available in version"""
        feature = self.features.get(feature_name)
        if not feature:
            return "unknown"

        if self.version_less_than(version, feature.introduced_in):
            return "not_available"
        elif feature.removed_in and self.version_greater_than(version, feature.removed_in):
            return "removed"
        elif feature.deprecated_in and self.version_greater_than(version, feature.deprecated_in):
            return "deprecated"
        else:
            return "available"

    def compare_versions(self, feature: str, v1: str, v2: str) -> dict:
        """Compare feature between two versions"""
        pass
```

### 2.5 Enhanced Retrieval Pipeline (Week 6-7)

**Tasks**:
- [ ] Implement query expansion
- [ ] Add reranking with cross-encoder models
- [ ] Context window optimization
- [ ] Parent-child chunk retrieval
- [ ] Implement query routing
- [ ] Add result explanation/confidence scores

**Advanced Retrieval**:
```python
# retrieval_pipeline.py
from sentence_transformers import CrossEncoder

class AdvancedRetriever:
    def __init__(self, hybrid_search, reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.hybrid_search = hybrid_search
        self.reranker = CrossEncoder(reranker_model)
        self.query_expander = QueryExpander()

    def retrieve(self, query: str, top_k: int = 5, filters: dict = None) -> List[Dict]:
        """
        Advanced retrieval pipeline

        Steps:
        1. Expand query with synonyms/related terms
        2. Hybrid search (retrieves 2x more results)
        3. Rerank with cross-encoder
        4. Fetch parent chunks for context
        5. Return top_k with confidence scores
        """
        # Query expansion
        expanded_queries = self.query_expander.expand(query)

        # Multi-query retrieval
        all_results = []
        for q in expanded_queries:
            results = self.hybrid_search.search(q, top_k=top_k * 2, filters=filters)
            all_results.extend(results)

        # Deduplicate
        unique_results = self.deduplicate(all_results)

        # Rerank
        reranked = self.rerank(query, unique_results)

        # Fetch parent chunks for context
        enriched = self.add_parent_context(reranked[:top_k])

        return enriched

    def rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """Rerank results using cross-encoder"""
        pairs = [[query, doc.page_content] for doc in results]
        scores = self.reranker.predict(pairs)

        # Combine with original scores
        for i, doc in enumerate(results):
            doc.metadata['rerank_score'] = scores[i]
            doc.metadata['combined_score'] = (
                0.3 * doc.metadata.get('search_score', 0) +
                0.7 * scores[i]
            )

        return sorted(results, key=lambda x: x.metadata['combined_score'], reverse=True)
```

### Phase 2 Deliverables

- ✅ Hierarchy-aware chunking
- ✅ Hybrid search (vector + keyword)
- ✅ Rich metadata extraction
- ✅ Version management
- ✅ Advanced retrieval with reranking
- ✅ Query expansion and routing

**Success Metrics**:
- Retrieval accuracy > 85% (measured with test queries)
- Can handle 100+ document library
- Version-aware search working
- Average query latency < 2 seconds

---

## Phase 3: Knowledge Graph Layer

**Duration**: 2-3 weeks
**Goal**: Add relationship-based retrieval
**Status**: Not Started

### 3.1 Document Graph Schema (Week 8)

**Tasks**:
- [ ] Design graph schema (nodes, edges)
- [ ] Choose graph database (NetworkX vs Neo4j)
- [ ] Setup graph storage on Modal
- [ ] Define entity types
- [ ] Define relationship types
- [ ] Create graph API

**Graph Schema**:
```python
# graph_schema.py
from enum import Enum
from dataclasses import dataclass
from typing import List

class NodeType(Enum):
    DOCUMENT = "document"
    SECTION = "section"
    COMMAND = "command"
    CONFIGURATION = "configuration"
    ERROR_CODE = "error_code"
    FEATURE = "feature"
    PRODUCT = "product"
    VERSION = "version"

class EdgeType(Enum):
    CONTAINS = "contains"  # document -> section
    REFERENCES = "references"  # section -> section
    REQUIRES = "requires"  # feature -> configuration
    CONFLICTS_WITH = "conflicts_with"  # config -> config
    SUPERSEDES = "supersedes"  # version -> version
    RELATED_TO = "related_to"  # general relationship
    DEPRECATED_BY = "deprecated_by"  # feature -> feature
    TROUBLESHOOTS = "troubleshoots"  # section -> error

@dataclass
class Node:
    id: str
    type: NodeType
    properties: dict

@dataclass
class Edge:
    source_id: str
    target_id: str
    type: EdgeType
    properties: dict

class DocumentGraph:
    def __init__(self, storage_path: str):
        import networkx as nx
        self.graph = nx.MultiDiGraph()
        self.storage_path = storage_path

    def add_node(self, node: Node):
        self.graph.add_node(node.id, type=node.type.value, **node.properties)

    def add_edge(self, edge: Edge):
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            type=edge.type.value,
            **edge.properties
        )

    def find_related(self, node_id: str, relationship_types: List[EdgeType] = None, max_depth: int = 2) -> List[Node]:
        """Find related nodes via graph traversal"""
        pass

    def find_path(self, start_id: str, end_id: str) -> List[Node]:
        """Find shortest path between nodes"""
        pass
```

### 3.2 Entity Extraction (Week 8-9)

**Tasks**:
- [ ] Extract entities from documents
- [ ] Use NER (Named Entity Recognition)
- [ ] Build command/config parser
- [ ] Extract hierarchical relationships
- [ ] Link entities across documents
- [ ] Resolve entity aliases

**Entity Extractor**:
```python
# entity_extractor.py
import spacy
from typing import List, Dict

class EntityExtractor:
    def __init__(self):
        # Load spaCy model or use custom NER
        self.nlp = spacy.load("en_core_web_sm")
        self.command_parser = CommandParser()
        self.config_parser = ConfigParser()

    def extract_entities(self, text: str, doc_metadata: dict) -> List[Dict]:
        """Extract all entities from text"""
        entities = []

        # NER for general entities
        doc = self.nlp(text)
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "type": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })

        # Domain-specific extraction
        entities.extend(self.command_parser.extract_commands(text))
        entities.extend(self.config_parser.extract_configs(text))
        entities.extend(self.extract_error_codes(text))
        entities.extend(self.extract_features(text))

        return entities

    def link_entities(self, entities: List[Dict], graph: DocumentGraph):
        """Create graph nodes and edges from entities"""
        for entity in entities:
            node = Node(
                id=self.generate_id(entity),
                type=self.get_node_type(entity),
                properties=entity
            )
            graph.add_node(node)

        # Find relationships between entities
        relationships = self.find_relationships(entities)
        for rel in relationships:
            edge = Edge(
                source_id=rel['source'],
                target_id=rel['target'],
                type=rel['type'],
                properties=rel.get('properties', {})
            )
            graph.add_edge(edge)
```

### 3.3 Relationship Mapping (Week 9)

**Tasks**:
- [ ] Build relationship extraction rules
- [ ] Map explicit relationships (references, requires)
- [ ] Infer implicit relationships (similarity, co-occurrence)
- [ ] Weight relationships by strength
- [ ] Build relationship index
- [ ] Validate relationship accuracy

**Relationship Extractor**:
```python
# relationship_extractor.py

class RelationshipExtractor:
    def __init__(self):
        self.rules = self.load_rules()

    def extract_relationships(self, entities: List[Dict], text: str) -> List[Dict]:
        """Extract relationships between entities"""
        relationships = []

        # Rule-based extraction
        relationships.extend(self.extract_references(entities, text))
        relationships.extend(self.extract_dependencies(entities, text))
        relationships.extend(self.extract_hierarchies(entities, text))

        # Pattern-based extraction
        relationships.extend(self.extract_by_patterns(entities, text))

        # Co-occurrence based
        relationships.extend(self.extract_cooccurrence(entities))

        return relationships

    def extract_references(self, entities: List[Dict], text: str) -> List[Dict]:
        """Extract explicit references like 'See Section 3.2'"""
        # Pattern: "See ...", "Refer to ...", "As described in ..."
        pass

    def extract_dependencies(self, entities: List[Dict], text: str) -> List[Dict]:
        """Extract requirement relationships"""
        # Pattern: "requires ...", "depends on ...", "prerequisite ..."
        pass

    def extract_hierarchies(self, entities: List[Dict], text: str) -> List[Dict]:
        """Extract parent-child relationships from config hierarchies"""
        # Parse [edit protocols bgp] structure
        pass
```

### 3.4 Graph-Based Retrieval (Week 9-10)

**Tasks**:
- [ ] Implement graph traversal queries
- [ ] Add graph-based search to retrieval pipeline
- [ ] Build "related topics" feature
- [ ] Implement dependency resolution
- [ ] Add graph visualization (optional)
- [ ] Optimize graph queries

**Graph Retrieval**:
```python
# graph_retrieval.py

class GraphRetriever:
    def __init__(self, graph: DocumentGraph, vector_retriever):
        self.graph = graph
        self.vector_retriever = vector_retriever

    def retrieve_with_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Combine vector search with graph traversal

        Steps:
        1. Vector search for initial results
        2. Extract entities from results
        3. Graph traversal to find related nodes
        4. Rank expanded results
        5. Return enriched context
        """
        # Initial vector search
        initial_results = self.vector_retriever.retrieve(query, top_k=top_k)

        # Extract entities from results
        entities = self.extract_entities_from_results(initial_results)

        # Graph expansion
        expanded_nodes = []
        for entity in entities:
            related = self.graph.find_related(
                entity.id,
                relationship_types=[
                    EdgeType.REFERENCES,
                    EdgeType.REQUIRES,
                    EdgeType.RELATED_TO
                ],
                max_depth=2
            )
            expanded_nodes.extend(related)

        # Merge and rank
        all_results = self.merge_results(initial_results, expanded_nodes)
        ranked = self.rank_with_graph_features(all_results, query)

        return ranked[:top_k]

    def find_dependencies(self, config_id: str) -> List[str]:
        """Find all dependencies for a configuration"""
        return self.graph.find_related(
            config_id,
            relationship_types=[EdgeType.REQUIRES],
            max_depth=5
        )

    def get_related_topics(self, topic_id: str, limit: int = 10) -> List[Dict]:
        """Get related topics via graph"""
        related = self.graph.find_related(
            topic_id,
            relationship_types=[EdgeType.RELATED_TO, EdgeType.REFERENCES],
            max_depth=2
        )
        return self.rank_by_relevance(related)[:limit]
```

### Phase 3 Deliverables

- ✅ Document knowledge graph
- ✅ Entity extraction from documents
- ✅ Relationship mapping
- ✅ Graph-based retrieval
- ✅ Related topics feature
- ✅ Dependency resolution

**Success Metrics**:
- Graph contains 1000+ nodes and 5000+ edges
- Graph traversal queries < 100ms
- Related topics accuracy > 80%
- Dependency resolution working

---

## Phase 4: Production Optimization

**Duration**: 2-3 weeks
**Goal**: Performance, caching, monitoring
**Status**: Not Started

### 4.1 Caching Layer (Week 11)

**Tasks**:
- [ ] Setup Redis on Modal
- [ ] Implement query result caching
- [ ] Cache embeddings
- [ ] Add cache invalidation logic
- [ ] Implement cache warming
- [ ] Monitor cache hit rates

**Caching Strategy**:
```python
# cache_manager.py
import redis
import hashlib
import json
from typing import Optional

class CacheManager:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        self.ttl = 3600  # 1 hour default

    def cache_query_result(self, query: str, filters: dict, result: dict):
        """Cache search results"""
        cache_key = self.generate_key("query", query, filters)
        self.redis.setex(cache_key, self.ttl, json.dumps(result))

    def get_cached_query(self, query: str, filters: dict) -> Optional[dict]:
        """Retrieve cached query result"""
        cache_key = self.generate_key("query", query, filters)
        cached = self.redis.get(cache_key)
        return json.loads(cached) if cached else None

    def cache_embedding(self, text: str, embedding: list):
        """Cache text embeddings"""
        cache_key = self.generate_key("embedding", text)
        self.redis.setex(cache_key, 86400, json.dumps(embedding))  # 24 hours

    def generate_key(self, prefix: str, *args) -> str:
        """Generate cache key"""
        content = json.dumps(args, sort_keys=True)
        hash_value = hashlib.sha256(content.encode()).hexdigest()
        return f"{prefix}:{hash_value}"
```

### 4.2 Reranking & Relevance Tuning (Week 11)

**Tasks**:
- [ ] Fine-tune cross-encoder reranker
- [ ] Build evaluation dataset
- [ ] Implement A/B testing framework
- [ ] Tune hybrid search weights
- [ ] Optimize ranking features
- [ ] Add user feedback loop

### 4.3 Monitoring & Observability (Week 12)

**Tasks**:
- [ ] Setup logging (Loguru)
- [ ] Add metrics collection (Prometheus)
- [ ] Create dashboards (Grafana)
- [ ] Setup alerting
- [ ] Add performance profiling
- [ ] Track cost metrics

**Monitoring**:
```python
# monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
query_counter = Counter('queries_total', 'Total queries')
query_duration = Histogram('query_duration_seconds', 'Query duration')
cache_hit_counter = Counter('cache_hits_total', 'Cache hits')
cache_miss_counter = Counter('cache_misses_total', 'Cache misses')
active_documents = Gauge('active_documents', 'Number of active documents')

class MetricsCollector:
    @staticmethod
    def track_query(func):
        def wrapper(*args, **kwargs):
            query_counter.inc()
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                query_duration.observe(duration)
                return result
            except Exception as e:
                # Track errors
                raise
        return wrapper

    @staticmethod
    def track_cache_hit():
        cache_hit_counter.inc()

    @staticmethod
    def track_cache_miss():
        cache_miss_counter.inc()
```

### 4.4 Advanced MCP Tools (Week 12-13)

**Tasks**:
- [ ] Add `get_config_hierarchy` tool
- [ ] Add `explain_command` tool
- [ ] Add `compare_versions` tool
- [ ] Add `find_troubleshooting` tool
- [ ] Add `get_code_examples` tool
- [ ] Document all MCP tools

**Enhanced MCP Tools**:
```python
# mcp_tools.py

@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "get_config_hierarchy":
        # Get full hierarchy for a configuration statement
        config_path = arguments["config_path"]
        hierarchy = graph_retriever.get_hierarchy(config_path)
        return [TextContent(type="text", text=json.dumps(hierarchy, indent=2))]

    elif name == "explain_command":
        # Detailed command explanation with syntax, options, examples
        command = arguments["command"]
        version = arguments.get("version")
        explanation = await explain_command_detailed(command, version)
        return [TextContent(type="text", text=explanation)]

    elif name == "compare_versions":
        # Compare feature/config between versions
        feature = arguments["feature"]
        v1 = arguments["version1"]
        v2 = arguments["version2"]
        comparison = version_manager.compare_versions(feature, v1, v2)
        return [TextContent(type="text", text=json.dumps(comparison, indent=2))]

    elif name == "find_troubleshooting":
        # Find troubleshooting docs for an error
        error = arguments["error"]
        docs = await search_troubleshooting(error)
        return [TextContent(type="text", text=json.dumps(docs, indent=2))]

    elif name == "get_code_examples":
        # Get code examples for a feature
        feature = arguments["feature"]
        examples = await get_examples(feature)
        return [TextContent(type="text", text=examples)]
```

### 4.5 Documentation & Testing (Week 13)

**Tasks**:
- [ ] Write deployment documentation
- [ ] Create user guide for MCP tools
- [ ] Write API documentation
- [ ] Add unit tests (pytest)
- [ ] Add integration tests
- [ ] Performance benchmarking
- [ ] Security audit

### Phase 4 Deliverables

- ✅ Redis caching layer
- ✅ Reranking optimization
- ✅ Monitoring & metrics
- ✅ 8-10 MCP tools
- ✅ Comprehensive documentation
- ✅ Test coverage > 70%

**Success Metrics**:
- Cache hit rate > 60%
- P95 query latency < 1.5 seconds
- System uptime > 99.5%
- Modal cost < $100/month (production)

---

## Technical Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Deployment** | Modal | Serverless Python functions |
| **API Server** | FastAPI | REST API and MCP server |
| **Protocol** | MCP (Model Context Protocol) | Claude integration |
| **LLM** | Claude 3.5 Sonnet (Anthropic) | Answer generation |
| **Vector DB** | FAISS (or Qdrant) | Semantic search |
| **Graph DB** | NetworkX (or Neo4j) | Relationship queries |
| **Cache** | Redis on Modal | Query/embedding cache |
| **Document Processing** | Docling | PDF/HTML parsing |
| **Embeddings** | HuggingFace Sentence Transformers | Text vectorization |
| **Reranking** | Cross-Encoder (sentence-transformers) | Result reranking |
| **Keyword Search** | BM25 (rank_bm25) | Exact match search |
| **NER** | spaCy | Entity extraction |
| **Monitoring** | Prometheus + Grafana | Metrics and dashboards |

---

## Cost Estimation

### Development (Phase 1-4)
- Modal compute: $20-50/month
- Anthropic API: $10-30/month (testing)
- Redis: Included in Modal
- **Total Dev Cost**: $30-80/month

### Production (Post-Phase 4)
- Modal compute: $50-150/month (depends on usage)
- Anthropic API: $100-500/month (depends on query volume)
- Redis: $20-50/month (if using dedicated)
- Storage (Modal Volumes): $10-20/month
- **Total Production Cost**: $180-720/month

**Scaling Assumptions**:
- 10,000 queries/month
- 500 documents
- 100GB storage

---

## Success Metrics

### Technical Metrics
- [ ] Retrieval accuracy > 85%
- [ ] P95 query latency < 2 seconds
- [ ] System uptime > 99.5%
- [ ] Cache hit rate > 60%
- [ ] Vector search recall@10 > 0.9

### Business Metrics
- [ ] Can handle 500+ document library
- [ ] Support 3+ product families with versions
- [ ] 10+ MCP tools available
- [ ] User satisfaction > 4.5/5
- [ ] Cost per query < $0.10

### Quality Metrics
- [ ] Answer relevance > 90%
- [ ] Correct version detection > 95%
- [ ] Command syntax accuracy > 98%
- [ ] Graph relationship accuracy > 85%

---

## Risk Mitigation

### Technical Risks
1. **Vector DB scalability**: Migrate to Qdrant if FAISS doesn't scale
2. **Graph DB performance**: Use Neo4j if NetworkX is too slow
3. **LLM cost**: Implement aggressive caching and prompt optimization
4. **Modal cold starts**: Keep functions warm with scheduled pings

### Product Risks
1. **Poor retrieval quality**: Build evaluation dataset early, iterate
2. **High latency**: Profile and optimize critical paths
3. **Cost overruns**: Monitor usage, set alerts, optimize prompts

---

## Maintenance & Iteration

### Weekly
- Monitor error rates and latency
- Review slow queries
- Check cache hit rates
- Review cost metrics

### Monthly
- Retrain reranker model
- Update document library
- Optimize search weights
- Review user feedback

### Quarterly
- Major version updates
- Architecture review
- Cost optimization
- Security audit

---

## Next Steps

1. **Review this roadmap** with stakeholders
2. **Set up development environment**:
   - Install Modal CLI
   - Create Anthropic API account
   - Clone BookTutor repo
3. **Start Phase 1**: Basic Modal + MCP integration
4. **Create test dataset**: 10-20 sample documents
5. **Define success criteria**: Specific to your use case

---

## Resources

### Documentation
- [Modal Docs](https://modal.com/docs)
- [MCP Protocol](https://github.com/anthropics/mcp)
- [LangChain Docs](https://python.langchain.com/)
- [Claude API Docs](https://docs.anthropic.com/)

### Related Projects
- [BookTutor AI](https://github.com/yourusername/booktutor-ai) - Current repo
- [Docling](https://github.com/DS4SD/docling) - Document processing
- [sentence-transformers](https://www.sbert.net/) - Embeddings

### Community
- Modal Discord
- LangChain Discord
- Anthropic Discord

---

**Document Version**: 1.0
**Last Updated**: 2025-10-27
**Author**: AI Roadmap Generator
**Status**: Draft - Awaiting Review
