"""
Pydantic Schemas for Document Ingestion, Semantic Search, and Question Answering.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DocumentIngestRequest(BaseModel):
    """
    Schema for submitting a document for chunking, vector embedding, and Pinecone indexing.
    """

    title: str = Field(..., min_length=1, max_length=200, description="Title of the source document")
    content: str = Field(..., min_length=10, description="Full text content of the document")
    category: str = Field(default="General", min_length=1, max_length=64, description="Document category (e.g. HR, Engineering, Legal)")
    source_url: Optional[str] = Field(default=None, description="Optional canonical reference URL")
    author: Optional[str] = Field(default="Staff", description="Author or publishing department")
    namespace: str = Field(default="default", description="Pinecone index partition namespace")
    chunk_size: Optional[int] = Field(default=400, ge=50, le=2000, description="Target character count per chunk")
    chunk_overlap: Optional[int] = Field(default=80, ge=0, le=500, description="Sliding window character overlap")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Custom metadata tags attached to vector chunks")


class DocumentChunk(BaseModel):
    """
    Represents an atomic chunk of a document stored with metadata.
    """

    id: str = Field(..., description="Unique chunk vector ID (e.g. doc_xyz#chunk_0)")
    doc_id: str = Field(..., description="Parent document ID")
    chunk_index: int = Field(..., ge=0, description="Zero-based sequence index")
    text: str = Field(..., description="Extracted chunk text payload")
    character_count: int = Field(..., ge=1)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata tags attached to vector record")


class DocumentResponse(BaseModel):
    """
    Public response schema for an ingested document.
    """

    id: str
    title: str
    category: str
    source_url: Optional[str] = None
    author: Optional[str] = None
    namespace: str
    chunk_count: int
    character_count: int
    created_at: str


class VectorRecord(BaseModel):
    """
    Pinecone vector record payload containing dense values and metadata payload.
    """

    id: str
    values: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UpsertVectorsRequest(BaseModel):
    """
    Schema for batch upserting raw vectors directly into Pinecone.
    """

    vectors: List[VectorRecord]
    namespace: str = Field(default="default")


class UpsertVectorsResponse(BaseModel):
    """
    Response schema after upserting vector records.
    """

    upserted_count: int
    namespace: str


class VectorQueryRequest(BaseModel):
    """
    Pinecone nearest-neighbor vector search query payload.
    Supports either pre-computed float vectors or raw natural language text.
    """

    vector: Optional[List[float]] = Field(default=None, description="Pre-computed dense float vector query")
    query_text: Optional[str] = Field(default=None, description="Natural language search phrase to embed on-the-fly")
    top_k: int = Field(default=4, ge=1, le=100, description="Number of nearest neighbors to retrieve")
    namespace: str = Field(default="default", description="Target namespace to search within")
    filter: Optional[Dict[str, Any]] = Field(default=None, description="Pinecone metadata filter ($eq, $in, $gt, etc.)")
    include_metadata: bool = Field(default=True, description="Whether to include metadata in matches")
    include_values: bool = Field(default=False, description="Whether to return float vector values")


class ScoredMatch(BaseModel):
    """
    Single retrieved vector match with similarity score and associated metadata.
    """

    id: str
    score: float
    values: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


class VectorQueryResponse(BaseModel):
    """
    Response payload containing retrieved nearest-neighbor vector matches.
    """

    matches: List[ScoredMatch]
    namespace: str


class Citation(BaseModel):
    """
    Source citation linking an AI answer back to the exact document chunk.
    """

    doc_id: str
    title: str
    category: str
    chunk_index: int
    similarity_score: float
    snippet: str


class QuestionAnsweringRequest(BaseModel):
    """
    Request payload for end-to-end Document Q&A RAG pipeline.
    """

    question: str = Field(..., min_length=3, description="User's natural language question")
    namespace: str = Field(default="default", description="Pinecone namespace to query")
    category_filter: Optional[str] = Field(default=None, description="Optional category restriction")
    metadata_filter: Optional[Dict[str, Any]] = Field(default=None, description="Custom Pinecone metadata filter")
    top_k: int = Field(default=4, ge=1, le=20, description="Number of context chunks to retrieve for synthesis")
    min_score_threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="Minimum similarity score cutoff")


class QuestionAnsweringResponse(BaseModel):
    """
    Complete synthesized answer with citations, confidence metrics, and context snippets.
    """

    question: str
    answer: str
    confidence_score: float
    citations: List[Citation]
    total_candidates_reviewed: int
    namespace: str
    processing_time_ms: float


class IndexStatsResponse(BaseModel):
    """
    Telemetry statistics describing the Pinecone index and vector counts per namespace.
    """

    index_name: str
    dimension: int
    metric: str
    total_vector_count: int
    namespaces: Dict[str, Dict[str, int]]
    engine: str
