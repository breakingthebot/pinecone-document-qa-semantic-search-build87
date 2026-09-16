"""
Document Ingestion and Vector Indexing REST Routes.
Handles document chunking, embedding generation, Pinecone upserts, and lifecycle management.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, status

from src.models.schema import (
    DocumentIngestRequest,
    DocumentResponse,
    UpsertVectorsRequest,
    UpsertVectorsResponse,
)
from src.services.document_service import DocumentService
from src.engine.pinecone_client import get_pinecone_engine

router = APIRouter(prefix="/api/documents", tags=["Document Ingestion"])
service = DocumentService()


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
def ingest_document(payload: DocumentIngestRequest) -> DocumentResponse:
    """
    Ingests a document: chunks the content, computes dense embeddings,
    and indexes the vectors into Pinecone with metadata tags.
    """
    return service.ingest_document(request=payload)


@router.get("", response_model=List[DocumentResponse])
def list_documents(
    namespace: Optional[str] = Query(default=None, description="Optional namespace filter"),
) -> List[DocumentResponse]:
    """
    Lists all ingested documents in the catalog.
    """
    return service.list_documents(namespace=namespace)


@router.get("/{doc_id}", response_model=DocumentResponse)
def get_document(doc_id: str) -> DocumentResponse:
    """
    Retrieves metadata for a specific ingested document.
    """
    doc = service.get_document(doc_id=doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found in catalog",
        )
    return doc


@router.delete("/{doc_id}")
def delete_document(doc_id: str) -> Dict[str, Any]:
    """
    Deletes a document from the catalog and purges all of its vector chunks from Pinecone.
    """
    deleted = service.delete_document(doc_id=doc_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found",
        )
    return {"ok": True, "deleted": doc_id}


@router.post("/vectors/upsert", response_model=UpsertVectorsResponse)
def upsert_raw_vectors(payload: UpsertVectorsRequest) -> UpsertVectorsResponse:
    """
    Direct low-level Pinecone vector upsert endpoint.
    Inserts or overwrites vector records with explicit values and metadata.
    """
    engine = get_pinecone_engine()
    count = engine.upsert(vectors=payload.vectors, namespace=payload.namespace)
    return UpsertVectorsResponse(upserted_count=count, namespace=payload.namespace)
