"""
Document Ingestion and Vector Lifecycle Service.
Coordinates document text parsing, chunking, embedding generation,
and Pinecone vector index upserts.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from src.config import settings
from src.engine.pinecone_client import get_pinecone_engine
from src.engine.embedder import get_embedder
from src.models.schema import (
    DocumentIngestRequest,
    DocumentResponse,
    VectorRecord,
)
from src.services.chunking_service import ChunkingService


class DocumentService:
    """
    Manages document ingestion, chunk vector embeddings, and Pinecone synchronization.
    """

    def __init__(self) -> None:
        self.engine = get_pinecone_engine()
        self.embedder = get_embedder()
        self.chunker = ChunkingService()
        # In-memory document metadata catalog: doc_id -> doc_dict
        self.documents_catalog: Dict[str, Dict[str, Any]] = {}

    def _now_iso(self) -> str:
        """
        Returns current UTC timestamp in ISO 8601 string format.
        """
        return datetime.now(timezone.utc).isoformat()

    def ingest_document(self, request: DocumentIngestRequest) -> DocumentResponse:
        """
        Ingests a document: chunks the text, computes dense embeddings,
        and upserts vector records into the configured Pinecone namespace.
        """
        doc_id = f"doc_{uuid.uuid4().hex[:10]}"
        now = self._now_iso()
        ns = request.namespace or settings.DEFAULT_NAMESPACE

        # 1. Split document into semantic chunks with overlap
        chunks = self.chunker.chunk_document(
            doc_id=doc_id,
            title=request.title,
            content=request.content,
            category=request.category,
            source_url=request.source_url or "",
            author=request.author or "Staff",
            chunk_size=request.chunk_size or settings.DEFAULT_CHUNK_SIZE,
            chunk_overlap=request.chunk_overlap or settings.DEFAULT_CHUNK_OVERLAP,
            extra_metadata=request.metadata,
        )

        # 2. Generate vector embeddings for all chunk texts
        chunk_texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_batch(chunk_texts)

        # 3. Build Pinecone vector records
        vector_records: List[VectorRecord] = []
        chunk_ids: List[str] = []

        for chunk, vec_values in zip(chunks, embeddings):
            chunk_ids.append(chunk.id)
            vector_records.append(
                VectorRecord(
                    id=chunk.id,
                    values=vec_values,
                    metadata=chunk.metadata,
                )
            )

        # 4. Upsert vectors into Pinecone index namespace
        self.engine.upsert(vectors=vector_records, namespace=ns)

        # 5. Persist document metadata in catalog
        doc_record = {
            "id": doc_id,
            "title": request.title,
            "category": request.category,
            "source_url": request.source_url,
            "author": request.author,
            "namespace": ns,
            "chunk_count": len(chunks),
            "chunk_ids": chunk_ids,
            "character_count": len(request.content),
            "created_at": now,
        }
        self.documents_catalog[doc_id] = doc_record

        return DocumentResponse(
            id=doc_id,
            title=request.title,
            category=request.category,
            source_url=request.source_url,
            author=request.author,
            namespace=ns,
            chunk_count=len(chunks),
            character_count=len(request.content),
            created_at=now,
        )

    def get_document(self, doc_id: str) -> Optional[DocumentResponse]:
        """
        Retrieves document metadata from the catalog.
        """
        rec = self.documents_catalog.get(doc_id)
        if not rec:
            return None

        return DocumentResponse(
            id=rec["id"],
            title=rec["title"],
            category=rec["category"],
            source_url=rec.get("source_url"),
            author=rec.get("author"),
            namespace=rec["namespace"],
            chunk_count=rec["chunk_count"],
            character_count=rec["character_count"],
            created_at=rec["created_at"],
        )

    def list_documents(self, namespace: Optional[str] = None) -> List[DocumentResponse]:
        """
        Lists ingested documents, optionally filtered by namespace.
        """
        docs: List[DocumentResponse] = []
        for rec in self.documents_catalog.values():
            if namespace and rec["namespace"] != namespace:
                continue

            docs.append(
                DocumentResponse(
                    id=rec["id"],
                    title=rec["title"],
                    category=rec["category"],
                    source_url=rec.get("source_url"),
                    author=rec.get("author"),
                    namespace=rec["namespace"],
                    chunk_count=rec["chunk_count"],
                    character_count=rec["character_count"],
                    created_at=rec["created_at"],
                )
            )

        return docs

    def delete_document(self, doc_id: str) -> bool:
        """
        Deletes a document from the catalog and purges all corresponding vector chunks from Pinecone.
        """
        rec = self.documents_catalog.get(doc_id)
        if not rec:
            return False

        chunk_ids = rec.get("chunk_ids", [])
        ns = rec["namespace"]

        # Delete chunk vectors from Pinecone
        if chunk_ids:
            self.engine.delete(ids=chunk_ids, namespace=ns)

        del self.documents_catalog[doc_id]
        return True
