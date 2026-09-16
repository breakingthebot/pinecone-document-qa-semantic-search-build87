"""
Document Chunking and Text Partitioning Service.
Splits documents using sentence-window boundaries with configurable sliding overlap
and enriches each chunk with lineage metadata.
"""

import re
from typing import List, Dict, Any, Optional

from src.models.schema import DocumentChunk


class ChunkingService:
    """
    Partitions long document texts into semantically coherent chunks for vector embedding.
    """

    def split_sentences(self, text: str) -> List[str]:
        """
        Splits text into sentences based on punctuation boundaries while preserving sentence integrity.
        """
        raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences if sentences else [text.strip()]

    def chunk_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        category: str,
        source_url: str = "",
        author: str = "",
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """
        Splits document text into windowed chunks with overlap.
        Preserves complete sentences within each chunk whenever possible.
        """
        sentences = self.split_sentences(content)
        chunks: List[DocumentChunk] = []

        current_chunk_sentences: List[str] = []
        current_len = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # If adding this sentence exceeds chunk_size and we already have text in the current chunk
            if current_len + sentence_len > chunk_size and current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                chunk_id = f"{doc_id}#chunk_{chunk_index}"

                chunk_metadata = {
                    "doc_id": doc_id,
                    "title": title,
                    "category": category,
                    "source_url": source_url or "",
                    "author": author or "",
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                }
                if extra_metadata:
                    chunk_metadata.update(extra_metadata)

                chunks.append(
                    DocumentChunk(
                        id=chunk_id,
                        doc_id=doc_id,
                        chunk_index=chunk_index,
                        text=chunk_text,
                        character_count=len(chunk_text),
                        metadata=chunk_metadata,
                    )
                )
                chunk_index += 1

                # Calculate overlap: retain trailing sentences whose length <= chunk_overlap
                overlap_sentences: List[str] = []
                overlap_len = 0

                for prev_sentence in reversed(current_chunk_sentences):
                    if overlap_len + len(prev_sentence) <= chunk_overlap:
                        overlap_sentences.insert(0, prev_sentence)
                        overlap_len += len(prev_sentence)
                    else:
                        break

                current_chunk_sentences = overlap_sentences
                current_len = sum(len(s) for s in current_chunk_sentences)

            current_chunk_sentences.append(sentence)
            current_len += sentence_len

        # Flush any remaining text as the final chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunk_id = f"{doc_id}#chunk_{chunk_index}"

            chunk_metadata = {
                "doc_id": doc_id,
                "title": title,
                "category": category,
                "source_url": source_url or "",
                "author": author or "",
                "chunk_index": chunk_index,
                "text": chunk_text,
            }
            if extra_metadata:
                chunk_metadata.update(extra_metadata)

            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    character_count=len(chunk_text),
                    metadata=chunk_metadata,
                )
            )

        return chunks
