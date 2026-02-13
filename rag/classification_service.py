"""Classification service for document classification business logic."""

import logging
import time
from typing import Any

from fastapi import HTTPException

from entities import ClassificationResult
from database.connection import AsyncDatabaseConnection
from embedding.embedder import DocumentEmbedder
from embedding.chunker import DocumentChunker
from rag.retriever import DocumentRetriever
from rag.classifier import DocumentClassifier
from config import Config

logger = logging.getLogger(__name__)


class ClassificationService:
    """Service class for document classification business logic."""
    
    def __init__(self, embedder: DocumentEmbedder, classifier: DocumentClassifier):
        """Initialize the classification service.
        
        Args:
            embedder: Document embedder instance
            classifier: Document classifier instance
        """
        self.embedder = embedder
        self.classifier = classifier
    
    def _chunk_document(self, text: str) -> list:
        """Chunk the input document using appropriate strategy.
        
        Args:
            text: Document text to chunk
            
        Returns:
            List of document chunks
        """
        chunker = DocumentChunker(
            strategy='semantic' if len(text) > 1000 else 'hybrid',
            max_chunk_size=Config.MAX_CHUNK_SIZE,
            overlap_size=Config.CHUNK_OVERLAP_SIZE,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        logger.info(f"Chunking document ({len(text)} chars)...")
        start_time = time.perf_counter()
        chunks = chunker.chunk_document(text)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"Created {len(chunks)} chunks in {elapsed_ms:.2f}ms")
        
        return chunks

    async def _retrieve_similar_documents(
        self, 
        chunks: list, 
        retriever: DocumentRetriever, 
        conn: Any
    ) -> tuple[list, dict]:
        """Retrieve similar documents for all chunks.
        
        Args:
            chunks: List of document chunks
            retriever: Document retriever instance
            conn: Database connection
            
        Returns:
            Tuple of (all_retrieved_docs, tag_frequency)
        """
        all_retrieved_docs = []
        tag_frequency = {}
        
        for idx, chunk in enumerate(chunks, 1):
            logger.debug(f"Processing chunk {idx}/{len(chunks)}...")
            
            start_time = time.perf_counter()
            similar_docs = await retriever.retrieve_similar_async(
                conn, chunk.text, top_k=Config.TOP_K_RETRIEVAL
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            logger.debug(f"  Retrieved {len(similar_docs)} similar chunks in {elapsed_ms:.2f}ms")
            
            # Aggregate tags from retrieved chunks
            self._aggregate_tags(similar_docs, all_retrieved_docs, tag_frequency)
        
        logger.info(f"Aggregated {len(all_retrieved_docs)} total retrieved chunks")
        logger.info(f"Found {len(tag_frequency)} unique tags")
        
        return all_retrieved_docs, tag_frequency

    def _aggregate_tags(
        self, 
        similar_docs: list, 
        all_retrieved_docs: list, 
        tag_frequency: dict
    ) -> None:
        """Aggregate tags from similar documents with similarity weighting.
        
        Args:
            similar_docs: List of similar documents
            all_retrieved_docs: Accumulator for all retrieved documents
            tag_frequency: Dictionary mapping tags to weighted frequency
        """
        for doc in similar_docs:
            all_retrieved_docs.append(doc)
            
            # Extract tags from retrieved chunks
            chunk_tags = doc.get('tags', [])
            if isinstance(chunk_tags, str):
                chunk_tags = [chunk_tags]
            
            for tag in chunk_tags:
                # Weight by similarity (1 - distance)
                similarity = 1 - doc.get('distance', 0)
                tag_frequency[tag] = tag_frequency.get(tag, 0) + similarity

    async def _fetch_available_tags(self, conn: Any) -> list[str]:
        """Fetch all available tags from database.
        
        Args:
            conn: Database connection
            
        Returns:
            List of available tag names
        """
        logger.debug("Fetching available tags from database...")
        available_tags_query = "SELECT DISTINCT tag_name FROM tags ORDER BY tag_name"
        tags_rows = await conn.fetch(available_tags_query)
        available_tags = [row['tag_name'] for row in tags_rows]
        logger.info(f"Found {len(available_tags)} available tags in database")
        
        return available_tags

    def _prepare_classification_context(
        self, 
        all_retrieved_docs: list, 
        tag_frequency: dict, 
        retriever: DocumentRetriever
    ) -> tuple[str, str]:
        """Prepare context and tag information for LLM classification.
        
        Args:
            all_retrieved_docs: All retrieved documents
            tag_frequency: Dictionary mapping tags to weighted frequency
            retriever: Document retriever for context formatting
            
        Returns:
            Tuple of (context, tag_context)
        """
        # Format aggregated context for LLM
        context = retriever.format_context_with_tags(all_retrieved_docs[:20])  # Limit to top 20
        logger.debug(f"Context size: {len(context)} characters")
        
        # Prepare tag examples for LLM
        top_tags = sorted(tag_frequency.items(), key=lambda x: x[1], reverse=True)[:15]
        tag_context = f"Most frequently retrieved tags:\n" + "\n".join(
            [f"- {tag}: {score:.2f}" for tag, score in top_tags]
        )
        
        return context, tag_context

    def _format_retrieved_documents(self, all_retrieved_docs: list) -> list[dict]:
        """Format retrieved documents for response.
        
        Args:
            all_retrieved_docs: All retrieved documents
            
        Returns:
            List of formatted document information
        """
        # Deduplicate and summarize
        doc_map = {}
        for doc in all_retrieved_docs:
            doc_id = doc.get('document_id')
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "title": doc.get('title'),
                    "tags": set(),
                    "avg_similarity": [],
                    "chunk_count": 0
                }
            
            doc_tags = doc.get('tags', [])
            if isinstance(doc_tags, str):
                doc_tags = [doc_tags]
            doc_map[doc_id]["tags"].update(doc_tags)
            doc_map[doc_id]["avg_similarity"].append(1 - doc.get('distance', 0))
            doc_map[doc_id]["chunk_count"] += 1
        
        retrieved_docs_info = [
            {
                "title": info["title"],
                "tags": list(info["tags"]),
                "chunks_matched": info["chunk_count"],
                "avg_similarity": round(
                    sum(info["avg_similarity"]) / len(info["avg_similarity"]), 3
                )
            }
            for info in sorted(
                doc_map.values(), 
                key=lambda x: sum(x["avg_similarity"]) / len(x["avg_similarity"]), 
                reverse=True
            )
        ][:10]  # Top 10 documents
        
        logger.debug(f"Including {len(retrieved_docs_info)} retrieved documents in response")
        
        return retrieved_docs_info

    async def classify_text(
        self, 
        text: str, 
        include_context: bool = False
    ) -> ClassificationResult:
        """Classify text using chunk-based RAG.
        
        Args:
            text: Document text to classify
            include_context: Include retrieved documents in response
            
        Returns:
            Classification result with tags
        """
        logger.info("Starting chunk-based classification...")
        
        # Chunk the input document
        chunks = self._chunk_document(text)
        
        # Connect to async database
        async with AsyncDatabaseConnection() as db:
            conn = db.conn
            retriever = DocumentRetriever(None, self.embedder)
            
            # Retrieve similar documents for each chunk
            all_retrieved_docs, tag_frequency = await self._retrieve_similar_documents(
                chunks, retriever, conn
            )
            
            if not all_retrieved_docs:
                logger.error("No similar documents found in database")
                raise HTTPException(
                    status_code=500,
                    detail="No similar documents found. Ensure training documents are ingested."
                )
            
            # Fetch all available tags from database
            available_tags = await self._fetch_available_tags(conn)
            
            # Prepare context and tag information for LLM
            context, tag_context = self._prepare_classification_context(
                all_retrieved_docs, tag_frequency, retriever
            )
            
            # Classify using LLM with tag-based approach
            logger.info("Classifying with LLM (tag-based)...")
            
            start_time = time.perf_counter()
            classification_result = await self.classifier.classify_tags(
                text, context, tag_context, available_tags
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            logger.info(f"Classification completed in {elapsed_ms:.2f}ms")
            logger.info(
                f"Tags: {classification_result['tags']} "
                f"({classification_result['confidence']:.2%})"
            )
            
            # Prepare retrieved documents for response if requested
            retrieved_docs_info = None
            if include_context:
                retrieved_docs_info = self._format_retrieved_documents(all_retrieved_docs)
            
            return ClassificationResult(
                tags=classification_result['tags'],
                confidence=classification_result['confidence'],
                justification=classification_result['justification'],
                document_length=len(text),
                chunks_processed=len(chunks),
                retrieved_documents_count=len(all_retrieved_docs),
                retrieved_documents=retrieved_docs_info
            )
