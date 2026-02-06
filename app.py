"""FastAPI application for military document classification."""

import os
import traceback
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
import uvicorn

from database.connection import DatabaseConnection, AsyncDatabaseConnection, get_pool, close_pool
from embedding.embedder import DocumentEmbedder
from embedding.chunker import DocumentChunker
from rag.retriever import DocumentRetriever
from rag.classifier import DocumentClassifier
from utils.document_parser import DocumentParser
from utils.debug_logger import setup_debug_logging, log_request
from config import Config

# Setup debug logging
logger = setup_debug_logging(logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO)

# Global instances (initialized on startup)
embedder = None
classifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global embedder, classifier
    
    logger.info("=" * 60)
    logger.info("Starting Military Document Classification API")
    logger.info("=" * 60)
    
    try:
        logger.info("Initializing database connection pool...")
        await get_pool()
        
        logger.info("Loading embedding model...")
        embedder = DocumentEmbedder()
        
        logger.info("Initializing classifier...")
        classifier = DocumentClassifier()
        
        logger.info("API ready!")
        logger.info(f"  - Embedding model: {Config.EMBEDDING_MODEL}")
        logger.info(f"  - LLM: {Config.LLM_PROVIDER}/{Config.LLM_MODEL}")
        logger.info(f"  - Database: {Config.DB_NAME}")
        logger.info(f"  - Database pool: 10-100 connections")
        logger.info(f"  - Debug mode: {os.getenv('DEBUG', 'false')}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown logic
    logger.info("Shutting down...")
    await close_pool()


# Initialize FastAPI app
app = FastAPI(
    title="Military Document Classification API",
    description="RAG-based document classification system using PostgreSQL pgvector",
    version="1.0.0",
    lifespan=lifespan,
    debug=os.getenv("DEBUG", "false").lower() == "true"
)


# Pydantic models for request/response
class ClassificationResult(BaseModel):
    """Classification result model."""
    tags: list[str]
    confidence: float
    justification: str
    document_length: int
    chunks_processed: int
    retrieved_documents_count: int
    retrieved_documents: Optional[list] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    database: str
    embedding_model: str
    llm_model: str


class TextClassificationRequest(BaseModel):
    """Request model for text classification."""
    text: str
    include_context: bool = False


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    logger.debug("Root endpoint accessed")
    return {
        "name": "Military Document Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "classify_file": "/classify/file",
            "classify_text": "/classify/text"
        },
        "supported_formats": list(DocumentParser.SUPPORTED_FORMATS),
        "documentation": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
@log_request
async def health_check():
    """Health check endpoint."""
    try:
        # Test database connection
        logger.debug("Testing database connection...")
        with DatabaseConnection() as db:
            cursor = db.cursor
            cursor.execute("SELECT COUNT(*) as count FROM documents")
            doc_count = cursor.fetchone()['count']
        
        logger.debug(f"Database connection successful: {doc_count} documents")
        
        return HealthResponse(
            status="healthy",
            database=f"connected ({doc_count} documents)",
            embedding_model=Config.EMBEDDING_MODEL,
            llm_model=f"{Config.LLM_PROVIDER}/{Config.LLM_MODEL}"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/classify/file", response_model=ClassificationResult, tags=["Classification"])
async def classify_file(
    file: UploadFile = File(...),
    include_context: bool = Form(False)
):
    """Classify a document from an uploaded file.
    
    Args:
        file: Uploaded file (TXT, PDF, or DOCX)
        include_context: Include retrieved documents in response
        
    Returns:
        Classification result with confidence and justification
    """
    # Validate file format
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in DocumentParser.SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_ext}. Supported: {', '.join(DocumentParser.SUPPORTED_FORMATS)}"
        )
    
    try:
        # Read file content
        file_bytes = await file.read()
        
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Parse document
        logger.info(f"Processing file: {file.filename} ({len(file_bytes)} bytes)")
        text, format_type = DocumentParser.parse_bytes(file_bytes, file.filename)
        
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="No text could be extracted from file")
        
        logger.info(f"Extracted {len(text)} characters from {format_type.upper()} file")
        
        # Classify document
        result = await _classify_text(text, include_context)
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.post("/classify/text", response_model=ClassificationResult, tags=["Classification"])
@log_request
async def classify_text(request: TextClassificationRequest):
    """Classify a document from raw text.
    
    Args:
        request: Text classification request with text and options
        
    Returns:
        Classification result with confidence and justification
    """
    logger.debug(f"Text length: {len(request.text)}, include_context: {request.include_context}")
    
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        return await _classify_text(request.text, request.include_context)
    except Exception as e:
        logger.error(f"Text classification error: {e}", exc_info=True)
        logger.error(f"Error: {e}", exc_info=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


async def _classify_text(text: str, include_context: bool = False) -> ClassificationResult:
    """Internal function to classify text using chunk-based RAG.
    
    Args:
        text: Document text to classify
        include_context: Include retrieved documents in response
        
    Returns:
        Classification result with tags
    """
    logger.info("Starting chunk-based classification...")
    
    # Initialize chunker for semantic chunking
    chunker = DocumentChunker(
        strategy='semantic' if len(text) > 1000 else 'hybrid',
        max_chunk_size=Config.MAX_CHUNK_SIZE,
        overlap_size=Config.CHUNK_OVERLAP_SIZE,
        openai_api_key=Config.OPENAI_API_KEY
    )
    
    # Chunk the input document
    logger.info(f"Chunking document ({len(text)} chars)...")
    start_time = time.perf_counter()
    chunks = chunker.chunk_document(text)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(f"Created {len(chunks)} chunks in {elapsed_ms:.2f}ms")
    
    # Connect to async database
    async with AsyncDatabaseConnection() as db:
        conn = db.conn
        retriever = DocumentRetriever(None, embedder)
        
        # Retrieve similar documents for each chunk
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
        
        logger.info(f"Aggregated {len(all_retrieved_docs)} total retrieved chunks")
        logger.info(f"Found {len(tag_frequency)} unique tags")
        
        if not all_retrieved_docs:
            logger.error("No similar documents found in database")
            raise HTTPException(
                status_code=500,
                detail="No similar documents found. Ensure training documents are ingested."
            )
        
        # Fetch all available tags from database
        logger.debug("Fetching available tags from database...")
        available_tags_query = "SELECT DISTINCT tag_name FROM tags ORDER BY tag_name"
        tags_rows = await conn.fetch(available_tags_query)
        available_tags = [row['tag_name'] for row in tags_rows]
        logger.info(f"Found {len(available_tags)} available tags in database")
        
        # Format aggregated context for LLM
        context = retriever.format_context_with_tags(all_retrieved_docs[:20])  # Limit to top 20
        logger.debug(f"Context size: {len(context)} characters")
        
        # Prepare tag examples for LLM
        top_tags = sorted(tag_frequency.items(), key=lambda x: x[1], reverse=True)[:15]
        tag_context = f"Most frequently retrieved tags:\n" + "\n".join([f"- {tag}: {score:.2f}" for tag, score in top_tags])
        
        # Classify using LLM with tag-based approach
        logger.info("Classifying with LLM (tag-based)...")
        
        start_time = time.perf_counter()
        classification_result = await classifier.classify_tags(text, context, tag_context, available_tags)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        logger.info(f"Classification completed in {elapsed_ms:.2f}ms")
        logger.info(f"Tags: {classification_result['tags']} ({classification_result['confidence']:.2%})")
        
        # Prepare retrieved documents for response if requested
        retrieved_docs_info = None
        if include_context:
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
                    "avg_similarity": round(sum(info["avg_similarity"]) / len(info["avg_similarity"]), 3)
                }
                for info in sorted(doc_map.values(), 
                                  key=lambda x: sum(x["avg_similarity"]) / len(x["avg_similarity"]), 
                                  reverse=True)
            ][:10]  # Top 10 documents
            
            logger.debug(f"Including {len(retrieved_docs_info)} retrieved documents in response")
        
        return ClassificationResult(
            tags=classification_result['tags'],
            confidence=classification_result['confidence'],
            justification=classification_result['justification'],
            document_length=len(text),
            chunks_processed=len(chunks),
            retrieved_documents_count=len(all_retrieved_docs),
            retrieved_documents=retrieved_docs_info
        )



def main():
    """Run the FastAPI application."""
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info("\nStarting server...")
    logger.info(f"Database: {Config.DB_NAME}@{Config.DB_HOST}:{Config.DB_PORT}")
    logger.info(f"LLM: {Config.LLM_PROVIDER}/{Config.LLM_MODEL}")
    logger.info(f"Debug mode: {debug_mode}\n")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug" if debug_mode else "info"
    )


if __name__ == "__main__":
    main()
