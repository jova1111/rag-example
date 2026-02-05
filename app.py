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
    classification: str
    confidence: float
    justification: str
    document_length: int
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
    """Internal function to classify text using async database operations.
    
    Args:
        text: Document text to classify
        include_context: Include retrieved documents in response
        
    Returns:
        Classification result
    """
    logger.info("Retrieving similar documents...")
    
    # Connect to async database
    async with AsyncDatabaseConnection() as db:
        conn = db.conn
        
        # Initialize retriever
        retriever = DocumentRetriever(None, embedder)
        
        # Retrieve similar documents using async method
        logger.debug(f"Retrieving top {Config.TOP_K_RETRIEVAL} similar documents...")
        
        start_time = time.perf_counter()
        similar_docs = await retriever.retrieve_similar_async(conn, text, top_k=Config.TOP_K_RETRIEVAL)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        logger.info(f"Retrieved similar documents in {elapsed_ms:.2f}ms")
        
        if not similar_docs:
            logger.error("No similar documents found in database")
            raise HTTPException(
                status_code=500,
                detail="No similar documents found. Ensure training documents are ingested."
            )
        
        logger.info(f"Retrieved {len(similar_docs)} similar documents")
        logger.debug(f"Top match: {similar_docs[0].get('title')} (distance: {similar_docs[0].get('distance', 0):.4f})")
        
        # Format context for LLM
        context = retriever.format_context(similar_docs)
        logger.debug(f"Context size: {len(context)} characters")
        
        # Classify
        logger.info("Classifying with LLM...")
        
        start_time = time.perf_counter()
        classification_result = await classifier.classify(text, context)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        logger.info(f"Classification completed in {elapsed_ms:.2f}ms")
        
        logger.info(f"Classification: {classification_result['classification']} ({classification_result['confidence']:.2%})")
        logger.debug(f"Justification: {classification_result['justification'][:100]}...")
        
        # Prepare retrieved documents for response if requested
        retrieved_docs_info = None
        if include_context:
            retrieved_docs_info = [
                {
                    "title": doc.get('title'),
                    "classification": doc.get('classification'),
                    "similarity": round(1 - doc.get('distance', 0), 3),
                    "confidence": round(doc.get('confidence', 0), 2)
                }
                for doc in similar_docs
            ]
            logger.debug(f"Including {len(retrieved_docs_info)} retrieved documents in response")
        
        return ClassificationResult(
            classification=classification_result['classification'],
            confidence=classification_result['confidence'],
            justification=classification_result['justification'],
            document_length=len(text),
            retrieved_documents_count=len(similar_docs),
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
