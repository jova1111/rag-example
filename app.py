"""FastAPI application for military document classification."""

import os
import traceback
import logging
import argparse
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import uvicorn

from database.connection import DatabaseConnection, get_pool, close_pool
from embedding.embedder import DocumentEmbedder
from rag.classifier import DocumentClassifier
from rag.classification_service import ClassificationService
from entities import ClassificationResult, HealthResponse, TextClassificationRequest
from utils.document_parser import DocumentParser
from utils.debug_logger import setup_debug_logging, log_request
from config import Config

# Setup debug logging
logger = setup_debug_logging(logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO)

# Global instances (initialized on startup)
classification_service: ClassificationService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global classification_service
    
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
        
        logger.info("Initializing classification service...")
        classification_service = ClassificationService(embedder, classifier)
        
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
            db.cursor.execute("SELECT COUNT(*) as count FROM documents")
            row = db.cursor.fetchone()
            doc_count = row['count']
        
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
    filename: str = file.filename or ""
    file_ext = os.path.splitext(filename)[1].lower()
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
        logger.info(f"Processing file: {filename} ({len(file_bytes)} bytes)")
        text, format_type = DocumentParser.parse_bytes(file_bytes, filename)
        
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="No text could be extracted from file")
        
        logger.info(f"Extracted {len(text)} characters from {format_type.upper()} file")
        
        # Classify document
        result = await classification_service.classify_text(text, include_context)
        
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
        return await classification_service.classify_text(request.text, request.include_context)
    except Exception as e:
        logger.error(f"Text classification error: {e}", exc_info=True)
        logger.error(f"Error: {e}", exc_info=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


def main():
    """Run the FastAPI application."""
    parser = argparse.ArgumentParser(description="Military Document Classification API")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on file changes")
    args = parser.parse_args()
    
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info("\nStarting server...")
    logger.info(f"Database: {Config.DB_NAME}@{Config.DB_HOST}:{Config.DB_PORT}")
    logger.info(f"LLM: {Config.LLM_PROVIDER}/{Config.LLM_MODEL}")
    logger.info(f"Debug mode: {debug_mode}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Auto-reload: {args.reload}\n")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=args.port,
        reload=args.reload,
        log_level="debug" if debug_mode else "info"
    )


if __name__ == "__main__":
    main()
