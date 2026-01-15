"""FastAPI application for military document classification."""

import os
import traceback
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from database.connection import DatabaseConnection
from embedding.embedder import DocumentEmbedder
from rag.retriever import DocumentRetriever
from rag.classifier import DocumentClassifier
from utils.document_parser import DocumentParser
from config import Config


# Global instances (initialized on startup)
embedder = None
classifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global embedder, classifier
    
    print("=" * 60)
    print("Starting Military Document Classification API")
    print("=" * 60)
    
    try:
        print("\n📊 Loading embedding model...")
        embedder = DocumentEmbedder()
        
        print("🤖 Initializing classifier...")
        classifier = DocumentClassifier()
        
        print("\n✓ API ready!")
        print(f"  - Embedding model: {Config.EMBEDDING_MODEL}")
        print(f"  - LLM: {Config.LLM_PROVIDER}/{Config.LLM_MODEL}")
        print(f"  - Database: {Config.DB_NAME}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Startup failed: {e}")
        traceback.print_exc()
        raise
    
    yield
    
    # Shutdown logic (if needed in the future)
    print("\nShutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="Military Document Classification API",
    description="RAG-based document classification system using PostgreSQL pgvector",
    version="1.0.0",
    lifespan=lifespan
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
async def health_check():
    """Health check endpoint."""
    try:
        # Test database connection
        with DatabaseConnection() as db:
            cursor = db.cursor
            cursor.execute("SELECT COUNT(*) as count FROM documents")
            doc_count = cursor.fetchone()['count']
        
        return HealthResponse(
            status="healthy",
            database=f"connected ({doc_count} documents)",
            embedding_model=Config.EMBEDDING_MODEL,
            llm_model=f"{Config.LLM_PROVIDER}/{Config.LLM_MODEL}"
        )
    except Exception as e:
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
        print(f"\n📄 Processing: {file.filename} ({len(file_bytes)} bytes)")
        text, format_type = DocumentParser.parse_bytes(file_bytes, file.filename)
        
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="No text could be extracted from file")
        
        print(f"✓ Extracted {len(text)} characters from {format_type.upper()} file")
        
        # Classify document
        result = await _classify_text(text, include_context)
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.post("/classify/text", response_model=ClassificationResult, tags=["Classification"])
async def classify_text(request: TextClassificationRequest):
    """Classify a document from raw text.
    
    Args:
        request: Text classification request with text and options
        
    Returns:
        Classification result with confidence and justification
    """
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        return await _classify_text(request.text, request.include_context)
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


async def _classify_text(text: str, include_context: bool = False) -> ClassificationResult:
    """Internal function to classify text.
    
    Args:
        text: Document text to classify
        include_context: Include retrieved documents in response
        
    Returns:
        Classification result
    """
    print(f"🔍 Retrieving similar documents...")
    
    # Connect to database
    with DatabaseConnection() as db:
        cursor = db.cursor
        
        # Initialize retriever
        retriever = DocumentRetriever(cursor, embedder)
        
        # Retrieve similar documents
        similar_docs = retriever.retrieve_similar(text, top_k=Config.TOP_K_RETRIEVAL)
        
        if not similar_docs:
            raise HTTPException(
                status_code=500,
                detail="No similar documents found. Ensure training documents are ingested."
            )
        
        print(f"✓ Retrieved {len(similar_docs)} similar documents")
        
        # Format context for LLM
        context = retriever.format_context(similar_docs)
        
        # Classify
        print("🤖 Classifying with LLM...")
        classification_result = classifier.classify(text, context)
        
        print(f"✓ Classification: {classification_result['classification']} ({classification_result['confidence']:.2%})")
        
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
    print("\nStarting server...")
    print(f"Database: {Config.DB_NAME}@{Config.DB_HOST}:{Config.DB_PORT}")
    print(f"LLM: {Config.LLM_PROVIDER}/{Config.LLM_MODEL}\n")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
