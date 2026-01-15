"""Document retriever for RAG system."""

from typing import List, Dict
from embedding.embedder import DocumentEmbedder
from database.models import DocumentModel


class DocumentRetriever:
    """Retrieve similar documents from vector database."""
    
    def __init__(self, cursor, embedder: DocumentEmbedder):
        """Initialize retriever.
        
        Args:
            cursor: Database cursor.
            embedder: Document embedder instance.
        """
        self.cursor = cursor
        self.embedder = embedder
    
    def retrieve_similar(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Retrieve most similar documents to the query.
        
        Args:
            query_text: Text to find similar documents for.
            top_k: Number of similar documents to retrieve.
            
        Returns:
            List of similar documents with metadata.
        """
        # Generate embedding for query
        query_embedding = self.embedder.embed_text(query_text)
        
        # Search in vector database
        similar_docs = DocumentModel.search_similar_documents(
            self.cursor, 
            query_embedding.tolist(), 
            top_k
        )
        
        return similar_docs
    
    def format_context(self, retrieved_docs: List[Dict]) -> str:
        """Format retrieved documents into context string for LLM.
        
        Args:
            retrieved_docs: List of retrieved documents.
            
        Returns:
            Formatted context string.
        """
        if not retrieved_docs:
            return "No similar documents found."
        
        context_parts = []
        for idx, doc in enumerate(retrieved_docs, 1):
            title = doc.get('title', 'Unknown')
            classification = doc.get('classification', 'Unknown')
            text = doc.get('text_content', '')
            distance = doc.get('distance', 0)
            confidence = doc.get('confidence', 0)
            
            # Truncate text if too long
            if len(text) > 300:
                text = text[:300] + "..."
            
            context_parts.append(
                f"[Document {idx}]\n"
                f"Title: {title}\n"
                f"Classification: {classification}\n"
                f"Confidence: {confidence:.2f}\n"
                f"Similarity: {1 - distance:.3f}\n"
                f"Content: {text}\n"
            )
        
        return "\n".join(context_parts)
