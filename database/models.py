"""Database models and operations."""

import uuid
from datetime import datetime
from typing import List, Optional


class DocumentModel:
    """Handle document-related database operations."""
    
    @staticmethod
    def insert_document(cursor, title: str, text_content: str, 
                       classification_label: str, source_unit: Optional[str] = None):
        """Insert a new document and return its ID."""
        document_id = str(uuid.uuid4())
        
        # Insert into documents table
        cursor.execute("""
            INSERT INTO documents (document_id, title, source_format, raw_file_path, 
                                  source_unit, status, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (document_id, title, 'text', f'/data/{document_id}.txt', 
              source_unit, 'processed', datetime.now()))
        
        # Insert text content as a single page
        cursor.execute("""
            INSERT INTO document_pages (document_id, page_number, text_content, ocr_confidence)
            VALUES (%s, %s, %s, %s)
        """, (document_id, 1, text_content, 1.0))
        
        # Insert classification
        cursor.execute("""
            SELECT class_id FROM document_classes WHERE class_name = %s
        """, (classification_label,))
        
        result = cursor.fetchone()
        if result:
            class_id = result['class_id']
        else:
            # Create new class if it doesn't exist
            cursor.execute("""
                INSERT INTO document_classes (class_name, description)
                VALUES (%s, %s)
                RETURNING class_id
            """, (classification_label, f'Classification level: {classification_label}'))
            class_id = cursor.fetchone()['class_id']
        
        cursor.execute("""
            INSERT INTO document_classification (document_id, class_id, confidence, 
                                                classified_by, classified_at)
            VALUES (%s, %s, %s, %s, %s)
        """, (document_id, class_id, 1.0, 'training_data', datetime.now()))
        
        return document_id
    
    @staticmethod
    def insert_embedding(cursor, document_id: str, text_content: str, 
                        embedding: List[float], chunk_index: int = 0):
        """Insert document embedding."""
        cursor.execute("""
            INSERT INTO document_embeddings (document_id, page_number, chunk_index, 
                                            text_content, embedding, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (document_id, 1, chunk_index, text_content, embedding, datetime.now()))
    
    @staticmethod
    def insert_chunks(cursor, document_id: str, chunks: List, embeddings: list,
                     strategy: str, max_chunk_size: int, overlap_size: int,
                     embedding_model: str):
        """Insert document chunks with embeddings.
        
        Args:
            cursor: Database cursor
            document_id: Document UUID
            chunks: List of Chunk objects
            embeddings: List of embedding vectors
            strategy: Chunking strategy used
            max_chunk_size: Max chunk size parameter
            overlap_size: Overlap size parameter
            embedding_model: Name of embedding model
        """
        # Insert chunking configuration
        cursor.execute("""
            INSERT INTO chunking_configurations 
            (document_id, strategy, max_chunk_size, overlap_size, total_chunks)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (document_id, strategy) DO UPDATE SET
                max_chunk_size = EXCLUDED.max_chunk_size,
                overlap_size = EXCLUDED.overlap_size,
                total_chunks = EXCLUDED.total_chunks
        """, (document_id, strategy, max_chunk_size, overlap_size, len(chunks)))
        
        chunk_ids = []
        
        # Insert each chunk
        for chunk, embedding in zip(chunks, embeddings):
            cursor.execute("""
                INSERT INTO document_embeddings 
                (document_id, page_number, chunk_index, chunk_strategy, 
                 chunk_size, overlap_size, start_position, end_position,
                 text_content, token_count, embedding, embedding_model)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING embedding_id
            """, (
                document_id, chunk.page_number, chunk.chunk_index, strategy,
                max_chunk_size, overlap_size, chunk.start_position, chunk.end_position,
                chunk.text, chunk.token_count, embedding, embedding_model
            ))
            
            chunk_id = cursor.fetchone()['embedding_id']
            chunk_ids.append(chunk_id)
        
        # Insert overlap relationships
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # Calculate overlap
            overlap_chars = max(0, current_chunk.end_position - next_chunk.start_position)
            
            if overlap_chars > 0:
                cursor.execute("""
                    INSERT INTO chunk_relationships 
                    (source_chunk_id, target_chunk_id, relationship_type, overlap_chars)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, (chunk_ids[i], chunk_ids[i + 1], 'overlaps', overlap_chars))
                
                # Also add adjacent relationship
                cursor.execute("""
                    INSERT INTO chunk_relationships 
                    (source_chunk_id, target_chunk_id, relationship_type, overlap_chars)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, (chunk_ids[i], chunk_ids[i + 1], 'adjacent', 0))
    
    @staticmethod
    def search_similar_documents(cursor, query_embedding: List[float], top_k: int = 5):
        """Search for similar documents using vector similarity."""
        cursor.execute("""
            SELECT 
                de.document_id,
                de.text_content,
                d.title,
                dc.class_name as classification,
                dclass.confidence,
                (de.embedding <-> %s::vector) as distance
            FROM document_embeddings de
            JOIN documents d ON de.document_id = d.document_id
            JOIN document_classification dclass ON d.document_id = dclass.document_id
            JOIN document_classes dc ON dclass.class_id = dc.class_id
            ORDER BY de.embedding <-> %s::vector
            LIMIT %s
        """, (query_embedding, query_embedding, top_k))
        
        return cursor.fetchall()
    
    @staticmethod
    def get_classification_rules(cursor):
        """Get all classification rules/levels."""
        cursor.execute("""
            SELECT class_name, description FROM document_classes
            ORDER BY class_id
        """)
        return cursor.fetchall()


class ClassificationModel:
    """Handle classification results."""
    
    @staticmethod
    def save_classification_result(cursor, document_id: str, predicted_class: str,
                                   confidence: float, classified_by: str):
        """Save classification result for a document."""
        # Get class_id
        cursor.execute("""
            SELECT class_id FROM document_classes WHERE class_name = %s
        """, (predicted_class,))
        
        result = cursor.fetchone()
        if not result:
            # Create new class if it doesn't exist
            cursor.execute("""
                INSERT INTO document_classes (class_name)
                VALUES (%s)
                RETURNING class_id
            """, (predicted_class,))
            class_id = cursor.fetchone()['class_id']
        else:
            class_id = result['class_id']
        
        # Update or insert classification
        cursor.execute("""
            INSERT INTO document_classification (document_id, class_id, confidence, 
                                                classified_by, classified_at)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (document_id) 
            DO UPDATE SET 
                class_id = EXCLUDED.class_id,
                confidence = EXCLUDED.confidence,
                classified_by = EXCLUDED.classified_by,
                classified_at = EXCLUDED.classified_at
        """, (document_id, class_id, confidence, classified_by, datetime.now()))
