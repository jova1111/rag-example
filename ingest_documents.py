"""Ingest training documents into the database."""

import json
import sys
import uuid
from datetime import datetime
from typing import List, Optional
from database.connection import DatabaseConnection
from database.models import DocumentModel
from embedding.embedder import DocumentEmbedder
from config import Config


def ingest_documents(data_file: str = 'data/sample_documents.json'):
    """Ingest documents from JSON file into database.
    
    Args:
        data_file: Path to JSON file with training documents.
    """
    print("=" * 60)
    print("Ingesting Training Documents")
    print("=" * 60)
    
    # Load training data
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        print(f"\n✓ Loaded {len(training_data)} documents from {data_file}")
    except FileNotFoundError:
        print(f"✗ Error: {data_file} not found")
        print("  Run this script from the project root directory")
        return False
    except json.JSONDecodeError as e:
        print(f"✗ Error parsing JSON: {e}")
        return False
    
    # Initialize embedder (no chunker needed - chunks come from JSON)
    print("\n📊 Initializing embedding model...")
    embedder = DocumentEmbedder()
    print(f"  Embedding model: {Config.EMBEDDING_MODEL}")
    
    # Connect to database
    try:
        with DatabaseConnection() as db:
            conn = db.conn
            cursor = db.cursor
            
            print(f"\n📥 Ingesting documents...")
            ingested_count = 0
            total_chunks = 0
            total_tags = 0
            
            for idx, doc in enumerate(training_data, 1):
                title = doc.get('title', f'Document {idx}')
                source_unit = doc.get('source_unit', None)
                chunks = doc.get('chunks', [])
                
                if not chunks:
                    print(f"  ⚠️  Skipping document {idx}: No chunks")
                    continue
                
                try:
                    # Insert document (no classification in new structure)
                    document_id = _insert_document_only(
                        cursor, title, source_unit
                    )
                    
                    print(f"  📄 Processing {len(chunks)} chunks for '{title}'")
                    
                    # Process each chunk
                    chunk_texts = [chunk['text'] for chunk in chunks]
                    embeddings = embedder.embed_batch(chunk_texts)
                    
                    # Insert chunks with embeddings and tags
                    chunk_count = 0
                    for chunk_idx, (chunk_data, embedding) in enumerate(zip(chunks, embeddings.tolist())):
                        # Insert chunk embedding
                        embedding_id = _insert_chunk_embedding(
                            cursor, document_id, chunk_idx, 
                            chunk_data['text'], embedding
                        )
                        
                        # Insert tags for this chunk
                        chunk_tags = chunk_data.get('tags', [])
                        for tag_name in chunk_tags:
                            _ensure_tag_and_link(cursor, embedding_id, tag_name)
                            total_tags += 1
                        
                        chunk_count += 1
                    
                    total_chunks += chunk_count
                    conn.commit()
                    ingested_count += 1
                    print(f"  ✓ [{ingested_count}/{len(training_data)}] {title} - {chunk_count} chunks, {len(chunk_tags) if chunks else 0} tags on last chunk")
                    
                except Exception as e:
                    print(f"  ✗ Error ingesting '{title}': {e}")
                    conn.rollback()
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Verify ingestion
            cursor.execute("SELECT COUNT(*) as count FROM documents")
            total_docs = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM document_embeddings")
            total_embeddings = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM tags")
            unique_tags = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM chunk_tags")
            total_chunk_tags = cursor.fetchone()['count']
            
            print(f"\n✓ Ingestion complete!")
            print(f"  - Documents in database: {total_docs}")
            print(f"  - Chunks/Embeddings in database: {total_embeddings}")
            print(f"  - Unique tags: {unique_tags}")
            print(f"  - Chunk-tag relationships: {total_chunk_tags}")
            print(f"  - Successfully ingested: {ingested_count}/{len(training_data)}")
            
            return ingested_count > 0
            
    except Exception as e:
        print(f"\n✗ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def _insert_document_only(cursor, title: str, source_unit: Optional[str] = None):
    """Insert a new document and return its ID."""
    document_id = str(uuid.uuid4())
    
    cursor.execute("""
        INSERT INTO documents (document_id, title, source_format, raw_file_path, 
                              source_unit, status, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (document_id, title, 'json', f'/data/{document_id}.json', 
          source_unit, 'processed', datetime.now()))
    
    return document_id


def _insert_chunk_embedding(cursor, document_id: str, chunk_index: int, 
                            text_content: str, embedding: List[float]):
    """Insert chunk embedding and return embedding_id."""
    cursor.execute("""
        INSERT INTO document_embeddings 
        (document_id, chunk_index, text_content, embedding, 
         chunk_strategy, embedding_model)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING embedding_id
    """, (document_id, chunk_index, text_content, embedding, 
          'predefined', Config.EMBEDDING_MODEL))
    
    result = cursor.fetchone()
    return result['embedding_id']


def _ensure_tag_and_link(cursor, embedding_id: str, tag_name: str):
    """Ensure tag exists and link it to chunk."""
    # Insert tag if it doesn't exist
    cursor.execute("""
        INSERT INTO tags (tag_name)
        VALUES (%s)
        ON CONFLICT (tag_name) DO NOTHING
    """, (tag_name,))
    
    # Get tag_id
    cursor.execute("""
        SELECT tag_id FROM tags WHERE tag_name = %s
    """, (tag_name,))
    
    tag_id = cursor.fetchone()['tag_id']
    
    # Link chunk to tag
    cursor.execute("""
        INSERT INTO chunk_tags (embedding_id, tag_id, confidence, assigned_by)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (embedding_id, tag_id) DO NOTHING
    """, (embedding_id, tag_id, 1.0, 'training_data'))


if __name__ == "__main__":
    data_file = sys.argv[1] if len(sys.argv) > 1 else 'data/sample_documents.json'
    
    print(f"\nData file: {data_file}")
    print(f"Embedding model: {Config.EMBEDDING_MODEL}\n")
    
    success = ingest_documents(data_file)
    
    if success:
        print("\n" + "=" * 60)
        print("✓ Documents ingested successfully!")
        print("=" * 60)
        print("\nNext step:")
        print("  Run: python classify_document.py \"Your document text here\"")
        sys.exit(0)
    else:
        print("\n✗ Ingestion failed. Please check the errors above.")
        sys.exit(1)
