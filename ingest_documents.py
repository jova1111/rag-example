"""Ingest training documents into the database."""

import json
import sys
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
    
    # Initialize embedder
    print("\n📊 Initializing embedding model...")
    embedder = DocumentEmbedder()
    
    # Connect to database
    try:
        with DatabaseConnection() as db:
            conn = db.conn
            cursor = db.cursor
            
            print(f"\n📥 Ingesting documents...")
            ingested_count = 0
            
            for idx, doc in enumerate(training_data, 1):
                title = doc.get('title', f'Document {idx}')
                text = doc.get('text', '')
                classification = doc.get('classification', 'UNCLASSIFIED')
                source_unit = doc.get('source_unit', None)
                
                if not text:
                    print(f"  ⚠️  Skipping document {idx}: No text content")
                    continue
                
                try:
                    # Insert document
                    document_id = DocumentModel.insert_document(
                        cursor, title, text, classification, source_unit
                    )
                    
                    # Generate and insert embedding
                    embedding = embedder.embed_text(text)
                    DocumentModel.insert_embedding(
                        cursor, document_id, text, embedding.tolist()
                    )
                    
                    conn.commit()
                    ingested_count += 1
                    print(f"  ✓ [{ingested_count}/{len(training_data)}] {title} ({classification})")
                    
                except Exception as e:
                    print(f"  ✗ Error ingesting '{title}': {e}")
                    conn.rollback()
                    continue
            
            # Verify ingestion
            cursor.execute("SELECT COUNT(*) as count FROM documents")
            total_docs = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM document_embeddings")
            total_embeddings = cursor.fetchone()['count']
            
            print(f"\n✓ Ingestion complete!")
            print(f"  - Documents in database: {total_docs}")
            print(f"  - Embeddings in database: {total_embeddings}")
            print(f"  - Successfully ingested: {ingested_count}/{len(training_data)}")
            
            return ingested_count > 0
            
    except Exception as e:
        print(f"\n✗ Ingestion failed: {e}")
        return False


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
