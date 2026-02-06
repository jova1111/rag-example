-- =====================================================
-- PostgreSQL Schema for Military Document Classification
-- Includes pgvector for embeddings (RAG)
-- =====================================================

-- 1️⃣ Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- =====================================================
-- 2️⃣ Documents table (raw file metadata)
-- =====================================================
CREATE TABLE documents (
    document_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title              TEXT,
    source_format      VARCHAR(20) NOT NULL,  -- pdf, docx, image
    raw_file_path      TEXT NOT NULL,
    language           VARCHAR(10),
    created_at         TIMESTAMP,
    ingested_at        TIMESTAMP DEFAULT NOW(),
    source_unit        TEXT,
    status             VARCHAR(20) DEFAULT 'processed'  -- processed, pending, review
);

-- =====================================================
-- 3️⃣ Document pages (extracted text, optional OCR confidence)
-- =====================================================
CREATE TABLE document_pages (
    page_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id    UUID REFERENCES documents(document_id) ON DELETE CASCADE,
    page_number    INT NOT NULL,
    text_content   TEXT,
    ocr_confidence FLOAT,
    UNIQUE (document_id, page_number)
);

-- =====================================================
-- 4️⃣ Document embeddings table (pgvector) - WITH CHUNKING SUPPORT
-- =====================================================
CREATE TABLE document_embeddings (
    embedding_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id    UUID REFERENCES documents(document_id) ON DELETE CASCADE,
    page_number    INT,
    chunk_index    INT NOT NULL,  -- Sequential chunk number within document
    
    -- Chunking metadata
    chunk_strategy VARCHAR(50),           -- 'semantic', 'token', 'hybrid'
    chunk_size     INT,                   -- Target size (tokens or chars)
    overlap_size   INT DEFAULT 0,         -- Overlap with previous chunk
    
    -- Chunk position tracking
    start_position INT,                   -- Starting character position in full text
    end_position   INT,                   -- Ending character position
    
    -- Chunk content
    text_content   TEXT NOT NULL,
    token_count    INT,                   -- Actual token count of chunk
    
    -- Vector embedding
    embedding      vector(768),          -- Adjust dimension to your embedding model
    
    -- Metadata
    created_at     TIMESTAMP DEFAULT NOW(),
    embedding_model VARCHAR(100),         -- Track which model created embedding
    
    UNIQUE (document_id, chunk_index)
);

-- Index for fast similarity search using IVF (approximate nearest neighbors)
CREATE INDEX idx_document_embeddings_vector 
ON document_embeddings USING ivfflat (embedding vector_l2_ops) 
WITH (lists = 100);

-- Additional indexes for chunk retrieval
CREATE INDEX idx_document_embeddings_doc_chunk ON document_embeddings(document_id, chunk_index);
CREATE INDEX idx_document_embeddings_strategy ON document_embeddings(chunk_strategy);


-- =====================================================
-- 4️⃣.2 Chunking configurations (track chunking strategies)
-- =====================================================
CREATE TABLE chunking_configurations (
    config_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id    UUID REFERENCES documents(document_id) ON DELETE CASCADE,
    strategy       VARCHAR(50) NOT NULL,  -- 'semantic', 'token', 'hybrid'
    
    -- Strategy parameters
    max_chunk_size INT NOT NULL,           -- Max tokens/chars per chunk
    overlap_size   INT DEFAULT 0,          -- Overlap between chunks
    min_chunk_size INT DEFAULT 100,        -- Minimum chunk size
    
    -- Semantic chunking parameters (if applicable)
    similarity_threshold FLOAT,            -- For semantic boundary detection
    use_sentence_boundaries BOOLEAN DEFAULT true,
    
    -- Token-based parameters
    tokenizer      VARCHAR(100),           -- e.g., 'cl100k_base', 'gpt2'
    
    -- Metadata
    created_at     TIMESTAMP DEFAULT NOW(),
    total_chunks   INT,                    -- Total chunks created
    
    UNIQUE (document_id, strategy)
);

CREATE INDEX idx_chunking_config_document ON chunking_configurations(document_id);

-- =====================================================
-- 5️⃣ Tags table (predefined document tags)
-- =====================================================
CREATE TABLE tags (
    tag_id      SERIAL PRIMARY KEY,
    tag_name    VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    category    VARCHAR(50),  -- e.g., 'Operations', 'Logistics', 'Intelligence'
    created_at  TIMESTAMP DEFAULT NOW()
);

-- =====================================================
-- 6️⃣ Chunk-Tags relationship (many-to-many)
-- =====================================================
CREATE TABLE chunk_tags (
    chunk_tag_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    embedding_id  UUID REFERENCES document_embeddings(embedding_id) ON DELETE CASCADE,
    tag_id        INT REFERENCES tags(tag_id) ON DELETE CASCADE,
    confidence    FLOAT,  -- Confidence score for tag assignment
    assigned_by   VARCHAR(50),  -- 'RAG', 'manual', or model name
    assigned_at   TIMESTAMP DEFAULT NOW(),
    
    UNIQUE (embedding_id, tag_id)
);

-- =====================================================
-- 7️⃣ Document classes (kept for later use)
-- =====================================================
CREATE TABLE document_classes (
    class_id    SERIAL PRIMARY KEY,
    class_name  VARCHAR(50) UNIQUE NOT NULL,
    description TEXT
);

-- =====================================================
-- 8️⃣ Document classification results (kept for later use)
-- =====================================================
CREATE TABLE document_classification (
    document_id   UUID PRIMARY KEY REFERENCES documents(document_id),
    class_id      INT REFERENCES document_classes(class_id),
    confidence    FLOAT,
    classified_by VARCHAR(50),  -- model name or analyst
    classified_at TIMESTAMP DEFAULT NOW()
);

-- =====================================================
-- ✅ Additional indexes for faster querying
-- =====================================================
CREATE INDEX idx_documents_source_unit ON documents(source_unit);
CREATE INDEX idx_document_pages_document_id_page_number ON document_pages(document_id, page_number);
CREATE INDEX idx_document_embeddings_document_id ON document_embeddings(document_id);
CREATE INDEX idx_chunk_tags_embedding_id ON chunk_tags(embedding_id);
CREATE INDEX idx_chunk_tags_tag_id ON chunk_tags(tag_id);
CREATE INDEX idx_tags_category ON tags(category);
CREATE INDEX idx_document_classification_class_id ON document_classification(class_id);

-- =====================================================
-- ✅ Schema ready for:
--    - Tag-based document classification
--    - Many-to-many chunk-tag relationships
--    - RAG + Embedding-based retrieval
--    - Semantic, token-based, and hybrid chunking
--    - Chunk overlap and relationship tracking
-- =====================================================
