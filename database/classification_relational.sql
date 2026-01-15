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
-- 4️⃣ Document embeddings table (pgvector)
-- =====================================================
CREATE TABLE document_embeddings (
    embedding_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id    UUID REFERENCES documents(document_id) ON DELETE CASCADE,
    page_number    INT,
    chunk_index    INT,
    text_content   TEXT,
    embedding      vector(1536),  -- Adjust dimension to your embedding model
    created_at     TIMESTAMP DEFAULT NOW()
);

-- Index for fast similarity search using IVF (approximate nearest neighbors)
CREATE INDEX idx_document_embeddings_vector 
ON document_embeddings USING ivfflat (embedding vector_l2_ops) 
WITH (lists = 100);

-- =====================================================
-- 5️⃣ Document classes
-- =====================================================
CREATE TABLE document_classes (
    class_id    SERIAL PRIMARY KEY,
    class_name  VARCHAR(50) UNIQUE NOT NULL,
    description TEXT
);

-- =====================================================
-- 6️⃣ Document classification results
-- =====================================================
CREATE TABLE document_classification (
    document_id   UUID PRIMARY KEY REFERENCES documents(document_id),
    class_id      INT REFERENCES document_classes(class_id),
    confidence    FLOAT,
    classified_by VARCHAR(50),  -- model name or analyst
    classified_at TIMESTAMP DEFAULT NOW()
);

-- =====================================================
-- 7️⃣ Confidentiality policies
-- =====================================================
CREATE TABLE confidentiality_policies (
    policy_id    VARCHAR(50) PRIMARY KEY,
    title        TEXT,
    description  TEXT,
    severity     VARCHAR(20)  -- confidential, secret, etc.
);

-- =====================================================
-- 8️⃣ Confidentiality assessment
-- =====================================================
CREATE TABLE confidentiality_assessment (
    document_id   UUID PRIMARY KEY REFERENCES documents(document_id),
    confidential  BOOLEAN NOT NULL,
    confidence    FLOAT,
    assessed_by   VARCHAR(50),  -- model or analyst
    assessed_at   TIMESTAMP DEFAULT NOW()
);

-- =====================================================
-- 9️⃣ Confidentiality triggers (why confidential)
-- =====================================================
CREATE TABLE confidentiality_triggers (
    trigger_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id    UUID REFERENCES documents(document_id) ON DELETE CASCADE,
    policy_id      VARCHAR(50) REFERENCES confidentiality_policies(policy_id),
    trigger_type   VARCHAR(50),  -- e.g., Operational Detail
    evidence_text  TEXT,
    page_number    INT
);

-- =====================================================
-- 10️⃣ Model runs / traceability
-- =====================================================
CREATE TABLE model_runs (
    run_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id   UUID REFERENCES documents(document_id),
    model_name    TEXT,
    model_version TEXT,
    prompt_hash   TEXT,
    run_time      TIMESTAMP DEFAULT NOW()
);

-- =====================================================
-- ✅ Optional indexes for faster querying
-- =====================================================
CREATE INDEX idx_documents_source_unit ON documents(source_unit);
CREATE INDEX idx_document_pages_document_id_page_number ON document_pages(document_id, page_number);
CREATE INDEX idx_document_embeddings_document_id ON document_embeddings(document_id);
CREATE INDEX idx_document_classification_class_id ON document_classification(class_id);

-- =====================================================
-- Script ready for RAG + Embedding-based retrieval
-- =====================================================
