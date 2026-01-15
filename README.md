# Military Document Classification - RAG System

A Python REST API that classifies military documents using **Retrieval-Augmented Generation (RAG)** with PostgreSQL (pgvector) and LLM.

## 🎯 Features

- **REST API**: FastAPI endpoints for file upload and classification
- **Multiple File Formats**: Supports TXT, PDF, and Word (DOCX) documents
- **RAG-based Classification**: Uses similar documents to guide classification
- **Vector Database**: PostgreSQL with pgvector for semantic search
- **Offline Capability**: Works offline after initial setup (embedding model cached locally)
- **Confidence Scoring**: Returns confidence scores and requires human review for low confidence
- **Strict Rule Adherence**: LLM cannot invent new classification rules
- **Comprehensive Logging**: Detailed justifications referencing retrieved documents

## 📋 Classification Levels

1. **UNCLASSIFIED** - General information, no sensitive content
2. **CONFIDENTIAL** - Could cause damage if disclosed
3. **SECRET** - Could cause serious damage if disclosed
4. **TOP SECRET** - Could cause exceptionally grave damage

## 🏗️ Project Structure

```
RAG.TestClassification/
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment configuration template
├── config.py                        # Configuration management
├── app.py                          # FastAPI application (main entry point)
├── ingest_documents.py             # Load training documents
├── test_classification.py          # API endpoint testing script
│
├── database/                        # Database layer
│   ├── __init__.py
│   ├── connection.py               # PostgreSQL connection management
│   ├── models.py                   # Database models and operations
|   ├── classification_relational.sql    # Database schema (PostgreSQL + pgvector)
│
├── embedding/                       # Document embedding
│   ├── __init__.py
│   └── embedder.py                 # Sentence transformer embeddings
│
├── rag/                            # RAG components
│   ├── __init__.py
│   ├── retriever.py                # Vector similarity search
│   └── classifier.py               # LLM-based classification (OpenAI/Ollama)
│
├── utils/                          # Utilities
│   ├── __init__.py
│   └── document_parser.py          # TXT, PDF, DOCX parser
│
└── data/                           # Training data
    └── sample_documents.json       # 33 comprehensive military documents
```

## 🚀 Quick Start (API)

1. **Ingest training data**: `python ingest_documents.py`
2. **Start API server**: `python app.py`
3. **Access API docs**: http://localhost:8000/docs
4. **Classify documents**: Upload files via API

## 📋 Installation & Setup

### Prerequisites

- Python 3.8+
- PostgreSQL 14+ with pgvector extension
- OpenAI API key (or configure local LLM)

### Step 1: Install PostgreSQL with pgvector

**Windows:**
```powershell
# Install PostgreSQL (if not already installed)
# Download from: https://www.postgresql.org/download/windows/

# Install pgvector extension
# Follow instructions at: https://github.com/pgvector/pgvector#windows
```

**Alternative (Docker):**
```powershell
docker run --name postgres-pgvector -e POSTGRES_PASSWORD=mypassword -p 5432:5432 -d ankane/pgvector
```

### Step 2: Create Database

Run classification_relational.sql script on a postgres server.

### Step 3: Install Python Dependencies

```powershell
# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Configure Environment

```powershell
# Copy example config
cp .env.example .env

# Edit .env with your settings
notepad .env
```

Required settings in `.env`:
```ini
DB_HOST=localhost
DB_PORT=5432
DB_NAME=military_classification
DB_USER=postgres
DB_PASSWORD=your_password_here

OPENAI_API_KEY=sk-your-key-here
LLM_PROVIDER=openai
LLM_MODEL=gpt-4

EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
```

### Step 5: Initialize Database

```powershell
python setup_db.py
```

Expected output:
```
✓ Database schema created successfully
✓ Inserted 4 classification levels
✓ Database setup complete!
```

### Step 6: Ingest Training Documents

```powershell
python ingest_documents.py
```
API Usage

### Start the API Server

```powershell
python app.py
```

Server will start at: **http://localhost:8000**

Interactive API documentation: **http://localhost:8000/docs**

### API Endpoints

#### 1. Health Check
```powershell
curl http://localhost:8000/health
```

#### 2. Classify File Upload (TXT, PDF, DOCX)
```powershell
# Upload a file for classification
curl -X POST "http://localhost:8000/classify/file" \
  -F "file=@document.pdf" \
  -F "include_context=true"
```

**PowerShell:**
```powershell
$file = "C:\path\to\document.pdf"
$uri = "http://localhost:8000/classify/file"
Invoke-RestMethod -Uri $uri -Method Post -Form @{file=Get-Item $file; include_context="false"}
```

#### 3. Classify Raw Text
```powershell
curl -X POST "http://localhost:8000/classify/text" \
  -H "Content-Type: application/json" \
  -d '{"text": "This report contains operational details...", "include_context": false}'
```

**PowerShell:**
```powershell
$body = @{
    text = "This report contains operational details about troop movements and coordination."
    include_context = $false
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/classify/text" -Method Post -Body $body -ContentType "application/json"
```

### Example API Response

```json
{
  "classification": "SECRET",
  "confidence": 0.87,
  "justification": "This document describes detailed operational plans including troop movements and air support, which aligns with retrieved Document 'Operation Thunder Strike Plan' classified as SECRET. The content could cause serious damage if disclosed.",
  "document_length": 156,
  "retrieved_documents_count": 5,
  "retrieved_documents": [
    {
      "title": "Operation Thunder Strike Plan",
      "classification": "SECRET",
      "similarity": 0.782,
      "confidence": 0.95
    }
  ]
}
```

### Supported File Formats

- **TXT** - Plain text files
- **PDF** - Portable Document Format
- **DOCX** - Microsoft Word documents

### Command Line Tool (Optional)

The original CLI tool is still available:

```powershell
python classify_document.py "Document text here"
python classify_document.py --file path/to/document.txt
💬 Justification:
   This document describes detailed operational plans including
   troop movements and air support, which aligns with Document 1
   (Operation Thunder Strike Plan) classified as SECRET. The
   content could cause serious damage if disclosed, matching
   the SECRET classification criteria.

============================================================
```

## 🔧 How It Works

1. **Document Ingestion**
   - Training documents are embedded using Sentence Transformers
   - Embeddings stored in PostgreSQL with pgvector
   - Classification labels stored with each document

2. **Classification Process**
   ```
   New Document → Embed → Vector Search → Top 5 Similar Docs
                                           ↓
   LLM ← Classification Rules + Retrieved Context + New Doc
                                           ↓
   Classification + Confidence + Justification
   ```

3. **RAG Components**
   - **Retriever**: Finds semantically similar documents using cosine similarity
   - **Classifier**: LLM analyzes document based on:
     - Predefined classification rules
     - Retrieved similar documents as examples
     - Cannot invent new rules

4. **Confidence Threshold**
   - If confidence < 0.7 (configurable), returns "REQUIRES HUMAN REVIEW"
   - Ensures high-stakes decisions get human oversight

## ⚙️ Configuration Options

Edit `config.py` or `.env`:

| Setting | Description | Default |
|---------|-------------|---------|
| `EMBEDDING_ the API

### Using the Interactive Documentation

1. Navigate to http://localhost:8000/docs
2. Expand any endpoint (e.g., `/classify/file`)
3. Click "Try it out"
4. Upload a file or enter text
5. Click "Execute"

### Using PowerShell

Create test files:

```powershell
# Create test documents
"Weekly training schedule for the platoon." | Out-File -Encoding UTF8 test_unclassified.txt
"Our unit will deploy to FOB Delta on March 15th with 120 personnel." | Out-File -Encoding UTF8 test_confidential.txt
"Operational plan includes air support coordination and pre-planned artillery strikes." | Out-File -Encoding UTF8 test_secret.txt

# Test classification
Invoke-RestMethod -Uri "http://localhost:8000/classify/file" -Method Post -Form @{file=Get-Item "test_unclassified.txt"}
```

### Using curl

```bash
# Test with different classification levels
curl -X POST "http://localhost:8000/classify/text" \
  -H "Content-Type: application/json" \
  -d '{"text": "Weekly training schedule for the platoon.", "include_context": false}'

curl -X POST "http://localhost:8000/classify/file" \
  -F "file=@test_document.pdf
   - **OpenAI**: Requires internet (API calls)
   - **Local LLM** (future): Uncomment in `requirements.txt`:
     ```
     transformers==4.36.0
     torch==2.1.0
     ```

## 📊 Database Schema

Based on `classification_relational.sql`:

- **documents**: Document metadata
- **document_pages**: Extracted text content
- **document_embeddings**: Vector embeddings (pgvector)
- **document_classes**: Classification levels
- **document_classification**: Classification results

## 🧪 Testing

Test with different classification levels:

```powershell
# UNCLASSIFIED
python classify_document.py "Weekly training schedule for the platoon."

# C

### Custom File Upload Limits

Edit `app.py` to adjust file size limits:
```python
# Add to app configuration
app = FastAPI(
    ...
)
app.add_middleware(
    # Add file size limit middleware
)ONFIDENTIAL
python classify_document.py "Our unit will deploy to FOB Delta on March 15th with 120 personnel."

# SECRET
python classify_document.py "Operational plan includes air support coordination and pre-planned artillery strikes on designated targets."

# TOP SECRET
python classify_document.py "Intelligence from classified HUMINT source reveals enemy command structure and requires compartmented access."
```

## 🛠️ Troubleshooting

**Database connection error:**
```powershell
# Verify PostgreSQL is running
psql -U postgres -c "SELECT version();"

# Check pgvector extension
psql -U postgres -d military_classification -c "SELECT * FROM pg_extension WHERE extname='vector';"
```

**Embedding model download:**
```powershell
# First run downloads model (~90MB)
# If slow, check internet connection
# Model cached at: %USERPROFILE%\.cache\torch\sentence_transformers\
```

**OpenAI API errors:**
```powershell
# Verify API key in .env
# Check API quota and billing
```

## 📈 Extending the System

### Add More Training Documents

Edit `data/sample_documents.json`:
```json
{
  "title": "Your Document Title",
  "text": "Document content...",
  "classification": "CONFIDENTIAL",
  "source_unit": "Your Unit"
}
```

Then re-run:
```powershell
python ingest_documents.py
```

### Use Local LLM with Ollama (Fully Offline)

**Option 1: Using Ollama (Recommended for local deployment)**

1. Install Ollama:
   - Download from https://ollama.ai/download
   - Or use: `winget install Ollama.Ollama` (Windows)

2. Pull a model:
   ```powershell
   ollama pull llama3:8b
   # or other models: mistral, llama2, codellama, etc.
   ```

3. Update `.env`:
   ```ini
   LLM_PROVIDER=local
   LLM_MODEL=llama3:8b
   ```

4. Install the Ollama Python SDK:
   ```powershell
   pip install ollama
   ```

5. Start Ollama (runs automatically on installation, or run `ollama serve`)

6. Restart the API server:
   ```powershell
   python app.py
   ```

**Option 2: Using Transformers (Advanced)**

1. Uncomment in `requirements.txt`:
   ```
   transformers==4.36.0
   torch==2.1.0
   ```

2. Update `.env`:
## 🌐 Deployment

### Run in Production

```powershell
# Install production server
pip install gunicorn

# Run with gunicorn (Linux/Mac)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:8000

# Run with uvicorn (Windows)
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment (Optional)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

**Built with**: Python • FastAPI/Mistral-7B-Instruct-v0.2
   ```

3. Implement in `rag/classifier.py` (see TODOs in code)

## 📝 Notes

- **Important**: This is a demonstration system. For production use:
  - Implement proper access controls
  - Add audit logging
  - Use secure credential management
  - Consider data encryption at rest
  - Implement rate limiting for API calls
  
- **PostgreSQL Note**: The original requirement mentioned MongoDB, but the provided SQL schema uses PostgreSQL with pgvector, which is more suitable for vector operations.

## 📄 License

Educational/demonstration purposes only.

## 🤝 Support

For issues or questions:
1. Check database connection and configuration
2. Verify all dependencies installed
3. Review logs for specific errors
4. Ensure training documents are ingested before classification

---

**Built with**: Python • PostgreSQL • pgvector • Sentence Transformers • OpenAI GPT-4
