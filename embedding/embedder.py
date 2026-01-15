"""Document embedding using OpenAI or sentence transformers."""

from sentence_transformers import SentenceTransformer
import numpy as np
from config import Config


class DocumentEmbedder:
    """Generate embeddings for documents using OpenAI or sentence transformers."""
    
    def __init__(self, model_name: str = None):
        """Initialize the embedding model.
        
        Args:
            model_name: Name of the embedding model.
        """
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.use_openai = self.model_name.startswith("text-embedding")
        
        if self.use_openai:
            print(f"Loading OpenAI embedding model: {self.model_name}")
            from openai import OpenAI
            self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
            self.dimension = Config.EMBEDDING_DIMENSION
            print(f"✓ OpenAI embedding model ready (dimension: {self.dimension})")
        else:
            print(f"Loading embedding model: {self.model_name}")
            print("(First run will download the model, then it works offline)")
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            print(f"✓ Embedding model loaded (dimension: {self.dimension})")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.
        
        Args:
            text: Input text to embed.
            
        Returns:
            Embedding vector as numpy array.
        """
        if self.use_openai:
            response = self.client.embeddings.create(
                input=text,
                model=self.model_name
            )
            embedding = np.array(response.data[0].embedding)
        else:
            embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_batch(self, texts: list) -> np.ndarray:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            Array of embedding vectors.
        """
        if self.use_openai:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model_name
            )
            embeddings = np.array([item.embedding for item in response.data])
        else:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings
    
    def chunk_text(self, text: str, max_length: int = 500) -> list:
        """Split text into chunks for embedding.
        
        Args:
            text: Input text to chunk.
            max_length: Maximum number of characters per chunk.
            
        Returns:
            List of text chunks.
        """
        # Simple chunking by character count
        # For production, consider sentence-aware chunking
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > max_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [text]
