"""Document chunking strategies for embeddings."""

from typing import List
from dataclasses import dataclass

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    chunk_index: int
    start_position: int
    end_position: int
    token_count: int
    page_number: int = 1


class DocumentChunker:
    """Chunk documents using various strategies with LangChain."""
    
    def __init__(self, strategy: str = 'hybrid', max_chunk_size: int = 512, 
                 overlap_size: int = 50, openai_api_key: str = None):
        """Initialize chunker.
        
        Args:
            strategy: 'semantic', 'token', or 'hybrid'
            max_chunk_size: Maximum tokens per chunk
            overlap_size: Number of tokens to overlap
            openai_api_key: OpenAI API key for semantic chunking
        """
        self.strategy = strategy
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.openai_api_key = openai_api_key
        
        # Initialize the appropriate splitter
        self._init_splitter()
    
    def _init_splitter(self):
        """Initialize the text splitter based on strategy."""
        if self.strategy == 'semantic':
            if not self.openai_api_key:
                raise ValueError("OpenAI API key required for semantic chunking")
            embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            self.splitter = SemanticChunker(embeddings)
        
        elif self.strategy == 'token':
            self.splitter = TokenTextSplitter(
                chunk_size=self.max_chunk_size,
                chunk_overlap=self.overlap_size
            )
        
        elif self.strategy == 'hybrid':
            # Use RecursiveCharacterTextSplitter for hybrid approach
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.max_chunk_size,
                chunk_overlap=self.overlap_size,
                length_function=len,
                is_separator_regex=False
            )
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def chunk_document(self, text: str, page_number: int = 1) -> List[Chunk]:
        """Chunk a document based on the configured strategy.
        
        Args:
            text: Document text to chunk
            page_number: Page number for tracking
            
        Returns:
            List of Chunk objects
        """
        # Use LangChain splitter to create chunks
        text_chunks = self.splitter.split_text(text)
        
        # Convert to Chunk objects with metadata
        chunks = []
        current_pos = 0
        
        for idx, chunk_text in enumerate(text_chunks):
            # Find the position in the original text
            start_pos = text.find(chunk_text, current_pos)
            if start_pos == -1:
                # Fallback if exact match not found
                start_pos = current_pos
            
            end_pos = start_pos + len(chunk_text)
            
            # Estimate token count (simple word count)
            token_count = len(chunk_text.split())
            
            chunks.append(Chunk(
                text=chunk_text,
                chunk_index=idx,
                start_position=start_pos,
                end_position=end_pos,
                token_count=token_count,
                page_number=page_number
            ))
            
            current_pos = end_pos
        
        return chunks
