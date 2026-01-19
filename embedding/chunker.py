"""Document chunking strategies for embeddings."""

import re
from typing import List
from dataclasses import dataclass


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
    """Chunk documents using various strategies."""
    
    def __init__(self, strategy: str = 'hybrid', max_chunk_size: int = 512, 
                 overlap_size: int = 50):
        """Initialize chunker.
        
        Args:
            strategy: 'semantic', 'token', or 'hybrid'
            max_chunk_size: Maximum tokens per chunk
            overlap_size: Number of tokens to overlap
        """
        self.strategy = strategy
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
    
    def chunk_document(self, text: str, page_number: int = 1) -> List[Chunk]:
        """Chunk a document based on the configured strategy.
        
        Args:
            text: Document text to chunk
            page_number: Page number for tracking
            
        Returns:
            List of Chunk objects
        """
        if self.strategy == 'semantic':
            return self._semantic_chunking(text, page_number)
        elif self.strategy == 'token':
            return self._token_based_chunking(text, page_number)
        elif self.strategy == 'hybrid':
            return self._hybrid_chunking(text, page_number)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _token_based_chunking(self, text: str, page_number: int) -> List[Chunk]:
        """Simple token-based chunking with overlap.
        
        Args:
            text: Text to chunk
            page_number: Page number
            
        Returns:
            List of chunks
        """
        # Simple word-based tokenization (for production, use tiktoken or similar)
        words = text.split()
        chunks = []
        chunk_index = 0
        
        i = 0
        while i < len(words):
            # Get chunk of max_chunk_size words
            chunk_words = words[i:i + self.max_chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            # Calculate positions
            start_pos = len(' '.join(words[:i]))
            if i > 0:
                start_pos += 1  # Account for space
            end_pos = start_pos + len(chunk_text)
            
            chunks.append(Chunk(
                text=chunk_text,
                chunk_index=chunk_index,
                start_position=start_pos,
                end_position=end_pos,
                token_count=len(chunk_words),
                page_number=page_number
            ))
            
            chunk_index += 1
            # Move forward with overlap
            i += max(1, self.max_chunk_size - self.overlap_size)
        
        return chunks
    
    def _semantic_chunking(self, text: str, page_number: int) -> List[Chunk]:
        """Semantic chunking based on sentence boundaries.
        
        Args:
            text: Text to chunk
            page_number: Page number
            
        Returns:
            List of chunks
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        chunk_index = 0
        current_chunk = []
        current_length = 0
        start_pos = 0
        
        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_length = len(sentence_words)
            
            # If adding this sentence exceeds max size, save current chunk
            if current_length + sentence_length > self.max_chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                end_pos = start_pos + len(chunk_text)
                
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_position=start_pos,
                    end_position=end_pos,
                    token_count=current_length,
                    page_number=page_number
                ))
                
                chunk_index += 1
                
                # Handle overlap - keep last few sentences
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s.split()) <= self.overlap_size:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s.split())
                    else:
                        break
                
                # Calculate new start position
                if overlap_sentences:
                    overlap_text = ' '.join(overlap_sentences)
                    start_pos = end_pos - len(overlap_text)
                else:
                    start_pos = end_pos
                
                current_chunk = overlap_sentences + [sentence]
                current_length = overlap_length + sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                chunk_index=chunk_index,
                start_position=start_pos,
                end_position=start_pos + len(chunk_text),
                token_count=current_length,
                page_number=page_number
            ))
        
        return chunks
    
    def _hybrid_chunking(self, text: str, page_number: int) -> List[Chunk]:
        """Hybrid approach: semantic boundaries with token limits.
        
        Args:
            text: Text to chunk
            page_number: Page number
            
        Returns:
            List of chunks
        """
        # Use semantic chunking but enforce token limits
        chunks = self._semantic_chunking(text, page_number)
        
        # If any chunk exceeds max_chunk_size, split it further
        final_chunks = []
        for chunk in chunks:
            if chunk.token_count > self.max_chunk_size:
                # Split large chunk using token-based method
                sub_chunks = self._token_based_chunking(chunk.text, page_number)
                # Adjust positions
                for sub_chunk in sub_chunks:
                    sub_chunk.start_position += chunk.start_position
                    sub_chunk.end_position += chunk.start_position
                    final_chunks.append(sub_chunk)
            else:
                final_chunks.append(chunk)
        
        # Re-index chunks
        for i, chunk in enumerate(final_chunks):
            chunk.chunk_index = i
        
        return final_chunks
