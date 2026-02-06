"""Document classifier using RAG and LLM."""

import json
import re
from typing import Dict, List
from config import Config

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

class DocumentClassifier:
    """Classify documents using RAG and LLM."""
    
    # Classification rules that the LLM must follow
    CLASSIFICATION_RULES = """
CLASSIFICATION RULES (MUST FOLLOW EXACTLY):

1. UNCLASSIFIED
   - General information with no sensitive content
   - Public records, announcements, general communications
   - No operational details, no personnel information
   
2. CONFIDENTIAL
   - Information that could cause damage if disclosed
   - Unit movements, training schedules
   - Non-critical operational details
   - Basic personnel rosters
   
3. SECRET
   - Information that could cause serious damage if disclosed
   - Detailed operational plans
   - Intelligence reports
   - Advanced weapons capabilities
   - Critical infrastructure details
   
4. TOP SECRET
   - Information that could cause exceptionally grave damage
   - Highly sensitive intelligence sources/methods
   - Nuclear weapons information
   - Covert operations details
   - High-level strategic military plans

CRITICAL: You MUST classify based ONLY on these rules and the retrieved examples.
DO NOT invent new classification criteria.
"""
    
    def __init__(self):
        """Initialize the classifier."""
        self.provider = Config.LLM_PROVIDER
        self.model = Config.LLM_MODEL
        self.confidence_threshold = Config.CONFIDENCE_THRESHOLD
        
        if self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
            if not Config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set in .env file")
            self.client = AsyncOpenAI(base_url=Config.OPENAI_API_ENDPOINT, api_key=Config.OPENAI_API_KEY)
        elif self.provider == "local":
            if not OLLAMA_AVAILABLE:
                raise ImportError("Ollama package not installed. Run: pip install ollama")
            # Test Ollama connection and model availability
            self._test_ollama_connection()
        else:
            raise NotImplementedError(f"LLM provider '{self.provider}' not yet implemented")
    
    async def classify(self, document_text: str, retrieved_context: str) -> Dict:
        """Classify a document using RAG (async).
        
        Args:
            document_text: The document to classify.
            retrieved_context: Context from retrieved similar documents.
            
        Returns:
            Dictionary with classification, confidence, and justification.
        """
        prompt = self._build_prompt(document_text, retrieved_context)
        
        if self.provider == "openai":
            result = await self._classify_with_openai(prompt)
        elif self.provider == "local":
            result = await self._classify_with_ollama(prompt)
        else:
            raise NotImplementedError(f"Provider {self.provider} not implemented")
        
        # Parse and validate result
        parsed_result = self._parse_llm_response(result)
        
        # Check confidence threshold
        if parsed_result['confidence'] < self.confidence_threshold:
            parsed_result['classification'] = "REQUIRES HUMAN REVIEW"
            parsed_result['justification'] += f"\n\n⚠️ Confidence ({parsed_result['confidence']:.2f}) below threshold ({self.confidence_threshold})."
        
        return parsed_result
    
    async def classify_tags(self, document_text: str, retrieved_context: str, tag_context: str = "", available_tags: List[str] = None) -> Dict:
        """Classify a document using RAG and return tags (async).
        
        Args:
            document_text: The document to classify.
            retrieved_context: Context from retrieved similar chunks.
            tag_context: Additional context about aggregated tag frequencies.
            available_tags: List of predefined tags that can be assigned.
            
        Returns:
            Dictionary with tags, confidence, and justification.
        """
        prompt = self._build_tag_prompt(document_text, retrieved_context, tag_context, available_tags)
        
        if self.provider == "openai":
            result = await self._classify_with_openai(prompt)
        elif self.provider == "local":
            result = await self._classify_with_ollama(prompt)
        else:
            raise NotImplementedError(f"Provider {self.provider} not implemented")
        
        # Parse and validate result for tags
        parsed_result = self._parse_tag_response(result)
        
        # Check confidence threshold
        if parsed_result['confidence'] < self.confidence_threshold:
            parsed_result['justification'] += f"\n\n⚠️ Confidence ({parsed_result['confidence']:.2f}) below threshold ({self.confidence_threshold})."
        
        return parsed_result
    
    def _build_prompt(self, document_text: str, retrieved_context: str) -> str:
        """Build the classification prompt for the LLM.
        
        Args:
            document_text: The document to classify.
            retrieved_context: Retrieved similar documents.
            
        Returns:
            Complete prompt string.
        """
        prompt = f"""You are a military document classification expert. Your task is to classify a document based ONLY on:
1. The provided classification rules
2. The retrieved similar documents as examples

{self.CLASSIFICATION_RULES}

RETRIEVED SIMILAR DOCUMENTS (as examples):
{retrieved_context}

DOCUMENT TO CLASSIFY:
{document_text}

INSTRUCTIONS:
- Analyze the document content carefully
- Compare it to the retrieved examples
- Apply ONLY the classification rules provided above
- DO NOT create new classification criteria
- Provide your confidence (0.0 to 1.0)
- Justify your decision by referencing specific retrieved documents and rules

OUTPUT FORMAT (respond ONLY with valid JSON):
{{
    "classification": "UNCLASSIFIED|CONFIDENTIAL|SECRET|TOP SECRET",
    "confidence": 0.85,
    "justification": "Brief explanation referencing retrieved documents and rules"
}}
"""
        return prompt
    
    def _build_tag_prompt(self, document_text: str, retrieved_context: str, tag_context: str = "", available_tags: List[str] = None) -> str:
        """Build the tag classification prompt for the LLM.
        
        Args:
            document_text: The document to classify.
            retrieved_context: Retrieved similar chunks with tags.
            tag_context: Additional context about aggregated tags.
            available_tags: List of predefined tags that can be assigned.
            
        Returns:
            Complete prompt string.
        """
        # Format available tags for the prompt
        if available_tags:
            tags_list = "\n".join([f"- {tag}" for tag in sorted(available_tags)])
            available_tags_section = f"""\nAVAILABLE TAGS (you MUST choose ONLY from these tags):
{tags_list}
"""
        else:
            available_tags_section = ""
        
        prompt = f"""You are a military document tagging specialist. Your task is to identify and assign relevant tags to a document based on:
1. The retrieved similar chunks and their tags
2. The aggregated tag frequencies from similar content
3. The document content
{available_tags_section}
RETRIEVED SIMILAR CHUNKS (with their tags):
{retrieved_context}

{tag_context}

DOCUMENT TO CLASSIFY:
{document_text}

CRITICAL INSTRUCTIONS:
- You MUST choose tags ONLY from the available tags list above
- DO NOT create or suggest new tags that are not in the available tags list
- Analyze the document content carefully
- Review the tags from retrieved similar chunks
- Consider the aggregated tag frequencies (more frequent tags are more relevant)
- Assign 1-5 most relevant tags that describe the document
- Provide your confidence (0.0 to 1.0)
- Justify your tag selection

OUTPUT FORMAT (respond ONLY with valid JSON):
{{
    "tags": ["tag1", "tag2", "tag3"],
    "confidence": 0.85,
    "justification": "Brief explanation of why these tags were selected"
}}
"""
        return prompt
    
    async def _classify_with_openai(self, prompt: str) -> str:
        """Call OpenAI API for classification (async).
        
        Args:
            prompt: The classification prompt.
            
        Returns:
            LLM response text.
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": "You are a military document classification expert. Always respond with valid JSON." + prompt}
                ],
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"✗ OpenAI API error: {e}")
            return json.dumps({
                "classification": "REQUIRES HUMAN REVIEW",
                "confidence": 0.0,
                "justification": f"Error during classification: {str(e)}"
            })
    
    def _test_ollama_connection(self):
        """Test connection to Ollama server and verify model availability.
        
        Raises:
            ConnectionError: If Ollama is not accessible or model not found.
        """
        try:
            # List available models using custom URL
            client = ollama.Client(host=Config.OLLAMA_URL)
            models_response = client.list()
            models_list = models_response.get('models', [])
            
            # Extract model names (handle both 'name' and 'model' keys)
            available_models = []
            for model in models_list:
                model_name = model.get('name') or model.get('model', '')
                if model_name:
                    available_models.append(model_name)
            
            # Check if requested model is available
            if not any(self.model in model_name for model_name in available_models):
                raise ValueError(
                    f"Model '{self.model}' not found. Available models: {', '.join(available_models)}. "
                    f"Pull the model with: ollama pull {self.model}"
                )
            
            print(f"✓ Connected to Ollama. Using model: {self.model}")
        except ollama.ResponseError as e:
            raise ConnectionError(
                f"Cannot connect to Ollama. Make sure Ollama is running. Error: {str(e)}"
            )
        except Exception as e:
            raise ConnectionError(
                f"Ollama connection test failed: {str(e)}"
            )
    
    async def _classify_with_ollama(self, prompt: str) -> str:
        """Call Ollama API for classification (async).
        
        Args:
            prompt: The classification prompt.
            
        Returns:
            LLM response text.
        """
        try:
            # Create async client with custom URL
            client = ollama.AsyncClient(host=Config.OLLAMA_URL)
            response = await client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.1,  # Low temperature for consistent classification
                    "num_predict": 500
                }
            )
            return response.get("response", "")
        except ollama.ResponseError as e:
            print(f"✗ Ollama API error: {e}")
            return json.dumps({
                "classification": "REQUIRES HUMAN REVIEW",
                "confidence": 0.0,
                "justification": f"Error during classification: {str(e)}"
            })
        except Exception as e:
            print(f"✗ Ollama error: {e}")
            return json.dumps({
                "classification": "REQUIRES HUMAN REVIEW",
                "confidence": 0.0,
                "justification": f"Error during classification: {str(e)}"
            })
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse and validate LLM response.
        
        Args:
            response: Raw LLM response.
            
        Returns:
            Parsed and validated result dictionary.
        """
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response)
            
            # Validate required fields
            classification = result.get('classification', 'REQUIRES HUMAN REVIEW')
            confidence = float(result.get('confidence', 0.0))
            justification = result.get('justification', 'No justification provided')
            
            # Validate classification level
            valid_levels = Config.CLASSIFICATION_LEVELS + ["REQUIRES HUMAN REVIEW"]
            if classification not in valid_levels:
                classification = "REQUIRES HUMAN REVIEW"
                justification += f"\n⚠️ Invalid classification level returned: {classification}"
                confidence = 0.0
            
            # Ensure confidence is in valid range
            confidence = max(0.0, min(1.0, confidence))
            
            return {
                'classification': classification,
                'confidence': confidence,
                'justification': justification
            }
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"✗ Error parsing LLM response: {e}")
            print(f"Response was: {response}")
            return {
                'classification': 'REQUIRES HUMAN REVIEW',
                'confidence': 0.0,
                'justification': f'Failed to parse LLM response: {str(e)}'
            }
    
    def _parse_tag_response(self, response: str) -> Dict:
        """Parse and validate LLM tag response.
        
        Args:
            response: Raw LLM response.
            
        Returns:
            Parsed and validated result dictionary with tags.
        """
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response)
            
            # Validate required fields
            tags = result.get('tags', [])
            if isinstance(tags, str):
                # If tags is a string, split by comma
                tags = [tag.strip() for tag in tags.split(',') if tag.strip()]
            elif not isinstance(tags, list):
                tags = []
            
            confidence = float(result.get('confidence', 0.0))
            justification = result.get('justification', 'No justification provided')
            
            # Ensure confidence is in valid range
            confidence = max(0.0, min(1.0, confidence))
            
            # Clean up tags
            tags = [str(tag).strip() for tag in tags if tag]
            
            return {
                'tags': tags,
                'confidence': confidence,
                'justification': justification
            }
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"✗ Error parsing tag response: {e}")
            print(f"Response was: {response}")
            return {
                'tags': [],
                'confidence': 0.0,
                'justification': f'Failed to parse LLM response: {str(e)}'
            }

