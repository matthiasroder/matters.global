"""
Embedding provider interface and implementations for matters.global

This module provides:
- An abstract base class for embedding providers
- Implementations for different embedding providers (OpenAI, etc.)
- A factory for creating providers based on configuration
"""

import os
import json
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field

# Try to import OpenAI for the OpenAI provider
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingConfig(BaseModel):
    """Base configuration for embedding providers."""
    model_config = {"protected_namespaces": ()}
    
    provider_type: str = Field(
        ..., 
        description="Type of embedding provider to use"
    )
    model_name: str = Field(
        ..., 
        description="Name of the embedding model to use"
    )
    dimension: int = Field(
        ..., 
        description="Dimension of the embedding vectors"
    )
    normalize: bool = Field(
        True, 
        description="Whether to normalize the embeddings to unit length"
    )


class OpenAIEmbeddingConfig(EmbeddingConfig):
    """Configuration for OpenAI embedding provider."""
    provider_type: str = "openai"
    api_key: Optional[str] = Field(
        None,
        description="OpenAI API key (uses environment variable if None)"
    )
    organization: Optional[str] = Field(
        None,
        description="OpenAI organization ID"
    )


class RandomEmbeddingConfig(EmbeddingConfig):
    """Configuration for random embedding provider (testing only)."""
    provider_type: str = "random"
    use_consistent_random: bool = Field(
        True,
        description="Whether to use consistent random vectors for the same text"
    )


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.
    
    This defines the interface that all embedding providers must implement.
    """
    
    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text.
        
        Args:
            text: Text to generate an embedding for
            
        Returns:
            List of floating point values representing the embedding
        """
        pass
    
    @abstractmethod
    def batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embeddings, one for each input text
        """
        pass
    
    @staticmethod
    @abstractmethod
    def similarity(embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score between 0 and 1
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get the dimension of the embeddings this provider generates.
        
        Returns:
            Dimension of the embeddings
        """
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Implementation of EmbeddingProvider using OpenAI's API."""
    
    def __init__(self, config: OpenAIEmbeddingConfig):
        """Initialize the OpenAI embedding provider.
        
        Args:
            config: Configuration for the OpenAI embedding provider
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package not installed. Please install it with: "
                "pip install openai"
            )
        
        self.config = config
        self.api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided in config or environment"
            )
        
        self.client = OpenAI(
            api_key=self.api_key,
            organization=config.organization
        )
        
        self._dimension = config.dimension
        self.model_name = config.model_name
        self.normalize = config.normalize
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text using OpenAI."""
        if not text.strip():
            # Return a zero embedding for empty text
            return [0.0] * self._dimension
        
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model_name
            )
            embedding = response.data[0].embedding
            
            if self.normalize:
                embedding = self._normalize_embedding(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating OpenAI embedding: {str(e)}")
            # Return zero embedding as fallback
            return [0.0] * self._dimension
    
    def batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batch using OpenAI."""
        if not texts:
            return []
        
        # Filter out empty strings and keep track of indices
        filtered_texts = []
        indices = []
        for i, text in enumerate(texts):
            if text.strip():
                filtered_texts.append(text)
                indices.append(i)
        
        if not filtered_texts:
            return [[0.0] * self._dimension] * len(texts)
        
        try:
            response = self.client.embeddings.create(
                input=filtered_texts,
                model=self.model_name
            )
            
            # Initialize with zero embeddings
            result = [[0.0] * self._dimension for _ in range(len(texts))]
            
            # Replace with actual embeddings for non-empty texts
            for i, embedding_data in enumerate(response.data):
                embedding = embedding_data.embedding
                if self.normalize:
                    embedding = self._normalize_embedding(embedding)
                result[indices[i]] = embedding
            
            return result
            
        except Exception as e:
            logger.error(f"Error batch generating OpenAI embeddings: {str(e)}")
            # Return zero embeddings as fallback
            return [[0.0] * self._dimension for _ in range(len(texts))]
    
    @staticmethod
    def similarity(embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        if not embedding1 or not embedding2:
            return 0.0
        
        # Convert to numpy arrays for efficient computation
        e1 = np.array(embedding1)
        e2 = np.array(embedding2)
        
        # Compute cosine similarity
        dot_product = np.dot(e1, e2)
        norm1 = np.linalg.norm(e1)
        norm2 = np.linalg.norm(e2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    @property
    def dimension(self) -> int:
        """Get the dimension of the embeddings this provider generates."""
        return self._dimension
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize an embedding to unit length."""
        e = np.array(embedding)
        norm = np.linalg.norm(e)
        if norm == 0:
            return embedding
        return (e / norm).tolist()


class EmbeddingProviderFactory:
    """Factory for creating embedding providers based on configuration."""
    
    @staticmethod
    def create_provider(config: Union[Dict[str, Any], EmbeddingConfig]) -> EmbeddingProvider:
        """Create an embedding provider from configuration.
        
        Args:
            config: Configuration for the embedding provider
                Either a dictionary or an EmbeddingConfig object
                
        Returns:
            An embedding provider instance
        """
        # Convert dictionary to config object if needed
        if isinstance(config, dict):
            provider_type = config.get("provider_type")
            if provider_type == "openai":
                config = OpenAIEmbeddingConfig(**config)
            else:
                raise ValueError(f"Unknown provider type: {provider_type}")
        
        # Create provider based on type
        if config.provider_type == "openai":
            if not isinstance(config, OpenAIEmbeddingConfig):
                config = OpenAIEmbeddingConfig(**config.dict())
            return OpenAIEmbeddingProvider(config)
        else:
            raise ValueError(f"Unknown provider type: {config.provider_type}")
    
    @staticmethod
    def load_from_file(file_path: str) -> EmbeddingProvider:
        """Load embedding provider from a JSON config file.
        
        Args:
            file_path: Path to the JSON config file
            
        Returns:
            An embedding provider instance
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        with open(path, 'r') as f:
            config = json.load(f)
        
        return EmbeddingProviderFactory.create_provider(config)
    
    @staticmethod
    def save_config(config: EmbeddingConfig, file_path: str) -> None:
        """Save embedding provider config to a JSON file.
        
        Args:
            config: Configuration to save
            file_path: Path to save the JSON config file
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(config.dict(), f, indent=2)