"""
Test script for the embedding providers.

This script will:
1. Load the embedding provider from the config file
2. Generate embeddings for sample texts
3. Calculate similarity between embeddings
"""

import os
import sys
from embedding_providers import EmbeddingProviderFactory
from openai import OpenAI

def verify_openai_key():
    """Verify that the OpenAI API key is valid by making a simple API call."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        return False

    try:
        # Create a client and make a simple API call
        client = OpenAI(api_key=api_key)
        # List models to test authentication (just get the first page of results)
        models = client.models.list()
        print("✓ OpenAI API key is valid")
        return True
    except Exception as e:
        print(f"Error: Invalid OpenAI API key: {str(e)}")
        return False

def main():
    """Main test function."""
    try:
        # Verify OpenAI API key
        if not verify_openai_key():
            print("Please set a valid OPENAI_API_KEY environment variable.")
            print("You can get a key from https://platform.openai.com/api-keys")
            return
        
        print("Loading embedding provider from config file...")
        provider = EmbeddingProviderFactory.load_from_file("config/embeddings.json")
        print(f"Loaded {provider.__class__.__name__} with dimension: {provider.dimension}")
        
        # Sample texts for testing
        sample_texts = [
            "Understanding graph databases for problem management",
            "Graph database concepts and applications",
            "Semantic similarity in text analysis",
            "Implementing vector embeddings for NLP",
            "Neo4j database administration and optimization"
        ]
        
        # Generate embeddings for each text
        print("\nGenerating embeddings for sample texts...")
        embeddings = provider.batch_generate_embeddings(sample_texts)
        
        print(f"Generated {len(embeddings)} embeddings, each with dimension {len(embeddings[0])}")
        
        # Calculate similarity between the first text and all others
        print("\nCalculating similarity with the first text:")
        base_embedding = embeddings[0]
        for i, embedding in enumerate(embeddings):
            similarity = provider.similarity(base_embedding, embedding)
            print(f"Similarity with text {i+1}: {similarity:.4f} - {sample_texts[i]}")
        
        # Calculate similarity between all pairs
        print("\nSimilarity matrix:")
        for i, embedding1 in enumerate(embeddings):
            for j, embedding2 in enumerate(embeddings):
                if j > i:  # Only print upper triangle
                    similarity = provider.similarity(embedding1, embedding2)
                    print(f"Texts {i+1} and {j+1}: {similarity:.4f}")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error testing embedding provider: {str(e)}")

if __name__ == "__main__":
    main()