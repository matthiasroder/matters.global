"""
Test script for graph embeddings functionality.

This script will:
1. Create problems and conditions with embeddings
2. Test vector similarity search
3. Verify that embeddings are properly stored and used
"""

import os
from graph_problem_manager import GraphProblemManager
from embedding_providers import EmbeddingProviderFactory

def create_test_data(manager):
    """Create test problems and conditions with embeddings."""
    print("\nCreating test problems...")
    
    # Create a set of related problems
    problem1 = manager.create_problem(
        "Understanding vector search algorithms in graph databases"
    )
    problem2 = manager.create_problem(
        "Implementing semantic search with embeddings in Neo4j"
    )
    problem3 = manager.create_problem(
        "Building a knowledge graph with vector similarity capabilities"
    )
    problem4 = manager.create_problem(
        "Optimizing query performance in large graph databases"
    )
    problem5 = manager.create_problem(
        "Developing intelligent problem management systems"
    )
    
    print(f"Created 5 test problems with embeddings")
    
    # Add conditions to problems
    print("\nAdding conditions to problems...")
    
    # Conditions for problem 1
    condition1 = manager.add_condition_to_problem(
        problem1.id,
        "Understand cosine similarity for vector comparisons"
    )
    condition2 = manager.add_condition_to_problem(
        problem1.id,
        "Learn about vector indexing techniques"
    )
    
    # Conditions for problem 2
    condition3 = manager.add_condition_to_problem(
        problem2.id,
        "Implement embeddings generation using OpenAI"
    )
    condition4 = manager.add_condition_to_problem(
        problem2.id,
        "Create vector indexes in Neo4j"
    )
    
    # Conditions for problem 3
    condition5 = manager.add_condition_to_problem(
        problem3.id,
        "Design schema for knowledge representation"
    )
    condition6 = manager.add_condition_to_problem(
        problem3.id,
        "Develop entity resolution system using vector similarity"
    )
    
    print(f"Added 6 conditions with embeddings")
    
    return {
        "problems": [problem1, problem2, problem3, problem4, problem5],
        "conditions": [condition1, condition2, condition3, condition4, condition5, condition6]
    }

def test_similar_problems(manager):
    """Test finding similar problems."""
    print("\nTesting similar problems search...")
    
    # Different search queries to test
    queries = [
        "Vector embeddings for semantic search",
        "Graph database performance optimization",
        "Knowledge representation with Neo4j",
        "Problem management systems"
    ]
    
    for query in queries:
        print(f"\nSearching for problems similar to: '{query}'")
        similar = manager.find_similar_problems(query, threshold=0.5)
        
        print(f"Found {len(similar)} similar problems:")
        for i, problem in enumerate(similar):
            print(f"{i+1}. [{problem['similarity']:.4f}] {problem['description']}")

def test_similar_conditions(manager):
    """Test finding similar conditions."""
    print("\nTesting similar conditions search...")
    
    # Different search queries to test
    queries = [
        "Creating vector embeddings with ML models",
        "Database indexing for fast retrieval",
        "Knowledge graph design patterns"
    ]
    
    for query in queries:
        print(f"\nSearching for conditions similar to: '{query}'")
        similar = manager.find_similar_conditions(query, threshold=0.5)
        
        print(f"Found {len(similar)} similar conditions:")
        for i, condition in enumerate(similar):
            print(f"{i+1}. [{condition['similarity']:.4f}] {condition['description']}")

def main():
    """Main test function."""
    try:
        # Create manager with default Neo4j connection
        print("Creating GraphProblemManager...")
        manager = GraphProblemManager(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="matters2025",  # Update with your password
            embedding_config_path="config/embeddings.json"
        )
        
        # Connect to Neo4j
        print("Connecting to Neo4j...")
        manager.connect()
        
        # Initialize schema
        print("Initializing schema with vector indexes...")
        manager.initialize_schema()
        
        # Create test data
        test_data = create_test_data(manager)
        
        # Test similarity search
        test_similar_problems(manager)
        test_similar_conditions(manager)
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Ensure Neo4j is running: brew services list | grep neo4j")
        print("2. Verify Neo4j credentials are correct")
        print("3. Make sure the OpenAI API key is set in your environment")
        print("4. Check that the embedding config file exists")
    
    finally:
        # Close the connection if it was established
        if 'manager' in locals() and manager.driver:
            manager.close()

if __name__ == "__main__":
    main()