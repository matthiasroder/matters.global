"""
Test script for the Entity Resolution System.

This script will:
1. Create test problems with similar descriptions
2. Use the Entity Resolution System to identify similar problems
3. Create and map canonical forms
4. Test canonical node functionality
"""

from graph_problem_manager import GraphProblemManager
from entity_resolution import EntityResolutionSystem
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_problems(manager):
    """Create test problems with similar descriptions."""
    print("\nCreating test problems with similar descriptions...")
    
    # Original problem
    problem1 = manager.create_problem(
        "Understanding vector embeddings for semantic similarity in NLP"
    )
    
    # Similar variants
    problem2 = manager.create_problem(
        "Using vector embeddings to measure semantic similarity in natural language"
    )
    problem3 = manager.create_problem(
        "Implementing semantic similarity with vector embeddings for NLP tasks"
    )
    
    # Somewhat similar but different focus
    problem4 = manager.create_problem(
        "Optimizing vector similarity search in large embedding databases"
    )
    
    # Less similar
    problem5 = manager.create_problem(
        "Training language models for natural language understanding"
    )
    
    print(f"Created 5 test problems")
    return [problem1, problem2, problem3, problem4, problem5]

def test_entity_resolution(manager, er_system, problems):
    """Test the entity resolution functionality."""
    print("\nTesting entity resolution for similar problems...")

    original_problem = problems[0]
    print(f"Finding matches for problem: {original_problem.description}")

    # Find similar entities
    matches = er_system.find_similar_entities(
        original_problem.id,
        entity_type="Problem",
        threshold=0.6,
        limit=10
    )

    print(f"Found {len(matches)} potential matches:")
    for i, match in enumerate(matches):
        print(f"{i+1}. [{match.similarity_score:.4f}] \"{match.target_id}\"")
        # Get the problem description for the matched ID
        target_problem = manager.get_problem_by_id(match.target_id)
        if target_problem:
            print(f"   Description: {target_problem.description}")

    if matches:
        # Create a canonical suggestion
        entity_ids = [original_problem.id] + [match.target_id for match in matches[:2]]
        suggestion = er_system.suggest_canonical_form(entity_ids, entity_type="Problem")

        if suggestion:
            print(f"\nSuggested canonical form:")
            print(f"Description: {suggestion.suggested_description}")
            print(f"Confidence: {suggestion.confidence}")
            print(f"Entities: {len(suggestion.entity_ids)}")

            # Create canonical node using entity resolution system
            canonical_id = er_system.create_canonical_node(suggestion)
            print(f"\nCreated canonical problem: {canonical_id}")

            # Get variants of the canonical form
            variants = manager.get_problem_variants(canonical_id)
            print(f"\nVariants of canonical form ({len(variants)}):")
            for i, variant in enumerate(variants):
                print(f"{i+1}. {variant['description']}")

def test_auto_resolution(manager, er_system):
    """Test automatic entity resolution."""
    print("\nTesting automatic entity resolution...")

    # Create additional problems with varying similarities
    print("Creating more test problems for auto-resolution...")

    # Group 1: ML-related problems
    p1 = manager.create_problem("Training machine learning models on large datasets")
    p2 = manager.create_problem("Machine learning model training with big data")
    p3 = manager.create_problem("Efficient training of ML models on massive datasets")

    # Group 2: Database-related problems
    p4 = manager.create_problem("Optimizing database queries for better performance")
    p5 = manager.create_problem("Improving performance of database query operations")
    p6 = manager.create_problem("Database performance tuning for complex queries")

    # Group 3: UI-related problems (more different from each other)
    p7 = manager.create_problem("Designing responsive user interfaces for mobile applications")
    p8 = manager.create_problem("Creating accessible UI components for web applications")
    p9 = manager.create_problem("User interface design patterns for cross-platform applications")

    print(f"Created 9 additional problems in 3 groups")

    # Run auto-resolution
    print("\nRunning auto-resolution with default parameters...")
    results = er_system.auto_resolve_entities(
        entity_type="Problem",
        threshold=0.7,  # Higher threshold for more confident grouping
        min_group_size=2,
        confidence_threshold=0.6
    )

    print(f"\nAuto-resolution created {len(results)} canonical groups:")
    for i, result in enumerate(results):
        print(f"\nGroup {i+1}:")
        print(f"Canonical: {result['canonical_description']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Method: {result['method']}")
        print(f"Entity count: {len(result['entity_ids'])}")

        # Get the actual descriptions for the entity IDs
        print("Entities in this group:")
        for entity_id in result['entity_ids']:
            entity = manager.get_problem_by_id(entity_id)
            if entity:
                print(f"- {entity.description}")

    # Test with different parameters
    print("\nRunning auto-resolution with stricter parameters...")
    strict_results = er_system.auto_resolve_entities(
        entity_type="Problem",
        threshold=0.8,  # Much higher threshold
        min_group_size=3,  # Require at least 3 problems in a group
        confidence_threshold=0.7  # Higher confidence required
    )

    print(f"Strict auto-resolution created {len(strict_results)} canonical groups")

def main():
    """Main test function."""
    try:
        # Create manager with default Neo4j connection
        print("Creating GraphProblemManager and EntityResolutionSystem...")
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
        print("Initializing schema...")
        manager.initialize_schema()

        # Create Entity Resolution System
        er_system = EntityResolutionSystem(manager)

        # Show available tests
        print("\nAvailable tests:")
        print("1. Basic entity resolution test")
        print("2. Automatic entity resolution test")
        print("3. Run all tests")

        choice = input("\nSelect test to run (1-3, default is 3): ").strip() or "3"

        if choice in ["1", "3"]:
            # Create test data
            problems = create_test_problems(manager)

            # Test entity resolution
            test_entity_resolution(manager, er_system, problems)

        if choice in ["2", "3"]:
            # Test automatic entity resolution
            test_auto_resolution(manager, er_system)

        print("\nTest completed successfully!")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
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