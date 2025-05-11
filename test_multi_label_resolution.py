"""
Test script for entity resolution with the multi-label schema.

This script will:
1. Create similar entities with different label combinations
2. Test finding similar entities across labels
3. Test grouping entities by similarity
4. Test creating canonical forms for mixed-label groups
5. Test mapping entities to canonical forms
"""

import time
from datetime import datetime
from typing import List, Dict, Any

from graph_problem_manager import (
    GraphManager, 
    MatterLabel,
    RelationshipType
)
from entity_resolution import EntityResolutionSystem

def setup_test_data(manager: GraphManager) -> Dict[str, List[str]]:
    """Create test data with similar entities of different types."""
    print("Creating test data for entity resolution...")
    
    # Track IDs for each entity type
    entity_ids = {
        "goals": [],
        "problems": [],
        "conditions": [],
        "solutions": [],
        "mixed": []  # Entities with multiple labels
    }
    
    # Create similar goals
    goal_descriptions = [
        "Improve the performance of the recommendation system",
        "Enhance the speed of the recommendation algorithm",
        "Make the recommendation engine run faster",
        "Boost performance of product recommendations"
    ]
    
    for desc in goal_descriptions:
        goal = manager.create_goal(description=desc, progress=0.0)
        entity_ids["goals"].append(goal.id)
        print(f"Created goal: {goal.id} - {desc}")
    
    # Create similar problems
    problem_descriptions = [
        "The system is experiencing slow database queries",
        "Database performance issues are affecting response time",
        "Queries to the database are taking too long to complete",
        "Slow SQL queries are impacting system performance"
    ]
    
    for desc in problem_descriptions:
        problem = manager.create_problem(description=desc)
        entity_ids["problems"].append(problem.id)
        print(f"Created problem: {problem.id} - {desc}")
    
    # Create similar conditions
    condition_descriptions = [
        "All API response times must be under 200ms",
        "API endpoints should respond in less than 200ms",
        "Response time for the API must not exceed 200ms",
        "The API must maintain sub-200ms response times"
    ]
    
    for desc in condition_descriptions:
        condition = manager.create_condition(description=desc)
        entity_ids["conditions"].append(condition.id)
        print(f"Created condition: {condition.id} - {desc}")
    
    # Create similar solutions
    solution_descriptions = [
        "Implement database query caching to improve performance",
        "Add caching layer for database queries to enhance speed",
        "Cache frequently used database queries to reduce response time",
        "Use query caching to optimize database performance"
    ]
    
    for desc in solution_descriptions:
        solution = manager.create_solution(description=desc)
        entity_ids["solutions"].append(solution.id)
        print(f"Created solution: {solution.id} - {desc}")
    
    # Create dual-label entities (Problem + Condition)
    dual_descriptions = [
        "Need to ensure all API responses include proper error codes",
        "API responses must have standardized error codes",
        "All API endpoints need to return appropriate error codes",
        "Implement consistent error codes in API responses"
    ]
    
    for desc in dual_descriptions:
        # Create as Problem first
        entity = manager.create_problem(description=desc)
        
        # Add Condition label via direct session
        with manager.driver.session() as session:
            session.run(
                """
                MATCH (m:Matter {id: $id})
                SET m:Condition, m.is_met = false
                RETURN m
                """,
                id=entity.id
            )
        
        entity_ids["mixed"].append(entity.id)
        print(f"Created mixed entity: {entity.id} - {desc}")
    
    return entity_ids

def test_find_similar_entities(resolver: EntityResolutionSystem):
    """Test finding similar entities with different label combinations."""
    print("\n=== Testing Find Similar Entities ===")
    
    # Test query descriptions
    queries = {
        "performance": "Improving system performance and response times",
        "api": "API response standards and error codes",
        "database": "Database query performance optimization"
    }
    
    for query_name, query_text in queries.items():
        print(f"\nSearch query: '{query_name}'")
        
        # Test with specific labels
        labels_to_test = [
            None,  # All labels
            [MatterLabel.GOAL.value],
            [MatterLabel.PROBLEM.value],
            [MatterLabel.CONDITION.value],
            [MatterLabel.SOLUTION.value],
            [MatterLabel.PROBLEM.value, MatterLabel.CONDITION.value]  # Multiple labels
        ]
        
        for labels in labels_to_test:
            label_str = ", ".join(labels) if labels else "all labels"
            matches = resolver.find_similar_entities(
                description=query_text,
                labels=labels,
                threshold=0.3,
                limit=10
            )
            
            print(f"- Found {len(matches)} matches with {label_str}")

def test_group_similar_entities(resolver: EntityResolutionSystem):
    """Test grouping similar entities by type and across types."""
    print("\n=== Testing Group Similar Entities ===")
    
    # Test grouping within each type
    labels_to_test = [
        None,  # All labels
        [MatterLabel.GOAL.value],
        [MatterLabel.PROBLEM.value],
        [MatterLabel.CONDITION.value],
        [MatterLabel.SOLUTION.value],
        [MatterLabel.PROBLEM.value, MatterLabel.CONDITION.value]  # Multiple labels
    ]
    
    for labels in labels_to_test:
        label_str = ", ".join(labels) if labels else "all labels"
        groups = resolver.group_similar_entities(
            labels=labels,
            threshold=0.5,
            min_group_size=2
        )
        
        print(f"Found {len(groups)} groups for {label_str}")
        
        for i, group in enumerate(groups):
            print(f"  Group {i+1}: {len(group)} entities")
            
            # Show labels distribution in the group
            label_counts = {}
            for entity in group:
                for label in entity.get("labels", []):
                    if label != MatterLabel.MATTER.value:  # Skip base Matter label
                        label_counts[label] = label_counts.get(label, 0) + 1
            
            print(f"  Label distribution: {label_counts}")

def test_canonical_suggestions(resolver: EntityResolutionSystem, entity_ids: Dict[str, List[str]]):
    """Test suggesting canonical forms for different entity types."""
    print("\n=== Testing Canonical Suggestions ===")
    
    # Test single-label entity groups
    for entity_type, ids in entity_ids.items():
        if entity_type in ("goals", "problems", "conditions", "solutions", "mixed"):
            # For testing, just use the first few entities
            test_ids = ids[:3]
            
            print(f"\nGenerating canonical suggestion for {entity_type}:")
            
            # For mixed entities, don't specify a target label
            target_label = None if entity_type == "mixed" else MatterLabel[entity_type[:-1].upper()].value
            
            # Get suggestion
            suggestion = resolver.suggest_canonical_form(test_ids, target_label)
            
            if suggestion:
                print(f"Suggestion: {suggestion.suggested_description[:100]}...")
                print(f"Confidence: {suggestion.confidence:.2f}")
                print(f"Labels: {suggestion.entity_labels}")
                print(f"Method: {suggestion.supporting_factors.get('method', 'unknown')}")
            else:
                print(f"No suggestion generated for {entity_type}")

def test_create_canonical_nodes(resolver: EntityResolutionSystem, entity_ids: Dict[str, List[str]]):
    """Test creating canonical nodes and mapping entities to them."""
    print("\n=== Testing Creating Canonical Nodes ===")
    
    # Test with different entity types
    for entity_type, ids in entity_ids.items():
        if entity_type in ("goals", "problems", "conditions", "solutions", "mixed"):
            # For testing, just use the first few entities
            test_ids = ids[:3]
            
            print(f"\nCreating canonical node for {entity_type}:")
            
            # For mixed entities, don't specify a target label
            target_label = None if entity_type == "mixed" else MatterLabel[entity_type[:-1].upper()].value
            
            # Get suggestion first
            suggestion = resolver.suggest_canonical_form(test_ids, target_label)
            
            if suggestion:
                # Create canonical node
                canonical_id = resolver.create_canonical_node(suggestion)
                
                if canonical_id:
                    print(f"Created canonical node: {canonical_id}")
                    print(f"Mapped {len(suggestion.entity_ids)} entities to it")
                    
                    # Verify mappings
                    manager = resolver.graph_manager
                    for entity_id in suggestion.entity_ids:
                        entity = manager.get_matter_by_id(entity_id)
                        canonical = manager.get_matter_canonical_form(entity_id)
                        
                        if canonical:
                            print(f"  - {entity_id} ({entity.description[:40]}...) -> {canonical.description[:40]}...")
                        else:
                            print(f"  - {entity_id} (mapping failed)")
                else:
                    print(f"Failed to create canonical node for {entity_type}")
            else:
                print(f"No suggestion generated for {entity_type}")

def main():
    """Main test function."""
    print("=== TESTING MULTI-LABEL ENTITY RESOLUTION ===\n")
    
    # Create the manager and connect
    manager = GraphManager(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="matters2025",
        embedding_config_path="config/embeddings.json"
    )
    
    try:
        print("Connecting to Neo4j...")
        manager.connect()
        
        print("Initializing schema...")
        manager.initialize_schema()
        
        # Initialize the entity resolution system
        resolver = EntityResolutionSystem(manager)
        
        # Create test data
        entity_ids = setup_test_data(manager)
        
        # Wait a moment for embeddings to be generated
        print("\nWaiting a moment for embeddings to be processed...")
        time.sleep(2)
        
        # Run the tests
        test_find_similar_entities(resolver)
        test_group_similar_entities(resolver)
        test_canonical_suggestions(resolver, entity_ids)
        test_create_canonical_nodes(resolver, entity_ids)
        
        print("\n=== ALL TESTS COMPLETE ===")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        
    finally:
        # Close the connection
        if manager.driver:
            manager.close()

if __name__ == "__main__":
    main()