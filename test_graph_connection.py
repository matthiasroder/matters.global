"""
Test script for the GraphProblemManager.

This script will:
1. Connect to the Neo4j database
2. Create a simple problem and conditions
3. Test problem relationships and connections
"""

from graph_problem_manager import GraphProblemManager

def main():
    """Main test function."""
    # Create the manager with default settings
    # Adjust these parameters as needed for your Neo4j setup
    manager = GraphProblemManager(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="matters2025"  # Replace with your new password
    )
    
    try:
        # Connect
        print("Connecting to Neo4j...")
        manager.connect()
        
        # Initialize schema
        print("Initializing schema...")
        manager.initialize_schema()
        
        # Create test problems
        print("\nCreating test problems...")
        problem1 = manager.create_problem(
            "Understanding graph databases for problem management"
        )
        problem2 = manager.create_problem(
            "Implementing semantic similarity in knowledge graphs"
        )
        
        print(f"Created problem 1: {problem1.id} - {problem1.description}")
        print(f"Created problem 2: {problem2.id} - {problem2.description}")
        
        # Add conditions
        print("\nAdding conditions to problems...")
        condition1 = manager.add_condition_to_problem(
            problem1.id,
            "Understand Neo4j fundamentals and Cypher query language"
        )
        condition2 = manager.add_condition_to_problem(
            problem1.id,
            "Design appropriate graph schema for problem representation"
        )
        condition3 = manager.add_condition_to_problem(
            problem2.id,
            "Implement vector embeddings for problem descriptions"
        )
        
        print(f"Added condition to problem 1: {condition1.description}")
        print(f"Added condition to problem 1: {condition2.description}")
        print(f"Added condition to problem 2: {condition3.description}")
        
        # Create dependency between problems
        print("\nCreating dependency between problems...")
        manager.add_dependency(problem2.id, problem1.id)
        print(f"Problem 2 now depends on Problem 1")
        
        # Update a condition
        print("\nUpdating condition status...")
        manager.update_condition(condition1.id, True)
        print(f"Marked condition as met: {condition1.description}")
        
        # Check if problem is solved
        print("\nChecking if problems are solved...")
        is_solved1 = manager.check_if_problem_solved(problem1.id)
        is_solved2 = manager.check_if_problem_solved(problem2.id)
        print(f"Problem 1 solved: {is_solved1}")
        print(f"Problem 2 solved: {is_solved2}")
        
        # Add a solution
        print("\nAdding a working solution...")
        solution = manager.add_working_solution(
            problem1.id,
            "Use Neo4j with custom Python wrapper for graph-based problem management"
        )
        print(f"Added solution: {solution.description}")
        
        # Find similar problems
        print("\nFinding similar problems (basic implementation)...")
        similar = manager.find_similar_problems("graph database knowledge")
        print(f"Found {len(similar)} similar problems")
        for prob in similar:
            print(f"- {prob.description}")
        
        # Get problem connections
        print("\nGetting problem connections...")
        connections = manager.find_problem_connections(problem1.id)
        print(f"Problem 1 has:")
        print(f"- {len(connections['conditions'])} conditions")
        print(f"- {len(connections['solutions'])} solutions")
        print(f"- {len(connections['depends_on'])} dependencies")
        print(f"- {len(connections['dependents'])} dependent problems")
        
        print("\nConnection successful and tests complete!")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Ensure Neo4j is running: brew services list | grep neo4j")
        print("2. Verify Neo4j credentials (default is usually neo4j/neo4j)")
        print("3. Check Neo4j logs: brew info neo4j (look for log file location)")
        print("4. Try accessing Neo4j Browser at: http://localhost:7474")
    
    finally:
        # Close the connection if it was established
        if manager.driver:
            manager.close()

if __name__ == "__main__":
    main()