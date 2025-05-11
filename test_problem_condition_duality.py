"""
Test script for the problem-condition duality in the multi-label schema.

This script demonstrates:
1. Creating entities that are both problems and conditions
2. Converting problems to conditions when they have no clear solution
3. Adding conditions to track goal progress
4. Showing how the same entity can participate in different relationship types
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from graph_problem_manager import (
    GraphManager, 
    Matter, 
    Goal, 
    Problem, 
    Condition, 
    Solution, 
    MatterLabel, 
    RelationshipType,
    ProblemState,
    SolutionState
)

def create_project_structure(manager: GraphManager) -> Dict[str, Any]:
    """Create a project structure with goals, problems and conditions."""
    print("\n=== Creating Project Structure ===")
    
    entities = {}
    
    # Create the main project goal
    main_goal = manager.create_goal(
        description="Implement a responsive mobile app for customer service",
        target_date=datetime.now() + timedelta(days=60),
        progress=0.0,
        tags=["project", "mobile", "customer-service"]
    )
    print(f"Created main goal: {main_goal.id} - {main_goal.description}")
    entities["main_goal"] = main_goal
    
    # Create sub-goals
    sub_goals = [
        "Design the UI/UX for the mobile app",
        "Implement the backend API services",
        "Create database schema and models",
        "Setup the CI/CD pipeline for builds and deployment"
    ]
    
    entities["sub_goals"] = []
    for i, desc in enumerate(sub_goals):
        sub_goal = manager.create_goal(
            description=desc,
            target_date=datetime.now() + timedelta(days=30 + i*5),
            progress=0.0
        )
        print(f"Created sub-goal: {sub_goal.id} - {sub_goal.description}")
        entities["sub_goals"].append(sub_goal)
        
        # Link to main goal (sub-goal is part of main goal)
        manager.set_matter_relationship(
            main_goal.id,
            RelationshipType.CONSISTS_OF,
            sub_goal.id
        )
    
    # Create problems (challenges to solve)
    problems = [
        "Need to optimize API response time for mobile network conditions",
        "Authentication process needs to be secure but user-friendly",
        "App must work offline and sync when connection is restored"
    ]
    
    entities["problems"] = []
    for desc in problems:
        problem = manager.create_problem(description=desc)
        print(f"Created problem: {problem.id} - {problem.description}")
        entities["problems"].append(problem)
        
        # Link to appropriate sub-goal
        manager.set_matter_relationship(
            entities["sub_goals"][1].id,  # Backend API sub-goal
            RelationshipType.CONSISTS_OF,
            problem.id
        )
    
    # Create dual-purpose entities (both Problem and Condition)
    dual_entities = [
        "Must support both iOS and Android platforms",
        "UI must follow accessibility guidelines",
        "All API endpoints must have comprehensive test coverage"
    ]
    
    entities["dual_entities"] = []
    for desc in dual_entities:
        # Create as Problem first
        entity = manager.create_problem(description=desc)
        
        # Add Condition label
        with manager.driver.session() as session:
            session.run(
                """
                MATCH (m:Matter {id: $id})
                SET m:Condition, m.is_met = false
                RETURN m
                """,
                id=entity.id
            )
        
        # Fetch updated entity
        updated_entity = manager.get_matter_by_id(entity.id)
        print(f"Created dual entity: {updated_entity.id} - {updated_entity.description}")
        print(f"  Labels: {updated_entity.labels}")
        entities["dual_entities"].append(updated_entity)
        
        # Link to appropriate sub-goal as a requirement
        manager.set_matter_relationship(
            entities["sub_goals"][0].id,  # UI/UX sub-goal
            RelationshipType.REQUIRES,
            entity.id
        )
    
    return entities

def demonstrate_problem_condition_conversion(manager: GraphManager, entities: Dict[str, Any]):
    """Demonstrate converting a problem into a condition when no direct solution exists."""
    print("\n=== Demonstrating Problem to Condition Conversion ===")
    
    # Create a problem that doesn't have a direct solution but becomes a condition for another problem
    problem = manager.create_problem(
        description="Need to ensure secure data transmission between app and backend",
        priority=1
    )
    print(f"Created problem: {problem.id} - {problem.description}")
    
    # Later we realize this is actually a condition for multiple other problems
    # So we'll add the Condition label
    with manager.driver.session() as session:
        session.run(
            """
            MATCH (m:Matter {id: $id})
            SET m:Condition, m.is_met = false,
                m.verification_method = 'Verify SSL/TLS encryption and certificate validation'
            RETURN m
            """,
            id=problem.id
        )
    
    # Now link it as a condition to one of our existing problems
    manager.set_matter_relationship(
        entities["problems"][0].id,  # API response time problem
        RelationshipType.REQUIRES,
        problem.id
    )
    
    # Get the updated entity
    converted_entity = manager.get_matter_by_id(problem.id)
    print(f"Converted to dual-purpose entity with labels: {converted_entity.labels}")
    
    # Show that we can still access it as both a problem and a condition
    as_problem = manager.get_problem_by_id(problem.id)
    as_condition = manager.get_condition_by_id(problem.id)
    
    print(f"Can access as problem: {as_problem is not None}")
    print(f"Can access as condition: {as_condition is not None}")
    
    # Show the entity's connections
    connections = manager.find_matter_connections(problem.id)
    print("Entity connections:")
    print(f"- Problems requiring this condition: {len(connections['prerequisites_for'])}")
    for prob in connections["prerequisites_for"]:
        print(f"  * {prob['description']}")

def track_goal_progress_with_conditions(manager: GraphManager, entities: Dict[str, Any]):
    """Demonstrate tracking goal progress using conditions."""
    print("\n=== Demonstrating Goal Progress Tracking ===")
    
    # Start with our UI/UX sub-goal
    ui_goal = entities["sub_goals"][0]
    
    # Check initial progress
    initial_progress = manager.check_goal_progress(ui_goal.id)
    print(f"Initial goal progress: {initial_progress:.2f}")
    
    # Create more conditions for the UI/UX goal
    conditions = [
        "Complete wireframe designs for all screens",
        "Get design approval from stakeholders",
        "Create component library for reusable UI elements"
    ]
    
    for desc in conditions:
        condition = manager.create_condition(description=desc, is_met=False)
        print(f"Created condition: {condition.id} - {condition.description}")
        
        # Link to the UI/UX goal
        manager.set_matter_relationship(
            ui_goal.id,
            RelationshipType.REQUIRES,
            condition.id
        )
    
    # Mark the dual-entity conditions as met
    for entity in entities["dual_entities"]:
        manager.update_condition(entity.id, True)
        print(f"Marked condition as met: {entity.description}")
    
    # Check updated progress
    updated_progress = manager.check_goal_progress(ui_goal.id)
    print(f"Updated goal progress: {updated_progress:.2f}")
    
    # Manually update the progress
    manager.set_goal_progress(ui_goal.id, updated_progress)
    
    # Update main goal progress based on sub-goals
    calculate_main_goal_progress(manager, entities["main_goal"], entities["sub_goals"])

def calculate_main_goal_progress(manager: GraphManager, main_goal: Goal, sub_goals: List[Goal]):
    """Calculate the main goal progress based on sub-goals."""
    # Get the progress of each sub-goal
    total_progress = 0.0
    
    for sub_goal in sub_goals:
        # Refresh from database to get current progress
        current = manager.get_goal_by_id(sub_goal.id)
        if current:
            total_progress += current.progress
    
    # Average progress across all sub-goals
    avg_progress = total_progress / len(sub_goals) if sub_goals else 0.0
    print(f"Calculated main goal progress: {avg_progress:.2f}")
    
    # Update the main goal
    manager.set_goal_progress(main_goal.id, avg_progress)
    print(f"Updated main goal progress to: {avg_progress:.2f}")

def create_solutions_and_resolve_problems(manager: GraphManager, entities: Dict[str, Any]):
    """Create solutions for problems and track resolution."""
    print("\n=== Creating Solutions for Problems ===")
    
    # Create solutions for our problems
    solutions = [
        "Implement API response caching for most frequent requests",
        "Use OAuth 2.0 with social login options for authentication",
        "Implement offline storage using SQLite with sync queue"
    ]
    
    for i, desc in enumerate(solutions):
        if i < len(entities["problems"]):
            problem = entities["problems"][i]
            
            # Create solution
            solution = manager.create_solution(
                description=desc,
                state=SolutionState.IN_PROGRESS
            )
            print(f"Created solution: {solution.id} - {solution.description}")
            
            # Link to problem
            manager.add_solution_to_matter(problem.id, solution.id)
            print(f"Added solution to problem: {problem.description}")
    
    # Check if any conditions for the problems are met
    # Mark some conditions as met
    if "dual_entities" in entities and entities["dual_entities"]:
        for entity in entities["dual_entities"]:
            # Mark as met
            manager.update_condition(entity.id, True)
            print(f"Marked condition as met: {entity.description}")
    
    # Check if problems are solved
    for problem in entities["problems"]:
        is_solved = manager.check_if_problem_solved(problem.id)
        print(f"Problem solved: {is_solved} - {problem.description}")
        
        # Update problem state if all conditions are met
        if is_solved:
            with manager.driver.session() as session:
                session.run(
                    """
                    MATCH (p:Problem {id: $id})
                    SET p.state = $state
                    """,
                    id=problem.id,
                    state=ProblemState.SOLVED.value
                )
                print(f"Updated problem state to SOLVED: {problem.description}")

def main():
    """Main test function."""
    print("=== TESTING PROBLEM-CONDITION DUALITY ===\n")
    
    # Create the manager
    manager = GraphManager(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="matters2025"
    )
    
    try:
        # Connect and initialize schema
        print("Connecting to Neo4j...")
        manager.connect()
        
        print("Initializing schema...")
        manager.initialize_schema()
        
        # Run the demonstrations
        entities = create_project_structure(manager)
        demonstrate_problem_condition_conversion(manager, entities)
        track_goal_progress_with_conditions(manager, entities)
        create_solutions_and_resolve_problems(manager, entities)
        
        print("\n=== TESTS COMPLETE ===")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Close the connection
        if manager.driver:
            manager.close()

if __name__ == "__main__":
    main()