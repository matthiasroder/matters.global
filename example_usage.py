#!/usr/bin/env python3
"""
Example usage of the matters.global package
This example demonstrates creating, updating, and analyzing problem definitions
"""

from mattersglobal import ProblemManager, Problem

def main():
    # Create a problem manager
    print("Creating problem manager...")
    manager = ProblemManager()
    
    # Create a new problem
    print("\nCreating a problem definition...")
    problem = manager.create_problem("I cannot lift 80kgs in deadlift")
    print(f"Created problem: {problem.description}")
    print(f"Current state: {problem.state}")
    
    # Add conditions
    print("\nAdding conditions to the problem...")
    problem.add_condition("I go train three times a week for 20 consecutive weeks")
    problem.add_condition("I get stronger every week as measured by the kgs I can lift")
    
    # Print all conditions
    print("\nProblem conditions:")
    for i, condition in enumerate(problem.conditions):
        print(f"  {i+1}. {condition.description} (Met: {condition.is_met})")
    
    # Check if problem is solved
    print("\nChecking if problem is solved...")
    is_solved = problem.check_if_solved()
    print(f"Is problem solved? {is_solved}")
    
    # Update conditions to indicate progress
    print("\nUpdating conditions as progress is made...")
    problem.update_condition(0, True)  # First condition is met
    print("First condition is now met")
    
    # Check if problem is solved (it shouldn't be yet)
    is_solved = problem.check_if_solved()
    print(f"Is problem solved now? {is_solved}")
    
    # Update the second condition
    print("\nUpdating the second condition...")
    problem.update_condition(1, True)  # Second condition is met
    print("Second condition is now met")
    
    # Check if problem is solved (it should be now)
    is_solved = problem.check_if_solved()
    print(f"Is problem solved now? {is_solved}")
    print(f"Problem state after updating: {problem.state}")
    
    # Add a solution
    print("\nAdding a solution to the solved problem...")
    problem.add_solution("Followed a progressive overload program with proper nutrition")
    
    # Print all solutions
    print("\nProblem solutions:")
    for i, solution in enumerate(problem.solutions):
        print(f"  {i+1}. {solution.description}")
    
    # Save the problem to a file
    print("\nSaving problem to file...")
    filepath = manager.save_problems()
    print(f"Problem saved to {filepath}")
    
    # Create a second problem to demonstrate connections
    print("\nCreating a second related problem...")
    problem2 = manager.create_problem("I cannot lift 100kgs in squat")
    problem2.add_condition("I train legs twice a week")
    
    # Save all problems
    print("\nSaving all problems to file...")
    manager.save_problems()
    
    # Clear the manager's problems and load from file
    print("\nLoading problems from file...")
    manager.problems = []
    problems = manager.load_problems()
    print(f"Loaded {len(problems)} problems")
    
    # Check if a statement is a problem
    print("\nAnalyzing if statements are problems...")
    statements = [
        "The website loads too slowly on mobile devices",
        "I enjoy running in the park",
        "The application crashes when clicking the save button"
    ]
    
    for statement in statements:
        is_problem = manager.is_statement_problem(statement)
        print(f"Is '{statement}' a problem? {is_problem}")
    
    # Find similar problems
    print("\nLooking for similar problems...")
    query = "I can't deadlift 80kg"
    similar = manager.find_similar_problem(query)
    if similar:
        print(f"Found similar problem: '{similar.description}'")
    else:
        print(f"No similar problem found for: '{query}'")
    
    # Compute and display problem connections
    print("\nComputing connections between problems...")
    connections = manager.compute_problem_connections()
    for problem_desc, related in connections.items():
        print(f"Problem: {problem_desc}")
        if related:
            for related_desc in related:
                print(f"  - Related to: {related_desc}")
        else:
            print("  - No related problems found")

if __name__ == "__main__":
    main()