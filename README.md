# matters.global

A Python package for managing problem definitions.

## Overview

`matters.global` is a tool for creating, storing, analyzing, and connecting problems, their conditions, and their solutions. It provides a structured approach to problem management with a focus on:

- Creating JSON problem definitions
- Storing JSON problem definitions
- Loading JSON problem definitions
- Computing connections between problems
- Determining if a statement is a problem
- Checking if similar problems already exist

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/matters.global.git
cd matters.global

# Set up the conda environment
conda create -n mattersglobal python=3.9
conda activate mattersglobal

# Install dependencies
pip install pydantic
```

## Problem Structure

Each problem is defined with the following structure:

```python
{
    "description": "A detailed description of the problem",
    "state": "solved | not_solved | obsolete",  # Initial state is not_solved
    "conditions": [
        {
            "description": "Detailed description of the condition",
            "is_met": False,  # Initial state, evaluates to True if condition is met
        },
    ],
    "solutions": [
        {
            "description": "Detailed description of the solution", # Added once the problem is solved
        },
    ]
}
```

## Usage Examples

### Creating and Managing Problems

```python
from mattersglobal import ProblemManager, Problem

# Create a problem manager
manager = ProblemManager()

# Create a new problem
problem = manager.create_problem("I cannot lift 80kgs in deadlift")

# Add conditions
problem.add_condition("I go train three times a week for 20 consecutive weeks")
problem.add_condition("I get stronger every week as measured by the kgs I can lift")

# Save problems to file
filepath = manager.save_problems()
print(f"Problems saved to {filepath}")

# Load problems from file
problems = manager.load_problems()
```

### Analyzing Problems

```python
# Check if a statement is a problem
statement = "The website loads too slowly on mobile devices"
is_problem = manager.is_statement_problem(statement)
print(f"Is '{statement}' a problem? {is_problem}")

# Find similar problems
similar = manager.find_similar_problem("I can't deadlift 80kg")
if similar:
    print(f"Similar problem found: {similar.description}")

# Compute connections between problems
connections = manager.compute_problem_connections()
for problem_desc, related in connections.items():
    print(f"Problem: {problem_desc}")
    for related_desc in related:
        print(f"  - Related to: {related_desc}")
```

## API Reference

### Problem Class

The core data model that represents a problem definition:

- `add_condition(description, is_met=False)`: Add a new condition
- `add_solution(description)`: Add a solution
- `update_condition(index, is_met)`: Update a condition's status
- `check_if_solved()`: Check if all conditions are met
- `mark_as_obsolete()`: Mark problem as obsolete

### ProblemManager Class

Manages collections of problems:

- `create_problem(description)`: Create a new problem
- `save_problems(filename)`: Save problems to JSON
- `load_problems(filename)`: Load problems from JSON
- `get_problem_by_description(description)`: Find problem by description
- `is_statement_problem(statement)`: Check if statement is a problem
- `find_similar_problem(description, threshold)`: Find similar existing problems
- `compute_problem_connections()`: Find connections between problems

## Future Enhancements

- Integration with AI models for better problem analysis
- Advanced similarity detection using embeddings
- Visualization of problem connections
- Collaborative problem solving features
- Web interface for problem management

## License

MIT
