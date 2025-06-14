"""
Assistant Functions for matters.global

This module provides function definitions and handlers for OpenAI Assistant integration.
It serves as a bridge between the OpenAI function calling format and our backend implementation.

This module supports the multi-label schema where entities can have multiple types:
- Matter (base type for all entities)
- Goal (entity representing a desired outcome)
- Problem (entity representing an issue to be resolved)
- Condition (entity representing a requirement)
- Solution (entity representing a way to solve a problem or achieve a goal)
"""

import json
import os
from typing import List, Dict, Any, Optional, Union
import logging

# Import our backend components with the new multi-label schema
from graph_problem_manager import GraphManager, Matter, Goal, Problem, Condition, Solution, MatterLabel, ProblemState, SolutionState
from entity_resolution import EntityResolutionSystem

# Setup logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize manager with environment variables for Railway deployment
graph_manager = GraphManager(
    uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
    username=os.environ.get("NEO4J_USERNAME", "neo4j"),
    password=os.environ.get("NEO4J_PASSWORD", "matters2025"),
    embedding_config_path="config/embeddings.json"
)

# Connect to the database
try:
    graph_manager.connect()
    graph_manager.initialize_schema()
    logger.info("Connected to graph database successfully")
except Exception as e:
    logger.error(f"Failed to connect to graph database: {str(e)}")

# Initialize entity resolution system
entity_resolver = EntityResolutionSystem(graph_manager)

# =====================================
# Function Definitions (OpenAI Schema)
# =====================================

FUNCTION_DEFINITIONS = [
    # ---- Matter (General Entity) Functions ----
    {
        "name": "list_matters",
        "description": "Get a list of all matters (goals, problems, conditions, solutions) with optional filtering by labels",
        "parameters": {
            "type": "object",
            "properties": {
                "labels": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["Goal", "Problem", "Condition", "Solution"]
                    },
                    "description": "Filter matters by their labels (types)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of matters to return"
                }
            },
            "required": []
        }
    },
    {
        "name": "get_matter_details",
        "description": "Get detailed information about a specific matter",
        "parameters": {
            "type": "object",
            "properties": {
                "matter_id": {
                    "type": "string",
                    "description": "The ID of the matter to retrieve details for"
                }
            },
            "required": ["matter_id"]
        }
    },
    {
        "name": "find_similar_matters",
        "description": "Find matters similar to a given description with optional label filtering",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Description to compare against existing matters"
                },
                "labels": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["Goal", "Problem", "Condition", "Solution"]
                    },
                    "description": "Filter by matter labels (types)"
                },
                "threshold": {
                    "type": "number",
                    "description": "Similarity threshold (0-1)",
                    "default": 0.7
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5
                }
            },
            "required": ["description"]
        }
    },
    {
        "name": "add_relationship",
        "description": "Add a relationship between two matters",
        "parameters": {
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "description": "ID of the source matter"
                },
                "target_id": {
                    "type": "string",
                    "description": "ID of the target matter"
                },
                "relationship_type": {
                    "type": "string",
                    "enum": ["REQUIRES", "BLOCKS", "ENABLES", "RELATES_TO", "PRECEDES",
                             "FOLLOWS", "PART_OF", "CONSISTS_OF", "SOLVED_BY", "ADDRESSES",
                             "FULFILLS", "MAPPED_TO", "DERIVED_FROM"],
                    "description": "Type of relationship to create between the matters"
                }
            },
            "required": ["source_id", "target_id", "relationship_type"]
        }
    },

    # ---- Goal Functions ----
    {
        "name": "create_goal",
        "description": "Create a new goal with description and optional target date",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "A detailed description of the goal"
                },
                "target_date": {
                    "type": "string",
                    "format": "date-time",
                    "description": "Optional target date for achieving the goal (ISO format)"
                },
                "progress": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Optional initial progress value (0-1)",
                    "default": 0
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for categorization"
                }
            },
            "required": ["description"]
        }
    },
    {
        "name": "set_goal_progress",
        "description": "Update a goal's progress value",
        "parameters": {
            "type": "object",
            "properties": {
                "goal_id": {
                    "type": "string",
                    "description": "The ID of the goal to update"
                },
                "progress": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "New progress value (0-1)"
                }
            },
            "required": ["goal_id", "progress"]
        }
    },

    # ---- Problem Functions (Legacy & New) ----
    {
        "name": "list_problems",
        "description": "Get a list of all problems or filter by criteria (legacy function)",
        "parameters": {
            "type": "object",
            "properties": {
                "state": {
                    "type": "string",
                    "enum": ["solved", "not_solved", "obsolete"],
                    "description": "Filter problems by their state"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of problems to return"
                }
            },
            "required": []
        }
    },
    {
        "name": "get_problem_details",
        "description": "Get detailed information about a specific problem (legacy function)",
        "parameters": {
            "type": "object",
            "properties": {
                "problem_id": {
                    "type": "string",
                    "description": "The ID of the problem to retrieve details for"
                }
            },
            "required": ["problem_id"]
        }
    },
    {
        "name": "create_problem",
        "description": "Create a new problem with description and optional state",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "A detailed description of the problem"
                },
                "state": {
                    "type": "string",
                    "enum": ["solved", "not_solved", "obsolete"],
                    "description": "The initial state of the problem",
                    "default": "not_solved"
                },
                "priority": {
                    "type": "integer",
                    "description": "Optional priority ranking (lower = higher priority)"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for categorization"
                }
            },
            "required": ["description"]
        }
    },
    {
        "name": "find_similar_problems",
        "description": "Find problems similar to a given description (legacy function)",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Description to compare against existing problems"
                },
                "threshold": {
                    "type": "number",
                    "description": "Similarity threshold (0-1)",
                    "default": 0.7
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5
                }
            },
            "required": ["description"]
        }
    },
    {
        "name": "add_problem_dependency",
        "description": "Add a prerequisite relationship between problems (legacy function)",
        "parameters": {
            "type": "object",
            "properties": {
                "problem_id": {
                    "type": "string",
                    "description": "ID of the dependent problem (that can only be solved after prerequisite)"
                },
                "depends_on_id": {
                    "type": "string",
                    "description": "ID of the prerequisite problem (that must be resolved before dependent problem)"
                }
            },
            "required": ["problem_id", "depends_on_id"]
        }
    },

    # ---- Condition Functions ----
    {
        "name": "add_condition_to_problem",
        "description": "Add a condition that must be met to solve the problem (legacy function)",
        "parameters": {
            "type": "object",
            "properties": {
                "problem_id": {
                    "type": "string",
                    "description": "The ID of the problem to add a condition to"
                },
                "condition_description": {
                    "type": "string",
                    "description": "A detailed description of the condition"
                },
                "is_met": {
                    "type": "boolean",
                    "description": "Whether the condition is already met",
                    "default": False
                }
            },
            "required": ["problem_id", "condition_description"]
        }
    },
    {
        "name": "create_condition",
        "description": "Create a new condition",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "A detailed description of the condition"
                },
                "is_met": {
                    "type": "boolean",
                    "description": "Whether the condition is already met",
                    "default": False
                },
                "verification_method": {
                    "type": "string",
                    "description": "Optional description of how to verify this condition"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for categorization"
                }
            },
            "required": ["description"]
        }
    },
    {
        "name": "update_condition",
        "description": "Update a condition's status (met or not met)",
        "parameters": {
            "type": "object",
            "properties": {
                "condition_id": {
                    "type": "string",
                    "description": "The ID of the condition to update"
                },
                "is_met": {
                    "type": "boolean",
                    "description": "The new status of the condition (true = met, false = not met)"
                }
            },
            "required": ["condition_id", "is_met"]
        }
    },

    # ---- Solution Functions ----
    {
        "name": "create_solution",
        "description": "Create a new solution",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "A detailed description of the solution"
                },
                "state": {
                    "type": "string",
                    "enum": ["theoretical", "in_progress", "implemented", "failed"],
                    "description": "The state of the solution implementation",
                    "default": "theoretical"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for categorization"
                }
            },
            "required": ["description"]
        }
    },
    {
        "name": "add_solution_to_matter",
        "description": "Add a solution to a matter (problem or goal)",
        "parameters": {
            "type": "object",
            "properties": {
                "matter_id": {
                    "type": "string",
                    "description": "The ID of the matter (problem or goal) the solution addresses"
                },
                "solution_id": {
                    "type": "string",
                    "description": "The ID of the solution"
                }
            },
            "required": ["matter_id", "solution_id"]
        }
    },
]

# =====================================
# Function Handlers (Implementation)
# =====================================

# ---- Matter (General Entity) Functions ----

def list_matters(labels: Optional[List[str]] = None, limit: Optional[int] = None) -> Dict[str, Any]:
    """Get a list of all matters with optional filtering by labels.

    Args:
        labels: Optional list of labels to filter by (e.g., ["Goal", "Problem"])
        limit: Maximum number of matters to return

    Returns:
        Dictionary with list of matters
    """
    try:
        with graph_manager.driver.session() as session:
            # Start with the base Matter label
            query = "MATCH (m:Matter)"
            params = {}

            # Add label filters if specified
            if labels and len(labels) > 0:
                label_clauses = []
                for i, label in enumerate(labels):
                    label_param = f"label_{i}"
                    label_clauses.append(f"m:{{{label_param}}}")
                    params[label_param] = label

                query += " WHERE " + " AND ".join(label_clauses)

            query += " RETURN m, labels(m) as node_labels ORDER BY m.created_at DESC"

            if limit:
                query += " LIMIT $limit"
                params["limit"] = limit

            result = session.run(query, **params)

            matters = []
            for record in result:
                node = record["m"]
                node_labels = record["node_labels"]

                # Create a base matter representation
                matter_data = {
                    "id": node["id"],
                    "description": node["description"],
                    "labels": node_labels,
                    "created_at": node.get("created_at"),
                    "updated_at": node.get("updated_at")
                }

                # Add label-specific properties
                if MatterLabel.PROBLEM.value in node_labels and "state" in node:
                    matter_data["state"] = node["state"]

                if MatterLabel.GOAL.value in node_labels:
                    if "target_date" in node:
                        matter_data["target_date"] = node["target_date"]
                    if "progress" in node:
                        matter_data["progress"] = node["progress"]

                if MatterLabel.CONDITION.value in node_labels and "is_met" in node:
                    matter_data["is_met"] = node["is_met"]

                if MatterLabel.SOLUTION.value in node_labels and "state" in node:
                    matter_data["state"] = node["state"]

                if "tags" in node:
                    matter_data["tags"] = node["tags"]

                matters.append(matter_data)

            return {
                "success": True,
                "matters": matters,
                "count": len(matters)
            }

    except Exception as e:
        logger.error(f"Error listing matters: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def get_matter_details(matter_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific matter.

    Args:
        matter_id: ID of the matter to retrieve

    Returns:
        Dictionary with matter details and connections
    """
    try:
        # Get the matter by ID
        matter = graph_manager.get_matter_by_id(matter_id)
        if not matter:
            return {
                "success": False,
                "error": f"Matter with ID {matter_id} not found"
            }

        # Get connections
        connections = graph_manager.find_matter_connections(matter_id)

        # Convert the matter object to a dictionary
        matter_data = matter.dict()

        # Format response
        return {
            "success": True,
            "matter": matter_data,
            "connections": connections
        }

    except Exception as e:
        logger.error(f"Error getting matter details: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def find_similar_matters(description: str, labels: Optional[List[str]] = None,
                       threshold: float = 0.7, limit: int = 5) -> Dict[str, Any]:
    """Find matters similar to the given description.

    Args:
        description: Description to compare against
        labels: Optional list of labels to filter by
        threshold: Similarity threshold (0-1)
        limit: Maximum number of results

    Returns:
        Dictionary with similar matters
    """
    try:
        similar_matters = graph_manager.find_similar_matters(
            description=description,
            labels=labels,
            threshold=threshold,
            limit=limit
        )

        return {
            "success": True,
            "similar_matters": similar_matters,
            "count": len(similar_matters)
        }

    except Exception as e:
        logger.error(f"Error finding similar matters: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def add_relationship(source_id: str, target_id: str, relationship_type: str) -> Dict[str, Any]:
    """Add a relationship between two matters.

    Args:
        source_id: ID of the source matter
        target_id: ID of the target matter
        relationship_type: Type of relationship

    Returns:
        Dictionary with result information
    """
    try:
        # Get both matters to verify they exist
        source = graph_manager.get_matter_by_id(source_id)
        target = graph_manager.get_matter_by_id(target_id)

        if not source:
            return {
                "success": False,
                "error": f"Source matter with ID {source_id} not found"
            }

        if not target:
            return {
                "success": False,
                "error": f"Target matter with ID {target_id} not found"
            }

        # Create the relationship
        success = graph_manager.set_matter_relationship(
            source_id=source_id,
            relationship_type=relationship_type,
            target_id=target_id
        )

        if not success:
            return {
                "success": False,
                "error": "Failed to create relationship in database"
            }

        return {
            "success": True,
            "source_id": source_id,
            "source_description": source.description,
            "target_id": target_id,
            "target_description": target.description,
            "relationship_type": relationship_type,
            "message": f"Successfully created {relationship_type} relationship"
        }

    except Exception as e:
        logger.error(f"Error adding relationship: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# ---- Goal Functions ----

def create_goal(description: str, target_date: Optional[str] = None,
              progress: float = 0.0, tags: Optional[List[str]] = None) -> Dict[str, Any]:
    """Create a new goal.

    Args:
        description: Description of the goal
        target_date: Optional target date (ISO format)
        progress: Initial progress (0-1)
        tags: Optional tags for categorization

    Returns:
        Dictionary with created goal information
    """
    try:
        # Convert target_date string to datetime if provided
        parsed_target_date = None
        if target_date:
            from datetime import datetime
            parsed_target_date = datetime.fromisoformat(target_date)

        # Create the goal
        goal = graph_manager.create_goal(
            description=description,
            target_date=parsed_target_date,
            progress=progress,
            tags=tags or []
        )

        return {
            "success": True,
            "goal": goal.dict()
        }

    except Exception as e:
        logger.error(f"Error creating goal: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def set_goal_progress(goal_id: str, progress: float) -> Dict[str, Any]:
    """Update a goal's progress value.

    Args:
        goal_id: ID of the goal
        progress: New progress value (0-1)

    Returns:
        Dictionary with result information
    """
    try:
        # First check if the goal exists
        goal = graph_manager.get_goal_by_id(goal_id)
        if not goal:
            return {
                "success": False,
                "error": f"Goal with ID {goal_id} not found"
            }

        # Update the progress
        success = graph_manager.set_goal_progress(goal_id, progress)

        if not success:
            return {
                "success": False,
                "error": f"Failed to update progress for goal {goal_id}"
            }

        return {
            "success": True,
            "goal_id": goal_id,
            "progress": progress,
            "message": f"Successfully updated progress for goal: {goal.description}"
        }

    except Exception as e:
        logger.error(f"Error updating goal progress: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# ---- Problem Functions (Legacy & New) ----

def list_problems(state: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
    """Get a list of all problems or filter by criteria (legacy function).

    Args:
        state: Optional filter for problem state
        limit: Maximum number of problems to return

    Returns:
        Dictionary with list of problems
    """
    try:
        # Use the more general list_matters with Problem label
        labels = [MatterLabel.PROBLEM.value]
        result = list_matters(labels=labels, limit=limit)

        if not result["success"]:
            return result

        # Filter by state if needed
        problems = result["matters"]
        if state:
            problems = [p for p in problems if p.get("state") == state]

        return {
            "success": True,
            "problems": problems,
            "count": len(problems)
        }

    except Exception as e:
        logger.error(f"Error listing problems: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def get_problem_details(problem_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific problem (legacy function).

    Args:
        problem_id: ID of the problem to retrieve

    Returns:
        Dictionary with problem details and connections
    """
    try:
        # Get the matter with the Problem label
        problem = graph_manager.get_problem_by_id(problem_id)
        if not problem:
            return {
                "success": False,
                "error": f"Problem with ID {problem_id} not found"
            }

        # Get connections using the new multi-label approach
        connections = graph_manager.find_matter_connections(problem_id)

        # Format response
        return {
            "success": True,
            "problem": problem.dict(),
            "connections": connections
        }

    except Exception as e:
        logger.error(f"Error getting problem details: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def create_problem(description: str, state: str = "not_solved",
                  priority: Optional[int] = None,
                  tags: Optional[List[str]] = None) -> Dict[str, Any]:
    """Create a new problem.

    Args:
        description: Description of the problem
        state: Initial state of the problem
        priority: Optional priority ranking (lower = higher priority)
        tags: Optional tags for categorization

    Returns:
        Dictionary with created problem information
    """
    try:
        # Create the problem
        problem = graph_manager.create_problem(
            description=description,
            state=ProblemState(state),
            priority=priority,
            tags=tags or []
        )

        return {
            "success": True,
            "problem": problem.dict()
        }

    except Exception as e:
        logger.error(f"Error creating problem: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def create_condition(description: str, is_met: bool = False,
                   verification_method: Optional[str] = None,
                   tags: Optional[List[str]] = None) -> Dict[str, Any]:
    """Create a new condition.

    Args:
        description: Description of the condition
        is_met: Whether the condition is already met
        verification_method: Method to verify if condition is met
        tags: Optional tags for categorization

    Returns:
        Dictionary with created condition information
    """
    try:
        # Create the condition
        condition = graph_manager.create_condition(
            description=description,
            is_met=is_met,
            verification_method=verification_method,
            tags=tags or []
        )

        return {
            "success": True,
            "condition": condition.dict()
        }

    except Exception as e:
        logger.error(f"Error creating condition: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def add_condition_to_problem(problem_id: str, condition_description: str, is_met: bool = False) -> Dict[str, Any]:
    """Add a condition to a problem (legacy function).

    Args:
        problem_id: ID of the problem
        condition_description: Description of the condition
        is_met: Whether the condition is already met

    Returns:
        Dictionary with created condition information
    """
    try:
        condition = graph_manager.add_condition_to_problem(
            problem_id=problem_id,
            condition_description=condition_description,
            is_met=is_met
        )

        if not condition:
            return {
                "success": False,
                "error": f"Failed to add condition. Problem with ID {problem_id} not found."
            }

        return {
            "success": True,
            "condition": condition.dict(),
            "problem_id": problem_id
        }

    except Exception as e:
        logger.error(f"Error adding condition: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def update_condition(condition_id: str, is_met: bool) -> Dict[str, Any]:
    """Update a condition's status.

    Args:
        condition_id: ID of the condition
        is_met: New status of the condition

    Returns:
        Dictionary with result information
    """
    try:
        success = graph_manager.update_condition(condition_id, is_met)

        if not success:
            return {
                "success": False,
                "error": f"Condition with ID {condition_id} not found"
            }

        # Get the condition to see its description
        condition = graph_manager.get_condition_by_id(condition_id)

        # Find matters that require this condition
        with graph_manager.driver.session() as session:
            # Find problems this condition belongs to (with the multi-label schema)
            result = session.run(
                """
                MATCH (m:Matter:Problem)-[:REQUIRES]->(c:Matter:Condition {id: $condition_id})
                RETURN m.id as problem_id, m.description as problem_description
                """,
                condition_id=condition_id
            )

            problems = []
            for record in result:
                problem_id = record["problem_id"]
                problem_description = record["problem_description"]
                problem_solved = graph_manager.check_if_problem_solved(problem_id)

                problems.append({
                    "id": problem_id,
                    "description": problem_description,
                    "solved": problem_solved
                })

        return {
            "success": True,
            "condition_id": condition_id,
            "description": condition.description if condition else None,
            "is_met": is_met,
            "affected_problems": problems
        }

    except Exception as e:
        logger.error(f"Error updating condition: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def create_solution(description: str, state: str = "theoretical",
                   tags: Optional[List[str]] = None) -> Dict[str, Any]:
    """Create a new solution.

    Args:
        description: Description of the solution
        state: Initial state of the solution
        tags: Optional tags for categorization

    Returns:
        Dictionary with created solution information
    """
    try:
        # Create the solution
        solution = graph_manager.create_solution(
            description=description,
            state=SolutionState(state),
            tags=tags or []
        )

        return {
            "success": True,
            "solution": solution.dict()
        }

    except Exception as e:
        logger.error(f"Error creating solution: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def add_solution_to_matter(matter_id: str, solution_id: str) -> Dict[str, Any]:
    """Add a solution to a matter (problem or goal).

    Args:
        matter_id: ID of the matter (problem or goal)
        solution_id: ID of the solution

    Returns:
        Dictionary with result information
    """
    try:
        # Check if both matters exist
        matter = graph_manager.get_matter_by_id(matter_id)
        solution = graph_manager.get_solution_by_id(solution_id)

        if not matter:
            return {
                "success": False,
                "error": f"Matter with ID {matter_id} not found"
            }

        if not solution:
            return {
                "success": False,
                "error": f"Solution with ID {solution_id} not found"
            }

        # Create the relationship
        success = graph_manager.add_solution_to_matter(matter_id, solution_id)

        if not success:
            return {
                "success": False,
                "error": f"Failed to add solution to matter"
            }

        return {
            "success": True,
            "matter_id": matter_id,
            "matter_description": matter.description,
            "solution_id": solution_id,
            "solution_description": solution.description
        }

    except Exception as e:
        logger.error(f"Error adding solution to matter: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def find_similar_problems(description: str, threshold: float = 0.7, limit: int = 5) -> Dict[str, Any]:
    """Find problems similar to the given description (legacy function).

    Args:
        description: Description to compare against
        threshold: Similarity threshold (0-1)
        limit: Maximum number of results

    Returns:
        Dictionary with similar problems
    """
    try:
        # Use the more general find_similar_matters with Problem label
        result = find_similar_matters(
            description=description,
            labels=[MatterLabel.PROBLEM.value],
            threshold=threshold,
            limit=limit
        )

        if not result["success"]:
            return result

        # Rename key for backward compatibility
        result["similar_problems"] = result.pop("similar_matters")

        return result

    except Exception as e:
        logger.error(f"Error finding similar problems: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def add_problem_dependency(problem_id: str, depends_on_id: str) -> Dict[str, Any]:
    """Add a dependency between problems (legacy function).

    Note: This function maintains the old parameter names and semantics,
    but internally uses the new relationship model.

    Args:
        problem_id: ID of the dependent problem
        depends_on_id: ID of the problem being depended on (must be resolved first)

    Returns:
        Dictionary with result status
    """
    try:
        # First verify both problems exist
        dependent = graph_manager.get_problem_by_id(problem_id)
        prerequisite = graph_manager.get_problem_by_id(depends_on_id)

        if not dependent:
            logger.error(f"Dependent problem not found with ID: {problem_id}")
            return {
                "success": False,
                "error": f"Dependent problem with ID {problem_id} not found in database"
            }

        if not prerequisite:
            logger.error(f"Prerequisite problem not found with ID: {depends_on_id}")
            return {
                "success": False,
                "error": f"Prerequisite problem with ID {depends_on_id} not found in database"
            }

        # Create both PRECEDES and legacy MUST_BE_RESOLVED_BEFORE relationships
        success = graph_manager.add_resolution_prerequisite(depends_on_id, problem_id)

        if not success:
            logger.error(f"Failed to add resolution prerequisite despite problems existing")
            return {
                "success": False,
                "error": "Failed to create relationship in database"
            }

        return {
            "success": True,
            "problem_id": problem_id,
            "problem_description": dependent.description,
            "depends_on_id": depends_on_id,
            "depends_on_description": prerequisite.description,
            "message": f"Successfully created relationship: '{prerequisite.description}' must be resolved before '{dependent.description}'"
        }

    except Exception as e:
        logger.error(f"Error adding dependency: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# Function dispatcher to map function names to implementations

FUNCTION_DISPATCH = {
    # Matter (General Entity) Functions
    "list_matters": list_matters,
    "get_matter_details": get_matter_details,
    "find_similar_matters": find_similar_matters,
    "add_relationship": add_relationship,

    # Goal Functions
    "create_goal": create_goal,
    "set_goal_progress": set_goal_progress,

    # Problem Functions (Legacy & New)
    "list_problems": list_problems,
    "get_problem_details": get_problem_details,
    "create_problem": create_problem,
    "find_similar_problems": find_similar_problems,
    "add_problem_dependency": add_problem_dependency,

    # Condition Functions
    "create_condition": create_condition,
    "add_condition_to_problem": add_condition_to_problem,
    "update_condition": update_condition,

    # Solution Functions
    "create_solution": create_solution,
    "add_solution_to_matter": add_solution_to_matter
}

def dispatch_function(function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch a function call to the appropriate handler.

    Args:
        function_name: Name of the function to call
        arguments: Arguments to pass to the function

    Returns:
        Results from the function
    """
    logger.info(f"Dispatching function: {function_name} with arguments: {arguments}")

    if function_name not in FUNCTION_DISPATCH:
        logger.error(f"Unknown function requested: {function_name}")
        return {
            "success": False,
            "error": f"Unknown function: {function_name}"
        }

    handler = FUNCTION_DISPATCH[function_name]
    logger.info(f"Calling handler for function: {function_name}")

    try:
        result = handler(**arguments)
        logger.info(f"Function {function_name} completed with result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error executing function {function_name}: {str(e)}")
        return {
            "success": False,
            "error": f"Error executing {function_name}: {str(e)}"
        }

# Function to get all function definitions for OpenAI
def get_function_definitions() -> List[Dict[str, Any]]:
    """Get all function definitions for OpenAI Assistant.
    
    Returns:
        List of function definitions
    """
    return FUNCTION_DEFINITIONS
