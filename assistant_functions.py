"""
Assistant Functions for matters.global

This module provides function definitions and handlers for OpenAI Assistant integration.
It serves as a bridge between the OpenAI function calling format and our backend implementation.
"""

import json
from typing import List, Dict, Any, Optional, Union
import logging

# Import our backend components
from graph_problem_manager import GraphProblemManager, Problem, Condition, ProblemState
from entity_resolution import EntityResolutionSystem

# Setup logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize manager with default settings
# In production, these would come from environment variables or config
graph_manager = GraphProblemManager(
    uri="bolt://localhost:7687",
    username="neo4j",
    password="matters2025",
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
    {
        "name": "list_problems",
        "description": "Get a list of all problems or filter by criteria",
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
        "description": "Get detailed information about a specific problem",
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
                }
            },
            "required": ["description"]
        }
    },
    {
        "name": "add_condition_to_problem",
        "description": "Add a condition that must be met to solve the problem",
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
    {
        "name": "find_similar_problems",
        "description": "Find problems similar to a given description",
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
        "description": "Add a prerequisite relationship between problems",
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
]

# =====================================
# Function Handlers (Implementation)
# =====================================

def list_problems(state: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
    """Get a list of all problems or filter by criteria.
    
    Args:
        state: Optional filter for problem state
        limit: Maximum number of problems to return
        
    Returns:
        Dictionary with list of problems
    """
    try:
        # In a real implementation, this would query all problems
        # For now, we'll implement a simple version
        with graph_manager.driver.session() as session:
            query = "MATCH (p:Problem)"
            params = {}
            
            if state:
                query += " WHERE p.state = $state"
                params["state"] = state
                
            query += " RETURN p ORDER BY p.id"
            
            if limit:
                query += " LIMIT $limit"
                params["limit"] = limit
                
            result = session.run(query, **params)
            
            problems = []
            for record in result:
                node = record["p"]
                problems.append({
                    "id": node["id"],
                    "description": node["description"],
                    "state": node["state"]
                })
            
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
    """Get detailed information about a specific problem.
    
    Args:
        problem_id: ID of the problem to retrieve
        
    Returns:
        Dictionary with problem details and connections
    """
    try:
        # Get basic problem info
        problem = graph_manager.get_problem_by_id(problem_id)
        if not problem:
            return {
                "success": False,
                "error": f"Problem with ID {problem_id} not found"
            }
        
        # Get connections
        connections = graph_manager.find_problem_connections(problem_id)
        
        # Format response
        return {
            "success": True,
            "problem": {
                "id": problem.id,
                "description": problem.description,
                "state": problem.state
            },
            "connections": connections
        }
        
    except Exception as e:
        logger.error(f"Error getting problem details: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def create_problem(description: str, state: str = "not_solved") -> Dict[str, Any]:
    """Create a new problem.
    
    Args:
        description: Description of the problem
        state: Initial state of the problem
        
    Returns:
        Dictionary with created problem information
    """
    try:
        # Create problem instance with the given state
        problem = Problem(description=description, state=ProblemState(state))
        
        # Create in database
        problem = graph_manager.create_problem(description)
        
        # Update state if different from default
        if state != "not_solved":
            with graph_manager.driver.session() as session:
                session.run(
                    "MATCH (p:Problem {id: $id}) SET p.state = $state",
                    id=problem.id,
                    state=state
                )
        
        return {
            "success": True,
            "problem": {
                "id": problem.id,
                "description": problem.description,
                "state": problem.state
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating problem: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def add_condition_to_problem(problem_id: str, condition_description: str, is_met: bool = False) -> Dict[str, Any]:
    """Add a condition to a problem.
    
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
            "condition": {
                "id": condition.id,
                "description": condition.description,
                "is_met": condition.is_met
            },
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
        
        # Also check if this updates the problem's solved status
        with graph_manager.driver.session() as session:
            # Find the problem this condition belongs to
            result = session.run(
                """
                MATCH (p:Problem)-[:REQUIRES]->(c:Condition {id: $condition_id})
                RETURN p.id as problem_id
                """,
                condition_id=condition_id
            )
            
            record = result.single()
            problem_solved = False
            problem_id = None
            
            if record:
                problem_id = record["problem_id"]
                problem_solved = graph_manager.check_if_problem_solved(problem_id)
        
        return {
            "success": True,
            "condition_id": condition_id,
            "is_met": is_met,
            "problem_id": problem_id,
            "problem_solved": problem_solved
        }
        
    except Exception as e:
        logger.error(f"Error updating condition: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def find_similar_problems(description: str, threshold: float = 0.7, limit: int = 5) -> Dict[str, Any]:
    """Find problems similar to the given description.
    
    Args:
        description: Description to compare against
        threshold: Similarity threshold (0-1)
        limit: Maximum number of results
        
    Returns:
        Dictionary with similar problems
    """
    try:
        similar_problems = graph_manager.find_similar_problems(
            description=description,
            threshold=threshold,
            limit=limit
        )
        
        return {
            "success": True,
            "similar_problems": similar_problems,
            "count": len(similar_problems)
        }
        
    except Exception as e:
        logger.error(f"Error finding similar problems: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def add_problem_dependency(problem_id: str, depends_on_id: str) -> Dict[str, Any]:
    """Add a dependency between problems.

    Note: This function maintains the old parameter names and semantics,
    but internally uses the new relationship model where the direction is reversed.

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

        # If both problems exist, attempt to create the relationship
        # Note: In the new model, the relationship direction is reversed
        # "A must be resolved before B" instead of "B depends on A"
        logger.info(f"Creating relationship: '{prerequisite.description}' must be resolved before '{dependent.description}'")

        # Use the new method with parameters in correct order
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
    "list_problems": list_problems,
    "get_problem_details": get_problem_details,
    "create_problem": create_problem,
    "add_condition_to_problem": add_condition_to_problem,
    "update_condition": update_condition,
    "find_similar_problems": find_similar_problems,
    "add_problem_dependency": add_problem_dependency
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
