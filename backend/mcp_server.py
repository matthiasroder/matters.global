"""MCP Server for matters.global knowledge graph."""

import sys, os, json, logging
from pathlib import Path
from typing import Optional, List

# Load .env before module-level env reads
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# Set cwd so config/embeddings.json resolves correctly
os.chdir(Path(__file__).parent)

from mcp.server.fastmcp import FastMCP
from assistant_functions import dispatch_function

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "matters-graph",
    instructions="Knowledge graph tools for matters.global. "
                 "Query and manage goals, problems, conditions, solutions, "
                 "and their relationships in a Neo4j graph database."
)


# ---- Matter (General Entity) Tools ----

@mcp.tool()
def list_matters(labels: Optional[List[str]] = None, limit: Optional[int] = None) -> str:
    """List all matters with optional label filtering (Goal, Problem, Condition, Solution)."""
    args = {}
    if labels is not None:
        args["labels"] = labels
    if limit is not None:
        args["limit"] = limit
    return json.dumps(dispatch_function("list_matters", args), default=str)


@mcp.tool()
def get_matter_details(matter_id: str) -> str:
    """Get detailed information about a specific matter including its connections."""
    return json.dumps(dispatch_function("get_matter_details", {"matter_id": matter_id}), default=str)


@mcp.tool()
def find_similar_matters(description: str, labels: Optional[List[str]] = None,
                         threshold: Optional[float] = None, limit: Optional[int] = None) -> str:
    """Find matters similar to a given description using vector similarity search."""
    args = {"description": description}
    if labels is not None:
        args["labels"] = labels
    if threshold is not None:
        args["threshold"] = threshold
    if limit is not None:
        args["limit"] = limit
    return json.dumps(dispatch_function("find_similar_matters", args), default=str)


@mcp.tool()
def add_relationship(source_id: str, target_id: str, relationship_type: str) -> str:
    """Add a relationship between two matters.

    Relationship types: REQUIRES, BLOCKS, ENABLES, RELATES_TO, PRECEDES,
    FOLLOWS, PART_OF, CONSISTS_OF, SOLVED_BY, ADDRESSES, FULFILLS,
    MAPPED_TO, DERIVED_FROM.
    """
    return json.dumps(dispatch_function("add_relationship", {
        "source_id": source_id,
        "target_id": target_id,
        "relationship_type": relationship_type,
    }), default=str)


@mcp.tool()
def remove_relationship(source_id: str, target_id: str, relationship_type: str) -> str:
    """Remove a relationship between two matters.

    Relationship types: REQUIRES, BLOCKS, ENABLES, RELATES_TO, PRECEDES,
    FOLLOWS, PART_OF, CONSISTS_OF, SOLVED_BY, ADDRESSES, FULFILLS,
    MAPPED_TO, DERIVED_FROM.
    """
    return json.dumps(dispatch_function("remove_relationship", {
        "source_id": source_id,
        "target_id": target_id,
        "relationship_type": relationship_type,
    }), default=str)


@mcp.tool()
def visualize_graph() -> str:
    """Generate an interactive HTML visualization of the entire matters graph. Returns the file path to open in a browser."""
    return json.dumps(dispatch_function("visualize_graph", {}), default=str)


@mcp.tool()
def delete_matter(matter_id: str) -> str:
    """Delete a matter (goal, problem, condition, or solution) and all its relationships."""
    return json.dumps(dispatch_function("delete_matter", {"matter_id": matter_id}), default=str)


# ---- Goal Tools ----

@mcp.tool()
def create_goal(description: str, target_date: Optional[str] = None,
                progress: float = 0.0, tags: Optional[List[str]] = None) -> str:
    """Create a new goal with description and optional target date."""
    args = {"description": description, "progress": progress}
    if target_date is not None:
        args["target_date"] = target_date
    if tags is not None:
        args["tags"] = tags
    return json.dumps(dispatch_function("create_goal", args), default=str)


@mcp.tool()
def set_goal_progress(goal_id: str, progress: float) -> str:
    """Update a goal's progress value (0-1)."""
    return json.dumps(dispatch_function("set_goal_progress", {
        "goal_id": goal_id,
        "progress": progress,
    }), default=str)


# ---- Problem Tools ----

@mcp.tool()
def list_problems(state: Optional[str] = None, limit: Optional[int] = None) -> str:
    """List problems with optional state filter (solved, not_solved, obsolete)."""
    args = {}
    if state is not None:
        args["state"] = state
    if limit is not None:
        args["limit"] = limit
    return json.dumps(dispatch_function("list_problems", args), default=str)


@mcp.tool()
def get_problem_details(problem_id: str) -> str:
    """Get detailed information about a specific problem."""
    return json.dumps(dispatch_function("get_problem_details", {"problem_id": problem_id}), default=str)


@mcp.tool()
def create_problem(description: str, state: Optional[str] = None,
                   priority: Optional[int] = None, tags: Optional[List[str]] = None) -> str:
    """Create a new problem with description and optional state/priority."""
    args = {"description": description}
    if state is not None:
        args["state"] = state
    if priority is not None:
        args["priority"] = priority
    if tags is not None:
        args["tags"] = tags
    return json.dumps(dispatch_function("create_problem", args), default=str)


@mcp.tool()
def find_similar_problems(description: str, threshold: Optional[float] = None,
                          limit: Optional[int] = None) -> str:
    """Find problems similar to a given description using vector similarity search."""
    args = {"description": description}
    if threshold is not None:
        args["threshold"] = threshold
    if limit is not None:
        args["limit"] = limit
    return json.dumps(dispatch_function("find_similar_problems", args), default=str)


@mcp.tool()
def add_problem_dependency(problem_id: str, depends_on_id: str) -> str:
    """Add a prerequisite relationship between problems (depends_on must be resolved first)."""
    return json.dumps(dispatch_function("add_problem_dependency", {
        "problem_id": problem_id,
        "depends_on_id": depends_on_id,
    }), default=str)


# ---- Condition Tools ----

@mcp.tool()
def create_condition(description: str, is_met: bool = False,
                     verification_method: Optional[str] = None,
                     tags: Optional[List[str]] = None) -> str:
    """Create a new condition with description and optional verification method."""
    args = {"description": description, "is_met": is_met}
    if verification_method is not None:
        args["verification_method"] = verification_method
    if tags is not None:
        args["tags"] = tags
    return json.dumps(dispatch_function("create_condition", args), default=str)


@mcp.tool()
def add_condition_to_problem(problem_id: str, condition_description: str,
                             is_met: bool = False) -> str:
    """Add a condition that must be met to solve a problem."""
    return json.dumps(dispatch_function("add_condition_to_problem", {
        "problem_id": problem_id,
        "condition_description": condition_description,
        "is_met": is_met,
    }), default=str)


@mcp.tool()
def update_condition(condition_id: str, is_met: bool) -> str:
    """Update a condition's status (met or not met)."""
    return json.dumps(dispatch_function("update_condition", {
        "condition_id": condition_id,
        "is_met": is_met,
    }), default=str)


# ---- Solution Tools ----

@mcp.tool()
def create_solution(description: str, state: Optional[str] = None,
                    tags: Optional[List[str]] = None) -> str:
    """Create a new solution (states: theoretical, in_progress, implemented, failed)."""
    args = {"description": description}
    if state is not None:
        args["state"] = state
    if tags is not None:
        args["tags"] = tags
    return json.dumps(dispatch_function("create_solution", args), default=str)


@mcp.tool()
def add_solution_to_matter(matter_id: str, solution_id: str) -> str:
    """Link a solution to a matter (problem or goal)."""
    return json.dumps(dispatch_function("add_solution_to_matter", {
        "matter_id": matter_id,
        "solution_id": solution_id,
    }), default=str)


if __name__ == "__main__":
    mcp.run(transport="stdio")
