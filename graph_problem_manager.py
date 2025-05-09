"""
matters.global graph-based problem manager

This module provides a Neo4j-based implementation for:
- Storing problems in a graph database
- Computing connections between problems
- Finding similar problems using vector embeddings
- Managing problem canonical forms and relationships
"""

from typing import List, Dict, Any, Optional, Set, Tuple, Union
from enum import Enum
import uuid
import json
from pathlib import Path
import logging
from pydantic import BaseModel, Field
from neo4j import GraphDatabase, Driver, Session, Transaction, Result

from embedding_providers import (
    EmbeddingProvider,
    EmbeddingProviderFactory
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reuse enums from the original implementation
class ProblemState(str, Enum):
    """Enum representing the possible states of a problem."""
    SOLVED = "solved"
    NOT_SOLVED = "not_solved"
    OBSOLETE = "obsolete"


class GraphNode(BaseModel):
    """Base model for all graph nodes."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class Problem(GraphNode):
    """Model representing a problem definition."""
    description: str = Field(
        ...,
        description="A detailed description of the problem"
    )
    state: ProblemState = Field(
        default=ProblemState.NOT_SOLVED,
        description="The state of the problem: solved, not_solved, or obsolete"
    )
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Vector embedding for semantic similarity comparison"
    )


class Condition(GraphNode):
    """Model representing a condition that must be met to solve the problem."""
    description: str = Field(
        ...,
        description="Detailed description of the condition"
    )
    is_met: bool = Field(
        False,
        description="Status of the condition; True if condition is met, False otherwise"
    )
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Vector embedding for semantic similarity comparison"
    )


class WorkingSolution(GraphNode):
    """Model representing a working solution to a problem."""
    description: str = Field(
        ...,
        description="Detailed description of the working solution"
    )
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Vector embedding for semantic similarity comparison"
    )


class CanonicalProblem(GraphNode):
    """Model representing a canonical form of similar problems."""
    description: str = Field(
        ...,
        description="Standard description representing a class of similar problems"
    )


class CanonicalCondition(GraphNode):
    """Model representing a canonical form of similar conditions."""
    description: str = Field(
        ...,
        description="Standard description representing a class of similar conditions"
    )


class GraphProblemManager:
    """Class for managing problems using a Neo4j graph database."""

    def __init__(self, uri: str = "bolt://localhost:7687",
                 username: str = "neo4j",
                 password: str = "password",
                 embedding_config_path: Optional[str] = "config/embeddings.json"):
        """Initialize the GraphProblemManager.

        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            embedding_config_path: Path to embedding provider config file
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        self.embedding_provider = None

        # Initialize embedding provider if config path is provided
        if embedding_config_path:
            try:
                self.embedding_provider = EmbeddingProviderFactory.load_from_file(
                    embedding_config_path
                )
                logger.info(f"Initialized embedding provider: {self.embedding_provider.__class__.__name__}")
            except Exception as e:
                logger.warning(f"Failed to initialize embedding provider: {str(e)}")
                logger.warning("Vector similarity search will not be available")
    
    def connect(self) -> None:
        """Establish connection to Neo4j database."""
        self.driver = GraphDatabase.driver(
            self.uri, 
            auth=(self.username, self.password)
        )
        # Test connection
        with self.driver.session() as session:
            result = session.run("RETURN 'Connection successful' AS message")
            print(result.single()["message"])
    
    def close(self) -> None:
        """Close the database connection."""
        if self.driver:
            self.driver.close()
    
    def initialize_schema(self) -> None:
        """Create constraints and indexes for the graph schema."""
        with self.driver.session() as session:
            # Create constraints for uniqueness
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Problem) REQUIRE p.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Condition) REQUIRE c.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Solution) REQUIRE s.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (cp:CanonicalProblem) REQUIRE cp.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (cc:CanonicalCondition) REQUIRE cc.id IS UNIQUE")

            # Create indexes for faster lookups
            session.run("CREATE INDEX IF NOT EXISTS FOR (p:Problem) ON (p.description)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (c:Condition) ON (c.description)")

            # Check if Neo4j supports vector indexes (Neo4j 5.11+)
            try:
                # Create vector index for problem embeddings
                session.run("""
                    CREATE VECTOR INDEX problem_embedding_index IF NOT EXISTS
                    FOR (p:Problem)
                    ON p.embedding
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'
                    }}
                """)

                # Create vector index for condition embeddings
                session.run("""
                    CREATE VECTOR INDEX condition_embedding_index IF NOT EXISTS
                    FOR (c:Condition)
                    ON c.embedding
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'
                    }}
                """)

                logger.info("Vector indexes created successfully")
            except Exception as e:
                logger.warning(f"Failed to create vector indexes: {str(e)}")
                logger.warning("This may happen if your Neo4j version doesn't support vector indexes")
                logger.warning("Vector similarity search will be slower")
    
    def create_problem(self, description: str) -> Problem:
        """Create a new problem in the database.

        Args:
            description: Description of the problem

        Returns:
            The created Problem object
        """
        problem = Problem(description=description)

        # Generate embedding if provider is available
        embedding = None
        if self.embedding_provider:
            try:
                embedding = self.embedding_provider.generate_embedding(description)
                problem.embedding = embedding
                logger.info(f"Generated embedding for problem: {problem.id}")
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {str(e)}")

        with self.driver.session() as session:
            # Create the problem node with embedding if available
            if embedding:
                session.run(
                    """
                    CREATE (p:Problem {
                        id: $id,
                        description: $description,
                        state: $state,
                        embedding: $embedding
                    })
                    """,
                    id=problem.id,
                    description=problem.description,
                    state=problem.state,
                    embedding=embedding
                )
            else:
                session.run(
                    """
                    CREATE (p:Problem {
                        id: $id,
                        description: $description,
                        state: $state
                    })
                    """,
                    id=problem.id,
                    description=problem.description,
                    state=problem.state
                )

        return problem
    
    def get_problem_by_id(self, problem_id: str) -> Optional[Problem]:
        """Get a problem by its ID.
        
        Args:
            problem_id: The ID of the problem to retrieve
            
        Returns:
            Problem if found, None otherwise
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Problem {id: $id})
                RETURN p
                """,
                id=problem_id
            )
            
            record = result.single()
            if record:
                node = record["p"]
                return Problem(
                    id=node["id"],
                    description=node["description"],
                    state=ProblemState(node["state"])
                )
            
            return None
    
    def add_condition_to_problem(self, problem_id: str,
                                condition_description: str,
                                is_met: bool = False) -> Optional[Condition]:
        """Add a condition to a problem.

        Args:
            problem_id: ID of the problem
            condition_description: Description of the condition
            is_met: Whether the condition is already met

        Returns:
            The created Condition object, or None if problem not found
        """
        condition = Condition(description=condition_description, is_met=is_met)

        # Generate embedding if provider is available
        embedding = None
        if self.embedding_provider:
            try:
                embedding = self.embedding_provider.generate_embedding(condition_description)
                condition.embedding = embedding
                logger.info(f"Generated embedding for condition: {condition.id}")
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {str(e)}")

        with self.driver.session() as session:
            # Check if problem exists
            problem = self.get_problem_by_id(problem_id)
            if not problem:
                return None

            # Create the condition and relationship
            if embedding:
                session.run(
                    """
                    MATCH (p:Problem {id: $problem_id})
                    CREATE (c:Condition {
                        id: $id,
                        description: $description,
                        is_met: $is_met,
                        embedding: $embedding
                    })
                    CREATE (p)-[:REQUIRES]->(c)
                    """,
                    problem_id=problem_id,
                    id=condition.id,
                    description=condition.description,
                    is_met=condition.is_met,
                    embedding=embedding
                )
            else:
                session.run(
                    """
                    MATCH (p:Problem {id: $problem_id})
                    CREATE (c:Condition {
                        id: $id,
                        description: $description,
                        is_met: $is_met
                    })
                    CREATE (p)-[:REQUIRES]->(c)
                    """,
                    problem_id=problem_id,
                    id=condition.id,
                    description=condition.description,
                    is_met=condition.is_met
                )

            return condition
    
    def update_condition(self, condition_id: str, is_met: bool) -> bool:
        """Update a condition's status.
        
        Args:
            condition_id: ID of the condition to update
            is_met: New status of the condition
            
        Returns:
            True if successful, False if condition not found
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Condition {id: $id})
                SET c.is_met = $is_met
                RETURN c
                """,
                id=condition_id,
                is_met=is_met
            )
            
            return result.single() is not None
    
    def check_if_problem_solved(self, problem_id: str) -> bool:
        """Check if all conditions for a problem are met.
        
        Args:
            problem_id: ID of the problem to check
            
        Returns:
            True if all conditions are met, False otherwise
        """
        with self.driver.session() as session:
            # Get all conditions for the problem
            result = session.run(
                """
                MATCH (p:Problem {id: $problem_id})-[:REQUIRES]->(c:Condition)
                WHERE c.is_met = false
                RETURN COUNT(c) as unmet_count
                """,
                problem_id=problem_id
            )
            
            unmet_count = result.single()["unmet_count"]
            
            # If no unmet conditions, update problem state
            if unmet_count == 0:
                session.run(
                    """
                    MATCH (p:Problem {id: $problem_id})
                    SET p.state = $solved
                    """,
                    problem_id=problem_id,
                    solved=ProblemState.SOLVED
                )
                return True
            
            return False
    
    def add_working_solution(self, problem_id: str, 
                            solution_description: str) -> Optional[WorkingSolution]:
        """Add a working solution to a problem.
        
        Args:
            problem_id: ID of the problem
            solution_description: Description of the solution
            
        Returns:
            The created WorkingSolution, or None if problem not found
        """
        solution = WorkingSolution(description=solution_description)
        
        with self.driver.session() as session:
            # Check if problem exists
            problem = self.get_problem_by_id(problem_id)
            if not problem:
                return None
            
            # Create the solution and relationship
            session.run(
                """
                MATCH (p:Problem {id: $problem_id})
                CREATE (s:Solution {
                    id: $id,
                    description: $description
                })
                CREATE (p)-[:SOLVED_BY]->(s)
                """,
                problem_id=problem_id,
                id=solution.id,
                description=solution.description
            )
            
            return solution
    
    def find_similar_problems(self, description: str,
                             threshold: float = 0.7,
                             limit: int = 10) -> List[Dict[str, Any]]:
        """Find problems similar to the given description.

        Uses vector similarity search if embeddings are available,
        otherwise falls back to text-based matching.

        Args:
            description: Description to compare against
            threshold: Similarity threshold (0-1)
            limit: Maximum number of results to return

        Returns:
            List of similar problems with similarity scores
        """
        # Check if embedding provider is available
        if not self.embedding_provider:
            logger.warning("Embedding provider not available, falling back to text search")
            return self._find_similar_by_text(description, limit)

        try:
            # Generate embedding for the query
            query_embedding = self.embedding_provider.generate_embedding(description)

            # Try to use vector index if available
            problems = self._find_similar_by_vector(query_embedding, threshold, limit)

            # If no results, fall back to text search
            if not problems:
                logger.info("No vector search results, falling back to text search")
                problems = self._find_similar_by_text(description, limit)

            return problems

        except Exception as e:
            logger.error(f"Error in vector similarity search: {str(e)}")
            return self._find_similar_by_text(description, limit)

    def _find_similar_by_vector(self,
                              query_embedding: List[float],
                              threshold: float = 0.7,
                              limit: int = 10) -> List[Dict[str, Any]]:
        """Find problems similar to the given embedding using vector search.

        Args:
            query_embedding: The embedding to compare against
            threshold: Similarity threshold (0-1)
            limit: Maximum number of results to return

        Returns:
            List of similar problems with similarity scores
        """
        with self.driver.session() as session:
            try:
                # Try using vector index with native similarity search (Neo4j 5.11+)
                result = session.run(
                    """
                    CALL db.index.vector.queryNodes('problem_embedding_index', $limit, $embedding)
                    YIELD node, score
                    WHERE score >= $threshold
                    RETURN node as p, score
                    ORDER BY score DESC
                    """,
                    embedding=query_embedding,
                    threshold=threshold,
                    limit=limit
                )
            except Exception:
                # Fall back to manual vector similarity calculation
                logger.info("Vector index not available, using manual similarity calculation")
                result = session.run(
                    """
                    MATCH (p:Problem)
                    WHERE p.embedding IS NOT NULL
                    WITH p, gds.similarity.cosine(p.embedding, $embedding) AS score
                    WHERE score >= $threshold
                    RETURN p, score
                    ORDER BY score DESC
                    LIMIT $limit
                    """,
                    embedding=query_embedding,
                    threshold=threshold,
                    limit=limit
                )

            problems = []
            for record in result:
                node = record["p"]
                score = record["score"]

                # Create problem object with embedding if available
                problem_data = {
                    "id": node["id"],
                    "description": node["description"],
                    "state": node["state"],
                    "similarity": score
                }

                if "embedding" in node:
                    problem_data["embedding"] = node["embedding"]

                problems.append(problem_data)

            return problems

    def _find_similar_by_text(self, description: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find problems similar to the given description using text search.

        Args:
            description: Text to search for
            limit: Maximum number of results to return

        Returns:
            List of similar problems
        """
        with self.driver.session() as session:
            # Simple text-based matching as fallback
            result = session.run(
                """
                MATCH (p:Problem)
                WHERE toLower(p.description) CONTAINS toLower($text)
                RETURN p
                LIMIT $limit
                """,
                text=description,
                limit=limit
            )

            problems = []
            for record in result:
                node = record["p"]

                # Create problem data without similarity score
                problem_data = {
                    "id": node["id"],
                    "description": node["description"],
                    "state": node["state"],
                    "similarity": 0.0  # No similarity score for text search
                }

                if "embedding" in node:
                    problem_data["embedding"] = node["embedding"]

                problems.append(problem_data)

            return problems
    
    def find_problem_connections(self, problem_id: str) -> Dict[str, Any]:
        """Find all connections for a given problem.
        
        Args:
            problem_id: ID of the problem
            
        Returns:
            Dictionary with connections information
        """
        with self.driver.session() as session:
            # Get conditions
            conditions_result = session.run(
                """
                MATCH (p:Problem {id: $problem_id})-[:REQUIRES]->(c:Condition)
                RETURN c
                """,
                problem_id=problem_id
            )
            
            conditions = []
            for record in conditions_result:
                node = record["c"]
                conditions.append({
                    "id": node["id"],
                    "description": node["description"],
                    "is_met": node["is_met"]
                })
            
            # Get solutions
            solutions_result = session.run(
                """
                MATCH (p:Problem {id: $problem_id})-[:SOLVED_BY]->(s:Solution)
                RETURN s
                """,
                problem_id=problem_id
            )
            
            solutions = []
            for record in solutions_result:
                node = record["s"]
                solutions.append({
                    "id": node["id"],
                    "description": node["description"]
                })
            
            # Get dependent problems
            depends_on_result = session.run(
                """
                MATCH (p:Problem {id: $problem_id})-[:DEPENDS_ON]->(dp:Problem)
                RETURN dp
                """,
                problem_id=problem_id
            )
            
            depends_on = []
            for record in depends_on_result:
                node = record["dp"]
                depends_on.append({
                    "id": node["id"],
                    "description": node["description"],
                    "state": node["state"]
                })
            
            # Get problems that depend on this one
            dependents_result = session.run(
                """
                MATCH (dp:Problem)-[:DEPENDS_ON]->(p:Problem {id: $problem_id})
                RETURN dp
                """,
                problem_id=problem_id
            )
            
            dependents = []
            for record in dependents_result:
                node = record["dp"]
                dependents.append({
                    "id": node["id"],
                    "description": node["description"],
                    "state": node["state"]
                })
            
            return {
                "conditions": conditions,
                "solutions": solutions,
                "depends_on": depends_on,
                "dependents": dependents
            }
            
    def add_dependency(self, problem_id: str, depends_on_id: str) -> bool:
        """Add a dependency between problems.

        Args:
            problem_id: ID of the dependent problem
            depends_on_id: ID of the problem being depended on

        Returns:
            True if successful, False otherwise
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p1:Problem {id: $problem_id})
                MATCH (p2:Problem {id: $depends_on_id})
                MERGE (p1)-[:DEPENDS_ON]->(p2)
                RETURN p1, p2
                """,
                problem_id=problem_id,
                depends_on_id=depends_on_id
            )

            return result.single() is not None

    def find_similar_conditions(self, description: str,
                              threshold: float = 0.7,
                              limit: int = 10) -> List[Dict[str, Any]]:
        """Find conditions similar to the given description.

        Uses vector similarity search if embeddings are available,
        otherwise falls back to text-based matching.

        Args:
            description: Description to compare against
            threshold: Similarity threshold (0-1)
            limit: Maximum number of results to return

        Returns:
            List of similar conditions with similarity scores
        """
        # Check if embedding provider is available
        if not self.embedding_provider:
            logger.warning("Embedding provider not available, falling back to text search")
            return self._find_similar_conditions_by_text(description, limit)

        try:
            # Generate embedding for the query
            query_embedding = self.embedding_provider.generate_embedding(description)

            # Try vector similarity search
            conditions = self._find_similar_conditions_by_vector(query_embedding, threshold, limit)

            # If no results, fall back to text search
            if not conditions:
                logger.info("No vector search results, falling back to text search")
                conditions = self._find_similar_conditions_by_text(description, limit)

            return conditions

        except Exception as e:
            logger.error(f"Error in vector similarity search: {str(e)}")
            return self._find_similar_conditions_by_text(description, limit)

    def _find_similar_conditions_by_vector(self,
                                        query_embedding: List[float],
                                        threshold: float = 0.7,
                                        limit: int = 10) -> List[Dict[str, Any]]:
        """Find conditions similar to the given embedding using vector search.

        Args:
            query_embedding: The embedding to compare against
            threshold: Similarity threshold (0-1)
            limit: Maximum number of results to return

        Returns:
            List of similar conditions with similarity scores
        """
        with self.driver.session() as session:
            try:
                # Try using vector index with native similarity search (Neo4j 5.11+)
                result = session.run(
                    """
                    CALL db.index.vector.queryNodes('condition_embedding_index', $limit, $embedding)
                    YIELD node, score
                    WHERE score >= $threshold
                    RETURN node as c, score
                    ORDER BY score DESC
                    """,
                    embedding=query_embedding,
                    threshold=threshold,
                    limit=limit
                )
            except Exception:
                # Fall back to manual vector similarity calculation
                logger.info("Vector index not available, using manual similarity calculation")
                result = session.run(
                    """
                    MATCH (c:Condition)
                    WHERE c.embedding IS NOT NULL
                    WITH c, gds.similarity.cosine(c.embedding, $embedding) AS score
                    WHERE score >= $threshold
                    RETURN c, score
                    ORDER BY score DESC
                    LIMIT $limit
                    """,
                    embedding=query_embedding,
                    threshold=threshold,
                    limit=limit
                )

            conditions = []
            for record in result:
                node = record["c"]
                score = record["score"]

                # Create condition data with score
                condition_data = {
                    "id": node["id"],
                    "description": node["description"],
                    "is_met": node["is_met"],
                    "similarity": score
                }

                if "embedding" in node:
                    condition_data["embedding"] = node["embedding"]

                conditions.append(condition_data)

            return conditions

    # Canonical node management methods

    def create_canonical_problem(self, description: str) -> CanonicalProblem:
        """Create a new canonical problem node.

        Args:
            description: Standard description for this class of problems

        Returns:
            The created CanonicalProblem object
        """
        canonical_problem = CanonicalProblem(description=description)

        with self.driver.session() as session:
            # Create the canonical problem node
            session.run(
                """
                CREATE (cp:CanonicalProblem {id: $id, description: $description})
                """,
                id=canonical_problem.id,
                description=canonical_problem.description
            )

        return canonical_problem

    def create_canonical_condition(self, description: str) -> CanonicalCondition:
        """Create a new canonical condition node.

        Args:
            description: Standard description for this class of conditions

        Returns:
            The created CanonicalCondition object
        """
        canonical_condition = CanonicalCondition(description=description)

        with self.driver.session() as session:
            # Create the canonical condition node
            session.run(
                """
                CREATE (cc:CanonicalCondition {id: $id, description: $description})
                """,
                id=canonical_condition.id,
                description=canonical_condition.description
            )

        return canonical_condition

    def map_problem_to_canonical(self, problem_id: str, canonical_id: str) -> bool:
        """Map a problem to its canonical form.

        Args:
            problem_id: ID of the problem to map
            canonical_id: ID of the canonical problem

        Returns:
            True if successful, False otherwise
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Problem {id: $problem_id})
                MATCH (cp:CanonicalProblem {id: $canonical_id})
                MERGE (p)-[:MAPPED_TO]->(cp)
                RETURN p, cp
                """,
                problem_id=problem_id,
                canonical_id=canonical_id
            )

            return result.single() is not None

    def map_condition_to_canonical(self, condition_id: str, canonical_id: str) -> bool:
        """Map a condition to its canonical form.

        Args:
            condition_id: ID of the condition to map
            canonical_id: ID of the canonical condition

        Returns:
            True if successful, False otherwise
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Condition {id: $condition_id})
                MATCH (cc:CanonicalCondition {id: $canonical_id})
                MERGE (c)-[:MAPPED_TO]->(cc)
                RETURN c, cc
                """,
                condition_id=condition_id,
                canonical_id=canonical_id
            )

            return result.single() is not None

    def get_canonical_problem(self, canonical_id: str) -> Optional[CanonicalProblem]:
        """Get a canonical problem by its ID.

        Args:
            canonical_id: ID of the canonical problem

        Returns:
            CanonicalProblem if found, None otherwise
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (cp:CanonicalProblem {id: $id})
                RETURN cp
                """,
                id=canonical_id
            )

            record = result.single()
            if record:
                node = record["cp"]
                return CanonicalProblem(
                    id=node["id"],
                    description=node["description"]
                )

            return None

    def get_canonical_condition(self, canonical_id: str) -> Optional[CanonicalCondition]:
        """Get a canonical condition by its ID.

        Args:
            canonical_id: ID of the canonical condition

        Returns:
            CanonicalCondition if found, None otherwise
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (cc:CanonicalCondition {id: $id})
                RETURN cc
                """,
                id=canonical_id
            )

            record = result.single()
            if record:
                node = record["cc"]
                return CanonicalCondition(
                    id=node["id"],
                    description=node["description"]
                )

            return None

    def get_problem_canonical_form(self, problem_id: str) -> Optional[CanonicalProblem]:
        """Get the canonical form of a problem.

        Args:
            problem_id: ID of the problem

        Returns:
            CanonicalProblem if mapped, None otherwise
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Problem {id: $problem_id})-[:MAPPED_TO]->(cp:CanonicalProblem)
                RETURN cp
                """,
                problem_id=problem_id
            )

            record = result.single()
            if record:
                node = record["cp"]
                return CanonicalProblem(
                    id=node["id"],
                    description=node["description"]
                )

            return None

    def get_condition_canonical_form(self, condition_id: str) -> Optional[CanonicalCondition]:
        """Get the canonical form of a condition.

        Args:
            condition_id: ID of the condition

        Returns:
            CanonicalCondition if mapped, None otherwise
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Condition {id: $condition_id})-[:MAPPED_TO]->(cc:CanonicalCondition)
                RETURN cc
                """,
                condition_id=condition_id
            )

            record = result.single()
            if record:
                node = record["cc"]
                return CanonicalCondition(
                    id=node["id"],
                    description=node["description"]
                )

            return None

    def get_problem_variants(self, canonical_id: str) -> List[Dict[str, Any]]:
        """Get all problems mapped to a canonical form.

        Args:
            canonical_id: ID of the canonical problem

        Returns:
            List of problems mapped to this canonical form
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Problem)-[:MAPPED_TO]->(cp:CanonicalProblem {id: $canonical_id})
                RETURN p
                """,
                canonical_id=canonical_id
            )

            problems = []
            for record in result:
                node = record["p"]
                problem_data = {
                    "id": node["id"],
                    "description": node["description"],
                    "state": node["state"]
                }

                if "embedding" in node:
                    problem_data["embedding"] = node["embedding"]

                problems.append(problem_data)

            return problems

    def get_condition_variants(self, canonical_id: str) -> List[Dict[str, Any]]:
        """Get all conditions mapped to a canonical form.

        Args:
            canonical_id: ID of the canonical condition

        Returns:
            List of conditions mapped to this canonical form
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Condition)-[:MAPPED_TO]->(cc:CanonicalCondition {id: $canonical_id})
                RETURN c
                """,
                canonical_id=canonical_id
            )

            conditions = []
            for record in result:
                node = record["c"]
                condition_data = {
                    "id": node["id"],
                    "description": node["description"],
                    "is_met": node["is_met"]
                }

                if "embedding" in node:
                    condition_data["embedding"] = node["embedding"]

                conditions.append(condition_data)

            return conditions