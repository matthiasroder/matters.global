"""
Matters.Global graph-based data manager

This module provides a Neo4j-based implementation for:
- Storing matters (goals, problems, conditions, solutions) in a graph database
- Computing connections between different entities
- Finding similar matters using hybrid vector/LLM similarity
- Managing entity relationships and canonical forms
"""

from typing import List, Dict, Any, Optional, Set, Tuple, Union
from enum import Enum
from datetime import datetime
import uuid
import json
from pathlib import Path
import logging
from pydantic import BaseModel, Field, root_validator
from neo4j import GraphDatabase, Driver, Session, Transaction, Result

from embedding_providers import (
    EmbeddingProvider,
    EmbeddingProviderFactory
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enums for entity states
class ProblemState(str, Enum):
    """Enum representing the possible states of a problem."""
    SOLVED = "solved"
    NOT_SOLVED = "not_solved"
    OBSOLETE = "obsolete"

class SolutionState(str, Enum):
    """Enum representing the possible states of a solution."""
    THEORETICAL = "theoretical"
    IN_PROGRESS = "in_progress"
    IMPLEMENTED = "implemented"
    FAILED = "failed"

class RelationshipType(str, Enum):
    """Enum representing the types of relationships between matters."""
    REQUIRES = "REQUIRES"
    BLOCKS = "BLOCKS"
    ENABLES = "ENABLES"
    RELATES_TO = "RELATES_TO"
    PRECEDES = "PRECEDES"
    FOLLOWS = "FOLLOWS"
    PART_OF = "PART_OF"
    CONSISTS_OF = "CONSISTS_OF"
    SOLVED_BY = "SOLVED_BY"
    ADDRESSES = "ADDRESSES"
    FULFILLS = "FULFILLS"
    MAPPED_TO = "MAPPED_TO"
    DERIVED_FROM = "DERIVED_FROM"

class MatterLabel(str, Enum):
    """Enum representing the possible labels for Matter nodes."""
    MATTER = "Matter"
    GOAL = "Goal"
    PROBLEM = "Problem"
    CONDITION = "Condition"
    SOLUTION = "Solution"

class Matter(BaseModel):
    """Base model for all graph entities in the Matters.Global system.

    A Matter is the core entity type that can have multiple labels
    representing its roles in the graph (Goal, Problem, Condition, Solution).
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str = Field(..., description="A detailed description of the matter")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Vector embedding for semantic similarity comparison"
    )
    tags: List[str] = Field(default_factory=list, description="Optional categorization tags")
    labels: List[str] = Field(default_factory=lambda: [MatterLabel.MATTER.value], description="Neo4j labels for this node")

    def add_label(self, label: Union[str, MatterLabel]) -> None:
        """Add a label to this matter if it doesn't already exist."""
        if isinstance(label, MatterLabel):
            label = label.value

        if label not in self.labels:
            self.labels.append(label)
            self.updated_at = datetime.now()

    def remove_label(self, label: Union[str, MatterLabel]) -> None:
        """Remove a label from this matter if it exists."""
        if isinstance(label, MatterLabel):
            label = label.value

        if label in self.labels and label != MatterLabel.MATTER.value:  # Cannot remove base Matter label
            self.labels.remove(label)
            self.updated_at = datetime.now()

    def has_label(self, label: Union[str, MatterLabel]) -> bool:
        """Check if this matter has a specific label."""
        if isinstance(label, MatterLabel):
            label = label.value

        return label in self.labels

    @root_validator
    def ensure_matter_label(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure the Matter label is always present."""
        if "labels" in values:
            if MatterLabel.MATTER.value not in values["labels"]:
                values["labels"].append(MatterLabel.MATTER.value)
        return values


class Goal(Matter):
    """Model representing a goal or desired outcome."""
    target_date: Optional[datetime] = Field(
        default=None,
        description="Optional target date for achieving the goal"
    )
    progress: float = Field(
        default=0.0,
        description="Progress indicator (0-1) for the goal",
        ge=0.0,
        le=1.0
    )

    def __init__(self, **data):
        super().__init__(**data)
        self.add_label(MatterLabel.GOAL)


class Problem(Matter):
    """Model representing a problem definition."""
    state: ProblemState = Field(
        default=ProblemState.NOT_SOLVED,
        description="The state of the problem: solved, not_solved, or obsolete"
    )
    priority: Optional[int] = Field(
        default=None,
        description="Optional priority ranking (lower number = higher priority)"
    )

    def __init__(self, **data):
        super().__init__(**data)
        self.add_label(MatterLabel.PROBLEM)


class Condition(Matter):
    """Model representing a condition that must be met."""
    is_met: bool = Field(
        default=False,
        description="Status of the condition; True if condition is met, False otherwise"
    )
    verification_method: Optional[str] = Field(
        default=None,
        description="Optional description of how to verify this condition"
    )

    def __init__(self, **data):
        super().__init__(**data)
        self.add_label(MatterLabel.CONDITION)


class Solution(Matter):
    """Model representing a solution to a problem or for achieving a goal."""
    state: SolutionState = Field(
        default=SolutionState.THEORETICAL,
        description="The state of the solution implementation"
    )
    implementation_date: Optional[datetime] = Field(
        default=None,
        description="When the solution was implemented"
    )

    def __init__(self, **data):
        super().__init__(**data)
        self.add_label(MatterLabel.SOLUTION)


class CanonicalMatter(BaseModel):
    """Base model for canonical representations of matters."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str = Field(..., description="A standardized description for this canonical entity")


class CanonicalProblem(CanonicalMatter):
    """Canonical representation of a problem class."""
    pass


class CanonicalCondition(CanonicalMatter):
    """Canonical representation of a condition class."""
    pass


class CanonicalGoal(CanonicalMatter):
    """Canonical representation of a goal class."""
    pass


class CanonicalSolution(CanonicalMatter):
    """Canonical representation of a solution class."""
    pass


class MatterRelationship(BaseModel):
    """Model representing a relationship between two matters."""
    source_id: str = Field(..., description="ID of the source matter")
    target_id: str = Field(..., description="ID of the target matter")
    type: str = Field(..., description="Type of relationship")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional relationship properties")


class GraphManager:
    """Class for managing matters using a Neo4j graph database."""

    def __init__(self, uri: str = "bolt://localhost:7687",
                 username: str = "neo4j",
                 password: str = "password",
                 embedding_config_path: Optional[str] = "config/embeddings.json"):
        """Initialize the GraphManager.

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
            # Create constraints for uniqueness across all Matter nodes
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Matter) REQUIRE m.id IS UNIQUE")

            # Create property existence constraints for required properties
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Matter) REQUIRE m.description IS NOT NULL")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Matter) REQUIRE m.created_at IS NOT NULL")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Matter) REQUIRE m.updated_at IS NOT NULL")

            # Create indexes for faster lookups
            session.run("CREATE INDEX IF NOT EXISTS FOR (m:Matter) ON (m.description)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (m:Matter) ON (m.created_at)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (m:Matter) ON (m.updated_at)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (m:Matter:Goal) ON (m.target_date)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (m:Matter:Problem) ON (m.state)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (m:Matter:Condition) ON (m.is_met)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (m:Matter:Solution) ON (m.state)")

            # Create full-text search index for descriptions
            try:
                session.run("""
                    CREATE FULLTEXT INDEX matter_description_index IF NOT EXISTS
                    FOR (m:Matter) ON EACH [m.description]
                    OPTIONS {indexConfig: {`fulltext.analyzer`: 'english'}}
                """)
                logger.info("Full-text search index created successfully")
            except Exception as e:
                logger.warning(f"Failed to create full-text search index: {str(e)}")
                logger.warning("This may happen if your Neo4j version doesn't support full-text search")

            # Check if Neo4j supports vector indexes (Neo4j 5.11+)
            try:
                # Create vector index for matter embeddings
                session.run("""
                    CREATE VECTOR INDEX matter_embedding_index IF NOT EXISTS
                    FOR (m:Matter)
                    ON m.embedding
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'
                    }}
                """)

                logger.info("Vector index created successfully")
            except Exception as e:
                logger.warning(f"Failed to create vector index: {str(e)}")
                logger.warning("This may happen if your Neo4j version doesn't support vector indexes")
                logger.warning("Vector similarity search will be slower")
    
    def create_matter(self, matter: Matter) -> Matter:
        """Create a new matter in the database with appropriate labels.

        This is the core method for creating any type of matter (Goal, Problem, Condition, Solution).

        Args:
            matter: Matter object with appropriate labels set

        Returns:
            The created Matter object with any updates (like embeddings)
        """
        # Generate embedding if provider is available
        if self.embedding_provider and matter.embedding is None:
            try:
                embedding = self.embedding_provider.generate_embedding(matter.description)
                matter.embedding = embedding
                logger.info(f"Generated embedding for matter: {matter.id}")
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {str(e)}")

        # Prepare properties for the database
        properties = matter.dict(exclude={"labels"})

        # Convert datetime objects to ISO format strings for Neo4j
        if "created_at" in properties and properties["created_at"]:
            properties["created_at"] = properties["created_at"].isoformat()
        if "updated_at" in properties and properties["updated_at"]:
            properties["updated_at"] = properties["updated_at"].isoformat()
        if "target_date" in properties and properties["target_date"]:
            properties["target_date"] = properties["target_date"].isoformat()
        if "implementation_date" in properties and properties["implementation_date"]:
            properties["implementation_date"] = properties["implementation_date"].isoformat()

        # Create the node with all appropriate labels
        with self.driver.session() as session:
            # Build the query dynamically based on labels
            label_str = ":".join(matter.labels)

            # Create the node with all properties
            query = f"""
            CREATE (m:{label_str} $properties)
            RETURN m
            """

            result = session.run(query, properties=properties)
            record = result.single()

            if record:
                logger.info(f"Created matter node with ID {matter.id} and labels {matter.labels}")
            else:
                logger.error(f"Failed to create matter node")

        return matter

    def create_goal(self, description: str, target_date: Optional[datetime] = None,
                   progress: float = 0.0, tags: List[str] = None) -> Goal:
        """Create a new goal in the database.

        Args:
            description: Description of the goal
            target_date: Optional target date for achieving the goal
            progress: Progress indicator (0-1)
            tags: Optional tags for categorization

        Returns:
            The created Goal object
        """
        goal = Goal(
            description=description,
            target_date=target_date,
            progress=progress,
            tags=tags or []
        )

        return self.create_matter(goal)

    def create_problem(self, description: str, state: ProblemState = ProblemState.NOT_SOLVED,
                      priority: Optional[int] = None, tags: List[str] = None) -> Problem:
        """Create a new problem in the database.

        Args:
            description: Description of the problem
            state: State of the problem
            priority: Optional priority ranking
            tags: Optional tags for categorization

        Returns:
            The created Problem object
        """
        problem = Problem(
            description=description,
            state=state,
            priority=priority,
            tags=tags or []
        )

        return self.create_matter(problem)

    def create_condition(self, description: str, is_met: bool = False,
                        verification_method: Optional[str] = None,
                        tags: List[str] = None) -> Condition:
        """Create a new condition in the database.

        Args:
            description: Description of the condition
            is_met: Whether the condition is already met
            verification_method: Method to verify if condition is met
            tags: Optional tags for categorization

        Returns:
            The created Condition object
        """
        condition = Condition(
            description=description,
            is_met=is_met,
            verification_method=verification_method,
            tags=tags or []
        )

        return self.create_matter(condition)

    def create_solution(self, description: str,
                       state: SolutionState = SolutionState.THEORETICAL,
                       implementation_date: Optional[datetime] = None,
                       tags: List[str] = None) -> Solution:
        """Create a new solution in the database.

        Args:
            description: Description of the solution
            state: State of the solution implementation
            implementation_date: When the solution was implemented
            tags: Optional tags for categorization

        Returns:
            The created Solution object
        """
        solution = Solution(
            description=description,
            state=state,
            implementation_date=implementation_date,
            tags=tags or []
        )

        return self.create_matter(solution)

    def get_matter_by_id(self, matter_id: str) -> Optional[Matter]:
        """Get a matter by its ID, regardless of labels.

        Args:
            matter_id: The ID of the matter to retrieve

        Returns:
            Matter if found, None otherwise
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (m:Matter {id: $id})
                RETURN m, labels(m) as node_labels
                """,
                id=matter_id
            )

            record = result.single()
            if not record:
                return None

            node = record["m"]
            node_labels = record["node_labels"]

            # Convert Neo4j date formats to Python datetime objects
            properties = dict(node)

            # Create appropriate object based on labels
            if MatterLabel.GOAL.value in node_labels:
                # Handle Goal-specific properties
                if "target_date" in properties and properties["target_date"]:
                    properties["target_date"] = datetime.fromisoformat(properties["target_date"])
                return Goal(**properties, labels=node_labels)

            elif MatterLabel.PROBLEM.value in node_labels:
                return Problem(**properties, labels=node_labels)

            elif MatterLabel.CONDITION.value in node_labels:
                return Condition(**properties, labels=node_labels)

            elif MatterLabel.SOLUTION.value in node_labels:
                # Handle Solution-specific properties
                if "implementation_date" in properties and properties["implementation_date"]:
                    properties["implementation_date"] = datetime.fromisoformat(properties["implementation_date"])
                return Solution(**properties, labels=node_labels)

            else:
                # Base Matter type if no specific label
                return Matter(**properties, labels=node_labels)

    def get_problem_by_id(self, problem_id: str) -> Optional[Problem]:
        """Get a problem by its ID.

        Args:
            problem_id: The ID of the problem to retrieve

        Returns:
            Problem if found, None otherwise
        """
        matter = self.get_matter_by_id(problem_id)

        if matter and matter.has_label(MatterLabel.PROBLEM):
            # If it's a matter with Problem label, convert it to Problem type
            if not isinstance(matter, Problem):
                problem_data = matter.dict()
                return Problem(**problem_data)
            return matter

        return None

    def get_goal_by_id(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by its ID.

        Args:
            goal_id: The ID of the goal to retrieve

        Returns:
            Goal if found, None otherwise
        """
        matter = self.get_matter_by_id(goal_id)

        if matter and matter.has_label(MatterLabel.GOAL):
            # If it's a matter with Goal label, convert it to Goal type
            if not isinstance(matter, Goal):
                goal_data = matter.dict()
                return Goal(**goal_data)
            return matter

        return None

    def get_condition_by_id(self, condition_id: str) -> Optional[Condition]:
        """Get a condition by its ID.

        Args:
            condition_id: The ID of the condition to retrieve

        Returns:
            Condition if found, None otherwise
        """
        matter = self.get_matter_by_id(condition_id)

        if matter and matter.has_label(MatterLabel.CONDITION):
            # If it's a matter with Condition label, convert it to Condition type
            if not isinstance(matter, Condition):
                condition_data = matter.dict()
                return Condition(**condition_data)
            return matter

        return None

    def get_solution_by_id(self, solution_id: str) -> Optional[Solution]:
        """Get a solution by its ID.

        Args:
            solution_id: The ID of the solution to retrieve

        Returns:
            Solution if found, None otherwise
        """
        matter = self.get_matter_by_id(solution_id)

        if matter and matter.has_label(MatterLabel.SOLUTION):
            # If it's a matter with Solution label, convert it to Solution type
            if not isinstance(matter, Solution):
                solution_data = matter.dict()
                return Solution(**solution_data)
            return matter

        return None

    def create_relationship(self, relationship: MatterRelationship) -> bool:
        """Create a relationship between two matters.

        Args:
            relationship: The relationship to create

        Returns:
            True if successful, False otherwise
        """
        # Validate that both source and target exist
        source = self.get_matter_by_id(relationship.source_id)
        target = self.get_matter_by_id(relationship.target_id)

        if not source or not target:
            logger.error(f"Cannot create relationship: Matter not found. source_id: {relationship.source_id}, target_id: {relationship.target_id}")
            return False

        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (source:Matter {id: $source_id})
                    MATCH (target:Matter {id: $target_id})
                    CREATE (source)-[r:$type]->(target)
                    SET r = $properties
                    RETURN r
                    """,
                    source_id=relationship.source_id,
                    target_id=relationship.target_id,
                    type=relationship.type,
                    properties=relationship.properties
                )

                record = result.single()
                if record:
                    logger.info(f"Created relationship: ({source.id})-[:{relationship.type}]->({target.id})")
                    return True
                else:
                    logger.error(f"Failed to create relationship despite matters existing")
                    return False

        except Exception as e:
            logger.error(f"Error creating relationship: {str(e)}")
            return False

    def add_condition_to_problem(self, problem_id: str,
                                condition_description: str,
                                is_met: bool = False,
                                verification_method: Optional[str] = None) -> Optional[Condition]:
        """Add a condition to a problem.

        Args:
            problem_id: ID of the problem
            condition_description: Description of the condition
            is_met: Whether the condition is already met
            verification_method: Method to verify if condition is met

        Returns:
            The created Condition object, or None if problem not found
        """
        # Check if problem exists
        problem = self.get_problem_by_id(problem_id)
        if not problem:
            logger.error(f"Problem with ID {problem_id} not found")
            return None

        # Create the condition
        condition = self.create_condition(
            description=condition_description,
            is_met=is_met,
            verification_method=verification_method
        )

        # Create the relationship
        relationship = MatterRelationship(
            source_id=problem_id,
            target_id=condition.id,
            type=RelationshipType.REQUIRES.value
        )

        success = self.create_relationship(relationship)

        if success:
            logger.info(f"Added condition {condition.id} to problem {problem_id}")
            return condition
        else:
            logger.error(f"Failed to create relationship between problem {problem_id} and condition {condition.id}")
            return None
    
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
                MATCH (c:Matter:Condition {id: $id})
                SET c.is_met = $is_met
                SET c.updated_at = $updated_at
                RETURN c
                """,
                id=condition_id,
                is_met=is_met,
                updated_at=datetime.now().isoformat()
            )

            return result.single() is not None

    def set_goal_condition(self, goal_id: str, condition_id: str) -> bool:
        """Set a condition as a requirement for a goal.

        Args:
            goal_id: ID of the goal
            condition_id: ID of the condition

        Returns:
            True if successful, False otherwise
        """
        relationship = MatterRelationship(
            source_id=goal_id,
            target_id=condition_id,
            type=RelationshipType.REQUIRES.value
        )

        return self.create_relationship(relationship)

    def add_solution_to_matter(self, matter_id: str, solution_id: str) -> bool:
        """Add a solution to a matter (problem or goal).

        Args:
            matter_id: ID of the matter (problem or goal)
            solution_id: ID of the solution

        Returns:
            True if successful, False otherwise
        """
        # First verify both matters exist
        matter = self.get_matter_by_id(matter_id)
        solution = self.get_solution_by_id(solution_id)

        if not matter:
            logger.error(f"Matter not found with ID: {matter_id}")
            return False

        if not solution:
            logger.error(f"Solution not found with ID: {solution_id}")
            return False

        # Create SOLVED_BY relationship from matter to solution
        relationship = MatterRelationship(
            source_id=matter_id,
            target_id=solution_id,
            type=RelationshipType.SOLVED_BY.value
        )

        # Also create the inverse ADDRESSES relationship
        inverse_relationship = MatterRelationship(
            source_id=solution_id,
            target_id=matter_id,
            type=RelationshipType.ADDRESSES.value
        )

        success1 = self.create_relationship(relationship)
        success2 = self.create_relationship(inverse_relationship)

        return success1 and success2

    def set_solution_fulfills_condition(self, solution_id: str, condition_id: str) -> bool:
        """Set a solution as fulfilling a condition.

        Args:
            solution_id: ID of the solution
            condition_id: ID of the condition

        Returns:
            True if successful, False otherwise
        """
        relationship = MatterRelationship(
            source_id=solution_id,
            target_id=condition_id,
            type=RelationshipType.FULFILLS.value
        )

        success = self.create_relationship(relationship)

        # If successful, also update the condition status to met
        if success:
            self.update_condition(condition_id, True)

        return success

    def set_goal_target_date(self, goal_id: str, target_date: datetime) -> bool:
        """Set the target date for a goal.

        Args:
            goal_id: ID of the goal
            target_date: Target date for achieving the goal

        Returns:
            True if successful, False otherwise
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (g:Matter:Goal {id: $id})
                SET g.target_date = $target_date
                SET g.updated_at = $updated_at
                RETURN g
                """,
                id=goal_id,
                target_date=target_date.isoformat(),
                updated_at=datetime.now().isoformat()
            )

            return result.single() is not None

    def set_goal_progress(self, goal_id: str, progress: float) -> bool:
        """Set the progress for a goal.

        Args:
            goal_id: ID of the goal
            progress: Progress value between 0 and 1

        Returns:
            True if successful, False otherwise
        """
        # Validate progress value
        if progress < 0 or progress > 1:
            logger.error(f"Invalid progress value: {progress}. Must be between 0 and 1.")
            return False

        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (g:Matter:Goal {id: $id})
                SET g.progress = $progress
                SET g.updated_at = $updated_at
                RETURN g
                """,
                id=goal_id,
                progress=progress,
                updated_at=datetime.now().isoformat()
            )

            return result.single() is not None

    def add_tags_to_matter(self, matter_id: str, tags: List[str]) -> bool:
        """Add tags to a matter.

        Args:
            matter_id: ID of the matter
            tags: List of tags to add

        Returns:
            True if successful, False otherwise
        """
        # Get current matter to merge tags
        matter = self.get_matter_by_id(matter_id)
        if not matter:
            logger.error(f"Matter not found with ID: {matter_id}")
            return False

        # Merge existing and new tags
        current_tags = set(matter.tags)
        new_tags = set(tags)
        combined_tags = list(current_tags.union(new_tags))

        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (m:Matter {id: $id})
                SET m.tags = $tags
                SET m.updated_at = $updated_at
                RETURN m
                """,
                id=matter_id,
                tags=combined_tags,
                updated_at=datetime.now().isoformat()
            )

            return result.single() is not None

    def set_matter_relationship(self, source_id: str, relationship_type: Union[str, RelationshipType],
                              target_id: str, properties: Dict[str, Any] = None) -> bool:
        """Create a relationship between two matters of any specified type.

        Args:
            source_id: ID of the source matter
            relationship_type: Type of relationship
            target_id: ID of the target matter
            properties: Optional properties for the relationship

        Returns:
            True if successful, False otherwise
        """
        # Convert RelationshipType enum to string if needed
        if isinstance(relationship_type, RelationshipType):
            relationship_type = relationship_type.value

        relationship = MatterRelationship(
            source_id=source_id,
            target_id=target_id,
            type=relationship_type,
            properties=properties or {}
        )

        return self.create_relationship(relationship)

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
                MATCH (p:Matter:Problem {id: $problem_id})-[:REQUIRES]->(c:Matter:Condition)
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
                    MATCH (p:Matter:Problem {id: $problem_id})
                    SET p.state = $solved
                    SET p.updated_at = $updated_at
                    """,
                    problem_id=problem_id,
                    solved=ProblemState.SOLVED.value,
                    updated_at=datetime.now().isoformat()
                )
                return True

            return False

    def check_goal_progress(self, goal_id: str) -> float:
        """Calculate and update progress for a goal based on met conditions.

        Args:
            goal_id: ID of the goal to check

        Returns:
            Current progress value between 0 and 1
        """
        with self.driver.session() as session:
            # Count total and met conditions
            result = session.run(
                """
                MATCH (g:Matter:Goal {id: $goal_id})-[:REQUIRES]->(c:Matter:Condition)
                WITH g, COUNT(c) as total_conditions,
                     COUNT(c.is_met = true) as met_conditions
                SET g.progress = CASE
                    WHEN total_conditions = 0 THEN 0.0
                    ELSE toFloat(met_conditions) / toFloat(total_conditions)
                END
                SET g.updated_at = $updated_at
                RETURN g.progress as progress
                """,
                goal_id=goal_id,
                updated_at=datetime.now().isoformat()
            )

            record = result.single()
            if record:
                return float(record["progress"])

            return 0.0
    
    def add_working_solution(self, problem_id: str,
                            solution_description: str) -> Optional[Solution]:
        """Add a working solution to a problem.

        This is a legacy method - use create_solution and add_solution_to_matter instead.

        Args:
            problem_id: ID of the problem
            solution_description: Description of the solution

        Returns:
            The created Solution, or None if problem not found
        """
        # Check if problem exists
        problem = self.get_problem_by_id(problem_id)
        if not problem:
            logger.error(f"Problem with ID {problem_id} not found")
            return None

        # Create solution
        solution = self.create_solution(
            description=solution_description,
            state=SolutionState.IMPLEMENTED
        )

        # Create relationship
        self.add_solution_to_matter(problem_id, solution.id)

        return solution
    
    def find_similar_matters(self, description: str,
                            labels: List[str] = None,
                            threshold: float = 0.7,
                            limit: int = 10) -> List[Dict[str, Any]]:
        """Find matters similar to the given description.

        Uses vector similarity search if embeddings are available,
        otherwise falls back to text-based matching.

        Args:
            description: Description to compare against
            labels: Optional list of labels to filter by (e.g., ["Goal", "Problem"])
            threshold: Similarity threshold (0-1)
            limit: Maximum number of results to return

        Returns:
            List of similar matters with similarity scores
        """
        # Default to all matter types if no labels specified
        if not labels:
            labels = [label.value for label in MatterLabel]

        # Convert label enums to strings if needed
        labels = [label.value if isinstance(label, MatterLabel) else label for label in labels]

        # Ensure Matter label is included
        if MatterLabel.MATTER.value not in labels:
            labels.append(MatterLabel.MATTER.value)

        # Check if embedding provider is available
        if not self.embedding_provider:
            logger.warning("Embedding provider not available, falling back to text search")
            return self._find_similar_by_text(description, labels, limit)

        try:
            # Generate embedding for the query
            query_embedding = self.embedding_provider.generate_embedding(description)

            # Try to use vector index if available
            matters = self._find_similar_by_vector(query_embedding, labels, threshold, limit)

            # If no results, fall back to text search
            if not matters:
                logger.info("No vector search results, falling back to text search")
                matters = self._find_similar_by_text(description, labels, limit)

            return matters

        except Exception as e:
            logger.error(f"Error in vector similarity search: {str(e)}")
            return self._find_similar_by_text(description, labels, limit)

    # Legacy method for compatibility
    def find_similar_problems(self, description: str,
                             threshold: float = 0.7,
                             limit: int = 10) -> List[Dict[str, Any]]:
        """Find problems similar to the given description (legacy method).

        Args:
            description: Description to compare against
            threshold: Similarity threshold (0-1)
            limit: Maximum number of results to return

        Returns:
            List of similar problems with similarity scores
        """
        return self.find_similar_matters(
            description=description,
            labels=[MatterLabel.PROBLEM.value],
            threshold=threshold,
            limit=limit
        )

    def _find_similar_by_vector(self,
                              query_embedding: List[float],
                              labels: List[str],
                              threshold: float = 0.7,
                              limit: int = 10) -> List[Dict[str, Any]]:
        """Find matters similar to the given embedding using vector search.

        Args:
            query_embedding: The embedding to compare against
            labels: List of labels to filter by
            threshold: Similarity threshold (0-1)
            limit: Maximum number of results to return

        Returns:
            List of similar matters with similarity scores
        """
        # Build dynamic label match clause based on provided labels
        label_clause = " AND ".join([f"m:{label}" for label in labels])

        with self.driver.session() as session:
            try:
                # Try using vector index with native similarity search (Neo4j 5.11+)
                result = session.run(
                    f"""
                    CALL db.index.vector.queryNodes('matter_embedding_index', $limit, $embedding)
                    YIELD node, score
                    WHERE score >= $threshold AND {label_clause}
                    RETURN node as m, score, labels(node) as node_labels
                    ORDER BY score DESC
                    """,
                    embedding=query_embedding,
                    threshold=threshold,
                    limit=limit
                )
            except Exception as e:
                # Fall back to manual vector similarity calculation
                logger.info(f"Vector index not available ({str(e)}), using manual similarity calculation")
                result = session.run(
                    f"""
                    MATCH (m:Matter)
                    WHERE {label_clause} AND m.embedding IS NOT NULL
                    WITH m, gds.similarity.cosine(m.embedding, $embedding) AS score
                    WHERE score >= $threshold
                    RETURN m, score, labels(m) as node_labels
                    ORDER BY score DESC
                    LIMIT $limit
                    """,
                    embedding=query_embedding,
                    threshold=threshold,
                    limit=limit
                )

            matters = []
            for record in result:
                node = record["m"]
                score = record["score"]
                node_labels = record["node_labels"]

                # Create data dictionary with type-specific properties
                matter_data = {
                    "id": node["id"],
                    "description": node["description"],
                    "labels": node_labels,
                    "similarity": score,
                    "created_at": node.get("created_at"),
                    "updated_at": node.get("updated_at"),
                    "tags": node.get("tags", [])
                }

                # Add type-specific properties based on labels
                if MatterLabel.PROBLEM.value in node_labels:
                    matter_data["state"] = node.get("state")
                    matter_data["priority"] = node.get("priority")

                if MatterLabel.GOAL.value in node_labels:
                    matter_data["target_date"] = node.get("target_date")
                    matter_data["progress"] = node.get("progress", 0.0)

                if MatterLabel.CONDITION.value in node_labels:
                    matter_data["is_met"] = node.get("is_met", False)
                    matter_data["verification_method"] = node.get("verification_method")

                if MatterLabel.SOLUTION.value in node_labels:
                    matter_data["state"] = node.get("state")
                    matter_data["implementation_date"] = node.get("implementation_date")

                if "embedding" in node:
                    matter_data["embedding"] = node["embedding"]

                matters.append(matter_data)

            return matters

    def _find_similar_by_text(self, description: str, labels: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Find matters similar to the given description using text search.

        Args:
            description: Text to search for
            labels: List of labels to filter by
            limit: Maximum number of results to return

        Returns:
            List of similar matters
        """
        # Build dynamic label match clause based on provided labels
        label_clause = " AND ".join([f"m:{label}" for label in labels])

        with self.driver.session() as session:
            # Try using full-text search index if available
            try:
                result = session.run(
                    f"""
                    CALL db.index.fulltext.queryNodes('matter_description_index', $text)
                    YIELD node, score
                    WHERE {label_clause}
                    RETURN node as m, score, labels(node) as node_labels
                    ORDER BY score DESC
                    LIMIT $limit
                    """,
                    text=description,
                    limit=limit
                )

                # If we get results, use them
                if result.peek():
                    pass  # Continue with processing
                else:
                    # Fall back to simple text matching
                    raise Exception("No full-text search results")

            except Exception:
                # Fall back to simple text-based matching
                logger.info("Full-text index not available, using simple text matching")
                result = session.run(
                    f"""
                    MATCH (m:Matter)
                    WHERE {label_clause} AND toLower(m.description) CONTAINS toLower($text)
                    RETURN m, 0.0 as score, labels(m) as node_labels
                    LIMIT $limit
                    """,
                    text=description,
                    limit=limit
                )

            matters = []
            for record in result:
                node = record["m"]
                score = record["score"]
                node_labels = record["node_labels"]

                # Create data dictionary with type-specific properties
                matter_data = {
                    "id": node["id"],
                    "description": node["description"],
                    "labels": node_labels,
                    "similarity": score,
                    "created_at": node.get("created_at"),
                    "updated_at": node.get("updated_at"),
                    "tags": node.get("tags", [])
                }

                # Add type-specific properties based on labels
                if MatterLabel.PROBLEM.value in node_labels:
                    matter_data["state"] = node.get("state")
                    matter_data["priority"] = node.get("priority")

                if MatterLabel.GOAL.value in node_labels:
                    matter_data["target_date"] = node.get("target_date")
                    matter_data["progress"] = node.get("progress", 0.0)

                if MatterLabel.CONDITION.value in node_labels:
                    matter_data["is_met"] = node.get("is_met", False)
                    matter_data["verification_method"] = node.get("verification_method")

                if MatterLabel.SOLUTION.value in node_labels:
                    matter_data["state"] = node.get("state")
                    matter_data["implementation_date"] = node.get("implementation_date")

                if "embedding" in node:
                    matter_data["embedding"] = node["embedding"]

                matters.append(matter_data)

            return matters
    
    def find_matter_connections(self, matter_id: str) -> Dict[str, Any]:
        """Find all connections for a given matter.

        Args:
            matter_id: ID of the matter

        Returns:
            Dictionary with connections information
        """
        with self.driver.session() as session:
            # Get matter labels to tailor the response
            labels_result = session.run(
                """
                MATCH (m:Matter {id: $matter_id})
                RETURN labels(m) as labels
                """,
                matter_id=matter_id
            )

            record = labels_result.single()
            if not record:
                return {"error": f"Matter with ID {matter_id} not found"}

            matter_labels = record["labels"]

            # Initialize result dictionary
            result = {
                "id": matter_id,
                "labels": matter_labels,
                "conditions": [],
                "solutions": [],
                "prerequisites": [],
                "prerequisites_for": [],
                "related_matters": [],
                "parts": [],
                "part_of": []
            }

            # Get conditions required by this matter
            conditions_result = session.run(
                """
                MATCH (m:Matter {id: $matter_id})-[:REQUIRES]->(c:Matter:Condition)
                RETURN c, labels(c) as labels
                """,
                matter_id=matter_id
            )

            for record in conditions_result:
                node = record["c"]
                labels = record["labels"]
                condition = {
                    "id": node["id"],
                    "description": node["description"],
                    "is_met": node.get("is_met", False),
                    "labels": labels
                }

                if "verification_method" in node:
                    condition["verification_method"] = node["verification_method"]

                result["conditions"].append(condition)

            # Get solutions addressing this matter
            solutions_result = session.run(
                """
                MATCH (m:Matter {id: $matter_id})-[:SOLVED_BY]->(s:Matter:Solution)
                RETURN s, labels(s) as labels
                """,
                matter_id=matter_id
            )

            for record in solutions_result:
                node = record["s"]
                labels = record["labels"]
                solution = {
                    "id": node["id"],
                    "description": node["description"],
                    "labels": labels
                }

                if "state" in node:
                    solution["state"] = node["state"]
                if "implementation_date" in node:
                    solution["implementation_date"] = node["implementation_date"]

                result["solutions"].append(solution)

            # Get prerequisites (matters that must be completed before this one)
            prerequisites_result = session.run(
                """
                MATCH (prereq:Matter)-[:PRECEDES]->(m:Matter {id: $matter_id})
                RETURN prereq, labels(prereq) as labels
                """
                +
                """ UNION
                MATCH (prereq:Matter)-[:MUST_BE_RESOLVED_BEFORE]->(m:Matter {id: $matter_id})
                RETURN prereq, labels(prereq) as labels
                """,  # Include legacy relationship
                matter_id=matter_id
            )

            for record in prerequisites_result:
                node = record["prereq"]
                labels = record["labels"]
                prerequisite = {
                    "id": node["id"],
                    "description": node["description"],
                    "labels": labels
                }

                # Add label-specific properties
                if MatterLabel.PROBLEM.value in labels and "state" in node:
                    prerequisite["state"] = node["state"]
                if MatterLabel.GOAL.value in labels and "progress" in node:
                    prerequisite["progress"] = node["progress"]

                result["prerequisites"].append(prerequisite)

            # Get matters that this one is a prerequisite for
            prerequisites_for_result = session.run(
                """
                MATCH (m:Matter {id: $matter_id})-[:PRECEDES]->(after:Matter)
                RETURN after, labels(after) as labels
                """
                +
                """ UNION
                MATCH (m:Matter {id: $matter_id})-[:MUST_BE_RESOLVED_BEFORE]->(after:Matter)
                RETURN after, labels(after) as labels
                """,  # Include legacy relationship
                matter_id=matter_id
            )

            for record in prerequisites_for_result:
                node = record["after"]
                labels = record["labels"]
                dependent = {
                    "id": node["id"],
                    "description": node["description"],
                    "labels": labels
                }

                # Add label-specific properties
                if MatterLabel.PROBLEM.value in labels and "state" in node:
                    dependent["state"] = node["state"]
                if MatterLabel.GOAL.value in labels and "progress" in node:
                    dependent["progress"] = node["progress"]

                result["prerequisites_for"].append(dependent)

            # Get matters that are related
            related_result = session.run(
                """
                MATCH (m:Matter {id: $matter_id})-[:RELATES_TO]-(related:Matter)
                RETURN related, labels(related) as labels
                """,
                matter_id=matter_id
            )

            for record in related_result:
                node = record["related"]
                labels = record["labels"]
                related = {
                    "id": node["id"],
                    "description": node["description"],
                    "labels": labels
                }
                result["related_matters"].append(related)

            # Get component parts
            parts_result = session.run(
                """
                MATCH (m:Matter {id: $matter_id})-[:CONSISTS_OF]->(part:Matter)
                RETURN part, labels(part) as labels
                """,
                matter_id=matter_id
            )

            for record in parts_result:
                node = record["part"]
                labels = record["labels"]
                part = {
                    "id": node["id"],
                    "description": node["description"],
                    "labels": labels
                }
                result["parts"].append(part)

            # Get what this is a part of
            part_of_result = session.run(
                """
                MATCH (m:Matter {id: $matter_id})-[:PART_OF]->(whole:Matter)
                RETURN whole, labels(whole) as labels
                """,
                matter_id=matter_id
            )

            for record in part_of_result:
                node = record["whole"]
                labels = record["labels"]
                whole = {
                    "id": node["id"],
                    "description": node["description"],
                    "labels": labels
                }
                result["part_of"].append(whole)

            # For backward compatibility, include old field names
            if MatterLabel.PROBLEM.value in matter_labels:
                result["depends_on"] = result["prerequisites"]
                result["dependents"] = result["prerequisites_for"]
                result["blocked_problems"] = result["prerequisites_for"]

            return result

    # Legacy method for backward compatibility
    def find_problem_connections(self, problem_id: str) -> Dict[str, Any]:
        """Find all connections for a given problem (legacy method).

        Args:
            problem_id: ID of the problem

        Returns:
            Dictionary with connections information
        """
        # Simply delegate to the new multi-label approach
        return self.find_matter_connections(problem_id)
            
    def add_resolution_prerequisite(self, prerequisite_id: str, dependent_id: str) -> bool:
        """Add a resolution prerequisite relationship between matters.

        Creates both PRECEDES and legacy MUST_BE_RESOLVED_BEFORE relationships for compatibility.

        Args:
            prerequisite_id: ID of the matter that must be resolved first
            dependent_id: ID of the matter that depends on the prerequisite

        Returns:
            True if successful, False otherwise
        """
        # Validate that both matters exist before attempting to create relationship
        prerequisite = self.get_matter_by_id(prerequisite_id)
        dependent = self.get_matter_by_id(dependent_id)

        if not prerequisite or not dependent:
            logger.error(f"Cannot create prerequisite relationship: Matter not found. prerequisite_id: {prerequisite_id}, dependent_id: {dependent_id}")
            return False

        try:
            with self.driver.session() as session:
                # Create both new PRECEDES relationship and legacy MUST_BE_RESOLVED_BEFORE for backward compatibility
                result = session.run(
                    """
                    MATCH (m1:Matter {id: $prerequisite_id})
                    MATCH (m2:Matter {id: $dependent_id})
                    MERGE (m1)-[:PRECEDES]->(m2)
                    MERGE (m1)-[:MUST_BE_RESOLVED_BEFORE]->(m2)
                    RETURN m1, m2
                    """,
                    prerequisite_id=prerequisite_id,
                    dependent_id=dependent_id
                )

                record = result.single()
                if record:
                    logger.info(f"Successfully created prerequisite relationship: {prerequisite_id} must be resolved before {dependent_id}")
                    return True
                else:
                    logger.error(f"Failed to create prerequisite relationship despite matters existing")
                    return False

        except Exception as e:
            logger.error(f"Error creating prerequisite relationship: {str(e)}")
            return False

    # Keep backward compatibility until all code is updated
    def add_dependency(self, problem_id: str, depends_on_id: str) -> bool:
        """Legacy method - use add_resolution_prerequisite instead.

        This now calls add_resolution_prerequisite with reversed parameters
        to maintain the same semantic meaning but with the new relationship direction.

        Args:
            problem_id: ID of the dependent problem
            depends_on_id: ID of the problem being depended on (prerequisite)

        Returns:
            True if successful, False otherwise
        """
        logger.warning("add_dependency is deprecated, use add_resolution_prerequisite instead")
        # Note the parameter order is reversed to maintain semantic meaning
        return self.add_resolution_prerequisite(depends_on_id, problem_id)

    def find_similar_conditions(self, description: str,
                              threshold: float = 0.7,
                              limit: int = 10) -> List[Dict[str, Any]]:
        """Find conditions similar to the given description (legacy method).

        This is a legacy method that uses the new multi-label approach with the Condition label.
        It is recommended to use find_similar_matters with appropriate labels instead.

        Args:
            description: Description to compare against
            threshold: Similarity threshold (0-1)
            limit: Maximum number of results to return

        Returns:
            List of similar conditions with similarity scores
        """
        # Delegate to the new multi-label approach, filtering for Condition label
        return self.find_similar_matters(
            description=description,
            labels=[MatterLabel.CONDITION.value],
            threshold=threshold,
            limit=limit
        )

    # Canonical node management methods

    def create_canonical_matter(self, description: str, canonical_type: str) -> CanonicalMatter:
        """Create a new canonical matter node of any type.

        Args:
            description: Standard description for this class of matters
            canonical_type: Type of canonical matter (Problem, Condition, Goal, Solution)

        Returns:
            The created CanonicalMatter object
        """
        canonical_matter = CanonicalMatter(description=description)

        with self.driver.session() as session:
            # Create the canonical matter node with specified type
            session.run(
                f"""
                CREATE (cm:CanonicalMatter:{canonical_type} {{id: $id, description: $description}})
                """,
                id=canonical_matter.id,
                description=canonical_matter.description
            )

        return canonical_matter

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
                CREATE (cp:CanonicalMatter:CanonicalProblem {id: $id, description: $description})
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
                CREATE (cc:CanonicalMatter:CanonicalCondition {id: $id, description: $description})
                """,
                id=canonical_condition.id,
                description=canonical_condition.description
            )

        return canonical_condition

    def create_canonical_goal(self, description: str) -> CanonicalGoal:
        """Create a new canonical goal node.

        Args:
            description: Standard description for this class of goals

        Returns:
            The created CanonicalGoal object
        """
        canonical_goal = CanonicalGoal(description=description)

        with self.driver.session() as session:
            # Create the canonical goal node
            session.run(
                """
                CREATE (cg:CanonicalMatter:CanonicalGoal {id: $id, description: $description})
                """,
                id=canonical_goal.id,
                description=canonical_goal.description
            )

        return canonical_goal

    def create_canonical_solution(self, description: str) -> CanonicalSolution:
        """Create a new canonical solution node.

        Args:
            description: Standard description for this class of solutions

        Returns:
            The created CanonicalSolution object
        """
        canonical_solution = CanonicalSolution(description=description)

        with self.driver.session() as session:
            # Create the canonical solution node
            session.run(
                """
                CREATE (cs:CanonicalMatter:CanonicalSolution {id: $id, description: $description})
                """,
                id=canonical_solution.id,
                description=canonical_solution.description
            )

        return canonical_solution

    def map_matter_to_canonical(self, matter_id: str, canonical_id: str) -> bool:
        """Map any matter to its canonical form.

        Args:
            matter_id: ID of the matter to map
            canonical_id: ID of the canonical matter

        Returns:
            True if successful, False otherwise
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (m:Matter {id: $matter_id})
                MATCH (cm:CanonicalMatter {id: $canonical_id})
                MERGE (m)-[:MAPPED_TO]->(cm)
                RETURN m, cm
                """,
                matter_id=matter_id,
                canonical_id=canonical_id
            )

            return result.single() is not None

    def map_problem_to_canonical(self, problem_id: str, canonical_id: str) -> bool:
        """Map a problem to its canonical form (legacy method).

        Args:
            problem_id: ID of the problem to map
            canonical_id: ID of the canonical problem

        Returns:
            True if successful, False otherwise
        """
        # Verify the matter has the Problem label
        problem = self.get_matter_by_id(problem_id)
        if not problem or not problem.has_label(MatterLabel.PROBLEM):
            logger.error(f"Matter {problem_id} does not have Problem label")
            return False

        # Verify the canonical matter has the CanonicalProblem label
        with self.driver.session() as session:
            canonical_check = session.run(
                """
                MATCH (cp:CanonicalMatter {id: $canonical_id})
                RETURN labels(cp) as labels
                """,
                canonical_id=canonical_id
            )

            record = canonical_check.single()
            if not record or "CanonicalProblem" not in record["labels"]:
                logger.error(f"Canonical matter {canonical_id} is not a CanonicalProblem")
                return False

        # Use the general mapping method
        return self.map_matter_to_canonical(problem_id, canonical_id)

    def map_condition_to_canonical(self, condition_id: str, canonical_id: str) -> bool:
        """Map a condition to its canonical form (legacy method).

        Args:
            condition_id: ID of the condition to map
            canonical_id: ID of the canonical condition

        Returns:
            True if successful, False otherwise
        """
        # Verify the matter has the Condition label
        condition = self.get_matter_by_id(condition_id)
        if not condition or not condition.has_label(MatterLabel.CONDITION):
            logger.error(f"Matter {condition_id} does not have Condition label")
            return False

        # Verify the canonical matter has the CanonicalCondition label
        with self.driver.session() as session:
            canonical_check = session.run(
                """
                MATCH (cc:CanonicalMatter {id: $canonical_id})
                RETURN labels(cc) as labels
                """,
                canonical_id=canonical_id
            )

            record = canonical_check.single()
            if not record or "CanonicalCondition" not in record["labels"]:
                logger.error(f"Canonical matter {canonical_id} is not a CanonicalCondition")
                return False

        # Use the general mapping method
        return self.map_matter_to_canonical(condition_id, canonical_id)

    def get_canonical_matter(self, canonical_id: str) -> Optional[CanonicalMatter]:
        """Get any canonical matter by its ID.

        Args:
            canonical_id: ID of the canonical matter

        Returns:
            CanonicalMatter if found, None otherwise
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (cm:CanonicalMatter {id: $id})
                RETURN cm, labels(cm) as labels
                """,
                id=canonical_id
            )

            record = result.single()
            if not record:
                return None

            node = record["cm"]
            labels = record["labels"]

            # Create the appropriate canonical type based on labels
            if "CanonicalProblem" in labels:
                return CanonicalProblem(
                    id=node["id"],
                    description=node["description"]
                )
            elif "CanonicalCondition" in labels:
                return CanonicalCondition(
                    id=node["id"],
                    description=node["description"]
                )
            elif "CanonicalGoal" in labels:
                return CanonicalGoal(
                    id=node["id"],
                    description=node["description"]
                )
            elif "CanonicalSolution" in labels:
                return CanonicalSolution(
                    id=node["id"],
                    description=node["description"]
                )
            else:
                # Base canonical matter type
                return CanonicalMatter(
                    id=node["id"],
                    description=node["description"]
                )

    def get_canonical_problem(self, canonical_id: str) -> Optional[CanonicalProblem]:
        """Get a canonical problem by its ID.

        Args:
            canonical_id: ID of the canonical problem

        Returns:
            CanonicalProblem if found, None otherwise
        """
        canonical_matter = self.get_canonical_matter(canonical_id)

        if canonical_matter and isinstance(canonical_matter, CanonicalProblem):
            return canonical_matter

        return None

    def get_canonical_condition(self, canonical_id: str) -> Optional[CanonicalCondition]:
        """Get a canonical condition by its ID.

        Args:
            canonical_id: ID of the canonical condition

        Returns:
            CanonicalCondition if found, None otherwise
        """
        canonical_matter = self.get_canonical_matter(canonical_id)

        if canonical_matter and isinstance(canonical_matter, CanonicalCondition):
            return canonical_matter

        return None

    def get_canonical_goal(self, canonical_id: str) -> Optional[CanonicalGoal]:
        """Get a canonical goal by its ID.

        Args:
            canonical_id: ID of the canonical goal

        Returns:
            CanonicalGoal if found, None otherwise
        """
        canonical_matter = self.get_canonical_matter(canonical_id)

        if canonical_matter and isinstance(canonical_matter, CanonicalGoal):
            return canonical_matter

        return None

    def get_canonical_solution(self, canonical_id: str) -> Optional[CanonicalSolution]:
        """Get a canonical solution by its ID.

        Args:
            canonical_id: ID of the canonical solution

        Returns:
            CanonicalSolution if found, None otherwise
        """
        canonical_matter = self.get_canonical_matter(canonical_id)

        if canonical_matter and isinstance(canonical_matter, CanonicalSolution):
            return canonical_matter

        return None

    def get_matter_canonical_form(self, matter_id: str) -> Optional[CanonicalMatter]:
        """Get the canonical form of any matter.

        Args:
            matter_id: ID of the matter

        Returns:
            CanonicalMatter if mapped, None otherwise
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (m:Matter {id: $matter_id})-[:MAPPED_TO]->(cm:CanonicalMatter)
                RETURN cm, labels(cm) as labels
                """,
                matter_id=matter_id
            )

            record = result.single()
            if not record:
                return None

            node = record["cm"]
            labels = record["labels"]

            # Create the appropriate canonical type based on labels
            if "CanonicalProblem" in labels:
                return CanonicalProblem(
                    id=node["id"],
                    description=node["description"]
                )
            elif "CanonicalCondition" in labels:
                return CanonicalCondition(
                    id=node["id"],
                    description=node["description"]
                )
            elif "CanonicalGoal" in labels:
                return CanonicalGoal(
                    id=node["id"],
                    description=node["description"]
                )
            elif "CanonicalSolution" in labels:
                return CanonicalSolution(
                    id=node["id"],
                    description=node["description"]
                )
            else:
                # Base canonical matter type
                return CanonicalMatter(
                    id=node["id"],
                    description=node["description"]
                )

    def get_problem_canonical_form(self, problem_id: str) -> Optional[CanonicalProblem]:
        """Get the canonical form of a problem (legacy method).

        Args:
            problem_id: ID of the problem

        Returns:
            CanonicalProblem if mapped, None otherwise
        """
        canonical_matter = self.get_matter_canonical_form(problem_id)

        if canonical_matter and isinstance(canonical_matter, CanonicalProblem):
            return canonical_matter

        return None

    def get_condition_canonical_form(self, condition_id: str) -> Optional[CanonicalCondition]:
        """Get the canonical form of a condition (legacy method).

        Args:
            condition_id: ID of the condition

        Returns:
            CanonicalCondition if mapped, None otherwise
        """
        canonical_matter = self.get_matter_canonical_form(condition_id)

        if canonical_matter and isinstance(canonical_matter, CanonicalCondition):
            return canonical_matter

        return None

    def get_goal_canonical_form(self, goal_id: str) -> Optional[CanonicalGoal]:
        """Get the canonical form of a goal.

        Args:
            goal_id: ID of the goal

        Returns:
            CanonicalGoal if mapped, None otherwise
        """
        canonical_matter = self.get_matter_canonical_form(goal_id)

        if canonical_matter and isinstance(canonical_matter, CanonicalGoal):
            return canonical_matter

        return None

    def get_solution_canonical_form(self, solution_id: str) -> Optional[CanonicalSolution]:
        """Get the canonical form of a solution.

        Args:
            solution_id: ID of the solution

        Returns:
            CanonicalSolution if mapped, None otherwise
        """
        canonical_matter = self.get_matter_canonical_form(solution_id)

        if canonical_matter and isinstance(canonical_matter, CanonicalSolution):
            return canonical_matter

        return None

    def get_matter_variants(self, canonical_id: str, label: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all matters mapped to a canonical form, optionally filtered by label.

        Args:
            canonical_id: ID of the canonical matter
            label: Optional label to filter by (e.g., "Problem", "Condition")

        Returns:
            List of matters mapped to this canonical form
        """
        with self.driver.session() as session:
            # Build the query based on whether a label filter is provided
            if label:
                query = f"""
                MATCH (m:Matter:{label})-[:MAPPED_TO]->(cm:CanonicalMatter {{id: $canonical_id}})
                RETURN m, labels(m) as labels
                """
            else:
                query = """
                MATCH (m:Matter)-[:MAPPED_TO]->(cm:CanonicalMatter {id: $canonical_id})
                RETURN m, labels(m) as labels
                """

            result = session.run(query, canonical_id=canonical_id)

            matters = []
            for record in result:
                node = record["m"]
                labels = record["labels"]

                # Core properties for all matter types
                matter_data = {
                    "id": node["id"],
                    "description": node["description"],
                    "labels": labels,
                    "created_at": node.get("created_at"),
                    "updated_at": node.get("updated_at"),
                    "tags": node.get("tags", [])
                }

                # Add label-specific properties
                if MatterLabel.PROBLEM.value in labels:
                    matter_data["state"] = node.get("state")
                    matter_data["priority"] = node.get("priority")

                if MatterLabel.CONDITION.value in labels:
                    matter_data["is_met"] = node.get("is_met", False)
                    matter_data["verification_method"] = node.get("verification_method")

                if MatterLabel.GOAL.value in labels:
                    matter_data["target_date"] = node.get("target_date")
                    matter_data["progress"] = node.get("progress", 0.0)

                if MatterLabel.SOLUTION.value in labels:
                    matter_data["state"] = node.get("state")
                    matter_data["implementation_date"] = node.get("implementation_date")

                if "embedding" in node:
                    matter_data["embedding"] = node["embedding"]

                matters.append(matter_data)

            return matters

    def get_problem_variants(self, canonical_id: str) -> List[Dict[str, Any]]:
        """Get all problems mapped to a canonical form (legacy method).

        Args:
            canonical_id: ID of the canonical problem

        Returns:
            List of problems mapped to this canonical form
        """
        return self.get_matter_variants(canonical_id, MatterLabel.PROBLEM.value)

    def get_condition_variants(self, canonical_id: str) -> List[Dict[str, Any]]:
        """Get all conditions mapped to a canonical form (legacy method).

        Args:
            canonical_id: ID of the canonical condition

        Returns:
            List of conditions mapped to this canonical form
        """
        return self.get_matter_variants(canonical_id, MatterLabel.CONDITION.value)

    def get_goal_variants(self, canonical_id: str) -> List[Dict[str, Any]]:
        """Get all goals mapped to a canonical form.

        Args:
            canonical_id: ID of the canonical goal

        Returns:
            List of goals mapped to this canonical form
        """
        return self.get_matter_variants(canonical_id, MatterLabel.GOAL.value)

    def get_solution_variants(self, canonical_id: str) -> List[Dict[str, Any]]:
        """Get all solutions mapped to a canonical form.

        Args:
            canonical_id: ID of the canonical solution

        Returns:
            List of solutions mapped to this canonical form
        """
        return self.get_matter_variants(canonical_id, MatterLabel.SOLUTION.value)

