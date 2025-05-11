# Multi-Label Schema Documentation

This document describes the multi-label schema implementation for the Matters.Global system. The schema uses Neo4j's label system to create flexible entity types that can have multiple roles in the knowledge graph.

## Core Concepts

### 1. Base Matter Type

All entities in the system derive from the base `Matter` type, which provides common properties and behaviors:

- **ID**: Unique identifier
- **Description**: Textual description of the matter
- **Created/Updated timestamps**: Track when entities are created and modified
- **Embedding**: Vector representation for semantic similarity
- **Tags**: Optional categorization tags
- **Labels**: Neo4j labels that define the entity's roles

### 2. Label-Based Type System

Instead of rigid single-type entities, our schema uses Neo4j's label system to assign multiple "types" to entities:

- **Matter**: Base label that all entities have
- **Goal**: Represents a desired outcome or objective
- **Problem**: Represents an issue that needs to be solved
- **Condition**: Represents a requirement that must be met
- **Solution**: Represents a way to solve a problem or achieve a goal

An entity can have **multiple labels**, allowing it to fulfill different roles in different contexts. For example, an entity can be both a Problem and a Condition.

### 3. Rich Relationship Types

Matters connect to each other through typed relationships that express how they interact:

- **REQUIRES**: Entity requires another entity to be completed/resolved
- **PRECEDES**: Entity must be completed before another
- **CONSISTS_OF**: Entity is composed of other entities
- **PART_OF**: Entity is a component of another entity
- **RELATES_TO**: General relationship between entities
- **SOLVED_BY**: Problem is solved by a solution
- **ADDRESSES**: Solution addresses a problem
- **FULFILLS**: Solution fulfills a condition
- And more...

## Schema Implementation

### Entity Classes

The core entity classes are:

```python
class Matter(BaseModel):
    id: str
    description: str
    created_at: datetime
    updated_at: datetime
    embedding: Optional[List[float]]
    tags: List[str]
    labels: List[str]
    
    def add_label(self, label)
    def remove_label(self, label)
    def has_label(self, label)
```

Specialized entities add type-specific properties:

```python
class Goal(Matter):
    target_date: Optional[datetime]
    progress: float
    
class Problem(Matter):
    state: ProblemState
    priority: Optional[int]
    
class Condition(Matter):
    is_met: bool
    verification_method: Optional[str]
    
class Solution(Matter):
    state: SolutionState
    implementation_date: Optional[datetime]
```

### Database Schema

In Neo4j, entities are stored with multiple labels:

```cypher
CREATE (m:Matter:Problem:Condition {
    id: "abc123",
    description: "API response times must be under 200ms",
    created_at: datetime(),
    updated_at: datetime(),
    state: "not_solved",
    is_met: false
})
```

## Key Features

### 1. Multi-Label Entities

Entities can have multiple labels, allowing them to serve different purposes:

- A requirement can be both a **Condition** (something to verify) and a **Problem** (something to solve)
- An achievement can be both a **Goal** (desired outcome) and a **Solution** (resolution to a problem)

### 2. Flexible Querying

Queries can filter by one or more labels:

```python
# Find matters with both Problem and Condition labels
manager.find_similar_matters(
    description="...",
    labels=["Problem", "Condition"]
)
```

### 3. Label Transitions

Entities can change their roles over time:

- A **Problem** with no clear solution can be converted to a **Condition** for a higher-level problem
- A **Goal** can become a **Problem** when obstacles are encountered
- A **Solution** can be refined into a new **Goal** for implementation

## Use Cases

### 1. Problem-Condition Duality

The schema elegantly handles cases where problems and conditions overlap:

- **Condition-First**: "API must be secure" is a condition that becomes a problem when we need to implement it
- **Problem-First**: "API is insecure" is a problem that, when solved, satisfies a condition

### 2. Goal Hierarchy and Progress Tracking

The schema supports hierarchical goal structures:

- Main goals consist of sub-goals
- Sub-goals require conditions to be met
- Conditions can be checked to calculate goal progress

### 3. Complex Problem Resolution

Complex problems can be modeled with various entity types:

- Problems may require conditions to be considered solved
- Problems can depend on other problems being solved first
- Solutions can address multiple problems simultaneously

## API Functions

The system provides functions for working with the multi-label schema:

### Creation Functions

```python
# Create entities of different types
manager.create_matter(matter)
manager.create_goal(description, target_date, progress)
manager.create_problem(description, state, priority)
manager.create_condition(description, is_met, verification_method)
manager.create_solution(description, state)
```

### Retrieval Functions

```python
# Get entities by ID and type
manager.get_matter_by_id(matter_id)
manager.get_goal_by_id(goal_id)
manager.get_problem_by_id(problem_id)
```

### Relationship Functions

```python
# Create relationships between entities
manager.create_relationship(relationship)
manager.set_matter_relationship(source_id, relationship_type, target_id)
manager.add_solution_to_matter(matter_id, solution_id)
```

### Search Functions

```python
# Find similar entities
manager.find_similar_matters(description, labels, threshold, limit)
```

### Status Update Functions

```python
# Update entity status
manager.update_condition(condition_id, is_met)
manager.set_goal_progress(goal_id, progress)
manager.check_if_problem_solved(problem_id)
```

## Migration from Previous Schema

The multi-label schema is backward compatible with the previous single-label schema:

- Legacy functions still work with the new implementation
- Existing Problem and Condition entities are treated as having single labels
- New entities can take advantage of multiple labels
- Queries automatically handle both old and new patterns

## Best Practices

1. **Start General**: Create entities with a single primary label first
2. **Add Labels as Needed**: Add additional labels when entities take on new roles
3. **Use Appropriate Relationships**: Choose relationship types that express the intended meaning
4. **Check Label Combinations**: Not all label combinations make sense (e.g., a Solution is rarely also a Problem)
5. **Update Status Carefully**: Status changes on dual-labeled entities affect all their roles

## Examples

### Creating a Dual-Purpose Entity

```python
# Create as a problem first
problem = manager.create_problem(
    description="API must use secure authentication"
)

# Add Condition label
with manager.driver.session() as session:
    session.run(
        """
        MATCH (m:Matter {id: $id})
        SET m:Condition, m.is_met = false
        RETURN m
        """,
        id=problem.id
    )
```

### Finding Entities with Multiple Labels

```python
# Find matters that are both Problems and Conditions
results = manager.find_similar_matters(
    description="API security requirements",
    labels=["Problem", "Condition"]
)
```

### Creating a Goal Hierarchy

```python
# Create main goal
main_goal = manager.create_goal(
    description="Implement secure API system"
)

# Create sub-goal
sub_goal = manager.create_goal(
    description="Implement OAuth authentication"
)

# Link them
manager.set_matter_relationship(
    main_goal.id,
    RelationshipType.CONSISTS_OF,
    sub_goal.id
)
```

## Conclusion

The multi-label schema provides a flexible foundation for representing complex knowledge structures in the Matters.Global system. By allowing entities to have multiple roles and rich relationships, it can model real-world scenarios where categories overlap and evolve over time.