# Matters.Global Data Schema

This document outlines the unified data schema for Matters.Global, using Neo4j's labeled property graph model with a flexible, multi-label approach.

## Core Concepts

The schema uses a multiple-label approach to represent the fluid nature of goals, problems, conditions, and solutions:

- A single node can have multiple labels reflecting its different roles in the graph
- Relationships express the connections between entities
- Properties store attributes specific to each entity and its roles

## Node Structure

### Base Node Type

All nodes share the base label `:Matter` with common properties:

```cypher
(m:Matter {
    id: String,                 // UUID
    description: String,        // Human-readable description
    created_at: DateTime,       // Creation timestamp
    updated_at: DateTime,       // Last update timestamp
    embedding: [Float],         // Vector embedding for similarity
    tags: [String]              // Optional categorization tags
})
```

### Specialized Labels

Nodes have additional labels reflecting their roles in different contexts:

- `:Goal` - Desired state or outcome
- `:Problem` - Something requiring resolution
- `:Condition` - Criterion that must be met
- `:Solution` - Approach to achieve a goal or resolve a problem

### Label-Specific Properties

```cypher
(:Goal {
    target_date: DateTime,  // Optional target for achievement
    progress: Float         // Optional progress indicator (0-1)
})

(:Problem {
    state: "solved" | "not_solved" | "obsolete",
    priority: Integer  // Optional priority ranking
})

(:Condition {
    is_met: Boolean,
    verification_method: String  // Optional description of how to verify
})

(:Solution {
    state: "theoretical" | "in_progress" | "implemented" | "failed",
    implementation_date: DateTime  // When solution was implemented
})
```

## Relationships

### Core Structural Relationships

```cypher
// Core structural relationships
(:Matter)-[:REQUIRES]->(:Matter)            // A requires B to be resolved/achieved
(:Matter)-[:BLOCKS]->(:Matter)              // A blocks progress on B
(:Matter)-[:ENABLES]->(:Matter)             // A enables or facilitates B
(:Matter)-[:RELATES_TO {strength: Float}]->(:Matter)  // Generic association with strength
```

### Temporal and Sequencing Relationships

```cypher
(:Matter)-[:PRECEDES]->(:Matter)            // A must be handled before B
(:Matter)-[:FOLLOWS]->(:Matter)             // A should be handled after B
```

### Compositional Relationships

```cypher
(:Matter)-[:PART_OF]->(:Matter)             // A is a component of B
(:Matter)-[:CONSISTS_OF]->(:Matter)         // A consists of B (inverse of PART_OF)
```

### Resolution Relationships

```cypher
(:Matter)-[:SOLVED_BY]->(:Matter:Solution)  // A is resolved by solution B
(:Matter:Solution)-[:ADDRESSES]->(:Matter)  // Solution A addresses matter B
(:Matter:Solution)-[:FULFILLS]->(:Matter:Condition)  // Solution A fulfills condition B
```

### Canonical Relationships

```cypher
(:Matter)-[:MAPPED_TO]->(:Matter)           // A is a variant of canonical form B
(:Matter)-[:DERIVED_FROM]->(:Matter)        // A is derived from or inspired by B
```

## Multi-Label Pattern Examples

### Dual-Role Nodes

```cypher
// A condition that is also a problem in its own right
(n:Matter:Condition:Problem {
    description: "Secure sufficient funding",
    is_met: false,
    state: "not_solved",
    priority: 1
})

// A goal that is represented as a problem to solve
(n:Matter:Goal:Problem {
    description: "Reach 10k monthly users",
    target_date: "2023-12-31",
    state: "not_solved"
})

// A solution that addresses multiple matters
(s:Matter:Solution)-[:ADDRESSES]->(g:Matter:Goal)
(s)-[:ADDRESSES]->(p:Matter:Problem)
(s)-[:FULFILLS]->(c:Matter:Condition)
```

## Example Graph Patterns

```cypher
// A goal with conditions that themselves are problems
(g1:Matter:Goal)-[:REQUIRES]->(c1:Matter:Condition:Problem)
(c1)-[:REQUIRES]->(c2:Matter:Condition)

// A goal expressed as a problem to solve
(g1:Matter:Goal)-[:MAPPED_TO]->(p1:Matter:Problem)

// A solution that addresses multiple related issues
(s1:Matter:Solution)-[:ADDRESSES]->(g1:Matter:Goal)
(s1)-[:ADDRESSES]->(p1:Matter:Problem)
(s1)-[:FULFILLS]->(c1:Matter:Condition)

// Prerequisite relationship between goals
(g1:Matter:Goal)-[:PRECEDES]->(g2:Matter:Goal)

// Complex solution with components
(s1:Matter:Solution)-[:CONSISTS_OF]->(s2:Matter:Solution)
(s1)-[:CONSISTS_OF]->(s3:Matter:Solution)
```

## Schema Indices and Constraints

```cypher
// Unique ID constraint
CREATE CONSTRAINT matter_id_unique IF NOT EXISTS
FOR (m:Matter) REQUIRE m.id IS UNIQUE;

// Embedding vector index (Neo4j 5.11+)
CREATE VECTOR INDEX matter_embedding_index IF NOT EXISTS
FOR (m:Matter) ON m.embedding
OPTIONS {indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
}};

// Full-text search index for descriptions
CREATE FULLTEXT INDEX matter_description_index IF NOT EXISTS
FOR (m:Matter) ON EACH [m.description]
OPTIONS {indexConfig: {`fulltext.analyzer`: 'english'}};
```

## Query Examples

### Finding Related Matters

```cypher
// Find all conditions required for a specific goal
MATCH (g:Matter:Goal {id: $goal_id})-[:REQUIRES]->(c:Matter:Condition)
RETURN c

// Find all goals blocked by a specific problem
MATCH (p:Matter:Problem {id: $problem_id})-[:BLOCKS]->(g:Matter:Goal)
RETURN g

// Find all solutions that fulfill a specific condition
MATCH (s:Matter:Solution)-[:FULFILLS]->(c:Matter:Condition {id: $condition_id})
RETURN s
```

### Finding Similar Matters

```cypher
// Find semantically similar matters using vector similarity
MATCH (m:Matter)
WHERE m.id <> $matter_id
WITH m, gds.similarity.cosine(m.embedding, $embedding) AS similarity
WHERE similarity > 0.7
RETURN m, similarity
ORDER BY similarity DESC
LIMIT 10
```

### Complex Traversals

```cypher
// Find all matters that must be resolved before achieving a specific goal
MATCH path = (m:Matter)-[:PRECEDES*]->(g:Matter:Goal {id: $goal_id})
RETURN nodes(path) as prerequisites
```

## Implementation Notes

1. **Flexibility**: This schema is designed to evolve. New labels and relationship types can be added without breaking existing queries.

2. **Migration Path**: When migrating from the previous schema:
   - Convert existing Problem nodes to (:Matter:Problem)
   - Convert existing Condition nodes to (:Matter:Condition)
   - Convert existing Solution nodes to (:Matter:Solution)
   - Preserve relationship directions and types

3. **Transaction Handling**: When creating complex structures involving multiple nodes and relationships, use transactions to ensure consistency.

4. **Query Optimization**: Use parameters in Cypher queries to allow for query plan caching and better performance.