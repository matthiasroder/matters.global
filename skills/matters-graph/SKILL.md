---
name: matters-graph
description: Query and manage the matters.global knowledge graph — goals, problems, conditions, solutions, and relationships in Neo4j.
---

# Matters Graph

## When to use
When the user asks about their goals, problems, conditions, solutions, knowledge graph, or matters.

## Available MCP Tools (via matters-graph server)

### Query
- list_matters(labels?, limit?) — list all matters, filter by Goal/Problem/Condition/Solution
- get_matter_details(matter_id) — details + connections for a matter
- find_similar_matters(description, labels?, threshold?, limit?) — vector similarity search
- list_problems(state?, limit?) — list problems by state
- get_problem_details(problem_id) — problem details
- find_similar_problems(description, threshold?, limit?) — similar problems

### Create
- create_goal(description, target_date?, progress?, tags?)
- create_problem(description, state?, priority?, tags?)
- create_condition(description, is_met?, verification_method?, tags?)
- create_solution(description, state?, tags?)

### Relationships
- add_relationship(source_id, target_id, relationship_type)
  Types: REQUIRES, BLOCKS, ENABLES, RELATES_TO, PRECEDES, FOLLOWS, PART_OF, CONSISTS_OF, SOLVED_BY, ADDRESSES, FULFILLS, MAPPED_TO, DERIVED_FROM
- add_problem_dependency(problem_id, depends_on_id)
- add_condition_to_problem(problem_id, condition_description, is_met?)
- add_solution_to_matter(matter_id, solution_id)

### Update
- set_goal_progress(goal_id, progress) — progress is 0-1
- update_condition(condition_id, is_met)

## Important
- All results have success: true/false — check before processing
- Always ask user permission before creating or modifying matters
- IDs are UUIDs
- Similarity search uses OpenAI embeddings
