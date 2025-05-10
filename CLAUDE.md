# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT: CONDA ENVIRONMENT USAGE

This project STRICTLY uses conda for dependency management:
- ALWAYS use conda commands for package management, NEVER use pip
- ALL dependencies should be managed through environment.yml
- ALWAYS verify you're in the activated conda environment before running commands
- If you need to add a package, add it to environment.yml and update the environment
- Run `conda env update -f environment.yml` to update the environment if changes are made

## Environment Setup
- Conda environment: `mattersglobal` (created using environment.yml)
- Python version: 3.9+

## Commands
- Create environment: `conda env create -f environment.yml`
- Activate environment: `conda activate mattersglobal`
- Update environment: `conda env update -f environment.yml --prune`
- Check installed packages: `conda list`
- Start the application: `./start.sh`
- Run WebSocket server: `python websocket_server.py`
- Run REST API server: `python server.py`
- Run UI development server: `cd ui && npm run dev`

## Package Management
- To install a package: Add it to environment.yml and run `conda env update -f environment.yml`
- To update a package: Change version in environment.yml and run `conda env update -f environment.yml`
- To check package versions: `conda list [package_name]`
- To search for package availability: `conda search -c conda-forge [package_name]`

## Code Style Guidelines
- Use Python 3.9+ features
- Follow PEP 8 conventions for formatting
- Use type hints with Python's typing module
- Use Pydantic for data validation and parsing
- Class naming: PascalCase (e.g., ProblemDefinition)
- Variable naming: snake_case (e.g., problem_definition)
- Handle errors with try/except blocks and clear error messages
- Use docstrings for functions and classes
- Keep code modular and functions focused on a single responsibility

## Implementation Progress Summary

We've successfully migrated from a JSON-based storage system to a graph database implementation using Neo4j. The implementation focuses on flexibility, particularly with embedding providers, and provides a foundation for advanced semantic similarity features.

### Completed Components

1. **Neo4j Integration** ✓
   - Set up Neo4j environment with appropriate constraints and indexes
   - Created GraphProblemManager class with CRUD operations
   - Implemented problem-condition-solution relationships in graph

2. **Embedding Architecture** ✓
   - Designed a modular embedding system with swappable providers
   - Implemented OpenAI embedding provider (with fallbacks)
   - Created configuration system using Pydantic models
   - Added provider factory pattern for runtime switching

3. **Vector Similarity Features** ✓
   - Added embedding storage in Neo4j nodes
   - Implemented vector-based similarity search with indexing
   - Created fallback mechanisms for text-based search

4. **Graph Relationships** ✓
   - Built problem dependency relationships and queries
   - Implemented methods for traversing problem connections
   - Added foundation for canonical form mapping

### Current Graph Schema
- **Node Types**:
  - `Problem`: {id, description, state, embedding}
  - `Condition`: {id, description, is_met, embedding}
  - `Solution`: {id, description, embedding}
  - `CanonicalProblem`: {id, description}
  - `CanonicalCondition`: {id, description}

- **Relationships**:
  - `(Problem)-[:REQUIRES]->(Condition)`: Problem requires condition
  - `(Problem)-[:MUST_BE_RESOLVED_BEFORE]->(Problem)`: Problem must be resolved before another problem
  - `(Problem)-[:SOLVED_BY]->(Solution)`: Solution that solved the problem
  - `(Problem)-[:MAPPED_TO]->(CanonicalProblem)`: Links variant to canonical form
  - `(Condition)-[:MAPPED_TO]->(CanonicalCondition)`: Links variant to canonical form

### Completed Tasks

1. **Entity Resolution System** ✓
   - Implemented algorithms for identifying similar problems/conditions
   - Created methods for suggesting canonical forms using multiple strategies
   - Built automatic merging functionality for similar entities
   - Added fallback mechanisms for different resolution approaches

2. **Canonical Node Mapping** ✓
   - Added methods to create and maintain canonical nodes
   - Implemented algorithms for mapping variants to canonical forms
   - Created system for automatically resolving entities into canonical groups

3. **Semantic Analysis Integration** ✓
   - Integrated OpenAI for semantic analysis of problems
   - Implemented intelligent canonical description generation
   - Added multiple resolution strategies with fallbacks

### Remaining Tasks

1. **User Interfaces**
   - Build user interfaces for reviewing and confirming mappings
   - Create forms for manual entity resolution management
   - Implement feedback mechanisms for suggested mappings

2. **Visualization Tools**
   - Create visualization tools for the problem network
   - Build interactive graph exploration interfaces
   - Implement visual indicators for canonical relationships

3. **Monitoring Dashboard**
   - Implement dashboard for monitoring problem relationships
   - Create statistics and metrics for system analysis
   - Build reporting tools for entity resolution effectiveness

### Files Created
- `graph_problem_manager.py`: Core Neo4j implementation
- `embedding_providers.py`: Modular embedding system
- `entity_resolution.py`: Entity resolution and canonical mapping
- `config/embeddings.json`: Configuration for embedding providers
- `test_graph_connection.py`: Testing Neo4j CRUD operations
- `test_embeddings.py`: Testing embedding generation
- `test_graph_embeddings.py`: Testing embedding storage and retrieval
- `test_entity_resolution.py`: Testing entity resolution and canonical mapping
- `README.md`: Comprehensive documentation
- `environment.yml`: Conda environment specification

### Required Dependencies
All dependencies are specified in the environment.yml file, including:
- neo4j-python-driver
- pydantic (for validation)
- openai (for embeddings and LLM integration)
- flask and flask-cors (for REST API)
- websockets (for WebSocket server)
- numpy (for numerical operations)
- pytest (for testing)

### Future Considerations
- Add support for authentication and user management
- Implement domain-specific knowledge organization
- Build visualization components for graph exploration
- Consider performance optimizations for large graph operations