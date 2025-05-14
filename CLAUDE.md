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
- NEVER mention Claude, Anthropic, AI, or LLMs in any code, comments, documentation, or commit messages
- Keep all code completely "AI-free" in terms of references or attributions

## Engineering Principles
- Always prioritize proper engineering solutions over quick fixes
- Identify and address root causes rather than symptoms
- Write sustainable, maintainable code that future developers can understand
- Avoid temporary workarounds or "hacks" that bypass validation, error checking, or type safety
- Design for flexibility and extensibility from the start
- When facing multi-faceted problems, take the time to design a comprehensive solution
- Use proper abstraction and encapsulation to manage complexity
- Write tests that validate both the happy path and edge cases
- Document architectural decisions and their rationales
- Consider performance implications, but prioritize correctness and maintainability first
- Refactor when necessary, rather than accumulating technical debt

## Current System Architecture

Matters.Global is a flexible knowledge management system that uses a multi-label schema in Neo4j, allowing entities to have multiple roles simultaneously. The system leverages semantic similarity for entity resolution and knowledge organization.

### Core Components

1. **Neo4j Graph Database** ✓
   - Flexible multi-label schema with Matter as the base entity type
   - Rich relationship types for connecting different matter types
   - Vector embeddings stored directly on nodes for similarity search
   - Constraints and indexes for efficient querying and data integrity

2. **Embedding Architecture** ✓
   - Modular embedding system with swappable providers
   - Primary OpenAI embedding provider with fallback mechanisms
   - Configuration system using Pydantic models
   - Provider factory pattern for runtime switching

3. **Semantic Similarity Features** ✓
   - Vector-based similarity search with Neo4j vector indexing
   - Hybrid approach combining vector search and LLM refinement
   - Multiple fallback mechanisms for robust matching
   - Cross-entity type similarity search

4. **Entity Resolution System** ✓
   - Smart algorithms for identifying semantically similar entities
   - Canonical form suggestion using multiple strategies
   - Automatic and user-guided resolution workflows
   - Fallback mechanisms for different resolution scenarios

5. **Multi-Label Schema** ✓
   - Flexible entity labeling system for dual-purpose entities
   - Support for entities that evolve roles over time
   - Problem-condition duality with appropriate properties
   - Goal-problem relationship modeling

### Current Graph Schema
- **Entity Types**:
  - `Matter`: Base type for all entities
  - `Goal`: Desired outcome with target date and progress
  - `Problem`: Issue requiring resolution
  - `Condition`: Requirement that must be met
  - `Solution`: Approach to solve a problem or achieve a goal
  - `CanonicalMatter`: Canonical representation of entities

- **Relationships**:
  - Core structural: `REQUIRES`, `BLOCKS`, `ENABLES`, `RELATES_TO`
  - Temporal: `PRECEDES`, `FOLLOWS`
  - Compositional: `PART_OF`, `CONSISTS_OF`
  - Resolution: `SOLVED_BY`, `ADDRESSES`, `FULFILLS`
  - Canonical: `MAPPED_TO`, `DERIVED_FROM`

### Recent Completed Tasks

1. **Multi-Label Schema Implementation** ✓
   - Implemented flexible entity model with multiple label support
   - Added methods for adding/removing labels from entities
   - Created test cases demonstrating multi-label capabilities
   - Updated OpenAI assistant prompt to properly support the schema

2. **OpenAI Assistant Integration** ✓
   - Created conversational interface for the knowledge system
   - Developed function calling for graph manipulation
   - Built thread management for conversation history
   - Implemented robust error handling and recovery

### Remaining Tasks

1. **Visualization and UI**
   - Build user interfaces for reviewing and confirming mappings
   - Create visualization tools for the multi-label graph
   - Implement visual indicators for entities with multiple roles

2. **Multi-User Support**
   - Add authentication and user management
   - Create permission systems for shared knowledge graphs
   - Implement collaborative workflows

### Key Files

- `graph_problem_manager.py`: Core Neo4j implementation with multi-label support
- `embedding_providers.py`: Modular embedding system with provider factory
- `entity_resolution.py`: Entity resolution and canonical mapping
- `assistant_manager.py`: OpenAI Assistant integration with conversational interface
- `assistant_functions.py`: Function definitions and handlers for AI assistant
- `websocket_server.py`: WebSocket server for real-time communication
- `server.py`: REST API server for HTTP communication
- `config/embeddings.json`: Configuration for embedding providers
- `test_multi_label.py`: Tests for multi-label schema implementation
- `test_problem_condition_duality.py`: Tests for problem-condition duality patterns
- `README.md`: System documentation

### Required Dependencies

All dependencies are specified in the environment.yml file, including:
- neo4j-python-driver: For graph database connectivity
- pydantic: For data validation and parsing
- openai: For embeddings and AI assistant integration
- flask and flask-cors: For REST API server
- websockets: For WebSocket server
- numpy: For numerical operations with embeddings
- pytest: For testing

## ROADMAP
- **Test multi label implementation**
- **Enhance and test AI Assistant Functions**: See AIFeatures.md
- **Move to pip-based**
- **Make multi-user**: Authentication and authorization for multi-user environments
- **Implement local llm**
- **Move to server for beta test**
- **Mobile Integration**: Support for mobile clients with offline capabilities
- **Public (local) vs private (cloud)graphs**
- **Integrate ocean protocol as persistent layer**
- **Performance Optimization**: Scaling strategies for large knowledge graphs
