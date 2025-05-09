# Matters.Global - Graph-Based Problem Management System

A flexible, Neo4j-powered knowledge graph system for managing problems, conditions, solutions, and their relationships with advanced semantic similarity capabilities.

## Overview

Matters.Global is a problem management system that uses a graph database (Neo4j) to store and organize problems, their conditions, and solutions. The system leverages vector embeddings for semantic similarity search and includes an entity resolution system that can identify similar problems and suggest canonical representations.

## Features

- **Neo4j Graph Database Integration**: Store problems in a flexible, connected graph structure
- **Vector Embeddings**: Generate and store embeddings for semantic similarity
- **Entity Resolution System**: Identify similar entities and map them to canonical forms
- **Canonical Form Management**: Create and maintain standardized representations of similar entities
- **Auto Mapping**: Automatically group similar problems/conditions and create canonical forms
- **Embedding Provider Flexibility**: Pluggable architecture for different embedding providers (OpenAI, etc.)
- **Dependency Management**: Track problem dependencies and relationships
- **Semantic Analysis**: Use NLP techniques and LLMs for problem analysis

## Architecture

### Core Components

1. **GraphProblemManager**: Main interface for CRUD operations on the graph database
2. **EmbeddingProviders**: System for generating and using vector embeddings
3. **EntityResolutionSystem**: Tools for identifying similar entities and suggesting canonical forms

### Data Model

- **Node Types**:
  - `Problem`: {id, description, state, embedding}
  - `Condition`: {id, description, is_met, embedding}
  - `Solution`: {id, description, embedding}
  - `CanonicalProblem`: {id, description}
  - `CanonicalCondition`: {id, description}

- **Relationships**:
  - `(Problem)-[:REQUIRES]->(Condition)`: Problem requires condition
  - `(Problem)-[:DEPENDS_ON]->(Problem)`: Problem depends on another problem
  - `(Problem)-[:SOLVED_BY]->(Solution)`: Solution that solved the problem
  - `(Problem)-[:MAPPED_TO]->(CanonicalProblem)`: Links variant to canonical form
  - `(Condition)-[:MAPPED_TO]->(CanonicalCondition)`: Links variant to canonical form

## Getting Started

### Prerequisites

- Python 3.9+
- Neo4j Database (version 5.x recommended, 5.11+ for vector indexing)
- Conda Environment Manager (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/matters.global.git
   cd matters.global
   ```

2. Create and activate the conda environment:
   ```bash
   conda create -n mattersglobal python=3.9
   conda activate mattersglobal
   ```

3. Install required packages:
   ```bash
   pip install neo4j pydantic openai numpy
   ```

4. Configure your Neo4j connection and OpenAI API key:
   - Update Neo4j credentials in your scripts or use environment variables
   - Set up the OpenAI API key in your environment:
     ```bash
     export OPENAI_API_KEY=your_api_key_here
     ```

5. Set up the embeddings configuration:
   ```bash
   mkdir -p config
   cp config/embeddings.json.example config/embeddings.json
   # Edit config/embeddings.json with your configuration
   ```

### Database Setup

1. Start Neo4j database
2. Run the initialization script to set up constraints and indexes:
   ```bash
   python test_graph_connection.py
   ```

## Usage

### Basic Operations

```python
from graph_problem_manager import GraphProblemManager

# Initialize manager
manager = GraphProblemManager(
    uri="bolt://localhost:7687",
    username="neo4j",
    password="your_password",
    embedding_config_path="config/embeddings.json"
)

# Connect to Neo4j
manager.connect()

# Initialize schema (creates indexes and constraints)
manager.initialize_schema()

# Create a problem
problem = manager.create_problem("Understanding vector embeddings for semantic similarity")

# Add conditions
condition1 = manager.add_condition_to_problem(
    problem.id,
    "Implement vector embedding generation"
)
condition2 = manager.add_condition_to_problem(
    problem.id,
    "Create similarity search functionality"
)

# Mark a condition as met
manager.update_condition(condition1.id, True)

# Check if problem is solved
is_solved = manager.check_if_problem_solved(problem.id)

# Find similar problems
similar_problems = manager.find_similar_problems("vector similarity search", threshold=0.7)
```

### Entity Resolution

```python
from entity_resolution import EntityResolutionSystem

# Create entity resolution system
er_system = EntityResolutionSystem(manager)

# Find similar entities for a specific problem
matches = er_system.find_similar_entities(problem.id, entity_type="Problem")

# Group similar problems
groups = er_system.group_similar_entities(entity_type="Problem", threshold=0.7)

# Get canonical suggestion for a group of problems
entity_ids = [problem1.id, problem2.id, problem3.id]
suggestion = er_system.suggest_canonical_form(entity_ids, entity_type="Problem")

# Create canonical node and map entities to it
canonical_id = er_system.create_canonical_node(suggestion)

# Auto-resolve similar entities
results = er_system.auto_resolve_entities(
    entity_type="Problem",
    threshold=0.7,
    min_group_size=2
)
```

### Running Tests

The repository includes several test scripts to demonstrate functionality:

1. **Test Graph Connection**:
   ```bash
   python test_graph_connection.py
   ```

2. **Test Embeddings**:
   ```bash
   python test_embeddings.py
   ```

3. **Test Graph Embeddings**:
   ```bash
   python test_graph_embeddings.py
   ```

4. **Test Entity Resolution**:
   ```bash
   python test_entity_resolution.py
   ```

## Configuration

### Embedding Configuration

The `config/embeddings.json` file controls the embedding provider settings:

```json
{
  "provider_type": "openai",
  "model_name": "text-embedding-3-small",
  "dimension": 1536,
  "normalize": true,
  "api_key": null,
  "organization": null
}
```

- `provider_type`: Type of embedding provider (e.g., "openai")
- `model_name`: Name of the embedding model to use
- `dimension`: Dimension of the embedding vectors
- `normalize`: Whether to normalize the embeddings to unit length
- `api_key`: API key (will use environment variable if null)
- `organization`: Organization ID for the API

## Advanced Features

### Canonical Node Management

The system includes tools for managing canonical representations of similar entities:

1. **Create Canonical Problems/Conditions**: Create standardized forms for similar entities
2. **Map to Canonical Forms**: Link variants to their canonical representations
3. **Get Variants**: Retrieve all problems/conditions mapped to a canonical form

### Automatic Entity Resolution

The entity resolution system can automatically:

1. Group similar entities based on semantic similarity
2. Suggest the best canonical description for each group
3. Create canonical nodes and map variants to them

### Vector Similarity Search

Search for similar entities using:

1. **Vector Indexes**: Fast similarity search using Neo4j's vector capabilities
2. **Fallback Text Search**: Text-based matching when vectors aren't available

## Future Enhancements

Planned features include:

1. **User Interface for Entity Resolution**: Review and approve suggested canonical forms
2. **Visualization Tools**: Visual exploration of the problem network
3. **Monitoring Dashboard**: Track problem relationships and statistics
4. **API Layer**: RESTful API for integration with other systems
5. **Multi-user Support**: Authentication and authorization for collaborative use

## Project Structure

- `graph_problem_manager.py`: Core Neo4j implementation
- `embedding_providers.py`: Modular embedding system
- `entity_resolution.py`: Entity resolution and canonical mapping
- `config/embeddings.json`: Configuration for embedding providers
- `test_*.py`: Test scripts for various components

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Acknowledgments

- This project uses Neo4j for graph database storage
- Embedding capabilities provided by OpenAI's API
- Built with Python and Pydantic