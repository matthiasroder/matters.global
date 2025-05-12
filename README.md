# Matters.Global Knowledge Management System

A flexible system for managing goals, problems, conditions, and solutions with semantic similarity features, entity resolution, and graph relationships.

## Architecture

The system consists of:

1. **Backend Implementation**
   - Graph database (Neo4j) with flexible multi-label schema
   - Embedding-based semantic similarity features
   - Entity resolution and canonical mapping system
   - Multiple relationship types for connecting matters

2. **Chat Interface**
   - Conversational UI for interacting with the system
   - OpenAI Assistant integration with multi-label awareness
   - Function calling to manage different matter types

## Setup Instructions

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) package manager
- Neo4j database (5.11+ recommended for vector search)
- OpenAI API key
- Node.js and npm (for the UI)

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/matters.global.git
   cd matters.global
   ```

2. Create the conda environment from the environment.yml file:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:
   ```bash
   conda activate mattersglobal
   ```

4. Set up environment variables:
   ```bash
   export OPENAI_API_KEY="your_openai_api_key"
   export NEO4J_URI="bolt://localhost:7687"
   export NEO4J_USERNAME="neo4j"
   export NEO4J_PASSWORD="password"
   ```

5. Quick start (automated setup and launch):
   ```bash
   ./start_new.sh
   ```
   This script will check dependencies, start the WebSocket server, and launch the UI.

### Neo4j Setup

1. Install and start Neo4j:
   - Download from [Neo4j website](https://neo4j.com/download/)
   - Or use Docker: `docker run -p 7474:7474 -p 7687:7687 neo4j:5.11`

2. Create a new database and set password

3. Ensure the database is accessible at the URI specified in your environment variables

### Running the Server

You have two server options:

#### Option 1: WebSocket Server (recommended for the UI)

1. Start the WebSocket server:
   ```bash
   python websocket_server.py
   ```

   The server will start on port 8090 by default.

2. You should see output confirming:
   - Connection to Neo4j
   - Creation or retrieval of the OpenAI Assistant
   - WebSocket server running and ready for connections

#### Option 2: REST API Server

1. Start the Flask server:
   ```bash
   python server.py
   ```

   The server will start on port 5000 by default.

2. You should see output confirming:
   - Connection to Neo4j
   - Creation or retrieval of the OpenAI Assistant
   - REST API server running and ready for connections

### Setting up the Chat UI

1. The chatbot UI is included in the `ui` directory:
   ```bash
   cd ui
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. The UI is already configured to connect to the WebSocket server at `ws://localhost:8091`. If you need to modify this:
   - Edit `src/config.ts` to change the `WEBSOCKET_ENDPOINT` value

4. Start the UI development server:
   ```bash
   npm run dev
   ```

5. Access the chat interface at: `http://localhost:5173` or `http://localhost:3000` (depending on Vite configuration)

## API Endpoints

The server exposes these main endpoints:

- `POST /api/chat/message` - Send a message to the assistant
- `GET /api/chat/history` - Get message history for a session
- `POST /api/chat/reset` - Reset or create a new session
- `GET /api/health` - Health check endpoint

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | None (Required) |
| `OPENAI_ASSISTANT_ID` | Optional existing assistant ID | None (Creates new) |
| `NEO4J_URI` | Neo4j connection URI | bolt://localhost:7687 |
| `NEO4J_USERNAME` | Neo4j username | neo4j |
| `NEO4J_PASSWORD` | Neo4j password | password |
| `PORT` | Server port | 5000 |
| `FLASK_SECRET_KEY` | Flask session secret | Random UUID |

## Customization

### System Message

To modify the assistant's behavior, edit the `SYSTEM_MESSAGE` in `assistant_manager.py`.

### Function Definitions

OpenAI function schemas are defined in `assistant_functions.py`. You can add new functions by:

1. Adding a function schema to the `FUNCTION_DEFINITIONS` list
2. Implementing a handler function
3. Adding the handler to the `FUNCTION_DISPATCH` dictionary

## Multi-Label Schema

The system uses a flexible multi-label schema that allows entities to serve multiple purposes simultaneously:

### Entity Types (Labels)

- **Matter**: Base type for all entities in the system
- **Goal**: A desired outcome or future state (e.g., "Launch mobile app by Q3")
- **Problem**: An issue that needs to be resolved (e.g., "API response times are too slow")
- **Condition**: A requirement that must be met (e.g., "System must pass security audit")
- **Solution**: An approach to solve a problem or achieve a goal (e.g., "Implement caching")

An entity can have multiple labels, reflecting its multiple roles in the system. For example:
- A requirement might be both a **Condition** and a **Problem** that needs solving
- A milestone might be both a **Goal** to achieve and a **Condition** for a larger goal
- An implementation task might be both a **Problem** to solve and a **Solution** to another problem

### Entity Properties

Each entity type has specific properties relevant to its role:

```
Matter {
    id: String,                 // Unique identifier
    description: String,        // Human-readable description
    created_at: DateTime,       // Creation timestamp
    updated_at: DateTime,       // Last update timestamp
    embedding: [Float],         // Vector embedding for similarity
    tags: [String]              // Optional categorization tags
}

Goal {
    target_date: DateTime,      // When the goal should be achieved
    progress: Float             // Progress toward completion (0-1)
}

Problem {
    problem_state: String,      // "solved", "not_solved", or "obsolete"
    priority: Integer           // Optional priority ranking
}

Condition {
    is_met: Boolean,            // Whether the condition is satisfied
    verification_method: String // How to verify the condition
}

Solution {
    solution_state: String,     // "theoretical", "in_progress", "implemented", or "failed"
    implementation_date: DateTime // When the solution was implemented
}
```

## Graph Relationships

The system uses Neo4j relationships to model connections between different matter types:

### Core Structural Relationships

- `(:Matter)-[:REQUIRES]->(:Matter)`: Entity A requires entity B to be resolved/achieved
- `(:Matter)-[:BLOCKS]->(:Matter)`: Entity A blocks progress on entity B
- `(:Matter)-[:ENABLES]->(:Matter)`: Entity A enables or facilitates entity B
- `(:Matter)-[:RELATES_TO]->(:Matter)`: Generic association between entities

### Temporal and Sequencing Relationships

- `(:Matter)-[:PRECEDES]->(:Matter)`: Entity A must be handled before entity B
- `(:Matter)-[:FOLLOWS]->(:Matter)`: Entity A should be handled after entity B

### Compositional Relationships

- `(:Matter)-[:PART_OF]->(:Matter)`: Entity A is a component of entity B
- `(:Matter)-[:CONSISTS_OF]->(:Matter)`: Entity A consists of entity B

### Resolution Relationships

- `(:Matter)-[:SOLVED_BY]->(:Matter:Solution)`: Entity A is resolved by solution B
- `(:Matter:Solution)-[:ADDRESSES]->(:Matter)`: Solution A addresses entity B
- `(:Matter:Solution)-[:FULFILLS]->(:Matter:Condition)`: Solution A fulfills condition B

### Canonical Relationships

- `(:Matter)-[:MAPPED_TO]->(:Matter)`: Entity A is a variant of canonical form B
- `(:Matter)-[:DERIVED_FROM]->(:Matter)`: Entity A is derived from or inspired by entity B

## Features

### Semantic Similarity

The system uses embedding-based semantic similarity to:
- Find similar entities regardless of exact wording
- Group related matters for canonical form mapping
- Suggest potential connections between entities
- Support "fuzzy" search across the knowledge graph

### Entity Resolution

The multi-stage entity resolution system helps maintain a clean knowledge graph by:
- Identifying semantically similar entities using vector embeddings
- Suggesting canonical forms for groups of similar entities
- Providing multiple resolution strategies with fallbacks
- Supporting both automatic and user-guided resolution

### Hybrid Vector + LLM Approach

The system combines vector embeddings and LLM capabilities:
- Vector similarity for efficient first-pass filtering
- LLM-based refinement for deeper semantic understanding
- Explanation generation for detected similarities
- Fallback mechanisms when vector search yields poor results

## Future Development

See the BRAINSTORMING.md file for plans on future development, including:
- Advanced visualization tools for the multi-label graph
- User interfaces for reviewing and confirming mappings
- Monitoring dashboard for knowledge graph analytics
- Multi-user support with authentication
- Native mobile applications with offline capabilities
