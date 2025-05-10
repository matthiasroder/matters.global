# Matters.Global Problem Management System

A system for managing, tracking, and resolving problems with semantic similarity features, entity resolution, and graph relationships.

## Architecture

The system consists of:

1. **Backend Implementation**
   - Graph database (Neo4j) for problem storage
   - Embedding-based similarity features
   - Entity resolution system

2. **Chat Interface**
   - Conversational UI for interacting with the system
   - OpenAI Assistant integration
   - Function calling to manage problems

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

## Graph Relationships

The system uses Neo4j relationships to model connections between problems:

- `(Problem)-[:REQUIRES]->(Condition)`: A problem requires a condition to be met
- `(Problem)-[:MUST_BE_RESOLVED_BEFORE]->(Problem)`: Problem A must be resolved before Problem B can be solved
- `(Problem)-[:SOLVED_BY]->(Solution)`: A problem is solved by a solution
- `(Problem)-[:MAPPED_TO]->(CanonicalProblem)`: Maps problem variants to canonical form
- `(Condition)-[:MAPPED_TO]->(CanonicalCondition)`: Maps condition variants to canonical form

The `MUST_BE_RESOLVED_BEFORE` relationship models a clear sequential dependency between problems. For example, if Problem A must be resolved before Problem B, there is a direct relationship: `(A)-[:MUST_BE_RESOLVED_BEFORE]->(B)`.

## Future Development

See the BRAINSTORMING.md file for plans on future development, including:
- User interfaces
- Visualization tools
- Monitoring dashboard
- Authentication system
