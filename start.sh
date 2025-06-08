#!/bin/bash
# Start script for matters.global with pip virtual environment

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set."
    echo "Please set it with: export OPENAI_API_KEY='your-key-here'"
    exit 1
fi

# Check for virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: Virtual environment is not active."
    echo "Please activate it with: source venv/bin/activate"
    echo "If the environment doesn't exist, create it with: python -m venv venv && pip install -r requirements.txt"
    exit 1
fi

# Check if Neo4j is running
echo "Checking Neo4j connection..."
CONNECTION_CHECK=0

# Try to check connection with Python
if python -c "import socket; s=socket.socket(); s.connect(('localhost', 7687)); s.close()" >/dev/null 2>&1; then
    CONNECTION_CHECK=1
fi

if [ $CONNECTION_CHECK -eq 0 ]; then
    echo "Warning: Cannot connect to Neo4j on port 7687."
    echo "Please make sure Neo4j is running before proceeding."
    read -p "Continue anyway? (y/n): " continue_anyway
    if [ "$continue_anyway" != "y" ]; then
        exit 1
    fi
else
    echo "Neo4j connection successful!"
fi

# Start the WebSocket server
echo "Starting WebSocket server..."
python websocket_server.py > websocket_server.log 2>&1 &
WEBSOCKET_PID=$!
echo "WebSocket server started with PID: $WEBSOCKET_PID"

# Give the server a moment to start
sleep 2

# Check if npm is installed
if ! command -v npm >/dev/null 2>&1; then
    echo "Error: npm is not installed. Please install Node.js and npm."
    echo "Visit https://nodejs.org/en/download/ for installation instructions."
    exit 1
fi

# Go to UI directory
if [ ! -d "ui" ]; then
    echo "Error: UI directory not found. Make sure you're in the project root directory."
    exit 1
fi

cd ui || { echo "Failed to change to UI directory"; exit 1; }

# Install npm dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing UI dependencies..."
    npm install || { echo "Failed to install UI dependencies"; exit 1; }
fi

# Start the UI
echo "Starting UI development server..."
npm run dev || { echo "Failed to start UI development server"; exit 1; }

# Cleanup function
cleanup() {
    echo "Shutting down servers..."
    kill $WEBSOCKET_PID
    exit 0
}

# Set trap to catch Ctrl+C
trap cleanup INT TERM

# Wait for the UI to exit
wait