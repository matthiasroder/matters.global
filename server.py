"""
Simple REST API server for matters.global chatbot

This server provides endpoints for the chatbot UI to interact with the
OpenAI Assistant-powered backend. It handles message routing, session 
management, and API access.
"""

import os
import uuid
from flask import Flask, request, jsonify, session
from flask_cors import CORS
import logging
from assistant_manager import AssistantManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", str(uuid.uuid4()))

# Enable CORS for all routes to allow the frontend to access the API
CORS(app, supports_credentials=True)

# Initialize AssistantManager
api_key = os.environ.get("OPENAI_API_KEY")
assistant_id = os.environ.get("OPENAI_ASSISTANT_ID")  # Optional, will create new if not provided
assistant_manager = AssistantManager(api_key=api_key, assistant_id=assistant_id)

# We'll use this to track active user sessions
user_sessions = {}

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"})

@app.route('/api/chat/message', methods=['POST'])
def send_message():
    """Process a message from the user.
    
    Request body:
    {
        "message": "user message here",
        "session_id": "optional-session-id"  
    }
    
    If session_id is not provided, a new one will be created.
    """
    data = request.json
    
    if not data or 'message' not in data:
        return jsonify({"error": "Message is required"}), 400
    
    message = data.get('message')
    session_id = data.get('session_id')
    
    # Create or retrieve user ID
    if not session_id:
        # Create a new session
        session_id = str(uuid.uuid4())
        user_sessions[session_id] = {
            "user_id": f"user_{uuid.uuid4()}",
            "messages": []
        }
    elif session_id not in user_sessions:
        # Unknown session, create new one
        user_sessions[session_id] = {
            "user_id": f"user_{uuid.uuid4()}",
            "messages": []
        }
    
    user_id = user_sessions[session_id]["user_id"]
    
    # Process the message through the assistant
    try:
        messages = assistant_manager.process_message(user_id, message)
        
        # Update session with messages
        user_sessions[session_id]["messages"] = messages
        
        return jsonify({
            "session_id": session_id,
            "messages": messages
        })
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return jsonify({
            "error": "Failed to process message",
            "message": str(e)
        }), 500

@app.route('/api/chat/history', methods=['GET'])
def get_history():
    """Get message history for a session.
    
    Query parameters:
    - session_id: ID of the session to retrieve
    """
    session_id = request.args.get('session_id')
    
    if not session_id or session_id not in user_sessions:
        return jsonify({"error": "Invalid or missing session ID"}), 400
    
    return jsonify({
        "session_id": session_id,
        "messages": user_sessions[session_id]["messages"]
    })

@app.route('/api/chat/sessions', methods=['GET'])
def list_sessions():
    """List all active sessions.
    
    This endpoint is primarily for debugging/admin purposes.
    """
    sessions = []
    for session_id, data in user_sessions.items():
        sessions.append({
            "session_id": session_id,
            "user_id": data["user_id"],
            "message_count": len(data["messages"])
        })
    
    return jsonify({"sessions": sessions})

@app.route('/api/chat/reset', methods=['POST'])
def reset_session():
    """Reset a session or create a new one.
    
    Request body:
    {
        "session_id": "optional-existing-session"  
    }
    
    Returns a new session_id or resets an existing one.
    """
    data = request.json or {}
    session_id = data.get('session_id')
    
    if not session_id or session_id not in user_sessions:
        # Create brand new session
        session_id = str(uuid.uuid4())
    
    # Reset the session
    user_sessions[session_id] = {
        "user_id": f"user_{uuid.uuid4()}",
        "messages": []
    }
    
    return jsonify({
        "session_id": session_id,
        "messages": []
    })

if __name__ == '__main__':
    # Create or retrieve the assistant before starting the server
    assistant_id = assistant_manager.get_or_create_assistant()
    logger.info(f"Using assistant: {assistant_id}")
    
    # Note about setup
    logger.info("Server starting...")
    logger.info("Ensure you've set OPENAI_API_KEY in your environment variables")
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)