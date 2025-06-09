"""
WebSocket server for matters.global chatbot

This server provides a WebSocket interface for the chatbot UI to interact
with the OpenAI Assistant-powered backend.
"""

import asyncio
import json
import logging
import os
import uuid
import websockets
from assistant_manager import AssistantManager

# Setup logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize AssistantManager
api_key = os.environ.get("OPENAI_API_KEY")
assistant_id = os.environ.get("OPENAI_ASSISTANT_ID")

try:
    assistant_manager = AssistantManager(api_key=api_key, assistant_id=assistant_id)
    logger.info("AssistantManager initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize AssistantManager: {str(e)}")
    # Continue without assistant for now - you could implement fallback behavior
    assistant_manager = None

# We'll use this to track active user sessions
user_sessions = {}

async def handle_message(websocket):
    """Handle WebSocket connections."""
    # Generate a unique session ID for this connection
    session_id = str(uuid.uuid4())
    user_id = f"user_{session_id}"

    # Store the session
    user_sessions[session_id] = {
        "user_id": user_id,
        "messages": []
    }

    logger.info(f"New connection established: {session_id}")

    try:
        async for message in websocket:
            logger.info(f"Received message from {session_id}: {message}")

            # Process the message through the assistant
            try:
                logger.info(f"Processing message from {user_id}: '{message}'")

                if not assistant_manager:
                    await websocket.send("Assistant not available. Please check configuration.")
                    await websocket.send("[END]")
                    continue

                # This is a synchronous call - in a production app, you'd want to make this non-blocking
                messages = assistant_manager.process_message(user_id, message)

                # Log retrieved messages
                logger.info(f"Got {len(messages)} messages from thread for user {user_id}")

                # Update session with messages
                user_sessions[session_id]["messages"] = messages
                logger.info(f"Updated session {session_id} with new messages")

                # Send the assistant's response
                if messages:
                    logger.info(f"Processing {len(messages)} messages to find assistant response")

                    # Sort messages by created_at timestamp in descending order (newest first)
                    sorted_messages = sorted(messages, key=lambda m: m.get('created_at', 0), reverse=True)
                    logger.info(f"Sorted {len(sorted_messages)} messages by timestamp")

                    # Find the latest assistant message
                    for msg in sorted_messages:
                        logger.info(f"Checking message with role: {msg['role']} from timestamp: {msg.get('created_at')}")
                        if msg["role"] == "assistant":
                            # Send response in chunks to simulate streaming
                            # For a real implementation, you'd use the OpenAI streaming API
                            response = msg["content"]
                            logger.info(f"Found latest assistant response: '{response}'")

                            # For demonstration, just sending the whole message
                            await websocket.send(response)
                            logger.info(f"Sent response to client")

                            # Signal end of message with a separate message
                            # Make sure this is sent as a separate WebSocket frame
                            await asyncio.sleep(0.1)  # Small delay to ensure separate frames
                            await websocket.send("[END]")
                            logger.info(f"Sent [END] marker to client")
                            break
                    else:
                        # No assistant message found
                        logger.warning(f"No assistant message found in thread {user_id}")
                        await websocket.send("I didn't get a response. Please try again.")
                        await websocket.send("[END]")
                else:
                    logger.warning(f"No messages returned for user {user_id}")
                    await websocket.send("No response received from the assistant.")
                    await websocket.send("[END]")

            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                await websocket.send(f"Error: {str(e)}")
                await websocket.send("[END]")

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Connection closed: {session_id}")
    finally:
        # Clean up the session when the connection is closed
        if session_id in user_sessions:
            del user_sessions[session_id]

async def main():
    # Create or retrieve the assistant before starting the server
    # and store it in the manager to be reused
    if assistant_manager:
        try:
            assistant_id = assistant_manager.get_or_create_assistant()
            assistant_manager.assistant_id = assistant_id
            logger.info(f"Using assistant: {assistant_id}")
        except Exception as e:
            logger.error(f"Failed to initialize assistant: {str(e)}")
    else:
        logger.warning("Assistant manager not available - running in degraded mode")

    # Note about setup
    logger.info("WebSocket server starting...")
    logger.info("Ensure you've set OPENAI_API_KEY in your environment variables")

    port = int(os.environ.get("PORT", 8091))

    async with websockets.serve(handle_message, "0.0.0.0", port):
        logger.info(f"WebSocket server running on port {port}")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())