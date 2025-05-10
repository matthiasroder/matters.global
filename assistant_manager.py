"""
Assistant Manager for matters.global

This module creates and manages an OpenAI Assistant for the matters.global problem tracking system.
It handles the integration between the chatbot UI, OpenAI Assistant API, and our backend functions.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
import logging
from openai import OpenAI
from openai.types.beta.threads import Run
from assistant_functions import (
    get_function_definitions, 
    dispatch_function
)

# Setup logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class AssistantManager:
    """Manages the OpenAI Assistant for matters.global problem management."""
    
    # System message that defines the assistant's behavior
    SYSTEM_MESSAGE = """
You are a helpful Problem Management Assistant for matters.global, an advanced problem tracking and resolution system.

Your purpose is to help users document, track, and explore THEIR problems in a conversational manner. You should be thoughtful, precise, and focused on reflecting the user's perspective and understanding.

# When gathering information about a new problem:

1. Ask open-ended questions to understand the problem's core nature as the USER sees it
2. Help users break down complex problems into clearer descriptions in THEIR own words
3. Help users identify conditions that must be met to solve the problem
4. Suggest checking for similar problems but always prioritize the user's unique perspective
5. Ask users about potential connections to other problems they've mentioned

# Guidelines for collecting quality information:

- Encourage the user to provide specific, focused, and actionable problem descriptions
- Help users formulate conditions that are clear, measurable, and verifiable to them
- Support the user's own language and terminology while gently suggesting clarity
- Help users distinguish between problems (desired outcomes) and solutions (specific approaches)

# How to use your functions:

- ALWAYS ASK PERMISSION before writing anything to the database
- Use list_problems when users want to see what's already in the system
- Use get_problem_details when discussing a specific existing problem
- Use create_problem ONLY AFTER explicit confirmation from the user
- Use add_condition_to_problem ONLY AFTER explicit confirmation from the user
- Use update_condition ONLY AFTER explicit confirmation from the user
- Use find_similar_problems to help users discover relevant existing problems
- Use add_problem_dependency ONLY AFTER explicitly confirming the prerequisite relationship with the user (for when one problem must be resolved before another)

Ask for clear confirmation before taking any action that modifies the database, such as:
"I can add this problem to the system for you. Would you like me to do that now?"
"Should I record this condition as part of this problem?"
"Would you like me to mark this condition as met?"
"It sounds like Problem A needs to be resolved before you can tackle Problem B. Should I create this prerequisite relationship between them?"

Maintain a conversational, helpful tone while respecting that the problems belong to the user. Ask clarifying questions when information is ambiguous, but never assume. Always reflect the user's perspective rather than imposing your own understanding.
"""
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 assistant_id: Optional[str] = None,
                 assistant_name: str = "Matters Global Problem Assistant"):
        """Initialize the AssistantManager.
        
        Args:
            api_key: OpenAI API key (uses environment variable if None)
            assistant_id: Existing assistant ID to use (creates new if None)
            assistant_name: Name for the assistant if creating new
        """
        # Initialize OpenAI client
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided or in environment variables")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Store or create assistant
        self.assistant_id = assistant_id
        self.assistant_name = assistant_name
        
        # Thread management (maps user IDs to thread IDs)
        self.user_threads = {}
        
    def create_assistant(self) -> str:
        """Create a new assistant with our functions.
        
        Returns:
            ID of the created assistant
        """
        logger.info(f"Creating new assistant: {self.assistant_name}")
        
        # Get function definitions
        function_definitions = get_function_definitions()
        
        # Create the assistant
        assistant = self.client.beta.assistants.create(
            name=self.assistant_name,
            instructions=self.SYSTEM_MESSAGE,
            model="gpt-4-turbo",  # Use appropriate model
            tools=[{"type": "function", "function": fn} for fn in function_definitions]
        )
        
        self.assistant_id = assistant.id
        logger.info(f"Created assistant with ID: {self.assistant_id}")
        
        return self.assistant_id
    
    def get_or_create_assistant(self) -> str:
        """Get an existing assistant or create a new one.

        Returns:
            ID of the assistant
        """
        # Use existing assistant if available
        if self.assistant_id:
            # Check if the assistant exists
            try:
                assistant = self.client.beta.assistants.retrieve(self.assistant_id)
                logger.info(f"Using existing assistant: {self.assistant_id}")
                return self.assistant_id
            except Exception as e:
                logger.warning(f"Assistant not found: {str(e)}")
                # Continue to create a new one

        # Create new assistant
        return self.create_assistant()
    
    def get_or_create_thread(self, user_id: str) -> str:
        """Get existing thread for user or create a new one.
        
        Args:
            user_id: Identifier for the user
            
        Returns:
            Thread ID for the conversation
        """
        if user_id in self.user_threads:
            thread_id = self.user_threads[user_id]
            logger.info(f"Using existing thread for user {user_id}: {thread_id}")
            return thread_id
        
        # Create a new thread
        thread = self.client.beta.threads.create()
        thread_id = thread.id
        
        # Store the thread ID for this user
        self.user_threads[user_id] = thread_id
        logger.info(f"Created new thread for user {user_id}: {thread_id}")
        
        return thread_id
    
    def add_message(self, thread_id: str, content: str, user_id: str) -> None:
        """Add a user message to the thread.
        
        Args:
            thread_id: ID of the thread
            content: Message content
            user_id: ID of the user (for metadata)
        """
        self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=content,
            metadata={"user_id": user_id}
        )
        logger.info(f"Added message from user {user_id} to thread {thread_id}")
    
    def run_assistant(self, thread_id: str) -> Run:
        """Run the assistant on the thread.
        
        Args:
            thread_id: ID of the thread
            
        Returns:
            The Run object from the API
        """
        # Use existing assistant ID and don't create a new one each time
        if not self.assistant_id:
            self.assistant_id = self.get_or_create_assistant()
        
        # Start the run
        run = self.client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=self.assistant_id
        )
        
        logger.info(f"Started run {run.id} on thread {thread_id} with assistant {self.assistant_id}")
        return run
    
    def wait_for_run(self, thread_id: str, run_id: str, 
                    check_interval: float = 1.0, 
                    timeout: float = 120.0) -> Run:
        """Wait for a run to complete.
        
        Args:
            thread_id: ID of the thread
            run_id: ID of the run
            check_interval: How often to check status (seconds)
            timeout: Maximum time to wait (seconds)
            
        Returns:
            The completed Run object
        """
        start_time = time.time()
        
        while True:
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )
            
            # Check if run is complete or requires action
            if run.status in ["completed", "failed", "cancelled", "expired"]:
                logger.info(f"Run {run_id} ended with status: {run.status}")
                return run
            
            # Check if function calling is required
            if run.status == "requires_action":
                if run.required_action.type == "submit_tool_outputs":
                    # Handle function calls
                    self._handle_tool_calls(thread_id, run_id, run.required_action.submit_tool_outputs.tool_calls)
                    
            # Check for timeout
            if time.time() - start_time > timeout:
                logger.warning(f"Timeout waiting for run {run_id}")
                # Cancel the run
                self.client.beta.threads.runs.cancel(
                    thread_id=thread_id,
                    run_id=run_id
                )
                raise TimeoutError(f"Run {run_id} timed out after {timeout} seconds")
            
            # Wait before checking again
            time.sleep(check_interval)
    
    def _handle_tool_calls(self, thread_id: str, run_id: str, tool_calls: List[Dict[str, Any]]) -> None:
        """Handle function calls from the assistant.
        
        Args:
            thread_id: ID of the thread
            run_id: ID of the run
            tool_calls: List of tool calls from the assistant
        """
        logger.info(f"Handling {len(tool_calls)} tool calls for run {run_id}")
        
        tool_outputs = []
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            logger.info(f"Function call: {function_name} with args: {arguments}")
            
            # Dispatch to our function handler
            result = dispatch_function(function_name, arguments)
            
            tool_outputs.append({
                "tool_call_id": tool_call.id,
                "output": json.dumps(result)
            })
        
        # Submit the results back to the assistant
        self.client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread_id,
            run_id=run_id,
            tool_outputs=tool_outputs
        )
        
        logger.info(f"Submitted {len(tool_outputs)} tool outputs for run {run_id}")
    
    def get_messages(self, thread_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent messages from a thread.
        
        Args:
            thread_id: ID of the thread
            limit: Maximum number of messages to return
            
        Returns:
            List of messages
        """
        logger.info(f"Retrieving messages from thread {thread_id}")
        
        messages = self.client.beta.threads.messages.list(
            thread_id=thread_id,
            limit=limit
        )
        
        # Log raw message data
        logger.info(f"Raw thread messages: {messages}")
        
        # Convert to simpler format for the UI
        formatted_messages = []
        for msg in messages.data:
            # Extract text content
            content = []
            for content_item in msg.content:
                if content_item.type == "text":
                    content.append(content_item.text.value)
            
            # Create formatted message
            formatted_msg = {
                "id": msg.id,
                "role": msg.role,
                "content": "\n".join(content),
                "created_at": msg.created_at
            }
            
            # Log each formatted message
            logger.info(f"Formatted message: {formatted_msg}")
            
            formatted_messages.append(formatted_msg)
        
        logger.info(f"Returning {len(formatted_messages)} messages")
        return formatted_messages
    
    def process_message(self, user_id: str, message: str) -> List[Dict[str, Any]]:
        """Process a user message and get the assistant's response.
        
        This is the main entry point for the UI integration.
        
        Args:
            user_id: Identifier for the user
            message: The user's message
            
        Returns:
            List of messages in the thread after processing
        """
        logger.info(f"Processing message from {user_id}: '{message}'")
        
        # Get or create thread for this user
        thread_id = self.get_or_create_thread(user_id)
        logger.info(f"Using thread ID: {thread_id} for user: {user_id}")
        
        # Add the user message to the thread
        self.add_message(thread_id, message, user_id)
        logger.info(f"Added message to thread: {thread_id}")
        
        # Ensure we have an assistant ID
        if not self.assistant_id:
            logger.info("No assistant ID found, creating or retrieving one")
            self.assistant_id = self.get_or_create_assistant()
        
        logger.info(f"Using assistant ID: {self.assistant_id}")
            
        # Run the assistant
        run = self.run_assistant(thread_id)
        logger.info(f"Started run with ID: {run.id}")
        
        # Wait for the run to complete
        try:
            logger.info(f"Waiting for run {run.id} to complete...")
            completed_run = self.wait_for_run(thread_id, run.id)
            logger.info(f"Run {run.id} completed with status: {completed_run.status}")
            
            if completed_run.status != "completed":
                logger.warning(f"Run ended with non-completed status: {completed_run.status}")
                # You might want to add a message to the thread about the failure
        except Exception as e:
            logger.error(f"Error during run: {str(e)}")
            # You might want to add a message to the thread about the error
        
        # Get the updated messages
        logger.info(f"Retrieving messages from thread {thread_id} after run completion")
        messages = self.get_messages(thread_id)
        logger.info(f"Retrieved {len(messages)} messages from thread {thread_id}")
        
        return messages

# Example usage in a simple API
if __name__ == "__main__":
    # This code would not run directly, but demonstrates usage
    
    # Initialize the manager
    manager = AssistantManager()
    
    # Example integration with a web server
    def handle_message(user_id, message):
        """Handle a message from the UI."""
        return manager.process_message(user_id, message)