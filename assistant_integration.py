#!/usr/bin/env python3
"""
Integration with OpenAI Assistant for matters.global

This script demonstrates how to use an OpenAI Assistant to generate problem
definitions and import them into our mattersglobal system.
"""

import os
import time
import json
import subprocess
import mattersglobal as mg

# Try to get the OpenAI API key from keychain
def get_api_key_from_keychain():
    try:
        # Use the security command-line tool to access the macOS keychain
        result = subprocess.run(
            ["security", "find-generic-password", "-s", "openai-api-key", "-w"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            # Try a different keychain item name
            result = subprocess.run(
                ["security", "find-generic-password", "-s", "OPENAI_API_KEY", "-w"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
            
            # If both failed, try one more common name
            result = subprocess.run(
                ["security", "find-generic-password", "-s", "openai", "-w"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
                
    except Exception as e:
        print(f"Error accessing keychain: {e}")
    
    # If we got here, all methods failed
    print("Could not find OpenAI API key in keychain")
    # Enter a temporary API key directly for testing
    key = input("Enter your OpenAI API key (or press Enter to skip): ")
    if key.strip():
        return key.strip()
        
    return None

# Set the OpenAI API key from keychain if available
api_key = get_api_key_from_keychain()
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    print("Successfully loaded OpenAI API key from keychain")

from openai import OpenAI

class AssistantIntegration:
    """Class for integrating with OpenAI Assistant for problem definitions."""
    
    def __init__(self, assistant_id="asst_3CJPySTMQ0d3RCf7ECLKCY2e"):
        """Initialize the integration with OpenAI Assistant.
        
        Args:
            assistant_id: ID of the OpenAI Assistant to use
        """
        # Initialize the OpenAI client
        self.client = OpenAI()
        self.assistant_id = assistant_id
        self.problem_manager = mg.ProblemManager()
        
        # Check if the OpenAI API key is set
        if not os.getenv("OPENAI_API_KEY"):
            print("WARNING: OPENAI_API_KEY environment variable is not set")
            print("You will not be able to use the OpenAI API without setting it")
    
    def create_thread(self):
        """Create a new thread for conversation with the assistant."""
        return self.client.beta.threads.create()
    
    def add_message_to_thread(self, thread_id, message_content):
        """Add a user message to a thread.
        
        Args:
            thread_id: ID of the thread to add the message to
            message_content: Content of the message to add
            
        Returns:
            The created message
        """
        return self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=message_content
        )
    
    def run_assistant(self, thread_id, instructions=None):
        """Run the assistant on a thread.
        
        Args:
            thread_id: ID of the thread to run the assistant on
            instructions: Optional instructions for the assistant
            
        Returns:
            The completed run
        """
        # Create a run
        run = self.client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=self.assistant_id,
            instructions=instructions
        )
        
        # Poll for the run to complete
        while True:
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
            
            if run.status == "completed":
                break
            elif run.status in ["failed", "cancelled", "expired"]:
                raise Exception(f"Run ended with status: {run.status}")
            elif run.status == "requires_action":
                # Handle required actions (tool calls)
                if run.required_action and run.required_action.type == "submit_tool_outputs":
                    tool_calls = run.required_action.submit_tool_outputs.tool_calls
                    tool_outputs = []
                    
                    for tool_call in tool_calls:
                        # We don't actually need to do anything with the function call
                        # since we're just extracting the parameters
                        # But we need to submit a response to continue
                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "output": json.dumps({"status": "success"})
                        })
                    
                    # Submit the tool outputs
                    self.client.beta.threads.runs.submit_tool_outputs(
                        thread_id=thread_id,
                        run_id=run.id,
                        tool_outputs=tool_outputs
                    )
            
            # Wait a moment before polling again
            time.sleep(1)
        
        return run
    
    def get_messages(self, thread_id):
        """Get all messages from a thread.
        
        Args:
            thread_id: ID of the thread to get messages from
            
        Returns:
            List of messages in the thread
        """
        return self.client.beta.threads.messages.list(thread_id=thread_id)
    
    def extract_tool_calls_from_run(self, thread_id, run_id):
        """Extract tool calls from a run.
        
        Args:
            thread_id: ID of the thread containing the run
            run_id: ID of the run to extract tool calls from
            
        Returns:
            List of tool calls from the run
        """
        # Get the run steps to find function calls
        run_steps = self.client.beta.threads.runs.steps.list(
            thread_id=thread_id,
            run_id=run_id
        )
        
        tool_calls = []
        
        # Look for function call steps
        for step in run_steps.data:
            if hasattr(step.step_details, 'tool_calls'):
                tool_calls.extend(step.step_details.tool_calls)
        
        return tool_calls
    
    def extract_problem_from_run(self, thread_id, run_id):
        """Extract problem definition from a run with a function call.
        
        Args:
            thread_id: ID of the thread containing the run
            run_id: ID of the run to extract the problem from
            
        Returns:
            Problem definition from the function call, or None if not found
        """
        tool_calls = self.extract_tool_calls_from_run(thread_id, run_id)
        
        # Look for the define_problem_parameters function call
        for tool_call in tool_calls:
            if tool_call.type == "function" and tool_call.function.name == "define_problem_parameters":
                # Extract the function arguments as JSON
                func_args = json.loads(tool_call.function.arguments)
                return func_args
        
        # Also check the run's required actions (for in-progress runs)
        run = self.client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id
        )
        
        if run.status == "requires_action" and run.required_action:
            if run.required_action.type == "submit_tool_outputs":
                for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                    if tool_call.type == "function" and tool_call.function.name == "define_problem_parameters":
                        # Extract the function arguments as JSON
                        func_args = json.loads(tool_call.function.arguments)
                        return func_args
        
        return None
    
    def create_problem_from_user_input(self, user_input, instructions=None):
        """Create a problem definition from user input using the assistant.
        
        Args:
            user_input: The user's problem description
            instructions: Optional instructions for the assistant
            
        Returns:
            The created Problem object, or None if creation failed
        """
        try:
            # Create a new thread
            thread = self.create_thread()
            
            # Add the user's message
            self.add_message_to_thread(thread.id, user_input)
            
            # Run the assistant
            run = self.run_assistant(thread.id, instructions)
            
            # Extract the problem definition
            problem_data = self.extract_problem_from_run(thread.id, run.id)
            
            if not problem_data:
                # If no function call was made, look for a regular message
                messages = self.get_messages(thread.id)
                for message in messages.data:
                    if message.role == "assistant":
                        for content_part in message.content:
                            if content_part.type == "text":
                                print(f"Assistant message: {content_part.text.value}")
                
                print("No problem definition found in the assistant's response")
                return None
            
            # Convert the problem definition to our model
            return self.convert_assistant_problem_to_model(problem_data)
            
        except Exception as e:
            print(f"Error creating problem: {e}")
            return None
    
    def convert_assistant_problem_to_model(self, problem_data):
        """Convert an assistant problem definition to our model.
        
        Args:
            problem_data: Problem definition from the assistant
            
        Returns:
            A Problem object
        """
        # Create a new problem with the description
        problem = self.problem_manager.create_problem(problem_data["description"])
        
        # Set the state directly
        problem.state = mg.ProblemState(problem_data["state"])
        
        # Clear any default conditions that might have been added
        problem.conditions = []
        
        # Add the conditions
        for condition in problem_data["conditions"]:
            problem.add_condition(
                description=condition["description"],
                is_met=condition["is_met"]
            )
        
        # Clear any default working solutions
        problem.working_solutions = []
        
        # Add the working solutions
        for solution in problem_data.get("working_solutions", []):
            problem.add_working_solution(solution["description"])
        
        # If the problem is solved, make sure all conditions are met
        if problem.state == mg.ProblemState.SOLVED:
            for condition in problem.conditions:
                condition.is_met = True
                
        return problem
    
    def save_problem(self, problem):
        """Save a problem to the problem manager.
        
        Args:
            problem: The Problem object to save
            
        Returns:
            Path to the saved file
        """
        # Add the problem to the manager (it's already added in create_problem)
        # But we need to make sure it's in the manager's list
        if problem not in self.problem_manager.problems:
            self.problem_manager.problems.append(problem)
            
        # Save all problems
        return self.problem_manager.save_problems()


# Example usage
if __name__ == "__main__":
    # Check if API key is set (either from environment or from keychain)
    if not os.getenv("OPENAI_API_KEY"):
        # Try getting it from keychain one more time
        api_key = get_api_key_from_keychain()
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            print("Successfully loaded OpenAI API key from keychain")
    
    # Final check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environment variable")
        print("For demonstration, we'll use mock data instead of calling the API")
        
        # Create a problem manager
        manager = mg.ProblemManager()
        
        # Create a mock problem similar to what the assistant would return
        problem = manager.create_problem(
            "Difficulty in emotionally letting go of your wife who has left, leading to continued contact and attempts to rekindle the relationship."
        )
        problem.state = mg.ProblemState.NOT_SOLVED
        problem.add_condition("Feeling at peace with being on your own.", False)
        problem.add_condition("Releasing the belief that your wife belongs to you in any way.", False)
        
        # Save the problem
        filepath = manager.save_problems()
        print(f"Mock problem saved to {filepath}")
        
        # Print the problem
        print(f"\nProblem: {problem.description}")
        print(f"State: {problem.state}")
        print("Conditions:")
        for i, condition in enumerate(problem.conditions):
            print(f"  {i+1}. {condition.description} (Met: {condition.is_met})")
        print("Working Solutions:")
        for i, solution in enumerate(problem.working_solutions):
            print(f"  {i+1}. {solution.description}")
        
    else:
        # Initialize the integration
        integration = AssistantIntegration()
        
        # Example user input - use a default value to avoid input issues in automation
        try:
            user_input = input("Describe your problem (or press Enter for default): ")
            if not user_input.strip():
                user_input = "I feel anxious in social situations and often avoid meeting new people."
                print(f"Using default problem: {user_input}")
        except EOFError:
            # In case of EOF error (common in some environments), use the default
            user_input = "I feel anxious in social situations and often avoid meeting new people."
            print(f"Using default problem: {user_input}")
        
        # Optional instructions to guide the assistant
        instructions = """
        Analyze the user's problem and create a structured problem definition. 
        Focus on identifying the root problem and the conditions that need to be met to solve it.
        Use the 'define_problem_parameters' function to structure the problem definition.
        Include a detailed problem description, appropriate state, relevant conditions, and 
        working solutions if applicable (or an empty array if none).
        """
        
        # Create a problem from the user input
        problem = integration.create_problem_from_user_input(user_input, instructions)
        
        if problem:
            # Save the problem
            filepath = integration.save_problem(problem)
            print(f"Problem saved to {filepath}")
            
            # Print the problem
            print(f"\nProblem: {problem.description}")
            print(f"State: {problem.state}")
            print("Conditions:")
            for i, condition in enumerate(problem.conditions):
                print(f"  {i+1}. {condition.description} (Met: {condition.is_met})")
            print("Working Solutions:")
            for i, solution in enumerate(problem.working_solutions):
                print(f"  {i+1}. {solution.description}")
        else:
            print("Failed to create problem from assistant")