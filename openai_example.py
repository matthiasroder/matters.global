#!/usr/bin/env python3
"""
Example showing how to create a problem from OpenAI API response
"""

import json
import os
from openai import OpenAI
import mattersglobal as mg

def create_problem_from_openai(user_input):
    """Create a problem definition using the OpenAI API."""
    
    # Check for API key
    if "OPENAI_API_KEY" not in os.environ:
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("For demonstration purposes, we'll use mock data.")
        # Mock data for demonstration
        return {
            "description": "I cannot run a 5K without stopping",
            "conditions": [
                {
                    "description": "Train 3 times per week for 8 weeks",
                    "is_met": False
                },
                {
                    "description": "Gradually increase running duration each week",
                    "is_met": False
                }
            ]
        }
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # System prompt for OpenAI
    system_prompt = """You will be provided with a problem description. 
    Your goal is to create a structured problem definition with possible conditions.
    Return only valid JSON that matches this structure:
    {
        "description": "A detailed description of the problem",
        "conditions": [
            {
                "description": "First condition that needs to be met",
                "is_met": false
            },
            {
                "description": "Second condition that needs to be met",
                "is_met": false
            }
        ]
    }"""
    
    try:
        # Try with JSON response format first
        try:
            response = client.chat.completions.create(
                model="gpt-4",  # Or any appropriate model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                response_format={"type": "json_object"}
            )
        except Exception as format_error:
            # Fall back to standard response if JSON format not supported
            print(f"JSON format not supported: {format_error}")
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Fall back to a different model
                messages=[
                    {"role": "system", "content": system_prompt + "\nIf you cannot output JSON directly, wrap your JSON in ```json and ``` markers."},
                    {"role": "user", "content": user_input}
                ]
            )
            
        # Parse JSON response
        content = response.choices[0].message.content
        
        # Check if the response is wrapped in code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        problem_json = json.loads(content)
        return problem_json
        
    except Exception as e:
        print(f"Error getting response from OpenAI: {e}")
        # Return mock data as fallback
        return {
            "description": user_input,
            "conditions": [
                {
                    "description": "This is a placeholder condition since OpenAI API call failed",
                    "is_met": False
                }
            ]
        }

def main():
    # Example problem statement
    user_input = "I cannot run a 5K without stopping."
    
    # Get problem definition (either from OpenAI or mock data)
    problem_json = create_problem_from_openai(user_input)
    
    try:
        # Create a ProblemManager
        manager = mg.ProblemManager()
        
        # Create a new problem with the description
        problem = manager.create_problem(problem_json["description"])
        
        # Add conditions from the JSON data
        for condition_data in problem_json.get("conditions", []):
            problem.add_condition(
                description=condition_data["description"],
                is_met=condition_data.get("is_met", False)  # Default to False if not specified
            )
        
        # Save the problem
        filepath = manager.save_problems()
        print(f"Problem saved to {filepath}")
        
        # Print the created problem
        print(f"Problem: {problem.description}")
        print(f"State: {problem.state}")
        print("Conditions:")
        for i, condition in enumerate(problem.conditions):
            print(f"  {i+1}. {condition.description} (Met: {condition.is_met})")
            
        # Alternative method: Create directly with Pydantic model
        print("\nAlternative method - Creating directly with Pydantic:")
        
        # Create conditions objects
        conditions = [
            mg.Condition(**condition_data) 
            for condition_data in problem_json.get("conditions", [])
        ]
        
        # Create problem directly
        direct_problem = mg.Problem(
            description=problem_json["description"],
            conditions=conditions
        )
        
        print(f"Problem: {direct_problem.description}")
        print(f"State: {direct_problem.state}")
        
    except Exception as e:
        print(f"Error creating problem: {e}")

if __name__ == "__main__":
    main()