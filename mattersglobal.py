"""
matters.global package for working with problem definitions

This module provides tools to:
- Create JSON problem definitions
- Store JSON problem definitions
- Load JSON problem definitions
- Compute connections between problems
- Determine if a statement is a problem
- Check if a similar problem already exists
"""

import json
import os
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field


class ProblemState(str, Enum):
    """Enum representing the possible states of a problem."""
    SOLVED = "solved"
    NOT_SOLVED = "not_solved"
    OBSOLETE = "obsolete"


class Condition(BaseModel):
    """Model representing a condition that must be met to solve the problem."""
    description: str = Field(
        ...,
        description="Detailed description of the condition"
    )
    is_met: bool = Field(
        False,
        description="Status of the condition; True if condition is met, False otherwise"
    )


class WorkingSolution(BaseModel):
    """Model representing a working solution to a problem."""
    description: str = Field(
        ...,
        description="Detailed description of the working solution"
    )


class Problem(BaseModel):
    """Model representing a problem definition."""
    description: str = Field(
        ...,
        description="A detailed description of the problem"
    )
    state: ProblemState = Field(
        default=ProblemState.NOT_SOLVED,
        description="The state of the problem: solved, not_solved, or obsolete"
    )
    conditions: List[Condition] = Field(
        default_factory=list,
        description="List of conditions that need to be met to solve the problem"
    )
    working_solutions: List[WorkingSolution] = Field(
        default_factory=list,
        alias="working_solutions",
        description="Working solutions for the problem, added only once the problem is solved"
    )
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "description": "A detailed description of the problem",
                "state": "not_solved",
                "conditions": [
                    {
                        "description": "Detailed description of the condition",
                        "is_met": False
                    }
                ],
                "working_solutions": [
                    {
                        "description": "Detailed description of the working solution"
                    }
                ]
            }
        }

    def add_condition(self, description: str, is_met: bool = False) -> None:
        """Add a new condition to the problem."""
        self.conditions.append(Condition(
            description=description,
            is_met=is_met
        ))

    def add_working_solution(self, description: str) -> None:
        """Add a new working solution to the problem."""
        self.working_solutions.append(WorkingSolution(
            description=description
        ))

    def update_condition(self, index: int, is_met: bool) -> None:
        """Update the status of a condition by index."""
        if 0 <= index < len(self.conditions):
            self.conditions[index].is_met = is_met
        else:
            raise IndexError(f"Condition index {index} out of range")

    def check_if_solved(self) -> bool:
        """Check if all conditions are met and update state if needed."""
        if all(condition.is_met for condition in self.conditions):
            self.state = ProblemState.SOLVED
            return True
        return False

    def mark_as_obsolete(self) -> None:
        """Mark the problem as obsolete."""
        self.state = ProblemState.OBSOLETE


class ProblemManager:
    """Class for managing collections of problems."""
    
    def __init__(self, data_dir: str = "./data"):
        """Initialize the ProblemManager.
        
        Args:
            data_dir: Directory to store problem files
        """
        self.data_dir = data_dir
        self.problems: List[Problem] = []
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def create_problem(self, description: str) -> Problem:
        """Create a new problem and add it to the collection.
        
        Args:
            description: Description of the problem
            
        Returns:
            The created Problem object
        """
        problem = Problem(description=description)
        self.problems.append(problem)
        return problem
    
    def save_problems(self, filename: str = "problems.json") -> str:
        """Save all problems to a JSON file.
        
        Args:
            filename: Name of the file to save problems to
            
        Returns:
            Path to the saved file
        """
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump([p.model_dump() for p in self.problems], f, indent=4)
        
        return filepath
    
    def load_problems(self, filename: str = "problems.json") -> List[Problem]:
        """Load problems from a JSON file.
        
        Args:
            filename: Name of the file to load problems from
            
        Returns:
            List of loaded Problem objects
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            return []
        
        with open(filepath, 'r') as f:
            problem_data = json.load(f)
        
        self.problems = [Problem(**data) for data in problem_data]
        return self.problems
    
    def get_problem_by_description(self, description: str) -> Optional[Problem]:
        """Find a problem by its description.
        
        Args:
            description: Description to search for
            
        Returns:
            Problem if found, None otherwise
        """
        for problem in self.problems:
            if problem.description == description:
                return problem
        return None
    
    def is_statement_problem(self, statement: str) -> bool:
        """Determine if a statement is a problem.
        
        A basic implementation - in a real-world scenario this might use
        NLP or AI to analyze the statement.
        
        Args:
            statement: Statement to analyze
            
        Returns:
            True if the statement appears to be a problem, False otherwise
        """
        # Simple heuristic - check if statement contains problem indicators
        problem_indicators = [
            "cannot", "can't", "won't", "doesn't", "isn't", "failed",
            "impossible", "difficult", "hard", "challenge", "issue",
            "problem", "trouble", "need to", "should be", "crashes", "too slow",
            "breaks", "error", "not working"
        ]
        
        return any(indicator in statement.lower() for indicator in problem_indicators)
    
    def find_similar_problem(self, description: str, threshold: float = 0.6) -> Optional[Problem]:
        """Find a problem similar to the given description.
        
        This is a placeholder implementation. In a real-world scenario,
        this would use more sophisticated text similarity algorithms.
        
        Args:
            description: Description to compare against
            threshold: Similarity threshold (0-1)
            
        Returns:
            Most similar Problem if above threshold, None otherwise
        """
        # This is a very basic implementation
        # In a real implementation, you might use:
        # - Text embeddings
        # - Cosine similarity
        # - Semantic similarity models
        
        from difflib import SequenceMatcher
        
        # Normalize the input text for better matching
        def normalize_text(text):
            text = text.lower()
            # Replace common synonyms
            replacements = {
                "can't": "cannot",
                "cant": "cannot",
                "won't": "will not",
                "wont": "will not",
                "isn't": "is not",
                "isnt": "is not",
                "kg": "kgs",
                "kilogram": "kg",
                "kilograms": "kgs",
            }
            for old, new in replacements.items():
                text = text.replace(old, new)
            return text
        
        normalized_description = normalize_text(description)
        
        best_match = None
        highest_similarity = 0
        
        for problem in self.problems:
            normalized_problem = normalize_text(problem.description)
            
            # Calculate similarity
            similarity = SequenceMatcher(
                None, 
                normalized_description, 
                normalized_problem
            ).ratio()
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = problem
        
        if highest_similarity >= threshold:
            return best_match
        
        return None
    
    def compute_problem_connections(self) -> Dict[str, List[str]]:
        """Compute connections between problems based on shared keywords.
        
        Returns:
            Dictionary mapping problem descriptions to lists of related problem descriptions
        """
        # This is a basic implementation based on shared words
        # In a real implementation, you might use:
        # - Topic modeling
        # - Entity extraction
        # - Graph-based relationship algorithms
        
        connections: Dict[str, List[str]] = {}
        
        # Extract significant words from each problem description
        for problem in self.problems:
            # Skip common words, keep words longer than 3 characters
            words = set(word.lower() for word in problem.description.split() 
                      if len(word) > 3 and word.lower() not in 
                      {"the", "and", "but", "for", "with", "that", "this"})
            
            connections[problem.description] = []
            
            # Find problems with shared words
            for other in self.problems:
                if other.description == problem.description:
                    continue
                    
                other_words = set(word.lower() for word in other.description.split()
                               if len(word) > 3 and word.lower() not in
                               {"the", "and", "but", "for", "with", "that", "this"})
                
                # If there are shared significant words, consider them connected
                if words.intersection(other_words):
                    connections[problem.description].append(other.description)
        
        return connections


# Example usage
if __name__ == "__main__":
    # Create a problem manager
    manager = ProblemManager()
    
    # Create a new problem
    problem = manager.create_problem("I cannot lift 80kgs in deadlift")
    
    # Add conditions
    problem.add_condition("I go train three times a week for 20 consecutive weeks")
    problem.add_condition("I get stronger every week as measured by the kgs I can lift")
    
    # Check if problem is solved (it's not yet)
    is_solved = problem.check_if_solved()
    print(f"Is problem solved? {is_solved}")
    
    # Update conditions
    problem.update_condition(0, True)  # First condition met
    problem.update_condition(1, True)  # Second condition met
    
    # Check if problem is solved now
    is_solved = problem.check_if_solved()
    print(f"Is problem solved now? {is_solved}")
    
    # Add a working solution since the problem is solved
    problem.add_working_solution("Followed a progressive overload program with proper nutrition")
    
    # Save problems to file
    filepath = manager.save_problems()
    print(f"Problems saved to {filepath}")
    
    # Print the problem as it would appear in JSON
    import json
    problem_json = json.dumps(problem.model_dump(), indent=2)
    print(f"\nProblem as JSON:\n{problem_json}")
    
    # Check if a statement is a problem
    statement = "The website loads too slowly on mobile devices"
    is_problem = manager.is_statement_problem(statement)
    print(f"Is '{statement}' a problem? {is_problem}")