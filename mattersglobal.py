problem_definition = {
    "description": "A detailed description of the problem",
    "state": "solved | not_solved | obsolete", # Initial state is not_solved
    "conditions": [
        {
            "description": "Detailed description of the condition",
            "is_met": False,  # Initial state, evaluates to True if condition is met
        },
    ],
    "solutions": [
        {
            "description": "Detailed description of the solution", # Solutions will be added only once the problem is solved.
        },
}
