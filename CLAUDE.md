# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Activate environment: `conda activate mattersglobal`
- Run Python: `python mattersglobal.py`
- Run Jupyter notebook: `jupyter notebook problems.ipynb`
- Install dependencies: `conda install -n mattersglobal <package>`

## Code Style Guidelines
- Use Python 3.9+ features
- Follow PEP 8 conventions for formatting
- Use type hints with Python's typing module
- Use Pydantic for data validation and parsing
- Class naming: PascalCase (e.g., ProblemDefinition)
- Variable naming: snake_case (e.g., problem_definition)
- Handle errors with try/except blocks and clear error messages
- Use docstrings for functions and classes
- Keep code modular and functions focused on a single responsibility