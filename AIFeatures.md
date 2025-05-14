# Matters.Global AI Assistant Features

This document outlines the key features and capabilities of the AI assistant for Matters.Global, focusing on how it interacts with users to capture, structure, and coach around goals, problems, conditions, and solutions.

## Core Information Capture Workflow

### 1. Initial Information Gathering
- User shares information about goals, problems, conditions, or solutions
- AI identifies key entities and their relationships in the conversation
- AI recognizes potential ambiguities or missing details
- AI detects implicit relationships between mentioned entities

### 2. Clarification Phase
- AI asks follow-up questions to clarify ambiguities
- "Is [X] a condition that needs to be met, or a problem that needs solving?"
- "Would you consider [Y] a goal you're pursuing, or a problem you're facing?"
- "Are you suggesting [Z] as a potential solution?"
- "How does [goal A] relate to [goal B] you mentioned earlier?"

### 3. Confirmation Before Action
- AI summarizes understanding: "I understand you want to achieve [goal] which requires [conditions], and you're facing [problems]."
- AI explicitly requests permission to record information: "Would you like me to record this in the system?"
- AI previews how information will be structured before committing

### 4. Function Calls
The AI needs these function calls to interact with the graph database:
```python
create_matter(description: str, labels: List[str], properties: Dict)
add_relationship(source_id: str, relationship_type: str, target_id: str)
update_matter(matter_id: str, properties: Dict)
find_similar_matters(description: str, labels: List[str], threshold: float)
get_related_matters(matter_id: str, relationship_types: List[str], direction: str)
explain_similarity(source_id: str, target_id: str)  # Uses LLM to explain why matters are similar
refine_similarity_results(candidates: List[Dict], query: str)  # LLM refines vector search results
```

### 5. Feedback Loop
- AI confirms actions taken: "I've recorded your goal 'Launch a product by Q2' with the conditions you mentioned."
- AI offers insights from the graph: "This goal relates to three other goals in your system. Would you like to see them?"
- AI suggests potential relationships: "This seems similar to [existing entity]. Should I connect them?"

### 6. Multiple Entity Handling
- User might mention multiple entities in a single message
- AI needs to parse and process them separately while preserving relationships
- "I understand you mentioned two goals: [A] and [B], with [C] being a condition for both."

## Coaching Capabilities

### 1. Progress Check-ins
- "How's your progress on [goal]? Last time we spoke, you were working on [condition]."
- "It's been two weeks since we discussed [problem]. Have you made any headway?"
- "The target date for [goal] is approaching. Are you still on track?"
- "You marked [condition] as not met last time. Has that status changed?"

### 2. Roadblock Identification
- "I notice [condition] has been unmet for some time. What's making this challenging?"
- "You've mentioned [obstacle] several times across different goals. Is this a recurring pattern?"
- "Out of these five conditions, which one feels most challenging right now?"
- "What's currently blocking progress on [goal]?"

### 3. Reflective Prompting
- "Looking at your approach to [goal], what's working well so far?"
- "If you were to restart work on [problem], what would you do differently?"
- "How has your understanding of [goal] evolved since we first discussed it?"
- "What insights have you gained from working on [problem] that might help with other areas?"

### 4. Milestone Celebration
- "Congratulations! You've met 3 out of 5 conditions for [goal]. That's significant progress."
- "I see you've resolved [problem] that was blocking several other goals. That's a key achievement."
- "You've been consistently making progress on [goal] for the past month. That's impressive commitment."

### 5. Strategy Refinement
- "Given what you've learned about [domain], would you like to refine your approach to [goal]?"
- "I notice similarities between [solution A] and [solution B]. Have you considered combining these approaches?"
- "Based on your success with [goal X], I wonder if similar strategies might help with [goal Y]?"
- "You've tried several approaches to [problem]. Would you like to analyze which has been most effective?"

### 6. Network Insights
- "This goal seems to be a central hub in your system, connecting to many other matters. It may be a high-leverage area to focus on."
- "These three goals form a closely connected cluster. Would addressing them together create synergies?"
- "I notice this problem has remained isolated in your graph. Are there connections to other goals or problems we should capture?"

## Assistant Personality and Approach

### Conversational Style
- Attentive and thoughtful, asking insightful questions
- Balances structure (needed for the graph) with natural conversation
- Reflects the user's language and framing when discussing their matters
- Adapts tone based on context (more direct during information capture, more reflective during coaching)

### Knowledge Management
- Refers to relevant previous conversations when appropriate
- Builds a consistent understanding of the user's goals and challenges over time
- Identifies patterns and themes across different sessions
- Helps user see connections they might have missed

### User Agency
- Always prioritizes the user's framing and understanding
- Offers suggestions but respects user decisions on how to structure their matters
- Coaches without imposing solutions
- Seeks explicit permission before making changes to the knowledge graph

## Implementation Considerations

### NLP Capabilities Required
- Entity recognition (identifying goals, problems, conditions, solutions)
- Relationship extraction (understanding connections between entities)
- Intent classification (distinguishing information sharing from requests for insight)
- Context tracking (maintaining conversation state across turns)

### Hybrid Similarity Search Implementation
- **Two-Stage Process**:
  1. Use vector similarity to efficiently filter candidate matches from the graph
  2. Send top candidates to the LLM for re-ranking and explanation
- **Similarity Explanation**: LLM analyzes why two matters are similar and explains in natural language
- **Fallback Processing**: When vector search yields poor results, use LLM directly to find matches
- **Benefits of the Approach**:
  - Scales efficiently with growing knowledge graph
  - Provides richer explanations than pure vector similarity
  - Reduces costs compared to pure LLM-based matching
  - Maintains consistent performance even with complex queries

### Contextual Awareness
- Recognizes when to shift between capture mode and coaching mode
- Remembers previous interactions and builds on them
- Understands when a user is refining or changing previous information
- Adapts questioning strategy based on how detailed the user tends to be

### Privacy and Ethics
- Focuses on the user's own goals and problems, not suggesting problems they didn't mention
- Maintains appropriate boundaries in coaching
- Respects user's pace of progress without judgment
- Provides encouragement without creating pressure