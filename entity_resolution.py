"""
Entity Resolution System for matters.global

This module provides functionality to:
- Identify similar matters (goals, problems, conditions, solutions) based on semantic similarity
- Suggest canonical forms for similar entities
- Merge or link related entities
- Manage the canonical representation of matters
"""

from typing import List, Dict, Any, Optional, Tuple, Set, Union
import logging
from pydantic import BaseModel, Field
import numpy as np
from graph_problem_manager import GraphManager, MatterLabel, Matter, CanonicalMatter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EntityMatch(BaseModel):
    """Model representing a potential match between two entities."""
    source_id: str = Field(..., description="ID of the source entity")
    target_id: str = Field(..., description="ID of the target entity")
    source_labels: List[str] = Field(..., description="Labels of the source entity (Goal, Problem, Condition, Solution)")
    target_labels: List[str] = Field(..., description="Labels of the target entity (Goal, Problem, Condition, Solution)")
    similarity_score: float = Field(..., description="Similarity score between the entities")
    confidence: float = Field(..., description="Confidence in the match")
    match_factors: Dict[str, float] = Field(
        default_factory=dict,
        description="Factors contributing to the match with their weights"
    )

class CanonicalSuggestion(BaseModel):
    """Model representing a suggested canonical form for a group of entities."""
    entity_ids: List[str] = Field(..., description="IDs of the entities in this group")
    entity_labels: List[str] = Field(..., description="Common labels of the entities in this group")
    suggested_description: str = Field(..., description="Suggested canonical description")
    confidence: float = Field(..., description="Confidence in the suggestion")
    supporting_factors: Dict[str, Any] = Field(
        default_factory=dict,
        description="Factors supporting this suggestion"
    )

class EntityResolutionSystem:
    """Class for identifying and resolving similar entities in the graph database."""

    def __init__(self, graph_manager: GraphManager):
        """Initialize the EntityResolutionSystem.

        Args:
            graph_manager: Instance of GraphManager for database access
        """
        self.graph_manager = graph_manager
        
    def find_similar_entities(self,
                             entity_id: str = None,
                             description: str = None,
                             labels: List[str] = None,
                             threshold: float = 0.7,
                             limit: int = 10) -> List[EntityMatch]:
        """Find similar entities to the given entity or description.

        Args:
            entity_id: ID of the entity to find matches for (optional if description is provided)
            description: Text description to search for (optional if entity_id is provided)
            labels: Optional list of labels to filter by (e.g., ["Problem", "Goal"])
            threshold: Similarity threshold (0-1)
            limit: Maximum number of matches to return

        Returns:
            List of potential entity matches
        """
        # Check that either entity_id or description is provided
        if entity_id is None and description is None:
            logger.error("Either entity_id or description must be provided")
            return []

        # Use either entity from database or direct text search
        entity = None
        entity_description = description

        if entity_id:
            # Get the entity from the database
            entity = self.graph_manager.get_matter_by_id(entity_id)
            if not entity:
                logger.error(f"Matter with ID {entity_id} not found")
                return []
            entity_description = entity.description

        # Use the entity's labels as the filter if none provided
        if entity and not labels:
            # Filter out the base Matter label which is always present
            search_labels = [label for label in entity.labels if label != MatterLabel.MATTER.value]
        else:
            search_labels = labels

        # Find similar matters
        similar_entities = self.graph_manager.find_similar_matters(
            entity_description,
            labels=search_labels,
            threshold=threshold,
            limit=limit + 1 if entity_id else limit  # +1 to allow for filtering out self if entity_id is provided
        )

        # Convert to EntityMatch objects
        matches = []
        for similar in similar_entities:
            # Skip the entity itself if we're searching by entity_id
            if entity_id and similar["id"] == entity_id:
                continue

            # For direct description search, we need a dummy source ID
            source_id = entity_id if entity_id else "description_search"
            source_labels = entity.labels if entity else [MatterLabel.MATTER.value]

            match = EntityMatch(
                source_id=source_id,
                target_id=similar["id"],
                source_labels=source_labels,
                target_labels=similar.get("labels", [MatterLabel.MATTER.value]),
                similarity_score=similar.get("similarity", 0.0),
                confidence=self._calculate_confidence(similar),
                match_factors={
                    "vector_similarity": similar.get("similarity", 0.0),
                    # Additional factors could be added here
                }
            )
            matches.append(match)

        return matches

    # Legacy method for backward compatibility
    def find_similar_problems(self,
                             problem_id: str,
                             threshold: float = 0.7,
                             limit: int = 10) -> List[EntityMatch]:
        """Legacy method to find similar problems.

        Args:
            problem_id: ID of the problem to find matches for
            threshold: Similarity threshold (0-1)
            limit: Maximum number of matches to return

        Returns:
            List of potential problem matches
        """
        return self.find_similar_entities(
            entity_id=problem_id,
            labels=[MatterLabel.PROBLEM.value],
            threshold=threshold,
            limit=limit
        )

    # Legacy method for backward compatibility
    def find_similar_conditions(self,
                             condition_id: str,
                             threshold: float = 0.7,
                             limit: int = 10) -> List[EntityMatch]:
        """Legacy method to find similar conditions.

        Args:
            condition_id: ID of the condition to find matches for
            threshold: Similarity threshold (0-1)
            limit: Maximum number of matches to return

        Returns:
            List of potential condition matches
        """
        return self.find_similar_entities(
            entity_id=condition_id,
            labels=[MatterLabel.CONDITION.value],
            threshold=threshold,
            limit=limit
        )
    
    def group_similar_entities(self,
                              labels: List[str] = None,
                              threshold: float = 0.7,
                              min_group_size: int = 2) -> List[List[Dict[str, Any]]]:
        """Group similar entities into clusters using a graph-based approach.

        Args:
            labels: List of labels to filter by (e.g., ["Problem", "Goal", "Condition"])
            threshold: Similarity threshold (0-1)
            min_group_size: Minimum number of entities in a group

        Returns:
            List of entity groups, where each group is a list of entities
        """
        # Use all labels if none provided
        if not labels:
            labels_str = "all matters"
            cypher_labels = "Matter"
        else:
            labels_str = ", ".join(labels)
            cypher_labels = ":".join(["Matter"] + labels)

        logger.info(f"Grouping similar matters ({labels_str}) with threshold {threshold}")

        # Step 1: Get all entities with the given labels
        entities = self._get_all_entities(cypher_labels)
        if not entities:
            return []

        # Step 2: Build similarity graph (adjacency list)
        # Key is entity ID, value is list of (target_id, similarity) tuples
        similarity_graph = self._build_similarity_graph(entities, threshold)

        # Step 3: Find connected components (groups of similar entities)
        groups = self._find_connected_components(similarity_graph)

        # Step 4: Filter groups by minimum size
        groups = [group for group in groups if len(group) >= min_group_size]

        # Step 5: Format result as list of entity dictionaries
        result = []
        for group in groups:
            group_entities = []
            for entity_id in group:
                # Find the entity data in the original list
                entity_data = next((e for e in entities if e["id"] == entity_id), None)
                if entity_data:
                    group_entities.append(entity_data)

            if group_entities:
                result.append(group_entities)

        logger.info(f"Found {len(result)} groups of similar matters ({labels_str})")
        return result

    # Legacy method for backward compatibility
    def group_similar_problems(self,
                              threshold: float = 0.7,
                              min_group_size: int = 2) -> List[List[Dict[str, Any]]]:
        """Group similar problems into clusters (legacy method).

        Args:
            threshold: Similarity threshold (0-1)
            min_group_size: Minimum number of entities in a group

        Returns:
            List of problem groups, where each group is a list of problems
        """
        return self.group_similar_entities(
            labels=[MatterLabel.PROBLEM.value],
            threshold=threshold,
            min_group_size=min_group_size
        )

    # Legacy method for backward compatibility
    def group_similar_conditions(self,
                              threshold: float = 0.7,
                              min_group_size: int = 2) -> List[List[Dict[str, Any]]]:
        """Group similar conditions into clusters (legacy method).

        Args:
            threshold: Similarity threshold (0-1)
            min_group_size: Minimum number of entities in a group

        Returns:
            List of condition groups, where each group is a list of conditions
        """
        return self.group_similar_entities(
            labels=[MatterLabel.CONDITION.value],
            threshold=threshold,
            min_group_size=min_group_size
        )

    def _get_all_entities(self, label_pattern: str) -> List[Dict[str, Any]]:
        """Get all entities with the given labels from the database.

        Args:
            label_pattern: Label pattern to match (e.g., "Matter:Problem", "Matter:Condition")

        Returns:
            List of entities
        """
        with self.graph_manager.driver.session() as session:
            # Use the provided label pattern in the Cypher query
            result = session.run(
                f"""
                MATCH (m:{label_pattern})
                RETURN m, labels(m) as labels
                """
            )

            entities = []
            for record in result:
                node = record["m"]
                node_labels = record["labels"]

                # Base entity data
                entity_data = {
                    "id": node["id"],
                    "description": node["description"],
                    "labels": node_labels,
                    "created_at": node.get("created_at"),
                    "updated_at": node.get("updated_at"),
                    "tags": node.get("tags", [])
                }

                # Add label-specific properties
                if MatterLabel.PROBLEM.value in node_labels and "state" in node:
                    entity_data["state"] = node["state"]

                if MatterLabel.CONDITION.value in node_labels and "is_met" in node:
                    entity_data["is_met"] = node["is_met"]

                if MatterLabel.GOAL.value in node_labels and "progress" in node:
                    entity_data["progress"] = node["progress"]

                if MatterLabel.SOLUTION.value in node_labels and "state" in node:
                    entity_data["state"] = node["state"]

                if "embedding" in node:
                    entity_data["embedding"] = node["embedding"]

                entities.append(entity_data)

            return entities

    def _build_similarity_graph(self,
                              entities: List[Dict[str, Any]],
                              threshold: float) -> Dict[str, List[Tuple[str, float]]]:
        """Build a similarity graph as an adjacency list.

        Args:
            entities: List of entity dictionaries
            threshold: Similarity threshold (0-1)

        Returns:
            Adjacency list where keys are entity IDs and values are lists of
            (target_id, similarity) tuples
        """
        graph = {entity["id"]: [] for entity in entities}

        # If we have embeddings, use vector similarity
        has_embeddings = all("embedding" in entity for entity in entities)

        # For small datasets, comparing all pairs is fine
        # For larger datasets, more efficient methods would be needed
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                # Skip self-comparisons
                if i == j:
                    continue

                similarity = 0.0

                if has_embeddings:
                    # Use vector similarity if embeddings are available
                    embedding1 = entity1.get("embedding")
                    embedding2 = entity2.get("embedding")
                    if embedding1 and embedding2:
                        try:
                            # Use the embedding provider for similarity calculation
                            similarity = self.graph_manager.embedding_provider.similarity(
                                embedding1, embedding2
                            )
                        except Exception as e:
                            logger.warning(f"Error calculating vector similarity: {str(e)}")
                            similarity = self._text_similarity(
                                entity1["description"], entity2["description"]
                            )
                else:
                    # Fall back to text similarity
                    similarity = self._text_similarity(
                        entity1["description"], entity2["description"]
                    )

                # Add edge if similarity exceeds threshold
                if similarity >= threshold:
                    graph[entity1["id"]].append((entity2["id"], similarity))

        return graph

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using a simple approach.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        # Simple implementation using Jaccard similarity of words
        # In a real implementation, more sophisticated methods would be used
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def _find_connected_components(self, graph: Dict[str, List[Tuple[str, float]]]) -> List[List[str]]:
        """Find connected components in the similarity graph.

        Args:
            graph: Adjacency list representation of the graph

        Returns:
            List of connected components, where each component is a list of entity IDs
        """
        # Track visited nodes
        visited = set()
        components = []

        # Process each node
        for node in graph:
            if node not in visited:
                # Found a new component
                component = []
                self._dfs(node, graph, visited, component)
                components.append(component)

        return components

    def _dfs(self, node: str, graph: Dict[str, List[Tuple[str, float]]],
           visited: Set[str], component: List[str]) -> None:
        """Depth-first search to find connected components.

        Args:
            node: Current node
            graph: Adjacency list representation of the graph
            visited: Set of visited nodes
            component: Current component being built
        """
        visited.add(node)
        component.append(node)

        # Visit all neighbors
        for neighbor, _ in graph.get(node, []):
            if neighbor not in visited:
                self._dfs(neighbor, graph, visited, component)
    
    def suggest_canonical_form(self,
                              entity_ids: List[str],
                              target_label: str = None) -> Optional[CanonicalSuggestion]:
        """Suggest a canonical form for a group of similar entities.

        Uses a combination of approaches:
        1. Try to use OpenAI to generate a consolidated description (if available)
        2. Fall back to text analysis methods if OpenAI is not available

        Args:
            entity_ids: List of entity IDs to create a canonical form for
            target_label: Optional specific label to focus on (e.g., "Problem", "Goal")
                          If None, uses the most common label among the entities

        Returns:
            A suggested canonical form, or None if no suggestion could be made
        """
        if not entity_ids:
            return None

        # Get all entities and find common labels
        entities = []
        label_counts = {}

        for entity_id in entity_ids:
            entity = self.graph_manager.get_matter_by_id(entity_id)
            if entity:
                entities.append(entity)
                # Count labels (excluding Matter which is on everything)
                for label in entity.labels:
                    if label != MatterLabel.MATTER.value:
                        label_counts[label] = label_counts.get(label, 0) + 1

        if not entities:
            return None

        # Determine the common labels for these entities
        if target_label:
            # Use the specified label if provided
            common_labels = [target_label]
        else:
            # Find the most common labels
            # Sort labels by frequency, highest first
            common_labels = sorted(
                label_counts.keys(),
                key=lambda label: label_counts[label],
                reverse=True
            )

            # Filter to labels present in at least half the entities
            min_count = len(entities) // 2
            common_labels = [label for label in common_labels
                            if label_counts[label] >= min_count]

            # Always include Matter as a fallback
            if not common_labels:
                common_labels = [MatterLabel.MATTER.value]

        # Try different methods for suggesting canonical forms
        methods_to_try = [
            self._suggest_canonical_with_openai,
            self._suggest_canonical_with_text_analysis,
            self._suggest_canonical_simple
        ]

        for method in methods_to_try:
            try:
                suggestion = method(entities, common_labels)
                if suggestion:
                    return suggestion
            except Exception as e:
                logger.warning(f"Error in canonical suggestion method {method.__name__}: {str(e)}")
                continue

        # If all methods fail, use the first entity description as canonical
        return CanonicalSuggestion(
            entity_ids=entity_ids,
            entity_labels=common_labels,
            suggested_description=entities[0].description,
            confidence=0.5,  # Low confidence since this is a fallback
            supporting_factors={
                "method": "fallback_first_entity",
                "num_entities": len(entities),
                "labels": common_labels
            }
        )

    # Legacy method for backward compatibility
    def suggest_canonical_problem_form(self, problem_ids: List[str]) -> Optional[CanonicalSuggestion]:
        """Suggest a canonical form for problems (legacy method).

        Args:
            problem_ids: List of problem IDs to create a canonical form for

        Returns:
            A suggested canonical form, or None if no suggestion could be made
        """
        return self.suggest_canonical_form(problem_ids, MatterLabel.PROBLEM.value)

    # Legacy method for backward compatibility
    def suggest_canonical_condition_form(self, condition_ids: List[str]) -> Optional[CanonicalSuggestion]:
        """Suggest a canonical form for conditions (legacy method).

        Args:
            condition_ids: List of condition IDs to create a canonical form for

        Returns:
            A suggested canonical form, or None if no suggestion could be made
        """
        return self.suggest_canonical_form(condition_ids, MatterLabel.CONDITION.value)

    def _suggest_canonical_with_openai(self,
                                     entities: List[Any],
                                     labels: List[str]) -> Optional[CanonicalSuggestion]:
        """Use OpenAI to generate a canonical description for a group of entities.

        Args:
            entities: List of entities (Matter objects)
            labels: Common labels for these entities

        Returns:
            A suggested canonical form, or None if generation failed
        """
        # Skip if no OpenAI embeddings are available (same provider is needed)
        if not self.graph_manager.embedding_provider or \
           not hasattr(self.graph_manager.embedding_provider, "client"):
            logger.info("OpenAI provider not available for canonical suggestion")
            return None

        try:
            # Extract the OpenAI client from the embedding provider
            client = self.graph_manager.embedding_provider.client

            # Prepare the descriptions as context
            descriptions = [entity.description for entity in entities]
            context = "\n\n".join([f"- {desc}" for desc in descriptions])

            # Format label text for prompt
            label_text = ", ".join(labels)

            # Format the prompt for canonical form generation
            prompt = f"""You are an expert in knowledge organization and entity resolution.

I have a group of similar descriptions that are classified as: {label_text}.
I need a single canonical description that accurately represents the core concept shared by all of these descriptions.

Here are the descriptions:
{context}

Please generate a consolidated canonical description for this group. The description should:
1. Capture the essential shared concept
2. Be clear and concise (aim for 1-2 sentences)
3. Use precise, general terminology
4. Be written in a consistent style
5. Be appropriate for entities with these labels: {label_text}

Return ONLY the canonical description, nothing else.
"""

            # Call the OpenAI API to generate the canonical description
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use a faster, cheaper model
                messages=[
                    {"role": "system", "content": "You are a helpful knowledge organization assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3  # Lower temperature for more focused output
            )

            # Extract the suggested description from the response
            suggested_description = response.choices[0].message.content.strip()

            # If the result is too long, truncate it
            if len(suggested_description) > 200:
                suggested_description = suggested_description[:197] + "..."

            # Calculate a confidence score based on the number of entities
            confidence = min(0.9, 0.6 + (len(entities) * 0.05))

            return CanonicalSuggestion(
                entity_ids=[entity.id for entity in entities],
                entity_labels=labels,
                suggested_description=suggested_description,
                confidence=confidence,
                supporting_factors={
                    "method": "openai_generation",
                    "num_entities": len(entities),
                    "model": "gpt-3.5-turbo",
                    "labels": labels
                }
            )

        except Exception as e:
            logger.error(f"Error using OpenAI for canonical suggestion: {str(e)}")
            return None

    def _suggest_canonical_with_text_analysis(self,
                                           entities: List[Any],
                                           labels: List[str]) -> Optional[CanonicalSuggestion]:
        """Use text analysis to generate a canonical description.

        Extracts key terms and patterns from the descriptions to create a canonical form.

        Args:
            entities: List of entities (Matter objects)
            labels: Common labels for these entities

        Returns:
            A suggested canonical form, or None if analysis failed
        """
        if len(entities) < 2:
            return None

        try:
            # Extract all descriptions
            descriptions = [entity.description.lower() for entity in entities]

            # Split descriptions into words
            all_words = [set(desc.split()) for desc in descriptions]

            # Find common words across descriptions
            common_words = set.intersection(*all_words)

            # Remove common stop words
            stop_words = {"the", "a", "an", "in", "on", "at", "to", "for", "with", "by", "of"}
            common_words = common_words - stop_words

            if not common_words:
                # If no common words, try a different approach
                return None

            # Build a frequency dict for words across all descriptions
            word_freq = {}
            for word_set in all_words:
                for word in word_set:
                    if word not in stop_words:
                        word_freq[word] = word_freq.get(word, 0) + 1

            # Find the most representative description based on common word coverage
            best_score = 0
            best_desc = descriptions[0]

            for desc in descriptions:
                desc_words = set(desc.split())
                # Score is based on coverage of common words and frequent words
                common_coverage = len(desc_words.intersection(common_words)) / max(1, len(common_words))
                # Also consider words that appear in multiple descriptions but not all
                freq_score = sum(word_freq.get(word, 0) for word in desc_words) / len(desc_words)
                score = (common_coverage * 0.7) + (freq_score * 0.3)

                if score > best_score:
                    best_score = score
                    best_desc = desc

            # Use the original case of the best description
            for entity in entities:
                if entity.description.lower() == best_desc:
                    suggested_description = entity.description
                    break
            else:
                # Fallback if original case not found
                suggested_description = best_desc.capitalize()

            # Calculate confidence based on common word coverage and number of entities
            confidence = min(0.85, 0.5 + (len(common_words) / 10) + (len(entities) * 0.05))

            return CanonicalSuggestion(
                entity_ids=[entity.id for entity in entities],
                entity_labels=labels,
                suggested_description=suggested_description,
                confidence=confidence,
                supporting_factors={
                    "method": "text_analysis",
                    "num_entities": len(entities),
                    "common_words": len(common_words),
                    "labels": labels
                }
            )

        except Exception as e:
            logger.error(f"Error in text analysis for canonical suggestion: {str(e)}")
            return None

    def _suggest_canonical_simple(self,
                               entities: List[Any],
                               labels: List[str]) -> Optional[CanonicalSuggestion]:
        """Simple method to generate a canonical form.

        Uses the longest description as the canonical form.

        Args:
            entities: List of entities (Matter objects)
            labels: Common labels for these entities

        Returns:
            A suggested canonical form
        """
        # Sort entities by description length (longer descriptions often have more info)
        entities_sorted = sorted(entities, key=lambda e: len(e.description), reverse=True)
        suggested_description = entities_sorted[0].description

        return CanonicalSuggestion(
            entity_ids=[entity.id for entity in entities],
            entity_labels=labels,
            suggested_description=suggested_description,
            confidence=0.7,  # Moderate confidence for this approach
            supporting_factors={
                "method": "longest_description",
                "num_entities": len(entities),
                "description_length": len(suggested_description),
                "labels": labels
            }
        )
    
    def create_canonical_node(self,
                             suggestion: CanonicalSuggestion,
                             user_approved: bool = False) -> Optional[str]:
        """Create a canonical node based on a suggestion.

        Args:
            suggestion: The canonical form suggestion
            user_approved: Whether the suggestion was approved by a user

        Returns:
            ID of the created canonical node, or None if creation failed
        """
        if not suggestion or not suggestion.entity_ids:
            logger.warning("Cannot create canonical node: invalid suggestion")
            return None

        try:
            # Determine the most specific canonical type based on labels
            # Order from most specific to least specific
            canonical_type = None
            for label in suggestion.entity_labels:
                if label == MatterLabel.PROBLEM.value:
                    canonical_type = "CanonicalProblem"
                    break
                elif label == MatterLabel.CONDITION.value:
                    canonical_type = "CanonicalCondition"
                    break
                elif label == MatterLabel.GOAL.value:
                    canonical_type = "CanonicalGoal"
                    break
                elif label == MatterLabel.SOLUTION.value:
                    canonical_type = "CanonicalSolution"
                    break

            # Default to CanonicalMatter if no specific label matches
            if not canonical_type:
                canonical_type = "CanonicalMatter"

            # Create the canonical node using the generic method
            canonical = self.graph_manager.create_canonical_matter(
                suggestion.suggested_description,
                canonical_type
            )

            # Map each entity to the canonical form
            for entity_id in suggestion.entity_ids:
                self.graph_manager.map_matter_to_canonical(entity_id, canonical.id)

            logger.info(f"Created canonical {canonical_type} '{canonical.id}' and mapped "
                       f"{len(suggestion.entity_ids)} matters to it")

            return canonical.id

        except Exception as e:
            logger.error(f"Error creating canonical node: {str(e)}")
            return None

    # Legacy method for backward compatibility
    def create_canonical_problem_node(self,
                             suggestion: CanonicalSuggestion,
                             user_approved: bool = False) -> Optional[str]:
        """Create a canonical problem node based on a suggestion (legacy method).

        Args:
            suggestion: The canonical form suggestion
            user_approved: Whether the suggestion was approved by a user

        Returns:
            ID of the created canonical problem node, or None if creation failed
        """
        # Ensure the suggestion has the Problem label
        if MatterLabel.PROBLEM.value not in suggestion.entity_labels:
            suggestion.entity_labels.append(MatterLabel.PROBLEM.value)

        return self.create_canonical_node(suggestion, user_approved)

    def auto_resolve_entities(self,
                           labels: List[str] = None,
                           threshold: float = 0.8,
                           min_group_size: int = 2,
                           confidence_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Automatically resolve similar entities by creating canonical forms.

        This is the main entry point for automatic entity resolution:
        1. Group similar entities
        2. Suggest canonical forms for each group
        3. Create canonical nodes and map entities to them

        Args:
            labels: Optional list of labels to filter by (e.g., ["Problem", "Goal"])
            threshold: Similarity threshold for grouping entities
            min_group_size: Minimum number of entities in a group
            confidence_threshold: Minimum confidence for accepting suggestions

        Returns:
            List of created canonical nodes with their mapped entities
        """
        labels_str = ", ".join(labels) if labels else "all matters"
        logger.info(f"Auto-resolving entities ({labels_str}) with threshold {threshold}")

        # Step 1: Group similar entities
        groups = self.group_similar_entities(
            labels=labels,
            threshold=threshold,
            min_group_size=min_group_size
        )

        if not groups:
            logger.info(f"No similar entity groups found for {labels_str}")
            return []

        logger.info(f"Found {len(groups)} groups of similar entities ({labels_str})")

        # Step 2 & 3: Suggest canonical forms and create canonical nodes
        results = []
        for i, group in enumerate(groups):
            # Get entity IDs for the group
            entity_ids = [entity["id"] for entity in group]

            # Suggest a canonical form
            suggestion = self.suggest_canonical_form(entity_ids, target_label=None)

            if not suggestion or suggestion.confidence < confidence_threshold:
                logger.info(f"Skipping group {i+1}: low confidence "
                           f"({suggestion.confidence if suggestion else 0:.2f} < {confidence_threshold})")
                continue

            # Create the canonical node
            canonical_id = self.create_canonical_node(suggestion)

            if canonical_id:
                # Add to results
                results.append({
                    "canonical_id": canonical_id,
                    "canonical_description": suggestion.suggested_description,
                    "entity_ids": suggestion.entity_ids,
                    "entity_labels": suggestion.entity_labels,
                    "confidence": suggestion.confidence,
                    "method": suggestion.supporting_factors.get("method", "unknown")
                })

        logger.info(f"Auto-resolved {len(results)} groups of entities ({labels_str})")
        return results

    # Legacy method for backward compatibility
    def auto_resolve_problems(self,
                           threshold: float = 0.8,
                           min_group_size: int = 2,
                           confidence_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Automatically resolve similar problems by creating canonical forms (legacy method).

        Args:
            threshold: Similarity threshold for grouping problems
            min_group_size: Minimum number of problems in a group
            confidence_threshold: Minimum confidence for accepting suggestions

        Returns:
            List of created canonical nodes with their mapped problems
        """
        return self.auto_resolve_entities(
            labels=[MatterLabel.PROBLEM.value],
            threshold=threshold,
            min_group_size=min_group_size,
            confidence_threshold=confidence_threshold
        )

    # Legacy method for backward compatibility
    def auto_resolve_conditions(self,
                           threshold: float = 0.8,
                           min_group_size: int = 2,
                           confidence_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Automatically resolve similar conditions by creating canonical forms (legacy method).

        Args:
            threshold: Similarity threshold for grouping conditions
            min_group_size: Minimum number of conditions in a group
            confidence_threshold: Minimum confidence for accepting suggestions

        Returns:
            List of created canonical nodes with their mapped conditions
        """
        return self.auto_resolve_entities(
            labels=[MatterLabel.CONDITION.value],
            threshold=threshold,
            min_group_size=min_group_size,
            confidence_threshold=confidence_threshold
        )

    def merge_duplicates(self,
                       labels: List[str] = None,
                       threshold: float = 0.9,
                       require_user_approval: bool = True) -> List[Dict[str, Any]]:
        """Identify and merge duplicate entities.

        For entities with very high similarity, we can consider them duplicates
        and merge them instead of just creating canonical mappings.

        Args:
            labels: Optional list of labels to filter by (e.g., ["Problem", "Goal"])
            threshold: Similarity threshold for considering duplicates (should be high)
            require_user_approval: Whether user approval is required for merging

        Returns:
            List of merge operations performed
        """
        labels_str = ", ".join(labels) if labels else "all matters"
        logger.info(f"Identifying duplicate entities ({labels_str}) with threshold {threshold}")

        # This is a placeholder for a future implementation
        # In a real system, this would:
        # 1. Find entities with very high similarity
        # 2. For each pair, determine if one should be considered the primary instance
        # 3. For attributes that differ, determine how to merge them
        # 4. If automatic merging is possible, perform the merge
        # 5. Otherwise, flag for user review

        if require_user_approval:
            logger.info(f"User approval required for merging - no automatic merges performed")
            return []

        # Placeholder for future implementation
        return []

    # Legacy method for backward compatibility
    def merge_duplicate_problems(self,
                               threshold: float = 0.9,
                               require_user_approval: bool = True) -> List[Dict[str, Any]]:
        """Identify and merge duplicate problems (legacy method).

        Args:
            threshold: Similarity threshold for considering duplicates (should be high)
            require_user_approval: Whether user approval is required for merging

        Returns:
            List of merge operations performed
        """
        return self.merge_duplicates(
            labels=[MatterLabel.PROBLEM.value],
            threshold=threshold,
            require_user_approval=require_user_approval
        )

    # Legacy method for backward compatibility
    def merge_duplicate_conditions(self,
                                threshold: float = 0.9,
                                require_user_approval: bool = True) -> List[Dict[str, Any]]:
        """Identify and merge duplicate conditions (legacy method).

        Args:
            threshold: Similarity threshold for considering duplicates (should be high)
            require_user_approval: Whether user approval is required for merging

        Returns:
            List of merge operations performed
        """
        return self.merge_duplicates(
            labels=[MatterLabel.CONDITION.value],
            threshold=threshold,
            require_user_approval=require_user_approval
        )
    
    def _calculate_confidence(self, entity_data: Dict[str, Any]) -> float:
        """Calculate confidence score for an entity match.
        
        Args:
            entity_data: Data about the entity
            
        Returns:
            Confidence score between 0 and 1
        """
        # Simple implementation based on similarity score
        # In a real implementation, this would consider multiple factors
        similarity = entity_data.get("similarity", 0.0)
        
        # Apply a sigmoid function to score more conservatively
        # Values below 0.5 get reduced confidence, values above get boosted
        if similarity > 0.9:
            return 0.95
        elif similarity > 0.8:
            return 0.85
        elif similarity > 0.7:
            return 0.75
        elif similarity > 0.6:
            return 0.6
        else:
            return 0.4