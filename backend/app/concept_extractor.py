"""
Concept Extractor for Multi-Token Attention Tracking
Automatically identifies key concepts (nouns/adjectives) from prompts.
"""

from typing import List, Tuple
import re


class ConceptExtractor:
    """
    Extracts key concepts from prompts for multi-token attention tracking.
    Uses simple heuristics to avoid heavy NLP dependencies.
    """
    
    # Common English stopwords
    STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'this', 'but', 'they', 'have', 'had',
        'what', 'when', 'where', 'who', 'which', 'why', 'how'
    }
    
    # Common prepositions and articles to filter
    FUNCTION_WORDS = {
        'above', 'about', 'across', 'after', 'against', 'along', 'among',
        'around', 'before', 'behind', 'below', 'beneath', 'beside', 'between',
        'beyond', 'during', 'inside', 'into', 'near', 'off', 'out', 'over',
        'through', 'toward', 'under', 'until', 'up', 'upon', 'within', 'without'
    }
    
    # Words that typically indicate attributes (adjectives/descriptors)
    ATTRIBUTE_INDICATORS = {
        'majestic', 'beautiful', 'dark', 'bright', 'large', 'small', 'red',
        'blue', 'green', 'yellow', 'golden', 'silver', 'wooden', 'metal',
        'ancient', 'modern', 'old', 'new', 'big', 'tiny', 'huge', 'massive',
        'delicate', 'rough', 'smooth', 'soft', 'hard', 'warm', 'cold',
        'neon', 'glowing', 'shiny', 'dull', 'vibrant', 'pale', 'snowy',
        'fiery', 'frozen', 'misty', 'foggy', 'sunny', 'cloudy', 'stormy',
        'robot', 'robotic', 'futuristic', 'vintage', 'rustic', 'elegant'
    }
    
    def __init__(self):
        """Initialize the concept extractor."""
        self.all_stopwords = self.STOPWORDS | self.FUNCTION_WORDS
    
    def extract_concepts(self, prompt: str, max_concepts: int = 4) -> List[Tuple[str, str]]:
        """
        Extract key concepts from a prompt.
        
        Args:
            prompt: The text prompt
            max_concepts: Maximum number of concepts to extract (default: 4)
        
        Returns:
            List of (word, category) tuples where category is 'noun' or 'attribute'
        """
        # Clean and tokenize
        words = re.findall(r'\b[a-z]+\b', prompt.lower())
        
        # Filter and categorize
        concepts = []
        seen = set()
        
        for word in words:
            # Skip if already seen, too short, or stopword
            if word in seen or len(word) <= 2 or word in self.all_stopwords:
                continue
            
            # Categorize as attribute or noun
            category = 'attribute' if word in self.ATTRIBUTE_INDICATORS else 'noun'
            concepts.append((word, category))
            seen.add(word)
            
            # Stop if we have enough
            if len(concepts) >= max_concepts:
                break
        
        return concepts
    
    def find_token_indices(self, concepts: List[Tuple[str, str]], tokens: List[str]) -> dict:
        """
        Find the token indices for each concept in the tokenized prompt.
        
        Args:
            concepts: List of (word, category) tuples
            tokens: List of tokens from the tokenizer
        
        Returns:
            Dictionary mapping concept words to their token indices
        """
        concept_indices = {}
        
        for word, category in concepts:
            # Try to find the word in tokens (handling subword tokenization)
            for idx, token in enumerate(tokens):
                # Clean token (remove special chars)
                token_clean = token.strip().lower()
                token_clean = re.sub(r'[^\w]', '', token_clean)
                
                # Check for exact match or if token contains the word
                if word == token_clean or word in token_clean:
                    if word not in concept_indices:
                        concept_indices[word] = []
                    concept_indices[word].append({
                        'index': idx,
                        'token': token,
                        'category': category
                    })
        
        return concept_indices
    
    def get_concept_summary(self, concepts: List[Tuple[str, str]]) -> str:
        """
        Generate a human-readable summary of extracted concepts.
        
        Args:
            concepts: List of (word, category) tuples
        
        Returns:
            Summary string
        """
        if not concepts:
            return "No key concepts detected"
        
        nouns = [word for word, cat in concepts if cat == 'noun']
        attributes = [word for word, cat in concepts if cat == 'attribute']
        
        parts = []
        if nouns:
            parts.append(f"Subjects: {', '.join(nouns)}")
        if attributes:
            parts.append(f"Attributes: {', '.join(attributes)}")
        
        return " | ".join(parts)


# Singleton instance for easy access
_extractor = ConceptExtractor()


def extract_concepts(prompt: str, max_concepts: int = 4) -> List[Tuple[str, str]]:
    """
    Convenience function to extract concepts from a prompt.
    
    Args:
        prompt: The text prompt
        max_concepts: Maximum number of concepts to extract
    
    Returns:
        List of (word, category) tuples
    """
    return _extractor.extract_concepts(prompt, max_concepts)


def find_token_indices(concepts: List[Tuple[str, str]], tokens: List[str]) -> dict:
    """
    Convenience function to find token indices for concepts.
    
    Args:
        concepts: List of (word, category) tuples
        tokens: List of tokens from the tokenizer
    
    Returns:
        Dictionary mapping concept words to their token indices
    """
    return _extractor.find_token_indices(concepts, tokens)
