"""
Concept Extractor for Multi-Token Attention Tracking
Automatically identifies key concepts (nouns/adjectives) from prompts
using NLTK POS tagging for linguistically grounded extraction.
"""

from typing import List, Tuple
import re

# Use NLTK for proper POS tagging — falls back to heuristic if unavailable
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag
    # Ensure required data is available
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False


class ConceptExtractor:
    """
    Extracts key concepts from prompts for multi-token attention tracking.

    Primary method: NLTK POS tagging (NN*, JJ* tags).
    Fallback method: Stopword-filtered regex (if NLTK unavailable).
    """

    # Common English stopwords and function words to filter
    STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'this', 'but', 'they', 'have', 'had',
        'what', 'when', 'where', 'who', 'which', 'why', 'how',
        'above', 'about', 'across', 'after', 'against', 'along', 'among',
        'around', 'before', 'behind', 'below', 'beneath', 'beside', 'between',
        'beyond', 'during', 'inside', 'into', 'near', 'off', 'out', 'over',
        'through', 'toward', 'under', 'until', 'up', 'upon', 'within', 'without',
        'very', 'really', 'just', 'also', 'still', 'even', 'much', 'more', 'most',
        'some', 'any', 'each', 'every', 'all', 'both', 'few', 'many', 'several',
        'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    }

    # POS tags that correspond to nouns
    NOUN_TAGS = {'NN', 'NNS', 'NNP', 'NNPS'}
    # POS tags that correspond to adjectives / attributes
    ADJ_TAGS = {'JJ', 'JJR', 'JJS'}

    def __init__(self):
        """Initialize the concept extractor."""
        self.use_nltk = _NLTK_AVAILABLE

    def extract_concepts(self, prompt: str, max_concepts: int = 4) -> List[Tuple[str, str]]:
        """
        Extract key concepts from a prompt using POS tagging.

        Args:
            prompt: The text prompt
            max_concepts: Maximum number of concepts to extract (default: 4)

        Returns:
            List of (word, category) tuples where category is 'noun' or 'attribute'
        """
        if self.use_nltk:
            return self._extract_with_nltk(prompt, max_concepts)
        else:
            return self._extract_with_heuristic(prompt, max_concepts)

    def _extract_with_nltk(self, prompt: str, max_concepts: int) -> List[Tuple[str, str]]:
        """
        Extract concepts using NLTK POS tagging.
        Nouns (NN*) → 'noun', Adjectives (JJ*) → 'attribute'.
        """
        tokens = word_tokenize(prompt.lower())
        tagged = pos_tag(tokens)

        concepts = []
        seen = set()

        # First pass: collect nouns (subjects are higher priority for tracking)
        for word, tag in tagged:
            if word in seen or word in self.STOPWORDS or len(word) <= 2:
                continue
            if tag in self.NOUN_TAGS:
                concepts.append((word, 'noun'))
                seen.add(word)

        # Second pass: collect adjectives
        for word, tag in tagged:
            if word in seen or word in self.STOPWORDS or len(word) <= 2:
                continue
            if tag in self.ADJ_TAGS:
                concepts.append((word, 'attribute'))
                seen.add(word)

        return concepts[:max_concepts]

    def _extract_with_heuristic(self, prompt: str, max_concepts: int) -> List[Tuple[str, str]]:
        """
        Fallback extraction using stopword filtering when NLTK is unavailable.
        All non-stopword tokens of length > 2 are treated as nouns.
        """
        words = re.findall(r'\b[a-z]+\b', prompt.lower())

        concepts = []
        seen = set()

        for word in words:
            if word in seen or len(word) <= 2 or word in self.STOPWORDS:
                continue
            concepts.append((word, 'noun'))
            seen.add(word)
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
