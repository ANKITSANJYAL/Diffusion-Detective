"""
Semantic Steering for Latent Space Intervention
Uses CLIP embedding algebra to create meaningful steering vectors.

v3.0: Proper text-embedding-space operations. All steering vectors operate
in the same 768-d space as UNet cross-attention conditioning, eliminating
the need for mathematically invalid latent-space projections.
"""

from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel


class SemanticSteering:
    """
    Implements semantic intervention using embedding algebra.

    Two modes of operation:
    1. EMBEDDING INJECTION (primary, used by pipeline.py):
       Directly modifies text embeddings fed to UNet cross-attention.
       This is mathematically grounded — it changes the K/V matrices
       that the cross-attention mechanism uses to condition generation.

    2. TEXT-SPACE STEERING (this module):
       Computes semantic difference vectors in CLIP text embedding space.
       These vectors operate in the same 768-d space as the UNet's
       cross-attention conditioning, so they can be added directly to
       per-token embeddings without needing projection to latent space.
    """

    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        device: str = "mps"
    ):
        """
        Initialize semantic steering.

        Args:
            tokenizer: CLIP tokenizer
            text_encoder: CLIP text encoder
            device: Device to run on
        """
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device

    def get_text_embedding(self, text: str) -> torch.Tensor:
        """
        Get the full sequence of text embeddings from the CLIP encoder.

        Args:
            text: Text to encode

        Returns:
            Text embedding tensor of shape [1, 77, 768]
        """
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(
                text_inputs.input_ids.to(self.device)
            )[0]

        return text_embeddings

    def get_token_embedding(self, word: str) -> torch.Tensor:
        """
        Get the embedding for a specific word's tokens, averaged if subword-tokenized.

        Args:
            word: Single word to encode

        Returns:
            Embedding vector of shape [768]
        """
        # Encode the word (without special tokens to find the actual content tokens)
        token_ids = self.tokenizer.encode(word, add_special_tokens=False)

        # Get the full sequence embedding and extract just the content tokens
        full_embedding = self.get_text_embedding(word)  # [1, 77, 768]

        # Content tokens start at index 1 (after <startoftext>)
        content_embeddings = full_embedding[0, 1:1+len(token_ids), :]  # [n_tokens, 768]

        # Average across subword tokens
        return content_embeddings.mean(dim=0)  # [768]

    def create_embedding_delta(
        self,
        target_concept: str,
        injection_attribute: str,
        strength: float = 1.0
    ) -> torch.Tensor:
        """
        Create a semantic steering delta in text embedding space.

        This computes: delta = embed(attribute) - embed(concept)
        The result lives in the same 768-d space as UNet cross-attention
        conditioning, making it mathematically valid for direct addition
        to per-token embeddings.

        Args:
            target_concept: The concept being modified (e.g., "tiger")
            injection_attribute: The attribute to inject (e.g., "blue")
            strength: Scaling factor

        Returns:
            Steering delta of shape [768], in text embedding space
        """
        target_emb = self.get_token_embedding(target_concept)
        attribute_emb = self.get_token_embedding(injection_attribute)

        delta = attribute_emb - target_emb

        # Normalize to unit direction, then scale
        delta = F.normalize(delta.unsqueeze(0), dim=1).squeeze(0)
        delta = delta * strength

        return delta

    def create_prompt_level_delta(
        self,
        base_prompt: str,
        target_concept: str,
        injection_attribute: str,
        strength: float = 1.0,
        method: str = "replacement"
    ) -> torch.Tensor:
        """
        Create a full-sequence steering delta by comparing two prompts.

        Methods:
        - "replacement": Compare embed("a blue tiger on a mountain") vs
          embed("a tiger on a mountain") — replaces target with
          "attribute target" in the full prompt context.
        - "addition": Compare embed("blue, a tiger on a mountain") vs
          embed("a tiger on a mountain") — prepends the attribute.

        Args:
            base_prompt: Original full prompt
            target_concept: Word to modify in the prompt
            injection_attribute: Attribute to inject
            strength: Scaling factor
            method: "replacement" or "addition"

        Returns:
            Full-sequence delta of shape [1, 77, 768]
        """
        if method == "replacement":
            modified_prompt = base_prompt.replace(
                target_concept,
                f"{injection_attribute} {target_concept}"
            )
        elif method == "addition":
            modified_prompt = f"{injection_attribute} {base_prompt}"
        else:
            raise ValueError(f"Unknown method: {method}")

        original_emb = self.get_text_embedding(base_prompt)    # [1, 77, 768]
        modified_emb = self.get_text_embedding(modified_prompt)  # [1, 77, 768]

        delta = modified_emb - original_emb  # [1, 77, 768]

        # Normalize per-token, then scale
        norms = delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        delta = delta / norms * strength

        return delta

    def compute_cosine_distance(self, concept_a: str, concept_b: str) -> float:
        """
        Compute cosine distance between two concepts in embedding space.
        Useful for measuring how semantically different two concepts are.

        Args:
            concept_a: First concept
            concept_b: Second concept

        Returns:
            Cosine distance (0 = identical, 2 = opposite)
        """
        emb_a = self.get_token_embedding(concept_a)
        emb_b = self.get_token_embedding(concept_b)

        similarity = F.cosine_similarity(
            emb_a.unsqueeze(0), emb_b.unsqueeze(0)
        ).item()

        return 1.0 - similarity
