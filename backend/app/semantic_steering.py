"""
Semantic Steering for Latent Space Intervention
Uses embedding algebra to create meaningful steering vectors.
"""

from typing import Optional, Tuple
import torch
from transformers import CLIPTokenizer, CLIPTextModel


class SemanticSteering:
    """
    Implements semantic intervention using embedding algebra.
    Creates steering vectors based on conceptual differences.
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
        Get text embedding from CLIP encoder.
        
        Args:
            text: Text to encode
        
        Returns:
            Text embedding tensor
        """
        # Tokenize
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Encode
        with torch.no_grad():
            text_embeddings = self.text_encoder(
                text_inputs.input_ids.to(self.device)
            )[0]
        
        return text_embeddings
    
    def create_steering_vector(
        self,
        target_concept: str,
        injection_attribute: str,
        latent_shape: Tuple[int, ...],
        strength: float = 1.0
    ) -> torch.Tensor:
        """
        Create a semantic steering vector using embedding algebra.
        
        The core idea: steering_vector = embedding(injection_attribute) - embedding(target_concept)
        This creates a direction in embedding space from the original concept to the desired attribute.
        
        Args:
            target_concept: The concept to modify (e.g., "tiger")
            injection_attribute: The attribute to inject (e.g., "blue", "neon", "robot")
            latent_shape: Shape of the latent tensor to match
            strength: Strength multiplier for the steering vector
        
        Returns:
            Steering vector for latent space intervention
        """
        # Get embeddings for both concepts
        target_embedding = self.get_text_embedding(target_concept)
        attribute_embedding = self.get_text_embedding(injection_attribute)
        
        # Compute semantic difference vector
        # This represents the "direction" from target to attribute in semantic space
        semantic_diff = attribute_embedding - target_embedding
        
        # Project to latent space dimensions
        # We need to convert from text embedding space [batch, 77, 768] to latent space [1, 4, 64, 64]
        # Average over token dimension and handle batch
        if semantic_diff.dim() == 3:
            # Shape: [batch, 77, 768] -> average tokens -> [batch, 768]
            semantic_direction = semantic_diff.mean(dim=1)  # [batch, 768]
        elif semantic_diff.dim() == 2:
            # Already [batch, features] or [features]
            semantic_direction = semantic_diff
        else:
            # Single dimension, unsqueeze
            semantic_direction = semantic_diff.unsqueeze(0)
        
        # Project to latent channels (4 for SD)
        # Use a simple linear projection (can be more sophisticated)
        batch_size, channels, height, width = latent_shape
        
        # Create projection by reducing dimensionality
        projection_dim = channels * height * width
        semantic_flat = semantic_direction.flatten()
        
        # Repeat and slice to match latent size
        if semantic_flat.shape[0] >= projection_dim:
            steering_flat = semantic_flat[:projection_dim]
        else:
            repeats = (projection_dim // semantic_flat.shape[0]) + 1
            steering_flat = semantic_flat.repeat(repeats)[:projection_dim]
        
        # Reshape to latent dimensions
        steering_vector = steering_flat.reshape(latent_shape)
        
        # Normalize and scale
        # Flatten, normalize along dim=0 (the only dimension), then reshape back
        flat_vec = steering_vector.flatten()
        normalized_vec = F.normalize(flat_vec.unsqueeze(0), dim=1).squeeze(0)
        steering_vector = normalized_vec.reshape(latent_shape)
        steering_vector = steering_vector * strength * 0.01  # Scale down for stability
        
        return steering_vector
    
    def create_advanced_steering(
        self,
        base_prompt: str,
        target_concept: str,
        injection_attribute: str,
        latent_shape: Tuple[int, ...],
        strength: float = 1.0,
        method: str = "difference"
    ) -> torch.Tensor:
        """
        Create an advanced steering vector with multiple methods.
        
        Args:
            base_prompt: Original full prompt
            target_concept: Concept to modify
            injection_attribute: Attribute to inject
            latent_shape: Shape of latent tensor
            strength: Strength multiplier
            method: Steering method - "difference", "addition", or "replacement"
        
        Returns:
            Steering vector for latent space intervention
        """
        if method == "difference":
            # Standard semantic difference
            return self.create_steering_vector(
                target_concept, injection_attribute, latent_shape, strength
            )
        
        elif method == "addition":
            # Add attribute to existing concept
            combined = f"{injection_attribute} {target_concept}"
            combined_embedding = self.get_text_embedding(combined)
            original_embedding = self.get_text_embedding(base_prompt)
            
            semantic_diff = combined_embedding - original_embedding
            
        elif method == "replacement":
            # Replace target with attribute in full context
            modified_prompt = base_prompt.replace(target_concept, injection_attribute)
            modified_embedding = self.get_text_embedding(modified_prompt)
            original_embedding = self.get_text_embedding(base_prompt)
            
            semantic_diff = modified_embedding - original_embedding
        
        else:
            raise ValueError(f"Unknown steering method: {method}")
        
        # Project to latent space (same as create_steering_vector)
        batch_size, channels, height, width = latent_shape
        
        # Handle different tensor dimensions
        if semantic_diff.dim() == 3:
            semantic_direction = semantic_diff.mean(dim=1)
        elif semantic_diff.dim() == 2:
            semantic_direction = semantic_diff
        else:
            semantic_direction = semantic_diff.unsqueeze(0)
        
        projection_dim = channels * height * width
        semantic_flat = semantic_direction.flatten()
        
        if semantic_flat.shape[0] >= projection_dim:
            steering_flat = semantic_flat[:projection_dim]
        else:
            repeats = (projection_dim // semantic_flat.shape[0]) + 1
            steering_flat = semantic_flat.repeat(repeats)[:projection_dim]
        
        steering_vector = steering_flat.reshape(latent_shape)
        
        # Normalize properly
        flat_vec = steering_vector.flatten()
        normalized_vec = torch.nn.functional.normalize(flat_vec.unsqueeze(0), dim=1).squeeze(0)
        steering_vector = normalized_vec.reshape(latent_shape)
        steering_vector = steering_vector * strength * 0.01
        
        return steering_vector
    
    def create_multi_concept_steering(
        self,
        modifications: list,
        latent_shape: Tuple[int, ...],
        strength: float = 1.0
    ) -> torch.Tensor:
        """
        Create steering vector for multiple concept modifications.
        
        Args:
            modifications: List of (target_concept, injection_attribute) tuples
            latent_shape: Shape of latent tensor
            strength: Global strength multiplier
        
        Returns:
            Combined steering vector
        """
        if not modifications:
            # Return zero vector
            return torch.zeros(latent_shape, device=self.device)
        
        # Accumulate steering vectors
        combined_steering = torch.zeros(latent_shape, device=self.device)
        
        for target, attribute in modifications:
            vector = self.create_steering_vector(
                target, attribute, latent_shape, strength / len(modifications)
            )
            combined_steering += vector
        
        return combined_steering


# Import F for normalization
import torch.nn.functional as F
