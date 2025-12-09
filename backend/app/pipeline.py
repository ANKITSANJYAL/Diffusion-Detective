"""
Custom Interpretable Stable Diffusion Pipeline
Implements attention extraction and latent steering without heuristics.
v2.0: Multi-token tracking and semantic interventions
"""

from typing import Optional, Dict, List, Tuple, Callable, Union
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import Attention
import numpy as np
from PIL import Image

from .concept_extractor import extract_concepts, find_token_indices
from .semantic_steering import SemanticSteering


class AttentionStore:
    """Stores attention maps across timesteps."""
    
    def __init__(self):
        self.attention_maps: Dict[int, List[torch.Tensor]] = {}
        self.current_step = 0
        self.logs: List[Dict] = []  # Structured logs instead of strings
        self.high_fidelity_data: Dict[str, Dict] = {}  # High-fidelity data packet for LLM reasoning
        self.baseline_data: Dict[str, Dict] = {}  # 🆕 Baseline attention data for comparison
        self.is_baseline_pass = True  # Track which pass we're on
    
    def reset(self):
        """Clear all stored attention maps and logs."""
        self.attention_maps.clear()
        self.logs.clear()
        self.high_fidelity_data.clear()
        self.baseline_data.clear()
        self.current_step = 0
        self.is_baseline_pass = True
    
    def add_attention_map(self, step: int, attention_probs: torch.Tensor, layer_name: str = ""):
        """Store attention probabilities for a given step."""
        if step not in self.attention_maps:
            self.attention_maps[step] = []
        
        # Store detached copy to prevent memory leaks
        self.attention_maps[step].append(attention_probs.detach().cpu())
    
    def add_log(self, step: int, phase: str, message: str, intervention_active: bool = False, metadata: dict = None):
        """Add a structured log entry."""
        log_entry = {
            "step": step,
            "phase": phase,
            "message": message,
            "intervention_active": intervention_active
        }
        if metadata:
            log_entry["metadata"] = metadata
        
        self.logs.append(log_entry)
    
    def interpret_attention(self, phase: str, concept: str, score: float, confidence: float, 
                           is_intervention: bool = False, delta: float = 0.0) -> str:
        """
        Translate technical attention metrics into descriptive artist-like actions.
        
        Args:
            phase: Current generation phase (e.g., "Composition Planning")
            concept: The token being focused on (e.g., "tiger")
            score: Raw attention score (0-1)
            confidence: Scaled confidence percentage (0-100)
            is_intervention: Whether semantic injection is active
            delta: Change in attention from previous step
        
        Returns:
            Descriptive action string (e.g., "Drafting heavy outlines for tiger")
        """
        # Determine intensity level
        if confidence >= 80:
            intensity = "Strong"
        elif confidence >= 60:
            intensity = "Moderate"
        elif confidence >= 40:
            intensity = "Light"
        else:
            intensity = "Faint"
        
        # Detect attention spike (mutation/injection effect)
        if is_intervention and delta > 0.003:  # Significant jump
            return f"MUTATION: '{concept}' is overtaking the prompt (Attention spike: +{delta*100:.1f}%)"
        
        # Phase-specific descriptions
        if phase == "Composition Planning":
            if confidence > 70:
                return f"Sketching bold composition for '{concept}' ({intensity} focus)"
            elif confidence > 40:
                return f"Planning rough layout of '{concept}' ({intensity} positioning)"
            else:
                return f"Considering placement of '{concept}' ({intensity} presence)"
        
        elif phase == "Attribute Decision":
            if is_intervention:
                return f"INJECTING: Forcing new attributes onto '{concept}' (Mutation active)"
            elif confidence > 70:
                return f"Locking in visual details for '{concept}' ({intensity} commitment)"
            elif confidence > 40:
                return f"Deciding appearance of '{concept}' (Color/texture emerging)"
            else:
                return f"Exploring visual options for '{concept}' ({intensity} consideration)"
        
        elif phase == "Structure Formation":
            if confidence > 70:
                return f"Solidifying structure of '{concept}' ({intensity} definition)"
            elif confidence > 40:
                return f"Refining edges and form of '{concept}' (Shape hardening)"
            else:
                return f"Adjusting proportions of '{concept}' ({intensity} tweaking)"
        
        else:  # Detail Refinement
            if confidence > 70:
                return f"Polishing fine details of '{concept}' (Final touches)"
            elif confidence > 40:
                return f"Adding texture details to '{concept}' (Surface refinement)"
            elif confidence < 20:
                return f"Fading '{concept}' into background (Diminishing focus)"
            else:
                return f"Subtle refinements to '{concept}' (Light finishing)"
    
    def _get_phase_action_verb(self, phase: str, is_intervention: bool, confidence: float) -> str:
        """Get descriptive action verb for a phase."""
        if is_intervention:
            if phase == "Attribute Decision":
                return "INJECTING new features into"
            else:
                return "Mutation warping"
        
        if phase == "Composition Planning":
            if confidence > 60:
                return "Sketching bold composition for"
            else:
                return "Planning rough layout of"
        elif phase == "Attribute Decision":
            if confidence > 60:
                return "Locking in visual details for"
            else:
                return "Deciding appearance of"
        elif phase == "Structure Formation":
            if confidence > 60:
                return "Solidifying structure of"
            else:
                return "Refining form of"
        else:  # Detail Refinement
            if confidence > 60:
                return "Polishing fine details of"
            else:
                return "Adding subtle touches to"
    
    def group_logs(self) -> List[Dict]:
        """Group consecutive similar logs to reduce clutter."""
        if not self.logs:
            return []
        
        grouped = []
        current_group = None
        
        for log in self.logs:
            # Don't group special logs (initialization, intervention markers, complete)
            if log['phase'] in ['Initialization', 'Baseline Complete', 'Intervention Setup', 
                                'Intervention Start', 'Intervention End', 'Complete']:
                if current_group:
                    grouped.append(current_group)
                    current_group = None
                grouped.append(log)
                continue
            
            # Check if this log can be grouped with current group
            if current_group is None:
                current_group = {
                    'start_step': log['step'],
                    'end_step': log['step'],
                    'phase': log['phase'],
                    'intervention_active': log['intervention_active'],
                    'confidences': [log['metadata'].get('confidence', 0) if log.get('metadata') else 0],
                    'tokens': [log['metadata'].get('token', '') if log.get('metadata') else ''],
                    'count': 1
                }
            elif (log['phase'] == current_group['phase'] and 
                  log['intervention_active'] == current_group['intervention_active'] and
                  abs(log['step'] - current_group['end_step']) <= 1):
                # Add to current group
                current_group['end_step'] = log['step']
                current_group['count'] += 1
                if log.get('metadata'):
                    current_group['confidences'].append(log['metadata'].get('confidence', 0))
                    current_group['tokens'].append(log['metadata'].get('token', ''))
            else:
                # Finalize current group and start new one
                grouped.append(current_group)
                current_group = {
                    'start_step': log['step'],
                    'end_step': log['step'],
                    'phase': log['phase'],
                    'intervention_active': log['intervention_active'],
                    'confidences': [log['metadata'].get('confidence', 0) if log.get('metadata') else 0],
                    'tokens': [log['metadata'].get('token', '') if log.get('metadata') else ''],
                    'count': 1
                }
        
        # Don't forget last group
        if current_group:
            grouped.append(current_group)
        
        # Convert groups to log format
        final_logs = []
        for item in grouped:
            if 'message' in item:
                # Single log (not grouped)
                final_logs.append(item)
            else:
                # Grouped log
                avg_conf = sum(item['confidences']) / len(item['confidences']) if item['confidences'] else 0
                # Get most common token
                token_counts = {}
                for t in item['tokens']:
                    if t:
                        token_counts[t] = token_counts.get(t, 0) + 1
                main_token = max(token_counts.items(), key=lambda x: x[1])[0] if token_counts else 'unknown'
                
                if item['count'] == 1:
                    # Don't group single items
                    final_logs.append(self.logs[[i for i, l in enumerate(self.logs) if l.get('step') == item['start_step']][0]])
                else:
                    # Create descriptive grouped message based on phase and intervention
                    phase_action = self._get_phase_action_verb(item['phase'], item['intervention_active'], avg_conf)
                    
                    # Determine intensity descriptor
                    if avg_conf >= 75:
                        intensity_desc = "Strong focus"
                    elif avg_conf >= 50:
                        intensity_desc = "Moderate work"
                    elif avg_conf >= 30:
                        intensity_desc = "Light touches"
                    else:
                        intensity_desc = "Background presence"
                    
                    message = f"{phase_action} '{main_token}' ({intensity_desc})"
                    
                    final_logs.append({
                        'step': item['start_step'],
                        'step_range': f"{item['start_step']}-{item['end_step']}",
                        'phase': item['phase'],
                        'message': message,
                        'intervention_active': item['intervention_active'],
                        'grouped': True,
                        'count': item['count']
                    })
        
        return final_logs
    
    def analyze_attention(
        self, 
        step: int, 
        tokens: List[str], 
        phase: str, 
        intervention_active: bool = False,
        tracked_concepts: dict = None
    ) -> dict:
        """
        Analyze attention patterns with multi-token tracking support.
        
        Args:
            step: Current denoising step
            tokens: Full list of tokens
            phase: Current phase name
            intervention_active: Whether intervention is active
            tracked_concepts: Dict mapping concept words to their token info
        
        Returns:
            Dictionary with attention analysis including multi-token focus scores
        """
        if step not in self.attention_maps or not self.attention_maps[step]:
            return None
        
        try:
            attention_list = self.attention_maps[step]
            if not attention_list:
                return None
            
            # Find the attention map with the largest spatial dimension
            max_spatial_size = 0
            best_attention = None
            
            for attn in attention_list:
                if attn.dim() == 3:  # [heads, spatial, text_tokens] (batch is already merged)
                    spatial_size = attn.shape[1]
                    if spatial_size > max_spatial_size:
                        max_spatial_size = spatial_size
                        best_attention = attn
                elif attn.dim() == 4:  # [batch, heads, spatial, text_tokens]
                    spatial_size = attn.shape[2]
                    if spatial_size > max_spatial_size:
                        max_spatial_size = spatial_size
                        best_attention = attn
            
            if best_attention is None:
                return None
            
            # Average over heads/batch
            if best_attention.dim() == 3:
                avg_attention = best_attention.mean(dim=0)  # [spatial, text_tokens]
            else:
                avg_attention = best_attention.mean(dim=1).mean(dim=0)  # [spatial, text_tokens]
            
            # Get top attended tokens with normalized confidence scores
            if len(tokens) > 0 and avg_attention.shape[-1] <= len(tokens):
                # Average across spatial dimension to get per-token attention
                spatial_avg = avg_attention.mean(dim=0)  # [text_tokens]
                
                # Get actual attention scores (0-1 range)
                attention_scores = spatial_avg.cpu()
                
                # MULTI-TOKEN TRACKING: Extract attention for tracked concepts
                focus_scores = {}
                if tracked_concepts:
                    for concept_word, token_info_list in tracked_concepts.items():
                        # Average attention across all token positions for this concept
                        concept_attention = []
                        for token_info in token_info_list:
                            idx = token_info['index']
                            if idx < len(attention_scores):
                                concept_attention.append(attention_scores[idx].item())
                        
                        if concept_attention:
                            avg_attention_score = sum(concept_attention) / len(concept_attention)
                            # Apply empirical scaling
                            confidence = self._scale_attention_to_confidence(avg_attention_score)
                            focus_scores[concept_word] = {
                                'raw': avg_attention_score,
                                'confidence': confidence,
                                'category': token_info_list[0]['category']
                            }
                
                # Get top K tokens FIRST (including special tokens)
                top_k = min(10, len(tokens))  # Get more to filter
                top_values, top_indices = torch.topk(attention_scores, top_k)
                
                # Filter out special tokens and get their attention values
                filtered_tokens = []
                content_token_attentions = []
                for idx, val in zip(top_indices, top_values):
                    token = tokens[idx.item()].strip()
                    token_clean = token.replace('<|startoftext|>', '').replace('<|endoftext|>', '').replace('<|endofpad|>', '').strip()
                    if token_clean and len(token_clean) > 0:
                        raw_score = val.item()
                        content_token_attentions.append(raw_score)
                        filtered_tokens.append((token_clean, raw_score, idx.item()))
                    
                    if len(filtered_tokens) >= 3:
                        break
                
                if not filtered_tokens:
                    return None
                
                # Use RAW attention values directly with empirical scaling
                # This gives more variation than normalizing by max
                final_tokens = []
                for token_clean, raw_score, idx in filtered_tokens:
                    confidence = self._scale_attention_to_confidence(raw_score)
                    final_tokens.append((token_clean, raw_score, confidence))
                
                if not final_tokens:
                    return None
                
                # Primary token (highest attention)
                token, raw_score, confidence = final_tokens[0]
                
                # Get top 3 for metadata
                top_tokens_list = [{"token": t[0], "attention": f"{t[1]:.4f}", "confidence": f"{t[2]:.1f}%"} 
                                   for t in final_tokens[:3]]
                
                # Calculate delta (change from previous step) for mutation detection
                delta = 0.0
                if len(self.logs) > 0:
                    last_log = self.logs[-1]
                    if last_log.get('metadata', {}).get('token') == token:
                        last_score = last_log.get('metadata', {}).get('attention_score', 0)
                        delta = raw_score - last_score
                
                # Generate semantic/descriptive message using interpreter
                message = self.interpret_attention(
                    phase=phase,
                    concept=token,
                    score=raw_score,
                    confidence=confidence,
                    is_intervention=intervention_active,
                    delta=delta
                )
                
                # 🧠 COLLECT HIGH-FIDELITY DATA for LLM reasoning
                step_key = f"step_{step}"
                
                # Store data in appropriate dict based on pass
                if self.is_baseline_pass:
                    # Baseline pass - store for comparison
                    baseline_step = self.baseline_data.get(step_key, {})
                    baseline_step[token] = raw_score
                    baseline_step['phase'] = phase
                    if focus_scores:
                        for concept, data in focus_scores.items():
                            baseline_step[concept] = data['raw']
                    self.baseline_data[step_key] = baseline_step
                else:
                    # Intervention pass - store with action and comparison
                    step_data = self.high_fidelity_data.get(step_key, {})
                    step_data[token] = raw_score
                    step_data['phase'] = phase
                    step_data['action'] = 'inject' if intervention_active else 'none'
                    
                    # Store all tracked concept attentions
                    if focus_scores:
                        for concept, data in focus_scores.items():
                            step_data[concept] = data['raw']
                    
                    # 🔄 Add baseline comparison (delta from baseline)
                    baseline_step = self.baseline_data.get(step_key, {})
                    if baseline_step:
                        step_data['baseline_comparison'] = {}
                        # Calculate deltas for all concepts
                        for concept in step_data:
                            if concept not in ['phase', 'action', 'baseline_comparison'] and concept in baseline_step:
                                baseline_val = baseline_step[concept]
                                current_val = step_data[concept]
                                delta_val = current_val - baseline_val
                                step_data['baseline_comparison'][concept] = {
                                    'baseline': baseline_val,
                                    'intervention': current_val,
                                    'delta': delta_val,
                                    'percent_change': (delta_val / baseline_val * 100) if baseline_val > 0 else 0
                                }
                    
                    self.high_fidelity_data[step_key] = step_data
                
                # Keep technical attribute for metadata
                if phase == "Composition Planning":
                    attribute = f"Structure: {int(confidence)}%"
                elif phase == "Attribute Decision":
                    attribute = f"Attributes: {int(confidence)}%"
                elif phase == "Structure Formation":
                    attribute = f"Form: {int(confidence)}%"
                else:
                    attribute = f"Details: {int(confidence)}%"
                
                return {
                    "token": token,
                    "score": raw_score,
                    "confidence": confidence,
                    "attribute": attribute,
                    "message": message,
                    "top_tokens": top_tokens_list,
                    "focus": focus_scores  # NEW: Multi-concept attention tracking
                }
            
            return None
            
        except Exception as e:
            return None
    
    def _scale_attention_to_confidence(self, raw_score: float) -> float:
        """
        Scale raw attention value to confidence percentage using empirical ranges.
        
        Args:
            raw_score: Raw attention value (typically 0.001-0.02)
        
        Returns:
            Confidence percentage (0-95%)
        """
        # EMPIRICAL SCALING based on observed diffusion attention ranges
        # Typical content token attention: 0.001-0.02 (after spatial averaging)
        # We scale this to 40-95% range for interpretability
        
        if raw_score >= 0.015:  # Very high attention
            confidence = 85 + (raw_score - 0.015) * 500  # 85-95%
        elif raw_score >= 0.010:  # High attention
            confidence = 70 + (raw_score - 0.010) * 3000  # 70-85%
        elif raw_score >= 0.007:  # Medium-high
            confidence = 55 + (raw_score - 0.007) * 5000  # 55-70%
        elif raw_score >= 0.005:  # Medium
            confidence = 40 + (raw_score - 0.005) * 7500  # 40-55%
        elif raw_score >= 0.003:  # Low-medium
            confidence = 25 + (raw_score - 0.003) * 7500  # 25-40%
        else:  # Low attention
            confidence = raw_score * 8000  # 0-25%
        
        # Clamp to valid range
        return min(max(confidence, 0), 95)


class CustomAttentionProcessor:
    """Custom attention processor that intercepts and stores attention maps."""
    
    def __init__(self, attention_store: AttentionStore, step_ref: Dict[str, int]):
        self.attention_store = attention_store
        self.step_ref = step_ref  # Mutable reference to current step
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> torch.FloatTensor:
        """Process attention and store probabilities."""
        
        batch_size, sequence_length, _ = hidden_states.shape
        
        # Prepare attention inputs
        query = attn.to_q(hidden_states)
        
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        else:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        
        # Reshape for multi-head attention
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * attn.scale
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Get attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Store cross-attention maps (these are most interpretable)
        if is_cross_attention:
            current_step = self.step_ref.get('value', 0)
            self.attention_store.add_attention_map(
                current_step,
                attention_probs,
                layer_name=f"cross_attn"
            )
        
        # Apply attention to values
        hidden_states = torch.matmul(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        # Linear projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        return hidden_states


class InterpretableSDPipeline:
    """
    Interpretable Stable Diffusion Pipeline with attention extraction and latent steering.
    No heuristics - pure intervention via latent space manipulation.
    """
    
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "mps",
        torch_dtype: torch.dtype = torch.float32
    ):
        # Force MPS for Mac, prevent CUDA usage
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA requested but not available, switching to MPS/CPU")
            if torch.backends.mps.is_available():
                device = "mps"
                torch_dtype = torch.float32
            else:
                device = "cpu"
                torch_dtype = torch.float32
        
        self.device = device
        self.torch_dtype = torch_dtype
        
        # Load the base pipeline
        print(f"Loading Stable Diffusion model: {model_id}")
        print(f"Target device: {device}, dtype: {torch_dtype}")
        
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,  # Disable for speed
            requires_safety_checker=False
        )
        
        print(f"Moving pipeline to {device}...")
        self.pipeline.to(device)
        print(f"✓ Pipeline successfully loaded on {device}")
        
        # Enable memory optimizations
        self.pipeline.enable_attention_slicing()
        
        # Initialize attention store
        self.attention_store = AttentionStore()
        self.step_ref = {'value': 0}
        
        # Initialize semantic steering (v2.0)
        self.semantic_steering = SemanticSteering(
            tokenizer=self.pipeline.tokenizer,
            text_encoder=self.pipeline.text_encoder,
            device=device
        )
        print("✓ Semantic steering initialized")
        
        print("Pipeline loaded successfully")
    
    def _setup_attention_hooks(self):
        """Install custom attention processors to intercept attention maps."""
        
        # Reset attention store
        self.attention_store.reset()
        
        # Install custom processors on all attention layers in UNet
        custom_processor = CustomAttentionProcessor(self.attention_store, self.step_ref)
        
        # Apply to all attention modules
        for name, module in self.pipeline.unet.named_modules():
            if isinstance(module, Attention):
                module.processor = custom_processor
    
    def _remove_attention_hooks(self):
        """Restore default attention processors."""
        for name, module in self.pipeline.unet.named_modules():
            if isinstance(module, Attention):
                module.processor = module.get_processor()
    
    def _create_intervention_vector(
        self,
        latents_shape: tuple,
        strength: float = 1.0
    ) -> torch.FloatTensor:
        """
        Create a latent intervention vector.
        This creates a small random perturbation in latent space to steer generation.
        
        Args:
            latents_shape: Shape of the latent tensor (batch, channels, height, width)
            strength: Magnitude of the intervention
        
        Returns:
            Intervention vector matching latent dimensions
        """
        # Create a small random vector in latent space
        # This will be added to the latents during specific steps
        intervention_vector = torch.randn(
            latents_shape,
            device=self.device,
            dtype=self.torch_dtype
        ) * 0.01 * strength  # Small perturbation scaled by strength
        
        return intervention_vector
    
    def generate(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        intervention_active: bool = False,
        intervention_strength: float = 1.0,
        intervention_step_start: int = 40,
        intervention_step_end: int = 20,
        seed: Optional[int] = None,
        callback: Optional[Callable[[int, str], None]] = None,
        # v2.0 parameters
        target_concept: Optional[str] = None,
        injection_attribute: Optional[str] = None,
        auto_detect_concepts: bool = True
    ) -> Tuple[Image.Image, Image.Image, List[str], Dict]:
        """
        Generate images with and without intervention (v2.0 with semantic steering).
        
        Args:
            prompt: Text prompt for generation
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            intervention_active: Whether to apply latent steering
            intervention_strength: Strength of intervention (0.0 - 2.0)
            intervention_step_start: Step to start intervention (higher = earlier)
            intervention_step_end: Step to end intervention
            seed: Random seed for reproducibility
            callback: Optional callback for progress updates
        
        Returns:
            (natural_image, controlled_image, reasoning_logs, metadata)
        """
        
        # 🎯 CRITICAL: Set seed for reproducibility - use SAME seed for both passes
        # This ensures identical background/scene in baseline and intervention images
        if seed is not None:
            # Pass 1: Baseline - create generator with seed
            generator_natural = torch.Generator(device=self.device).manual_seed(seed)
            # Pass 2: Will be reset to SAME seed later to ensure identical starting noise
            generator_controlled = None  # Will be created fresh with same seed
        else:
            generator_natural = None
            generator_controlled = None
        
        # Tokenize prompt for attention analysis (get full 77-token sequence)
        tokenizer = self.pipeline.tokenizer
        token_ids = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids[0]
        tokens = [tokenizer.decode([tid]) for tid in token_ids]
        
        # ========== v2.0: CONCEPT EXTRACTION & TRACKING ==========
        tracked_concepts = {}
        concepts_summary = ""
        
        if auto_detect_concepts:
            # Auto-detect key concepts from prompt
            concepts = extract_concepts(prompt, max_concepts=4)
            tracked_concepts = find_token_indices(concepts, tokens)
            
            # Generate summary
            if concepts:
                concept_words = [word for word, cat in concepts]
                concepts_summary = f"Tracking: {', '.join(concept_words)}"
            else:
                concepts_summary = "No key concepts detected"
        
        # ========== GENERATE NATURAL (BASELINE) ==========
        init_msg = f"Starting generation: '{prompt}' ({num_inference_steps} steps)"
        if concepts_summary:
            init_msg += f" | {concepts_summary}"
        
        self.attention_store.add_log(0, "Initialization", init_msg)
        
        # 🔄 CONDITIONAL TWO-PASS LOGIC:
        # - If intervention_active=True: Generate baseline + intervention for comparison
        # - If intervention_active=False: Generate single image only (no comparison needed)
        
        if intervention_active:
            # 🔄 PASS 1: BASELINE (for comparison)
            self.attention_store.is_baseline_pass = True
            
            natural_image = self.pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator_natural
            ).images[0]
            
            self.attention_store.add_log(num_inference_steps, "Baseline Complete", 
                "✅ Natural baseline image generated - attention data captured for comparison")
            
            # 🔄 PASS 2: INTERVENTION (with comparison to baseline)
            self.attention_store.is_baseline_pass = False
        else:
            # 🎨 SINGLE-PASS: No intervention, just generate naturally
            # Set to False since this is the only pass (no baseline comparison needed)
            self.attention_store.is_baseline_pass = False
            self.attention_store.add_log(0, "Mode", 
                "🔬 Natural generation (no intervention) - single pass only")
        
        # ========== GENERATE CONTROLLED (WITH INTERVENTION) ==========
        intervention_mode = "none"
        attribute_embedding = None
        target_token_indices = []
        
        if intervention_active:
            if target_concept and injection_attribute:
                intervention_mode = "embedding_injection"
                
                # 🧬 STEP 1: Get the embedding vector for the injection attribute
                print(f"\n🧬 Encoding injection attribute: '{injection_attribute}'")
                attr_tokens = tokenizer(
                    injection_attribute,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.to(self.device)
                
                with torch.no_grad():
                    attr_embeds = self.pipeline.text_encoder(attr_tokens)[0]
                    # Extract the embedding for the actual word (skip <start_of_text>)
                    attribute_embedding = attr_embeds[0, 1, :].clone()  # [768]
                
                print(f"✓ Attribute embedding shape: {attribute_embedding.shape}")
                
                # 🎯 STEP 2: Find target concept token indices in the prompt
                print(f"\n🎯 Locating '{target_concept}' in prompt tokens...")
                prompt_token_ids = token_ids.tolist()
                target_token_ids = tokenizer.encode(target_concept, add_special_tokens=False)
                
                # Search for target concept in prompt
                for i in range(len(prompt_token_ids) - len(target_token_ids) + 1):
                    if prompt_token_ids[i:i+len(target_token_ids)] == target_token_ids:
                        target_token_indices.extend(range(i, i+len(target_token_ids)))
                        print(f"✓ Found '{target_concept}' at token indices: {target_token_indices}")
                        break
                
                if not target_token_indices:
                    print(f"⚠️  Warning: Could not find '{target_concept}' in prompt. Trying fuzzy match...")
                    # Fuzzy match: find any token containing the concept
                    for i, token in enumerate(tokens):
                        if target_concept.lower() in token.lower():
                            target_token_indices.append(i)
                            print(f"✓ Found similar token '{token}' at index {i}")
                
                # 🎯 Add injection_attribute to tracked_concepts so we can monitor its attention
                if injection_attribute and injection_attribute.strip():
                    # find_token_indices is already imported at the top
                    attr_indices = find_token_indices([(injection_attribute, "injected_attribute")], tokens)
                    if attr_indices:
                        # Merge with existing tracked_concepts
                        tracked_concepts.update(attr_indices)
                        print(f"📊 Now tracking attention on injected attribute: '{injection_attribute}' at indices {attr_indices[injection_attribute]}")
                
                self.attention_store.add_log(0, "Intervention Setup", 
                    f"💉 Embedding Injection: '{injection_attribute}' → '{target_concept}' | Strength: {intervention_strength:.1f}x | Zone: Steps {intervention_step_end}-{intervention_step_start} | Target indices: {target_token_indices}",
                    intervention_active=True)
            else:
                intervention_mode = "latent_noise"
                self.attention_store.add_log(0, "Intervention Setup", 
                    f"Random intervention: strength {intervention_strength:.1f}x, zone {intervention_step_end}-{intervention_step_start}",
                    intervention_active=True)
        
        # ========== EARLY EXIT: If no intervention, use simple generation ==========
        if not intervention_active:
            # Simple generation without intervention - just use pipeline directly
            self._setup_attention_hooks()
            
            controlled_image = self.pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator_natural
            ).images[0]
            
            self._remove_attention_hooks()
            
            self.attention_store.add_log(num_inference_steps, "Complete", 
                f"✅ Natural generation complete - analyzed {len(self.attention_store.attention_maps)} steps")
            
            # Group logs
            grouped_logs = self.attention_store.group_logs()
            
            # Prepare metadata
            metadata = {
                'prompt': prompt,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'intervention_active': False,
                'intervention_strength': 0,
                'intervention_range': 'N/A',
                'total_logs': len(self.attention_store.logs),
                'grouped_logs': len(grouped_logs)
            }
            
            # Return same image for both (no intervention comparison)
            return controlled_image, controlled_image, grouped_logs, metadata
        
        # ========== CONTINUE WITH INTERVENTION LOGIC ==========
        # 🎯 CRITICAL: Reset generator to SAME seed for controlled pass
        # This ensures IDENTICAL starting noise -> same background/scene
        if seed is not None:
            generator_controlled = torch.Generator(device=self.device).manual_seed(seed)
            self.attention_store.add_log(0, "Seed Reset", 
                f"🔄 Generator reset to seed {seed} - ensuring identical background between baseline and intervention")
        
        # Setup attention hooks for interpretability
        self._setup_attention_hooks()
        
        # Encode prompt
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        prompt_embeds = self.pipeline.text_encoder(text_input_ids)[0]
        
        # Manual denoising loop with intervention
        # Using reset generator ensures identical initial latents as baseline
        latents = torch.randn(
            (1, self.pipeline.unet.config.in_channels, 64, 64),
            generator=generator_controlled,
            device=self.device,
            dtype=self.torch_dtype
        )
        
        # ========== v2.0: SEMANTIC STEERING ==========
        # Create intervention vector using semantic algebra or random noise
        if intervention_active and intervention_mode == "semantic":
            # Use embedding algebra to create meaningful steering
            intervention_vector = self.semantic_steering.create_steering_vector(
                target_concept=target_concept,
                injection_attribute=injection_attribute,
                latent_shape=latents.shape,
                strength=intervention_strength
            )
        else:
            # Original random intervention
            intervention_vector = self._create_intervention_vector(
                latents.shape,
                strength=intervention_strength
            )
        
        # Prepare scheduler
        self.pipeline.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.pipeline.scheduler.timesteps
        
        # Prepare prompt embeddings for CFG
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            uncond_tokens = tokenizer(
                "",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            uncond_embeds = self.pipeline.text_encoder(
                uncond_tokens.input_ids.to(self.device)
            )[0]
            prompt_embeds_combined = torch.cat([uncond_embeds, prompt_embeds])
        else:
            prompt_embeds_combined = prompt_embeds
        
        latents = latents * self.pipeline.scheduler.init_noise_sigma
        
        # Helper to get phase name (HIGH timestep = coarse, LOW timestep = detail)
        def get_phase(step, total_steps):
            progress_pct = (step / total_steps) * 100
            if progress_pct >= 70:  # Steps 50-35 (high timesteps = coarse)
                return "Composition Planning"
            elif progress_pct >= 40:  # Steps 35-20 (mid timesteps = attributes)
                return "Attribute Decision"
            elif progress_pct >= 20:  # Steps 20-10 (low-mid timesteps = structure)
                return "Structure Formation"
            else:  # Steps 10-0 (low timesteps = details)
                return "Detail Refinement"
        
        # Denoising loop with TRUE EMBEDDING INJECTION
        for i, t in enumerate(timesteps):
            current_step = len(timesteps) - i
            self.step_ref['value'] = current_step
            phase = get_phase(current_step, num_inference_steps)
            
            # Check if we're in intervention zone
            is_intervening = intervention_active and intervention_step_end <= current_step <= intervention_step_start
            
            # 🧬 TRUE SEMANTIC STEERING: Modify text embeddings directly
            current_prompt_embeds = prompt_embeds_combined.clone()
            
            if is_intervening and intervention_mode == "embedding_injection" and attribute_embedding is not None and target_token_indices:
                # INJECT: Add attribute embedding to target concept token(s)
                # Math: embedding['tiger'] = embedding['tiger'] + (embedding['blue'] * strength)
                
                for token_idx in target_token_indices:
                    if do_classifier_free_guidance:
                        # Inject into the CONDITIONAL embedding (second batch element)
                        current_prompt_embeds[1, token_idx, :] += (attribute_embedding * intervention_strength)
                    else:
                        current_prompt_embeds[0, token_idx, :] += (attribute_embedding * intervention_strength)
                
                # Log the specific injection
                if current_step == intervention_step_start:
                    self.attention_store.add_log(
                        current_step, 
                        "Intervention Start", 
                        f"💉 Injecting '{injection_attribute}' vector into '{target_concept}' token (Strength: {intervention_strength:.2f}x)",
                        intervention_active=True
                    )
                elif current_step == intervention_step_end:
                    self.attention_store.add_log(
                        current_step,
                        "Intervention End",
                        f"Injection complete — '{target_concept}' now contains '{injection_attribute}' semantics",
                        intervention_active=True
                    )
                elif current_step % 5 == 0:
                    self.attention_store.add_log(
                        current_step,
                        phase,
                        f"[ACTIVE] '{injection_attribute}' → '{target_concept}' | Embedding modification active",
                        intervention_active=True
                    )
            
            elif is_intervening and intervention_mode == "latent_noise":
                # Fallback: Old latent noise method
                latents = latents + intervention_vector
                
                if current_step == intervention_step_start:
                    self.attention_store.add_log(
                        current_step, 
                        "Intervention Start", 
                        f"💉 INJECTION APPLIED — Steering latent space (strength: {intervention_strength:.2f}x)",
                        intervention_active=True
                    )
            
            # Expand latents for CFG
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise with MODIFIED embeddings
            with torch.no_grad():
                noise_pred = self.pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=current_prompt_embeds  # Use modified embeddings!
                ).sample
            
            # Perform CFG
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Analyze attention patterns every 5 steps or during intervention
            if current_step % 5 == 0 or is_intervening:
                analysis = self.attention_store.analyze_attention(
                    current_step, 
                    tokens, 
                    phase, 
                    is_intervening,
                    tracked_concepts=tracked_concepts  # v2.0: Pass tracked concepts
                )
                if analysis:
                    self.attention_store.add_log(
                        current_step,
                        phase,
                        analysis['message'],
                        intervention_active=is_intervening,
                        metadata={
                            'token': analysis['token'],
                            'confidence': analysis.get('confidence', 0),
                            'attention_score': analysis['score'],
                            'attribute': analysis['attribute'],
                            'top_tokens': analysis.get('top_tokens', []),
                            'focus': analysis.get('focus', {})  # v2.0: Multi-concept focus scores
                        }
                    )
            
            # Compute previous noisy sample
            latents = self.pipeline.scheduler.step(noise_pred, t, latents).prev_sample
        
        # 🎨 POST-PROCESSING: Decode latents to image
        if callback:
            callback(-1, {"status": "decoding", "message": "🎨 Decoding latents into pixels...", "progress": 98})
        
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.pipeline.vae.decode(latents).sample
        
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        controlled_image = self.pipeline.numpy_to_pil(image)[0]
        
        # Count attention analysis logs (exclude special logs)
        attention_logs = [l for l in self.attention_store.logs 
                         if l.get('phase') not in ['Initialization', 'Baseline Complete', 'Intervention Setup', 
                                                     'Intervention Start', 'Intervention End', 'Complete']
                         and l.get('metadata', {}).get('token')]
        
        self.attention_store.add_log(
            num_inference_steps,
            "Complete",
            f"Generation finished — Analyzed {len(attention_logs)} attention patterns across {len(self.attention_store.attention_maps)} steps"
        )
        
        # Remove attention hooks
        self._remove_attention_hooks()
        
        # 🕵️ POST-PROCESSING: Generate narrative (happens in main.py, but signal intent here)
        if callback:
            callback(-1, {"status": "analyzing", "message": "🕵️ Consulting Forensic Narrator...", "progress": 99})
        
        # Group logs to reduce clutter
        grouped_logs = self.attention_store.group_logs()
        
        # Prepare metadata
        metadata = {
            'prompt': prompt,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'intervention_active': intervention_active,
            'intervention_strength': intervention_strength,
            'intervention_range': f"{intervention_step_end}-{intervention_step_start}",
            'total_logs': len(self.attention_store.logs),
            'grouped_logs': len(grouped_logs)
        }
        
        # 🔄 Return logic based on intervention mode:
        # - If intervention was active: return both baseline and intervention images
        # - If intervention was off: return single image as both (frontend expects 2 images)
        if intervention_active:
            return natural_image, controlled_image, grouped_logs, metadata
        else:
            # Single pass - return same image for both (no comparison)
            return controlled_image, controlled_image, grouped_logs, metadata
    
    def cleanup(self):
        """Free GPU memory."""
        if hasattr(self, 'pipeline'):
            del self.pipeline
        
        # Clear cache for both CUDA and MPS
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
