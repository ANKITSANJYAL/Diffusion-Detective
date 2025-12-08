"""
AI Narrator Service
Generates Sherlock Holmes-style investigation reports from attention logs.
"""

from typing import List, Optional
import os
from openai import OpenAI


class NarratorService:
    """
    Service for generating narrative explanations of the diffusion process.
    Uses OpenAI GPT-4o-mini or can be adapted for local LLMs.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the narrator service.
        
        Args:
            api_key: OpenAI API key. If None, reads from environment.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            print("Warning: No OpenAI API key provided. Narrative generation will be limited.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
    
    def generate_narrative(
        self,
        prompt: str,
        reasoning_logs: List[str],
        intervention_active: bool,
        intervention_strength: float = 1.0
    ) -> str:
        """
        Generate a Sherlock Holmes-style narrative from attention logs.
        
        Args:
            prompt: The original generation prompt
            reasoning_logs: List of step-by-step attention analysis logs
            intervention_active: Whether intervention was applied
            intervention_strength: Strength of intervention
        
        Returns:
            Narrative text in Sherlock Holmes style
        """
        
        if not self.client:
            return self._generate_fallback_narrative(
                prompt, reasoning_logs, intervention_active, intervention_strength
            )
        
        try:
            # Prepare context for the LLM
            log_summary = "\n".join(reasoning_logs[:20])  # Limit to avoid token overflow
            
            intervention_info = ""
            if intervention_active:
                intervention_info = f"\n\n🔬 INTERVENTION APPLIED:\nLatent steering with strength {intervention_strength:.2f} was applied during the generation process."
            
            system_prompt = """You are a technical forensic analyst investigating the internal workings of a Stable Diffusion model.

Your task is to provide a CONCISE, DATA-DRIVEN analysis report (3-4 sentences) that:
1. Identifies which tokens/concepts the model focused on during generation (use actual token names from logs)
2. Quantifies attention shifts with percentages from the logs
3. Explains any latent steering interventions and their measurable impact
4. Uses precise technical language, NOT flowery metaphors

Example style: "At Step 40, the model shifted primary attention from 'mountain' (65% confidence) to 'tiger' (82% confidence), indicating structure refinement. The intervention at Step 30 caused a 20% increase in color coherence for the target attribute."

Be factual, concise, and technical. Avoid poetry or vague language like "ghostly whispers" or "tumultuous sea"."""

            user_prompt = f"""CASE FILE: Image Generation Investigation

PROMPT: "{prompt}"

ATTENTION LOGS:
{log_summary}
{intervention_info}

Deduce what happened during this generation process, Detective Holmes!"""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300,
                temperature=0.8
            )
            
            narrative = response.choices[0].message.content
            return narrative
        
        except Exception as e:
            print(f"Error generating narrative with OpenAI: {e}")
            return self._generate_fallback_narrative(
                prompt, reasoning_logs, intervention_active, intervention_strength
            )
    
    def _generate_fallback_narrative(
        self,
        prompt: str,
        reasoning_logs: List[str],
        intervention_active: bool,
        intervention_strength: float
    ) -> str:
        """
        Generate a rule-based narrative when LLM is unavailable.
        """
        
        # Extract key tokens from logs
        key_concepts = set()
        for log in reasoning_logs:
            if "Focus on" in log:
                concepts = log.split("Focus on ")[-1].strip()
                key_concepts.add(concepts)
        
        concepts_str = ", ".join(list(key_concepts)[:5]) if key_concepts else "various visual elements"
        
        if intervention_active:
            narrative = f"""🔍 **Detective's Log**: Elementary, my dear Watson! The diffusion process revealed a fascinating pattern. 
            
The model's attention oscillated between {concepts_str}, each step refining the latent representation with mathematical precision. 
            
Most intriguingly, I applied a latent steering intervention (strength: {intervention_strength:.2f}), subtly manipulating the model's trajectory through the noise space. The result? A controlled deviation from the natural path—proof that we can indeed steer these generative beasts!

The game is afoot! 🕵️"""
        else:
            narrative = f"""🔍 **Detective's Log**: A straightforward case, Watson. The model proceeded naturally, focusing its attention on {concepts_str} throughout the denoising process. 

No interventions were applied—this is the baseline, the control experiment. Pure, unadulterated diffusion at work.

The natural order prevails. 🎩"""
        
        return narrative
    
    def generate_step_explanation(self, step: int, log_entry: str) -> str:
        """
        Generate a brief explanation for a single step.
        
        Args:
            step: Current generation step
            log_entry: Log entry for this step
        
        Returns:
            Brief explanation string
        """
        
        # Simple keyword highlighting
        keywords = ["Focus", "Shape", "Color", "Texture", "Intervention", "Steering"]
        
        explanation = log_entry
        for keyword in keywords:
            if keyword in explanation:
                explanation = explanation.replace(keyword, f"**{keyword}**")
        
        return explanation
