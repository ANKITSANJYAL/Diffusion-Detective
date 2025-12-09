"""
AI Narrator Service
Generates Sherlock Holmes-style investigation reports from attention logs.
"""

from typing import List, Optional, Dict
import os
import json
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
            
            # 🧬 Detect EMBEDDING INJECTION (v2.0.2)
            injection_detected = False
            injection_attribute = None
            injection_target = None
            
            for log in reasoning_logs:
                if "💉 Injecting" in log and "vector into" in log:
                    # Parse: "💉 Injecting 'blue' vector into 'tiger' token"
                    try:
                        parts = log.split("'")
                        if len(parts) >= 4:
                            injection_attribute = parts[1]  # 'blue'
                            injection_target = parts[3]     # 'tiger'
                            injection_detected = True
                            break
                    except:
                        pass
            
            intervention_info = ""
            if intervention_active:
                if injection_detected and injection_attribute and injection_target:
                    intervention_info = f"\n\n� EMBEDDING INJECTION APPLIED:\n'{injection_attribute}' semantics injected into '{injection_target}' token embeddings (strength: {intervention_strength:.2f}).\nLook for attention changes on '{injection_attribute}' before and after injection."
                else:
                    intervention_info = f"\n\n�🔬 INTERVENTION APPLIED:\nLatent steering with strength {intervention_strength:.2f} was applied during the generation process."
            
            system_prompt = """You are a technical forensic analyst investigating the internal workings of a Stable Diffusion model.

The logs now use DESCRIPTIVE ACTION VERBS instead of dry statistics. Look for these verbs:
- "Sketching", "Planning", "Drafting" → Early composition phase
- "Locking in", "Deciding", "Solidifying" → Mid-generation attribute phase
- "Refining", "Polishing", "Adding touches" → Late detail phase
- "INJECTING", "MUTATION", "Forcing features" → Semantic intervention active
- "Fading into background" → Concept losing attention

Your task is to provide a NARRATIVE analysis (3-4 sentences) that:
1. Describes the ARTISTIC PROCESS: What did the model sketch first? What did it refine later?
2. Identifies when MUTATIONS occurred (if injection happened): "At Step 30, the model stopped naturally drafting 'tiger' and began forcing 'blue' features onto it"
3. Explains the BALANCE: Did one concept dominate while others faded?
4. Uses the ACTION VERBS from the logs naturally in your narrative

Example (with injection): "The model began by sketching a bold composition for 'tiger' with strong focus. At Step 30, semantic injection activated - the system stopped natural rendering and began forcing 'blue' attributes onto the 'tiger' concept. This mutation caused 'blue' attention to spike from background levels to 65%, successfully warping the original concept. Final steps polished the hybrid result."

Example (no injection): "Initial steps focused on planning the rough layout of 'mountain' while 'tiger' remained in background. Mid-generation locked in visual details for 'tiger' with strong commitment, overshadowing 'mountain' which faded. Final refinement balanced both concepts moderately."

Write like you're watching an artist paint in real-time. Use the descriptive verbs from the logs."""

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
        
        # 🧬 Detect EMBEDDING INJECTION (v2.0.2)
        injection_detected = False
        injection_attribute = None
        injection_target = None
        
        for log in reasoning_logs:
            if "💉 Injecting" in log and "vector into" in log:
                try:
                    parts = log.split("'")
                    if len(parts) >= 4:
                        injection_attribute = parts[1]
                        injection_target = parts[3]
                        injection_detected = True
                        break
                except:
                    pass
        
        if intervention_active:
            if injection_detected and injection_attribute and injection_target:
                narrative = f"""🔍 **Detective's Log**: Elementary, my dear Watson! The diffusion process revealed a fascinating pattern. 
            
The model's attention oscillated between {concepts_str}, each step refining the latent representation with mathematical precision. 
            
Most intriguingly, I applied an EMBEDDING INJECTION at step ~30: '{injection_attribute}' semantics were directly injected into '{injection_target}' token embeddings (strength: {intervention_strength:.2f}). This is TRUE semantic steering—modifying the model's understanding at the embedding level, not just adding noise to latents.

The result? The '{injection_target}' concept now carries '{injection_attribute}' attributes. The game is afoot! 🕵️"""
            else:
                narrative = f"""🔍 **Detective's Log**: Elementary, my dear Watson! The diffusion process revealed a fascinating pattern. 
            
The model's attention oscillated between {concepts_str}, each step refining the latent representation with mathematical precision. 
            
Most intriguingly, I applied a latent steering intervention (strength: {intervention_strength:.2f}), subtly manipulating the model's trajectory through the noise space. The result? A controlled deviation from the natural path—proof that we can indeed steer these generative beasts!

The game is afoot! 🕵️"""
        else:
            narrative = f"""🔍 **Detective's Log**: A straightforward case, Watson. The model proceeded naturally, focusing its attention on {concepts_str} throughout the denoising process. 

No interventions were applied—this is the baseline, the control experiment. Pure, unadulterated diffusion at work.

The natural order prevails. 🎩"""
        
        return narrative
    
    def generate_step_by_step_reasoning(
        self,
        prompt: str,
        data_packet: Dict[str, Dict],
        intervention_info: Optional[Dict] = None
    ) -> List[str]:
        """
        Generate LLM-powered step-by-step reasoning that reacts to actual data changes.
        This replaces hardcoded templates with intelligent, context-aware analysis.
        
        Args:
            prompt: The original generation prompt
            data_packet: High-fidelity attention data per step
                Format: {
                    "step_40": {"tiger": 0.55, "mountain": 0.12, "action": "inject_cat"},
                    "step_35": {"tiger": 0.45, "mountain": 0.05, "action": "none"},
                    ...
                }
            intervention_info: Optional intervention metadata
                Format: {
                    "target": "tiger",
                    "attribute": "blue",
                    "step_start": 40,
                    "step_end": 20,
                    "strength": 1.5
                }
        
        Returns:
            List of step-by-step reasoning strings with emoji prefixes
        """
        
        if not self.client:
            return self._generate_fallback_step_reasoning(data_packet, intervention_info)
        
        try:
            # Prepare data summary for LLM
            data_summary = self._format_data_packet(data_packet)
            
            # Prepare intervention context
            intervention_context = ""
            if intervention_info and intervention_info.get('active', True):
                intervention_context = f"""
🔬 INTERVENTION APPLIED:
- Target Concept: '{intervention_info.get('target', 'unknown')}'
- Injected Attribute: '{intervention_info.get('attribute', 'unknown')}'
- Active Steps: {intervention_info.get('step_start', '?')} → {intervention_info.get('step_end', '?')}
- Strength: {intervention_info.get('strength', 1.0):.2f}

CRITICAL: You must explain how this injection CHANGED the attention patterns. Look for:
- Spikes in the injected attribute's attention
- Drops in competing concepts
- Collateral damage to unrelated concepts
"""
            
            # Adjust system prompt based on whether intervention was applied
            if intervention_info and intervention_info.get('active', True):
                mission_context = """You are a FORENSIC AI ANALYST investigating how a diffusion model's attention was manipulated. I will give you BASELINE vs INTERVENTION attention data.

🎯 YOUR MISSION: Do not just list data points. Explain the IMPLICATION of the data using natural language.

❌ BAD (Too Mathematical):
"Tiger 📉 55% → 20% (-35%). Dog 📈 0% → 40%."

✅ GOOD (Forensic Narrative):
"I observed a massive collapse in the 'Tiger' structure, dropping from 55% to 20% as the 'Dog' injection (now 40%) forced a violent mutation of the subject."

❌ BAD:
"Mountain 15% (stable)."

✅ GOOD:
"Remarkably, the background 'Mountain' remained stable at 15%, proving the intervention was surgically precise—only the subject mutated."
"""
            else:
                mission_context = """You are an AI ANALYST observing how a diffusion model naturally generates an image. I will give you attention data showing the model's natural creative process.

🎯 YOUR MISSION: Do not just list data points. Explain what the model is DOING and WHY using natural language.

❌ BAD (Too Mathematical):
"Tiger 55%, Mountain 20%."

✅ GOOD (Natural Process Narrative):
"I am establishing the 'Tiger' as the primary subject, allocating 55% of my attention to its features while grounding the scene with a 'Mountain' backdrop at 20%."

❌ BAD:
"Mountain decreased to 5%."

✅ GOOD:
"As I refined the 'Tiger' details, the 'Mountain' naturally faded to 5%, allowing the majestic subject to dominate the composition."
"""
            
            # Different examples for intervention vs natural generation
            if intervention_info and intervention_info.get('active', True):
                examples = """
{
  "logs": [
    {
      "range": "Steps 50-41",
      "type": "normal",
      "message": "I am drafting the scene. Both runs are identical so far; the 'Mountain' is stable at 15%, anchoring the background while the 'Tiger' begins to form.",
      "stats": {"tiger": 0.8, "mountain": 0.3}
    },
    {
      "range": "Step 40",
      "type": "injection_start",
      "message": "The intervention just hit. I detected a violent shift: 'Tiger' coherence collapsed by 63% (from 55% to 20%) as the 'Dog' vector (now 40%) overwrote the subject's identity.",
      "stats": {"tiger": 20, "dog": 40, "mountain": 15}
    },
    {
      "range": "Steps 39-21",
      "type": "conflict",
      "message": "The model is fighting itself. While the 'Dog' concept dominates (+38% growth), the original 'Tiger' structure continues to decay. Notice how the 'Mountain' is starting to fade (-66%) because the new subject requires a different scale.",
      "stats": {"tiger": 16, "dog": 38, "mountain": 5}
    },
    {
      "range": "Steps 20-10",
      "type": "collateral_damage",
      "message": "Collateral damage detected: The background 'Mountain' has nearly vanished (dropped to 5%) to accommodate the spatial requirements of the 'Dog'. The entire composition is restructuring.",
      "stats": {"tiger": 18, "dog": 35, "mountain": 5}
    },
    {
      "range": "Steps 9-0",
      "type": "normal",
      "message": "Final resolution phase. The 'Dog' features are locked in at 32%, and the mutation is complete. The baseline's 'Tiger' has been successfully replaced.",
      "stats": {"dog": 32}
    }
  ]
}"""
            else:
                examples = """
{
  "logs": [
    {
      "range": "Steps 50-41",
      "type": "normal",
      "message": "I am beginning to sketch the scene. The 'Tiger' is emerging as my primary focus at 0.8%, while I establish the 'Mountain' backdrop at 0.3% to create depth.",
      "stats": {"tiger": 0.8, "mountain": 0.3}
    },
    {
      "range": "Steps 40-30",
      "type": "normal",
      "message": "The composition is taking shape. I'm dedicating substantial attention to the 'Tiger' (now 55%), carefully positioning it against the mountain setting to emphasize its majestic presence.",
      "stats": {"tiger": 55, "mountain": 15}
    },
    {
      "range": "Steps 29-15",
      "type": "normal",
      "message": "I'm refining the details. The 'Tiger' remains my focal point at 45% while the 'Mountain' fades naturally to 10% as I prioritize the subject's intricate features—fur texture, eyes, and muscular form.",
      "stats": {"tiger": 45, "mountain": 10}
    },
    {
      "range": "Steps 14-0",
      "type": "normal",
      "message": "Final polish phase. I'm locking in the 'Tiger' at 35% attention, ensuring every detail conveys majesty. The sunset lighting and mountain serve as subtle complements to the powerful subject.",
      "stats": {"tiger": 35, "mountain": 5}
    }
  ]
}"""
            
            # Build writing rules based on intervention mode
            if intervention_info and intervention_info.get('active', True):
                rule_4 = "Compare baseline vs intervention EXPLICITLY in words"
                rule_6 = "Identify invariants as PROOF of precision: 'The Mountain stayed stable, proving...'"
                rule_7 = 'Use "type": "normal", "injection_start", "conflict", "collateral_damage"'
                tone = "Forensic detective explaining findings to a jury—point at the charts and tell the story."
            else:
                rule_4 = "Describe the NATURAL creative process—planning, composition, refinement"
                rule_6 = "Explain attention shifts as natural prioritization: 'The background fades as I focus on...'"
                rule_7 = 'Use "type": "normal"'
                tone = "Artist explaining creative process—describe what you're building and why."
            
            system_prompt = mission_context + f"""

OUTPUT FORMAT: JSON object with "logs" array:
{examples}

WRITING RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Maximum 5-7 entries (not 50!)
2. WEAVE numbers into sentences—don't list them
3. Use narrative language: "I observed", "I detected", "I am establishing", "I'm refining"
4. {rule_4}
5. Explain WHY changes matter (e.g., "because the subject requires more detail")
6. {rule_6}
7. {rule_7}
8. Include "stats" dict with attention values
9. 2-3 sentences per message, but make them MEANINGFUL
10. Use emojis sparingly in type labels only (💉 ⚠️ ⚡ 🎨)

TONE: {tone}"""

            user_prompt = f"""NEURAL STATE DUMP:

PROMPT: "{prompt}"

ATTENTION DATA BY TIMESTEP:
{data_summary}
{intervention_context}

Generate your internal monologue for each phase. Explain what you're thinking as you process this image."""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=800,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON response with new grouped structure
            try:
                parsed = json.loads(content)
                
                # Extract the logs array
                if isinstance(parsed, dict) and 'logs' in parsed:
                    logs = parsed['logs']
                elif isinstance(parsed, dict) and 'steps' in parsed:
                    # Fallback to old format
                    logs = [{"range": f"Step {i}", "type": "normal", "message": msg, "stats": {}} 
                            for i, msg in enumerate(parsed['steps'])]
                elif isinstance(parsed, list):
                    logs = parsed
                else:
                    logs = []
                
                # Validate structure
                validated_logs = []
                for log in logs:
                    if isinstance(log, dict) and 'message' in log:
                        # Ensure all required fields exist
                        validated_logs.append({
                            'range': log.get('range', 'Unknown'),
                            'type': log.get('type', 'normal'),
                            'message': log.get('message', '').strip(),
                            'stats': log.get('stats', {})
                        })
                    elif isinstance(log, str):
                        # Convert old string format to new structure
                        validated_logs.append({
                            'range': 'Unknown',
                            'type': 'normal',
                            'message': log.strip(),
                            'stats': {}
                        })
                
                return validated_logs if validated_logs else self._generate_fallback_step_reasoning(data_packet, intervention_info)
                
            except Exception as parse_error:
                print(f"JSON parsing error: {parse_error}")
                print(f"Raw content: {content[:200]}...")
                return self._generate_fallback_step_reasoning(data_packet, intervention_info)
        
        except Exception as e:
            print(f"Error generating step-by-step reasoning with OpenAI: {e}")
            return self._generate_fallback_step_reasoning(data_packet, intervention_info)
    
    def _format_data_packet(self, data_packet: Dict[str, Dict]) -> str:
        """Format data packet into readable text for LLM."""
        lines = []
        
        # Sort by step number (descending, since diffusion goes 50→0)
        sorted_steps = sorted(data_packet.keys(), key=lambda x: int(x.split('_')[1]), reverse=True)
        
        # Limit to key steps to avoid overwhelming LLM (sample every ~5 steps, plus injection points)
        injection_steps = set()
        for step_key in sorted_steps:
            if data_packet[step_key].get('action') != 'none':
                injection_steps.add(step_key)
        
        # Get every 5th step plus injection steps
        sampled_steps = []
        for i, step_key in enumerate(sorted_steps):
            step_num = int(step_key.split('_')[1])
            if step_key in injection_steps or i % 5 == 0 or step_num <= 10:
                sampled_steps.append(step_key)
        
        # Limit to max 10 steps
        sampled_steps = sampled_steps[:10]
        
        for step_key in sampled_steps:
            step_num = step_key.split('_')[1]
            data = data_packet[step_key]
            
            # Format attention scores
            concepts = []
            action = data.get('action', 'none')
            phase = data.get('phase', 'unknown')
            
            for key, value in data.items():
                if key not in ['action', 'phase'] and isinstance(value, (int, float)):
                    concepts.append(f"{key}: {value:.1%}")
            
            action_marker = " 💉 [INJECT]" if action != "none" else ""
            lines.append(f"Step {step_num} ({phase}){action_marker}: {', '.join(concepts)}")
        
        return '\n'.join(lines)
    
    def _generate_fallback_step_reasoning(
        self,
        data_packet: Dict[str, Dict],
        intervention_info: Optional[Dict] = None
    ) -> List[Dict]:
        """Generate rule-based step reasoning when LLM is unavailable. Returns grouped structure."""
        logs = []
        
        # Sort by step number (descending)
        sorted_steps = sorted(data_packet.keys(), key=lambda x: int(x.split('_')[1]), reverse=True)
        
        # Group steps into phases
        if len(sorted_steps) > 10:
            # Early phase
            early_steps = sorted_steps[:len(sorted_steps)//3]
            if early_steps:
                first_step = int(early_steps[0].split('_')[1])
                last_step = int(early_steps[-1].split('_')[1])
                stats = {}
                for step_key in early_steps[:3]:
                    data = data_packet[step_key]
                    for k, v in data.items():
                        if k not in ['action', 'phase'] and isinstance(v, (int, float)):
                            stats[k] = stats.get(k, 0) + v
                for k in stats:
                    stats[k] = stats[k] / min(3, len(early_steps)) * 100
                
                logs.append({
                    'range': f"Steps {first_step}-{last_step}",
                    'type': 'normal',
                    'message': f"🎨 Sketching initial composition. Establishing scene layout.",
                    'stats': stats
                })
            
            # Check for injection
            injection_step = None
            for step_key in sorted_steps:
                if data_packet[step_key].get('action') != 'none':
                    injection_step = step_key
                    break
            
            if injection_step:
                step_num = int(injection_step.split('_')[1])
                data = data_packet[injection_step]
                concepts = {k: v*100 for k, v in data.items() if k not in ['action', 'phase'] and isinstance(v, (int, float))}
                
                logs.append({
                    'range': f"Step {step_num}",
                    'type': 'injection_start',
                    'message': f"💉 INJECTION EVENT: Forcing '{intervention_info.get('attribute', 'new') if intervention_info else 'new'}' attributes onto '{intervention_info.get('target', 'target') if intervention_info else 'target'}'.",
                    'stats': concepts
                })
            
            # Final phase
            final_steps = sorted_steps[-len(sorted_steps)//3:]
            if final_steps:
                first_step = int(final_steps[0].split('_')[1])
                last_step = int(final_steps[-1].split('_')[1])
                stats = {}
                for step_key in final_steps[:3]:
                    data = data_packet[step_key]
                    for k, v in data.items():
                        if k not in ['action', 'phase'] and isinstance(v, (int, float)):
                            stats[k] = stats.get(k, 0) + v
                for k in stats:
                    stats[k] = stats[k] / min(3, len(final_steps)) * 100
                
                logs.append({
                    'range': f"Steps {first_step}-{last_step}",
                    'type': 'normal',
                    'message': "⚡ Polishing final details and locking in composition.",
                    'stats': stats
                })
        
        return logs if logs else [{'range': 'All Steps', 'type': 'normal', 'message': '🎨 Processing generation', 'stats': {}}]
    
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
