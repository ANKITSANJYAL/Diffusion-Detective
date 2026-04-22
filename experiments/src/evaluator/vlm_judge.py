"""
VLM Evaluator that uses LLaVA-1.5 to computationally rate the consistency of the extracted
attention logs and spatial heatmaps natively on the A100 GPUs.
"""
import torch
import json
import re
from PIL import Image

try:
    from transformers import LlavaForConditionalGeneration, AutoProcessor
except ImportError:
    pass

class VLMJudge:
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf", device="cuda"):
        self.device = device
        self.model_id = model_id
        self.processor = None
        self.model = None

    def load_model(self):
        """
        Loads the 7B parameter model into memory.
        Must be done strictly sequentially after the Stable Diffusion UNet to prevent OOM.
        """
        if self.model is None:
            print(f"Loading {self.model_id} into {self.device} (float16) for evaluation...")
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            ).to(self.device)

    def evaluate_consistency(self, attention_logs: dict, heatmap_img_path: str, steered_img_path: str):
        """
        Sends the heatmap and steered image to LLaVA alongside the mathematical attention logs
        to strictly score mechanistic consistency.
        """
        self.load_model()

        try:
            image_heatmap = Image.open(heatmap_img_path).convert("RGB")
            image_steered = Image.open(steered_img_path).convert("RGB")

            # Summarise attention logs for LLaVA's context window.
            # high_fidelity_data stores per-step dicts with token names as keys
            # and raw attention scores as float values + 'phase' / 'action' metadata.
            # We extract per-step statistics (max/mean over concept scores).
            compact_stats = {}
            for step, info in attention_logs.items():
                numeric_vals = [v for k, v in info.items()
                                if isinstance(v, (int, float)) and k not in ('phase', 'action')]
                if numeric_vals:
                    compact_stats[step] = {
                        "max": round(max(numeric_vals), 5),
                        "mean": round(sum(numeric_vals) / len(numeric_vals), 5),
                        "phase": info.get("phase", ""),
                        "action": info.get("action", "none"),
                    }
            stats_summary = json.dumps(compact_stats)

            # LLaVA prompt structure
            prompt = (
                "USER: <image>\n<image>\n"
                "You are an expert XAI researcher. The first image is an attention heatmap showing where exactly "
                "the model focused its mathematical logic. The second image is the final generated output. "
                "Here are the mathematical spikes for the associated concept: " + stats_summary + "\n\n"
                "Rate the mechanistic consistency (1-10). Does the concept mathematically spike exactly where the "
                "target object visually appears in the final image? "
                "You must strictly end your response with: 'Score: X/10'.\nASSISTANT:"
            )

            inputs = self.processor(
                text=prompt,
                images=[image_heatmap, image_steered],
                return_tensors="pt"
            ).to(self.device, torch.float16)

            # Generate evaluation
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=250,
                    temperature=0.2,
                    do_sample=True
                )

            # Decode the generated text (everything after ASSISTANT:)
            full_response = self.processor.decode(output_ids[0], skip_special_tokens=True)
            response_body = full_response.split("ASSISTANT:")[-1].strip()

            # Regex search for 'Score: X/10'
            score_match = re.search(r"Score:\s*(\d+)/10", response_body, re.IGNORECASE)
            final_score = int(score_match.group(1)) if score_match else 0

            return {
                "consistency_score": final_score,
                "reasoning": response_body
            }

        except Exception as e:
            print(f"LLaVA Scoring Failed: {e}")
            return {"consistency_score": 0, "reasoning": "VLM Exception raised."}

