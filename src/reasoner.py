"""Reasoner module: provides an interface to generate chain-of-thought style reasoning steps.

This implementation provides a small local fallback (rule-based) and a pluggable interface
to use a large LLM (via Hugging Face transformers) if available. The class is intentionally
minimal and returns a list of short strings (reasoning steps).
"""
from typing import List, Optional
import os

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    torch = None


class Reasoner:
    def __init__(self, model_name: Optional[str] = None, device: str = "cpu", load_in_4bit: bool = True):
        self.model_name = model_name
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.tokenizer = None
        self.model = None
        if model_name and AutoTokenizer is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            except Exception:
                self.tokenizer = None

    def _load_model(self):
        if not self.model_name or AutoModelForCausalLM is None:
            return
        if self.model is None:
            # attempt to load in 4-bit if requested (requires bitsandbytes and compatible HF setup)
            load_kwargs = {"trust_remote_code": True}
            if self.load_in_4bit:
                load_kwargs.update({"load_in_4bit": True, "device_map": "auto"})
            try:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
                # move to device if not using device_map
                if not hasattr(self.model, 'is_loaded_in_4bit'):
                    try:
                        self.model.to(self.device)
                    except Exception:
                        pass
            except Exception:
                # fallback: try loading without 4-bit
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
                try:
                    self.model.to(self.device)
                except Exception:
                    pass

    def generate_steps(self, prompt: str, n_steps: int = 4, max_tokens: int = 128) -> List[str]:
        """Return a list of short reasoning steps for the prompt.

        If an LLM model_name was provided and available, use it; otherwise fall back to a heuristic.
        """
        # Fallback heuristic: split on commas, then on spaces to create compact steps
        parts = [p.strip() for p in prompt.split(",") if p.strip()]
        if len(parts) >= n_steps:
            return parts[:n_steps]

        if not self.model_name or AutoModelForCausalLM is None or self.tokenizer is None:
            # simple chunking fallback
            words = prompt.split()
            if len(parts) == 0:
                chunk_size = max(1, len(words) // n_steps)
                steps = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
                return steps[:n_steps]
            while len(parts) < n_steps:
                parts.append(parts[-1])
            return parts[:n_steps]

        # use LLM
        self._load_model()
        if self.model is None:
            # fallback if loading failed
            while len(parts) < n_steps:
                parts.append(parts[-1] if parts else prompt)
            return parts[:n_steps]

        inputs = self.tokenizer(prompt + f"\nDecompose the prompt into {n_steps} short steps:", return_tensors="pt", truncation=True)
        input_ids = inputs.input_ids.to(self.device)
        try:
            out = self.model.generate(input_ids, max_new_tokens=max_tokens)
            text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        except Exception:
            text = ""

        # naive parse: split into lines, commas, or semicolons
        lines = [l.strip() for l in text.replace(";", "\n").split("\n") if l.strip()]
        steps = []
        for l in lines:
            for part in l.split(","):
                p = part.strip()
                if p:
                    steps.append(p)
                if len(steps) >= n_steps:
                    break
            if len(steps) >= n_steps:
                break
        if len(steps) < n_steps:
            while len(steps) < n_steps:
                steps.append(steps[-1] if steps else prompt)
        return steps[:n_steps]

    def compute_weights(self, steps: List[str]) -> List[float]:
        """Compute a simple weight per reasoning step based on token count normalized to mean 1.0.

        Returns a list of positive floats where mean is 1.0.
        """
        if not steps:
            return []
        lens = [len(s.split()) for s in steps]
        mean = sum(lens) / len(lens) if sum(lens) > 0 else 1.0
        weights = [max(0.1, l / mean) for l in lens]
        # normalize so mean is 1.0
        m = sum(weights) / len(weights)
        return [w / m for w in weights]
