"""
Dataset loader for CVPR benchmarks (COCO-Captions 2017, PartiPrompts, EditBench).
"""
import json
import sys
import os
import random
from torch.utils.data import Dataset
try:
    from datasets import load_dataset
except ImportError:
    pass  # Allow pipeline to initialize without HF datasets initially

# Add backend to path so we can reuse concept_extractor
_backend_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'backend')
if os.path.isdir(_backend_dir):
    sys.path.insert(0, os.path.abspath(_backend_dir))

from app.concept_extractor import ConceptExtractor

# Semantically diverse injection attributes, grouped by modality.
# During dataset construction we cycle through these so that experiments
# exercise a wide range of attribute types, not just colour.
INJECTION_ATTRIBUTES = [
    # Colour
    "red", "blue", "golden", "silver", "neon green",
    # Texture / material
    "wooden", "metallic", "glass", "stone", "velvet",
    # Style
    "futuristic", "ancient", "cyberpunk", "baroque", "minimalist",
    # Atmosphere
    "glowing", "frozen", "fiery", "misty", "shadowy",
]


class BenchmarkDataset(Dataset):
    """
    Unified Dataset for loading prompts from evaluation benchmarks.
    Supported suites: PartiPrompts, EditBench
    """
    def __init__(self, dataset_name: str, categories: list = None, max_samples: int = None,
                 injection_attributes: list = None, seed: int = 42):
        super().__init__()
        self.dataset_name = dataset_name.lower()
        self.categories = categories or ["Simple", "Compositional", "Conflicting"]
        self.max_samples = max_samples
        self.samples = []
        self._concept_extractor = ConceptExtractor()
        self._injection_attributes = injection_attributes or INJECTION_ATTRIBUTES
        self._rng = random.Random(seed)

        if self.dataset_name == "partiprompts":
            self._load_parti_prompts()
        elif self.dataset_name == "coco2017":
            self._load_coco_2017()
        elif self.dataset_name == "editbench":
            self._load_edit_bench()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                             f"Supported: partiprompts, coco2017, editbench")

        # Limit samples if requested (for smoke tests)
        if self.max_samples and len(self.samples) > self.max_samples:
            self.samples = self.samples[:self.max_samples]

    # ------------------------------------------------------------------
    # Target concept extraction
    # ------------------------------------------------------------------
    def _extract_target(self, prompt: str) -> str:
        """
        Extract the primary target concept from a prompt using POS-tag-aware
        ConceptExtractor.  Returns the first noun found, or the last word of
        the prompt as a last-resort fallback.
        """
        concepts = self._concept_extractor.extract_concepts(prompt, max_concepts=4)
        # Return the first noun
        for word, category in concepts:
            if category == 'noun':
                return word
        # Fallback: return first extracted concept regardless of category
        if concepts:
            return concepts[0][0]
        # Absolute fallback: last word, stripped of punctuation
        words = prompt.strip().rstrip('.!?,;:').split()
        return words[-1] if words else prompt

    # ------------------------------------------------------------------
    # Injection attribute selection
    # ------------------------------------------------------------------
    def _pick_injection_attribute(self) -> str:
        """
        Cycle through the diverse attribute pool with shuffled ordering to
        ensure every experiment run covers multiple attribute modalities.
        """
        return self._rng.choice(self._injection_attributes)

    # ------------------------------------------------------------------
    # Dataset loaders
    # ------------------------------------------------------------------
    def _load_parti_prompts(self):
        """
        Loads PartiPrompts from HuggingFace.
        Categorizes based on the defined heuristic or metadata.
        """
        dataset = load_dataset("nateraw/parti-prompts", split="train")

        for item in dataset:
            prompt = item['Prompt']
            category = item.get('Category', 'Simple')

            # Map PartiPrompts categories to our simplified taxonomy
            if category in ['abstract', 'world knowledge']:
                mapped_category = 'Conflicting'
            elif category in ['interactions', 'properties']:
                mapped_category = 'Compositional'
            else:
                mapped_category = 'Simple'

            if mapped_category in self.categories:
                self.samples.append({
                    "prompt": prompt,
                    "target_concept": self._extract_target(prompt),
                    "injection_attribute": self._pick_injection_attribute(),
                    "category": mapped_category,
                })

    # ------------------------------------------------------------------
    # COCO-Captions 2017
    # ------------------------------------------------------------------
    def _load_coco_2017(self):
        """
        Loads COCO-Captions 2017 validation split (5,000 images, ~25k captions).

        We pick ONE caption per image (the first) to get exactly 5,000 unique
        prompts.  Each is categorised with a heuristic based on word count and
        conjunction/preposition density:
          - Simple:        ≤8 words, no conjunctions
          - Compositional: contains 'and', 'with', 'near', 'on', 'in front of', etc.
          - Conflicting:   everything else (unusual / abstract descriptions)

        Source: HuggingFace  `HuggingFaceM4/COCO`  (captions_val2017)
        """
        # Primary source: yerevann/coco-karpathy (Parquet, no loading script)
        # HuggingFaceM4/COCO is deprecated — its loading script is blocked
        # by datasets ≥ 4.x which removed trust_remote_code support.
        try:
            dataset = load_dataset(
                "yerevann/coco-karpathy",
                split="validation",
            )
        except Exception as e:
            raise RuntimeError(
                "Could not load COCO-Captions from 'yerevann/coco-karpathy'. "
                "Install `datasets` (pip install datasets) and ensure internet "
                "access for HuggingFace downloads."
            ) from e

        # Compositional markers
        COMP_MARKERS = {
            "and", "with", "near", "next to", "in front of", "behind",
            "on top of", "beside", "between", "around", "holding",
            "riding", "sitting on", "standing on", "wearing",
        }

        seen_ids = set()
        for item in dataset:
            # De-duplicate by image id (pick first caption per image)
            img_id = item.get("image_id") or item.get("imgid") or id(item)
            if img_id in seen_ids:
                continue
            seen_ids.add(img_id)

            # Extract caption text — handle multiple dataset schemas
            # yerevann/coco-karpathy uses "sentences" (list of str)
            if "sentences" in item and item["sentences"]:
                caption = item["sentences"][0]
            elif "sentences_raw" in item and item["sentences_raw"]:
                caption = item["sentences_raw"][0]
            elif "caption" in item:
                caption = item["caption"] if isinstance(item["caption"], str) else item["caption"][0]
            elif "text" in item:
                caption = item["text"] if isinstance(item["text"], str) else item["text"][0]
            else:
                continue

            caption = caption.strip()
            if not caption or len(caption) < 5:
                continue

            # Categorise
            words = caption.lower().split()
            word_count = len(words)
            caption_lower = caption.lower()

            has_comp = any(m in caption_lower for m in COMP_MARKERS)

            if word_count <= 8 and not has_comp:
                category = "Simple"
            elif has_comp:
                category = "Compositional"
            else:
                category = "Conflicting"

            if category not in self.categories:
                continue

            self.samples.append({
                "prompt": caption,
                "target_concept": self._extract_target(caption),
                "injection_attribute": self._pick_injection_attribute(),
                "category": category,
            })

        # Shuffle so that max_samples truncation gets a diverse mix
        self._rng.shuffle(self.samples)

    def _load_edit_bench(self):
        """
        Stub for loading EditBench.
        """
        # EditBench requires a specific formatted JSON or huggingface dataset.
        pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
