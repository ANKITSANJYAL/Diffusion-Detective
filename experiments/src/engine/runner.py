"""
Unified experiment engine for CVPR benchmarks.

Key design:  For each prompt, run the baseline pass ONCE, then iterate over
all treatment conditions (strength × window × attribute).  This is the
k-fold-style layout the paper needs — every prompt sees every condition.

Usage:
    torchrun --nproc_per_node=2 -m experiments.run_experiment --config-name=ablation
"""
import torch
import wandb
import os
import itertools
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf, ListConfig
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor
import json
from datetime import datetime

# Import the core pipeline from the main web application
import sys
import pathlib
backend_app_path = str(pathlib.Path(__file__).parent.parent.parent.parent / "backend")
if backend_app_path not in sys.path:
    sys.path.append(backend_app_path)

from app.pipeline import InterpretableSDPipeline
from experiments.src.metrics import MetricEvaluator, AttentionFaithfulness


# ─────────────────────────────────────────────────────────────────
# Treatment grid builder
# ─────────────────────────────────────────────────────────────────
def _build_treatment_grid(cfg: DictConfig):
    """
    Build the Cartesian product of all active sweep axes.

    Returns a list of dicts, each describing one treatment:
        {strength: float, step_start: int, step_end: int,
         attribute: str | None, condition_tag: str}

    When an axis is disabled, we use the fixed default from cfg.intervention.
    """
    sweep = cfg.get("sweep", {})

    # --- strengths ---
    if sweep.get("sweep_strength", False):
        strengths = list(sweep.get("strengths", [1.0]))
    else:
        strengths = [cfg.intervention.strength]

    # --- windows ---
    if sweep.get("sweep_window", False):
        raw_windows = sweep.get("windows", [[cfg.intervention.step_start, cfg.intervention.step_end]])
        windows = [(int(w[0]), int(w[1])) for w in raw_windows]
    else:
        windows = [(int(cfg.intervention.step_start), int(cfg.intervention.step_end))]

    # --- attributes ---
    if sweep.get("sweep_attribute", False):
        attribute_groups = sweep.get("attribute_groups", {})
        extra_attrs = []
        for group_name, attrs in attribute_groups.items():
            extra_attrs.extend(list(attrs))
        # None means "use the dataset-assigned attribute"
        attributes = [None] + extra_attrs
    else:
        attributes = [None]  # dataset default only

    # Build cross-product
    grid = []
    for strength, (step_start, step_end), attr in itertools.product(strengths, windows, attributes):
        tag_parts = [f"s{strength:.2f}", f"w{step_start}-{step_end}"]
        if attr is not None:
            tag_parts.append(f"a={attr}")
        else:
            tag_parts.append("a=dataset")
        condition_tag = "_".join(tag_parts)

        grid.append({
            "strength": float(strength),
            "step_start": int(step_start),
            "step_end": int(step_end),
            "attribute_override": attr,   # None = use dataset-assigned attribute
            "condition_tag": condition_tag,
        })

    return grid


class ExperimentRunner:
    def __init__(self, cfg: DictConfig, local_rank: int = 0, world_size: int = 1):
        self.cfg = cfg
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main_process = (self.local_rank == 0)

        # Setup device for this process
        self.device = f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"

        # Initialize wandb only on main process
        if self.is_main_process and self.cfg.wandb.mode != "disabled":
            wandb.init(
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.get("entity"),
                mode=self.cfg.wandb.mode,
                config=dict(self.cfg),
                name=self.cfg.experiment_name
            )

        # Load Pipeline onto specific GPU
        if self.is_main_process:
            print(f"Loading Pipeline on {self.device}...")

        torch_dtype = torch.float16 if self.cfg.pipeline.torch_dtype == "float16" else torch.float32

        self.pipeline = InterpretableSDPipeline(
            model_id=self.cfg.pipeline.model_id,
            device=self.device,
            torch_dtype=torch_dtype
        )

        # Initialize Metric Evaluator if needed
        self.metric_evaluator = None
        if self.cfg.metrics.calculate_delta_clip or self.cfg.metrics.calculate_lpips:
            if self.is_main_process:
                print(f"Loading Metric Evaluator (CLIP/LPIPS) on {self.device}...")
            self.metric_evaluator = MetricEvaluator(device=self.device)

        # Setup local results directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_id = f"{self.cfg.experiment_name}_{timestamp}"
        self.out_dir = pathlib.Path(__file__).parent.parent.parent / "results" / self.experiment_id

        self.metrics_file_path = self.out_dir / "aggregated_metrics.jsonl"

        if self.is_main_process:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            # Save the explicit execution configuration for backward-tracking
            with open(self.out_dir / "run_config.yaml", "w") as f:
                f.write(OmegaConf.to_yaml(self.cfg))

        # ── Build the treatment grid ─────────────────────────────────
        self.treatment_grid = _build_treatment_grid(cfg)
        if self.is_main_process:
            print(f"\n{'='*60}")
            print(f"UNIFIED ABLATION: {len(self.treatment_grid)} treatments per prompt")
            for i, t in enumerate(self.treatment_grid):
                print(f"  [{i:3d}] {t['condition_tag']}")
            print(f"{'='*60}\n")

    # ==================================================================
    # Main run loop — baseline once, N interventions
    # ==================================================================
    def run(self, dataloader: DataLoader):
        """
        Unified ablation:  for each prompt, compute baseline once,
        then loop over every treatment in the grid.
        """
        total_prompts = len(dataloader.dataset)
        n_treatments = len(self.treatment_grid)
        if self.is_main_process:
            print(f"Starting unified ablation: {total_prompts} prompts × "
                  f"{n_treatments} treatments = {total_prompts * n_treatments} "
                  f"intervention passes  (+{total_prompts} baselines)")

        iterator = dataloader
        if self.is_main_process:
            iterator = tqdm(dataloader, desc="Prompts")

        global_sample_idx = 0

        for batch in iterator:
            prompts = batch["prompt"]
            targets = batch["target_concept"]
            injections = batch["injection_attribute"]
            categories = batch["category"]

            for i in range(len(prompts)):
                prompt = prompts[i]
                target = targets[i]
                dataset_injection = injections[i]
                category = categories[i]

                try:
                    # ── STEP 1: Baseline (one per prompt) ────────────
                    baseline_cache = self.pipeline.generate_baseline(
                        prompt=prompt,
                        num_inference_steps=self.cfg.generation.num_inference_steps,
                        guidance_scale=self.cfg.generation.guidance_scale,
                        seed=self.cfg.generation.seed,
                        auto_detect_concepts=self.cfg.generation.auto_detect_concepts,
                    )
                    baseline_img = baseline_cache["baseline_image"]

                    # ── STEP 2: Loop treatments ──────────────────────
                    for treatment in self.treatment_grid:
                        injection = treatment["attribute_override"] or dataset_injection

                        try:
                            steered_img, logs, meta = self.pipeline.generate_intervention(
                                baseline_cache=baseline_cache,
                                target_concept=target,
                                injection_attribute=injection,
                                intervention_strength=treatment["strength"],
                                intervention_step_start=treatment["step_start"],
                                intervention_step_end=treatment["step_end"],
                            )

                            # ── Perceptual Metrics ───────────────────
                            delta_clip = base_clip = steered_clip = lpips_score = None

                            if self.metric_evaluator is not None:
                                if self.cfg.metrics.calculate_delta_clip:
                                    delta_clip, base_clip, steered_clip = (
                                        self.metric_evaluator.calculate_delta_clip(
                                            baseline_img, steered_img, prompt, target, injection
                                        )
                                    )
                                if self.cfg.metrics.calculate_lpips:
                                    t_base = to_tensor(baseline_img).unsqueeze(0).to(self.device).float() * 2.0 - 1.0
                                    t_steered = to_tensor(steered_img).unsqueeze(0).to(self.device).float() * 2.0 - 1.0
                                    lpips_score = self.metric_evaluator.calculate_structure_preservation(
                                        t_base, t_steered
                                    )

                            # ── Attention Diagnostic Metrics ──────────
                            # These are the CORE metrics for the paper's thesis:
                            # "When/where does cross-attention predict visual change?"
                            attn_store = self.pipeline.attention_store

                            # Debug: print baseline_data keys on first treatment of first prompt
                            if global_sample_idx == 0 and treatment is self.treatment_grid[0] and self.is_main_process:
                                bl_keys = sorted(attn_store.baseline_data.keys())
                                iv_keys = sorted(attn_store.high_fidelity_data.keys())
                                print(f"\n  [DEBUG] baseline_data keys ({len(bl_keys)}): {bl_keys[:10]}")
                                print(f"  [DEBUG] high_fidelity_data keys ({len(iv_keys)}): {iv_keys[:10]}")
                                if bl_keys:
                                    sample_key = bl_keys[0]
                                    sample_data = attn_store.baseline_data[sample_key]
                                    print(f"  [DEBUG] baseline_data['{sample_key}'] keys: {list(sample_data.keys())}")
                                if iv_keys:
                                    sample_key = iv_keys[0]
                                    sample_data = attn_store.high_fidelity_data[sample_key]
                                    print(f"  [DEBUG] high_fidelity_data['{sample_key}'] keys: {list(sample_data.keys())}")
                                print(f"  [DEBUG] target_concept='{target}'\n")
                                print(f"  [DEBUG] baseline_latent_trajectory len: {len(attn_store.baseline_latent_trajectory)}")
                                print(f"  [DEBUG] intervention_latent_trajectory len: {len(attn_store.intervention_latent_trajectory)}")

                            # --- ACS (Attention Concentration Score) ---
                            acs_data = AttentionFaithfulness.attention_concentration_score(
                                baseline_data=attn_store.baseline_data,
                                intervention_data=attn_store.high_fidelity_data,
                                target_concept=target,
                                step_start=treatment["step_start"],
                                step_end=treatment["step_end"],
                            )

                            # --- ADC (original, kept for ablation) ---
                            adc = AttentionFaithfulness.attention_delta_correlation(
                                intervention_data=attn_store.high_fidelity_data,
                                target_concept=target,
                            )

                            # --- ★ P-ADC (Predictive, DECOUPLED) ---
                            p_adc = AttentionFaithfulness.predictive_attention_delta_correlation(
                                baseline_data=attn_store.baseline_data,
                                intervention_data=attn_store.high_fidelity_data,
                                target_concept=target,
                            )

                            # --- ★ L-ADC (Latent-Delta Correlation) ---
                            l_adc = None
                            if (attn_store.baseline_latent_trajectory and
                                    attn_store.intervention_latent_trajectory):
                                l_adc = AttentionFaithfulness.latent_delta_correlation(
                                    baseline_data=attn_store.baseline_data,
                                    intervention_data=attn_store.high_fidelity_data,
                                    target_concept=target,
                                    baseline_latent_trajectory=attn_store.baseline_latent_trajectory,
                                    intervention_latent_trajectory=attn_store.intervention_latent_trajectory,
                                )

                            # --- Spatial Faithfulness (SF-IoU + SF-IoU-HR) ---
                            sf_data = {"sf_iou": None, "sf_iou_hr": None,
                                       "sf_precision": None, "sf_recall": None}
                            try:
                                tokens = baseline_cache["tokens"]
                                t_idx = None
                                target_lower = target.lower()
                                for idx, tok in enumerate(tokens):
                                    tok_clean = tok.strip().lower()
                                    if tok_clean == target_lower or target_lower.startswith(tok_clean):
                                        t_idx = idx
                                        break

                                if t_idx is not None and attn_store.attention_maps:
                                    base_t = to_tensor(baseline_img)    # [C, H, W] in [0, 1]
                                    steer_t = to_tensor(steered_img)
                                    sf_data = AttentionFaithfulness.spatial_faithfulness(
                                        attention_maps=attn_store.attention_maps,
                                        target_token_idx=t_idx,
                                        baseline_image_tensor=base_t,
                                        steered_image_tensor=steer_t,
                                    )
                            except Exception:
                                pass  # Non-fatal — spatial faithfulness is best-effort

                            # --- ★ AKS (Attention Knockout Score) ---
                            # Causal ablation: only run if configured (expensive — 3rd forward pass)
                            aks_data = {"aks": None, "aks_raw": None, "l2_steered": None, "l2_knockout": None}
                            if self.cfg.metrics.get("calculate_knockout", False):
                                try:
                                    knockout_img = self.pipeline.generate_knockout(
                                        baseline_cache=baseline_cache,
                                        target_concept=target,
                                        injection_attribute=injection,
                                        intervention_strength=treatment["strength"],
                                        intervention_step_start=treatment["step_start"],
                                        intervention_step_end=treatment["step_end"],
                                    )
                                    base_t = to_tensor(baseline_img)
                                    steer_t = to_tensor(steered_img)
                                    ko_t = to_tensor(knockout_img)
                                    aks_data = AttentionFaithfulness.attention_knockout_score(
                                        baseline_image_tensor=base_t,
                                        steered_image_tensor=steer_t,
                                        knockout_image_tensor=ko_t,
                                    )
                                except Exception as e:
                                    if self.is_main_process:
                                        print(f"  ✗ Knockout failed: {e}")

                            # ── Result row ───────────────────────────
                            result_data = {
                                "run_id": self.experiment_id,
                                "sample_idx": global_sample_idx,
                                "prompt": prompt,
                                "category": category,
                                "target_concept": target,
                                "injection_attribute": injection,
                                "condition_tag": treatment["condition_tag"],
                                "hyperparameters": {
                                    "seed": self.cfg.generation.seed,
                                    "guidance_scale": self.cfg.generation.guidance_scale,
                                    "strength": treatment["strength"],
                                    "step_start": treatment["step_start"],
                                    "step_end": treatment["step_end"],
                                },
                                "metrics": {
                                    "delta_clip": delta_clip,
                                    "base_clip": base_clip,
                                    "steered_clip": steered_clip,
                                    "lpips": lpips_score,
                                    # Attention diagnostic metrics
                                    "attention_concentration_score": acs_data["acs"],
                                    "attention_mean_baseline": acs_data["mean_baseline"],
                                    "attention_mean_intervention": acs_data["mean_intervention"],
                                    "attention_steps_matched": acs_data["num_steps_matched"],
                                    "spatial_faithfulness_iou": sf_data["sf_iou"],
                                    "spatial_faithfulness_iou_hr": sf_data["sf_iou_hr"],
                                    "spatial_faithfulness_precision": sf_data["sf_precision"],
                                    "spatial_faithfulness_recall": sf_data["sf_recall"],
                                    "attention_delta_correlation": adc,
                                    # ★ New decoupled metrics
                                    "predictive_adc": p_adc,
                                    "latent_delta_correlation": l_adc,
                                    # ★ Causal knockout
                                    "attention_knockout_score": aks_data["aks"],
                                    "attention_knockout_score_raw": aks_data.get("aks_raw"),
                                    "knockout_l2_steered": aks_data["l2_steered"],
                                    "knockout_l2_knockout": aks_data["l2_knockout"],
                                },
                            }

                            # ── Reproduction params ──────────────────
                            if self.cfg.storage.get("save_reproduction_params", True):
                                result_data["reproduction"] = {
                                    "model_id": self.cfg.pipeline.model_id,
                                    "num_inference_steps": self.cfg.generation.num_inference_steps,
                                    "seed": self.cfg.generation.seed,
                                    "guidance_scale": self.cfg.generation.guidance_scale,
                                    "torch_dtype": self.cfg.pipeline.torch_dtype,
                                    "strength": treatment["strength"],
                                    "step_start": treatment["step_start"],
                                    "step_end": treatment["step_end"],
                                }

                            # ── Optional image/heatmap storage ───────
                            save_images = self.cfg.storage.save_images_locally
                            if save_images:
                                safe_tag = treatment["condition_tag"].replace("=", "")
                                safe_prompt = prompt[:30].replace(" ", "_").replace("/", "")
                                sample_dir = self.out_dir / f"sample_{global_sample_idx}_{safe_prompt}" / safe_tag
                                sample_dir.mkdir(parents=True, exist_ok=True)
                                baseline_img.save(sample_dir / "baseline.png")
                                steered_img.save(sample_dir / "steered.png")
                                result_data["paths"] = {
                                    "baseline": str(sample_dir / "baseline.png"),
                                    "steered": str(sample_dir / "steered.png"),
                                }

                                # Render heatmap
                                self._try_render_heatmap(sample_dir, prompt, target, result_data)

                            # ── JSONL (primary data store) ───────────
                            with open(self.metrics_file_path, "a") as f:
                                f.write(json.dumps(result_data) + "\n")

                            # ── wandb ────────────────────────────────
                            if self.is_main_process and self.cfg.wandb.mode != "disabled":
                                wb = {
                                    "prompt": prompt, "category": category,
                                    "condition": treatment["condition_tag"],
                                }
                                if delta_clip is not None:
                                    wb["Delta-CLIP"] = delta_clip
                                    wb["Base-CLIP"] = base_clip
                                    wb["Steered-CLIP"] = steered_clip
                                if lpips_score is not None:
                                    wb["LPIPS"] = lpips_score
                                if acs_data["acs"] is not None:
                                    wb["ACS"] = acs_data["acs"]
                                if sf_data["sf_iou"] is not None:
                                    wb["SF-IoU"] = sf_data["sf_iou"]
                                if sf_data.get("sf_iou_hr") is not None:
                                    wb["SF-IoU-HR"] = sf_data["sf_iou_hr"]
                                if adc is not None:
                                    wb["ADC"] = adc
                                if p_adc is not None:
                                    wb["P-ADC"] = p_adc
                                if l_adc is not None:
                                    wb["L-ADC"] = l_adc
                                if aks_data["aks"] is not None:
                                    wb["AKS"] = aks_data["aks"]
                                wandb.log(wb)

                        except Exception as e:
                            if self.is_main_process:
                                print(f"  ✗ Treatment {treatment['condition_tag']} failed "
                                      f"on '{prompt[:40]}': {e}")

                except Exception as e:
                    if self.is_main_process:
                        print(f"✗ Baseline failed on '{prompt[:40]}': {e}")
                        import traceback; traceback.print_exc()

                global_sample_idx += 1

        if self.is_main_process and self.cfg.wandb.mode != "disabled":
            wandb.finish()

        if self.is_main_process:
            print(f"\n{'='*60}")
            print(f"Done. Results: {self.metrics_file_path}")
            print(f"Total rows: {global_sample_idx * n_treatments}")
            print(f"{'='*60}")

    # ==================================================================
    # Heatmap rendering helper
    # ==================================================================
    def _try_render_heatmap(self, sample_dir, prompt, target, result_data):
        """Attempt to render an attention heatmap for the current intervention."""
        try:
            from experiments.src.evaluator.heatmap_renderer import HeatmapRenderer

            tokenizer = self.pipeline._tokenizer
            tokens = tokenizer.tokenize(prompt)

            t_idx = None
            target_lower = target.lower()
            for idx, tok in enumerate(tokens):
                tok_clean = tok.replace("</w>", "").lower()
                if tok_clean == target_lower or target_lower.startswith(tok_clean):
                    t_idx = idx + 1
                    break

            if t_idx is not None and hasattr(self.pipeline, 'attention_store'):
                attn_maps = self.pipeline.attention_store.attention_maps
                if attn_maps:
                    last_step = sorted(list(attn_maps.keys()))[0]
                    attn_list = attn_maps.get(last_step, [])

                    max_spatial = 0
                    best_attn = None
                    for attn in attn_list:
                        spatial = attn.shape[1] if attn.dim() == 3 else attn.shape[2]
                        if spatial > max_spatial:
                            max_spatial = spatial
                            best_attn = attn

                    if best_attn is not None:
                        avg_attn = best_attn.mean(dim=0) if best_attn.dim() == 3 else best_attn.mean(dim=1).mean(dim=0)
                        if t_idx < avg_attn.shape[-1]:
                            spatial_map = avg_attn[:, t_idx]
                            heatmap_path = str(sample_dir / "heatmap.png")
                            HeatmapRenderer.render_and_save(
                                spatial_map.cpu().numpy().tolist(),
                                str(sample_dir / "steered.png"),
                                heatmap_path,
                            )
                            result_data.setdefault("paths", {})["heatmap"] = heatmap_path
        except Exception as e:
            if self.is_main_process:
                print(f"  ✗ Heatmap failed: {e}")
