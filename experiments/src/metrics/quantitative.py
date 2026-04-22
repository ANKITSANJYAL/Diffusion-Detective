import torch
import torch.nn.functional as F
import numpy as np

try:
    from transformers import CLIPProcessor, CLIPModel
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
except ImportError:
    pass


# ======================================================================
# Attention Diagnostic Metrics
# ======================================================================
# These answer the paper's CORE question: "When and where do cross-
# attention maps accurately predict the visual consequences of embedding-
# space interventions — and when do they fail?"
#
# The framework is DIAGNOSTIC, not a faithfulness proof.  We present
# metrics that reveal the conditions under which attention is predictive,
# and surface the failure modes where it is not.

class AttentionFaithfulness:
    """
    Diagnostic metrics linking internal cross-attention patterns to
    observable visual changes under embedding-space interventions.

    Five complementary scores (3 original + 2 new causal/decoupled):

    1. **ACS — Attention Concentration Score**
       Relative change in target-token attention during intervention vs baseline.
       ACS = (mean_intervention − mean_baseline) / mean_baseline
       Negative → attention redistributed *away from* the target during steering.

    2. **SF-IoU — Spatial Faithfulness (IoU)**
       Overlap between the attention heatmap and actual pixel-change mask.
       Computed at BOTH native attention resolution AND upsampled image
       resolution (SF-IoU-HR) to isolate resolution-mismatch artifacts.

    3. **ADC — Attention-Delta Correlation** (ORIGINAL, kept for reference)
       Pearson r between per-step intervention attention and per-step
       attention delta.  KNOWN LIMITATION: partially tautological because
       both quantities derive from the same intervention pass.

    4. **P-ADC — Predictive Attention-Delta Correlation** ★ NEW
       Pearson r between BASELINE attention (before any intervention) and
       the resulting PIXEL-CHANGE magnitude per step.  This is DECOUPLED:
       the predictor (baseline attention) is computed without knowledge of
       the intervention, so a significant correlation is non-trivial.
       This is the paper's strongest causal-direction metric.

    5. **L-ADC — Latent-Delta Correlation** ★ NEW
       Pearson r between per-step attention intensity and per-step LATENT
       L2-norm change.  Uses actual intermediate latent snapshots (not
       attention deltas), giving a ground-truth measure of "how much the
       image changed at each step" that is independent of attention.
    """

    @staticmethod
    def attention_concentration_score(
        baseline_data: dict,
        intervention_data: dict,
        target_concept: str,
        step_start: int,
        step_end: int,
    ) -> dict:
        """
        Compute how much the target concept's attention changed during
        the intervention window vs. baseline.

        Args:
            baseline_data: attention_store.baseline_data (step_key → {concept: score})
            intervention_data: attention_store.high_fidelity_data
            target_concept: the concept being steered
            step_start / step_end: intervention window boundaries

        Returns:
            dict with:
                acs: relative change (positive = attention concentrated)
                mean_baseline: average attention in window during baseline
                mean_intervention: average during intervention
                step_deltas: per-step deltas for fine-grained analysis
        """
        target_lower = target_concept.lower().strip()
        baseline_scores = []
        intervention_scores = []
        step_deltas = {}

        # Only iterate over steps that actually exist in BOTH dicts.
        # Baseline captures every 5th step; intervention captures every
        # step inside the window plus every 5th outside it.
        # Intersect the keys to avoid misses.
        bl_keys = set(baseline_data.keys())
        iv_keys = set(intervention_data.keys())
        common_keys = bl_keys & iv_keys

        # Filter to the intervention window
        window_keys = []
        for key in common_keys:
            try:
                step_num = int(key.split("_")[1])
            except (IndexError, ValueError):
                continue
            if step_end <= step_num <= step_start:
                window_keys.append((step_num, key))
        window_keys.sort()

        for step_num, step_key in window_keys:
            bl_step = baseline_data.get(step_key, {})
            iv_step = intervention_data.get(step_key, {})

            bl_score = None
            iv_score = None

            _SKIP_KEYS = ('phase', 'action', 'baseline_comparison')

            def _clean(k):
                """Normalize a key for comparison: strip whitespace, BPE markers."""
                return k.lower().strip().replace("</w>", "").replace("Ġ", "")

            def _find_score(step_dict, target):
                """Find the target concept's score in a step dict."""
                # Pass 1: exact match after cleaning
                for key in step_dict:
                    if key in _SKIP_KEYS:
                        continue
                    if isinstance(step_dict[key], (int, float)) and _clean(key) == target:
                        return step_dict[key]
                # Pass 2: substring match
                for key in step_dict:
                    if key in _SKIP_KEYS:
                        continue
                    if isinstance(step_dict[key], (int, float)) and target in _clean(key):
                        return step_dict[key]
                return None

            bl_score = _find_score(bl_step, target_lower)
            iv_score = _find_score(iv_step, target_lower)

            if bl_score is not None and iv_score is not None:
                baseline_scores.append(bl_score)
                intervention_scores.append(iv_score)
                step_deltas[step_key] = {
                    "baseline": round(bl_score, 6),
                    "intervention": round(iv_score, 6),
                    "delta": round(iv_score - bl_score, 6),
                }

        if not baseline_scores:
            return {"acs": None, "mean_baseline": None, "mean_intervention": None,
                    "num_steps_matched": 0, "step_deltas": {}}

        mean_bl = sum(baseline_scores) / len(baseline_scores)
        mean_iv = sum(intervention_scores) / len(intervention_scores)

        # ACS: relative concentration change
        acs = (mean_iv - mean_bl) / mean_bl if mean_bl > 1e-8 else 0.0

        return {
            "acs": round(acs, 6),
            "mean_baseline": round(mean_bl, 6),
            "mean_intervention": round(mean_iv, 6),
            "num_steps_matched": len(baseline_scores),
            "step_deltas": step_deltas,
        }

    @staticmethod
    def spatial_faithfulness(
        attention_maps: dict,
        target_token_idx: int,
        baseline_image_tensor: torch.Tensor,
        steered_image_tensor: torch.Tensor,
        threshold_percentile: float = 75.0,
    ) -> dict:
        """
        Measure IoU between the attention heatmap (where the model looked)
        and the pixel-change mask (where the output actually changed).

        Computes at TWO resolutions to disentangle resolution artifacts:
          - sf_iou:    computed at native attention resolution (typ. 64×64)
          - sf_iou_hr: upsampled to image resolution (typ. 1024×1024),
                       matching the DAAM protocol (Tang et al., ICLR 2023)

        Args:
            attention_maps: attention_store.attention_maps (step → list of tensors)
            target_token_idx: index of the target concept token in the 77-token sequence
            baseline_image_tensor: [C, H, W] tensor in [0, 1]
            steered_image_tensor: [C, H, W] tensor in [0, 1]
            threshold_percentile: percentile for binarizing both maps

        Returns:
            dict with sf_iou, sf_iou_hr, sf_precision, sf_recall
        """
        _null = {"sf_iou": None, "sf_iou_hr": None, "sf_precision": None, "sf_recall": None}
        if not attention_maps or target_token_idx is None:
            return _null

        # --- Build aggregate attention heatmap for target token ---
        spatial_maps = []
        for step_key, attn_list in attention_maps.items():
            for attn in attn_list:
                if attn.dim() == 3:      # [heads, spatial, text_tokens]
                    avg = attn.mean(dim=0)  # [spatial, text_tokens]
                elif attn.dim() == 4:    # [batch, heads, spatial, text_tokens]
                    avg = attn.mean(dim=1).mean(dim=0)
                else:
                    continue
                if target_token_idx < avg.shape[-1]:
                    spatial_attn = avg[:, target_token_idx]  # [spatial]
                    side = int(spatial_attn.shape[0] ** 0.5)
                    if side * side == spatial_attn.shape[0]:
                        spatial_maps.append(spatial_attn.reshape(side, side))

        if not spatial_maps:
            return _null

        # Resize all to the largest attention resolution and average
        max_side = max(m.shape[0] for m in spatial_maps)
        resized = []
        for m in spatial_maps:
            if m.shape[0] != max_side:
                m = F.interpolate(
                    m.unsqueeze(0).unsqueeze(0).float(),
                    size=(max_side, max_side),
                    mode='bilinear', align_corners=False,
                ).squeeze()
            resized.append(m)
        attn_heatmap = torch.stack(resized).mean(dim=0)  # [max_side, max_side]

        # --- Build pixel-change mask at FULL image resolution ---
        diff_full = (steered_image_tensor - baseline_image_tensor).abs().mean(dim=0)  # [H, W]
        img_h, img_w = diff_full.shape

        # ════════════════════════════════════════════════════════
        # SF-IoU (native attention resolution) — original metric
        # ════════════════════════════════════════════════════════
        diff_resized = F.interpolate(
            diff_full.unsqueeze(0).unsqueeze(0).float(),
            size=(max_side, max_side),
            mode='bilinear', align_corners=False,
        ).squeeze()

        attn_thresh = torch.quantile(attn_heatmap.flatten().float(), threshold_percentile / 100)
        diff_thresh = torch.quantile(diff_resized.flatten().float(), threshold_percentile / 100)

        pred_mask = (attn_heatmap >= attn_thresh).float()
        change_mask = (diff_resized >= diff_thresh).float()

        intersection = (pred_mask * change_mask).sum()
        union = ((pred_mask + change_mask) > 0).float().sum()

        iou = (intersection / union).item() if union > 1e-8 else 0.0

        tp = intersection
        precision = (tp / pred_mask.sum()).item() if pred_mask.sum() > 0 else 0.0
        recall = (tp / change_mask.sum()).item() if change_mask.sum() > 0 else 0.0

        # ════════════════════════════════════════════════════════
        # SF-IoU-HR (image resolution) — DAAM-style upsampled
        # ════════════════════════════════════════════════════════
        attn_upsampled = F.interpolate(
            attn_heatmap.unsqueeze(0).unsqueeze(0).float(),
            size=(img_h, img_w),
            mode='bilinear', align_corners=False,
        ).squeeze()

        attn_thresh_hr = torch.quantile(attn_upsampled.flatten().float(), threshold_percentile / 100)
        diff_thresh_hr = torch.quantile(diff_full.flatten().float(), threshold_percentile / 100)

        pred_mask_hr = (attn_upsampled >= attn_thresh_hr).float()
        change_mask_hr = (diff_full >= diff_thresh_hr).float()

        intersection_hr = (pred_mask_hr * change_mask_hr).sum()
        union_hr = ((pred_mask_hr + change_mask_hr) > 0).float().sum()

        iou_hr = (intersection_hr / union_hr).item() if union_hr > 1e-8 else 0.0

        return {
            "sf_iou": round(iou, 6),
            "sf_iou_hr": round(iou_hr, 6),
            "sf_precision": round(precision, 6),
            "sf_recall": round(recall, 6),
        }

    @staticmethod
    def attention_delta_correlation(
        intervention_data: dict,
        target_concept: str,
    ) -> float:
        """
        ORIGINAL ADC (kept for backward compatibility and ablation).

        Pearson correlation between per-step intervention attention intensity
        and per-step attention delta magnitude.

        ⚠ KNOWN LIMITATION: Both quantities derive from the same intervention
        pass, so the correlation is partially tautological.  Use P-ADC
        (predictive_attention_delta_correlation) for the decoupled version.

        Returns:
            Pearson r in [-1, 1], or None if insufficient data.
        """
        target_lower = target_concept.lower().strip()

        def _clean(k):
            return k.lower().strip().replace("</w>", "").replace("Ġ", "")

        attention_intensities = []
        delta_magnitudes = []

        for step_key, step_data in sorted(intervention_data.items()):
            bc = step_data.get('baseline_comparison', {})
            # Find the target concept in baseline_comparison
            for concept_key, comparison in bc.items():
                if _clean(concept_key) == target_lower or \
                   target_lower in _clean(concept_key):
                    attention_intensities.append(comparison['intervention'])
                    delta_magnitudes.append(abs(comparison['delta']))
                    break

        if len(attention_intensities) < 3:
            return None

        # Pearson correlation
        a = np.array(attention_intensities)
        d = np.array(delta_magnitudes)

        a_mean = a.mean()
        d_mean = d.mean()
        a_std = a.std()
        d_std = d.std()

        if a_std < 1e-8 or d_std < 1e-8:
            return 0.0

        r = np.mean((a - a_mean) * (d - d_mean)) / (a_std * d_std)
        return round(float(r), 6)

    # ==================================================================
    # ★ P-ADC — Predictive (Decoupled) Attention-Delta Correlation
    # ==================================================================
    @staticmethod
    def predictive_attention_delta_correlation(
        baseline_data: dict,
        intervention_data: dict,
        target_concept: str,
    ) -> float:
        """
        DECOUPLED metric: correlate BASELINE attention (computed without
        any knowledge of the intervention) with the PIXEL/ATTENTION CHANGE
        that resulted from the intervention.

        This answers the causal question:
          "Does the model's natural attention on a concept predict how
           sensitive that concept is to embedding perturbation?"

        Concretely:
          X_t = baseline attention on target concept at step t
          Y_t = |intervention_attention_t − baseline_attention_t|  (change magnitude)
          P-ADC = Pearson(X, Y)

        A significant positive P-ADC means: steps where the model naturally
        attends more to the target are also the steps where the intervention
        has the largest effect.  This is non-trivial because X is measured
        from the baseline pass (no intervention), while Y measures the
        intervention's impact.

        Returns:
            Pearson r in [-1, 1], or None if insufficient data.
        """
        target_lower = target_concept.lower().strip()

        def _clean(k):
            return k.lower().strip().replace("</w>", "").replace("Ġ", "")

        def _find_score(step_dict, target):
            _SKIP_KEYS = ('phase', 'action', 'baseline_comparison')
            for key in step_dict:
                if key in _SKIP_KEYS:
                    continue
                if isinstance(step_dict[key], (int, float)) and _clean(key) == target:
                    return step_dict[key]
            for key in step_dict:
                if key in _SKIP_KEYS:
                    continue
                if isinstance(step_dict[key], (int, float)) and target in _clean(key):
                    return step_dict[key]
            return None

        baseline_attention = []   # X: attention from baseline pass (no intervention)
        change_magnitude = []     # Y: how much the intervention changed attention

        common_keys = set(baseline_data.keys()) & set(intervention_data.keys())
        for step_key in sorted(common_keys):
            bl_score = _find_score(baseline_data[step_key], target_lower)
            # Get intervention score from baseline_comparison in intervention_data
            iv_step = intervention_data[step_key]
            bc = iv_step.get('baseline_comparison', {})
            iv_score = None
            for concept_key, comparison in bc.items():
                if _clean(concept_key) == target_lower or \
                   target_lower in _clean(concept_key):
                    iv_score = comparison.get('intervention')
                    break

            if bl_score is not None and iv_score is not None:
                baseline_attention.append(bl_score)
                change_magnitude.append(abs(iv_score - bl_score))

        if len(baseline_attention) < 3:
            return None

        x = np.array(baseline_attention)
        y = np.array(change_magnitude)

        x_std = x.std()
        y_std = y.std()

        if x_std < 1e-8 or y_std < 1e-8:
            return 0.0

        r = np.corrcoef(x, y)[0, 1]
        return round(float(r), 6)

    # ==================================================================
    # ★ L-ADC — Latent-Delta Correlation
    # ==================================================================
    @staticmethod
    def latent_delta_correlation(
        baseline_data: dict,
        intervention_data: dict,
        target_concept: str,
        baseline_latent_trajectory: list,
        intervention_latent_trajectory: list,
    ) -> float:
        """
        Correlate per-step BASELINE attention intensity with per-step
        LATENT L2-norm change between baseline and intervention.

        This is the strongest decoupled metric because:
          X_t = baseline attention on target at step t (no intervention)
          Y_t = ||z^interv_t − z^base_t||_2  (actual latent divergence)

        Both are measured independently: X from baseline pass, Y from
        comparing the two latent trajectories.

        Args:
            baseline_data: attention_store.baseline_data
            intervention_data: attention_store.high_fidelity_data
            target_concept: the target concept
            baseline_latent_trajectory: list of (step, latent_tensor) from baseline
            intervention_latent_trajectory: list of (step, latent_tensor) from intervention

        Returns:
            Pearson r in [-1, 1], or None if insufficient data.
        """
        target_lower = target_concept.lower().strip()

        def _clean(k):
            return k.lower().strip().replace("</w>", "").replace("Ġ", "")

        def _find_score(step_dict, target):
            _SKIP_KEYS = ('phase', 'action', 'baseline_comparison')
            for key in step_dict:
                if key in _SKIP_KEYS:
                    continue
                if isinstance(step_dict[key], (int, float)) and _clean(key) == target:
                    return step_dict[key]
            for key in step_dict:
                if key in _SKIP_KEYS:
                    continue
                if isinstance(step_dict[key], (int, float)) and target in _clean(key):
                    return step_dict[key]
            return None

        # Build dict: step → latent for fast lookup
        bl_latents = {step: lat for step, lat in baseline_latent_trajectory}
        iv_latents = {step: lat for step, lat in intervention_latent_trajectory}

        attention_values = []
        latent_deltas = []

        common_steps = sorted(set(bl_latents.keys()) & set(iv_latents.keys()))
        for step in common_steps:
            step_key = f"step_{step}"
            bl_score = _find_score(baseline_data.get(step_key, {}), target_lower)
            if bl_score is None:
                continue

            # L2 norm of latent difference at this step
            l2_delta = (iv_latents[step].float() - bl_latents[step].float()).norm().item()

            attention_values.append(bl_score)
            latent_deltas.append(l2_delta)

        if len(attention_values) < 3:
            return None

        x = np.array(attention_values)
        y = np.array(latent_deltas)

        x_std = x.std()
        y_std = y.std()

        if x_std < 1e-8 or y_std < 1e-8:
            return 0.0

        r = np.corrcoef(x, y)[0, 1]
        return round(float(r), 6)

    # ==================================================================
    # ★ AKS — Attention Knockout Score (Causal Ablation)
    # ==================================================================
    @staticmethod
    def attention_knockout_score(
        baseline_image_tensor: torch.Tensor,
        steered_image_tensor: torch.Tensor,
        knockout_image_tensor: torch.Tensor,
    ) -> dict:
        """
        Causal ablation metric: if we zero out cross-attention for the
        target token during intervention (knockout), does the visual
        change disappear?

        Three images are compared:
          baseline  — original generation, no intervention
          steered   — intervention applied normally
          knockout  — intervention applied BUT attention for target token
                      is zeroed out

        AKS = 1 − (ΔLPIPS_knockout / ΔLPIPS_steered)

        If attention is truly causal:
          - ΔLPIPS_steered should be large (intervention changed the image)
          - ΔLPIPS_knockout should be small (blocking attention blocks the change)
          - AKS → 1.0 (high causality)

        If attention is epiphenomenal:
          - Knockout doesn't reduce the change
          - AKS → 0.0 (attention didn't matter)

        Args:
            baseline_image_tensor: [C, H, W] in [0, 1]
            steered_image_tensor:  [C, H, W] in [0, 1]
            knockout_image_tensor: [C, H, W] in [0, 1]

        Returns:
            dict with aks, lpips_steered, lpips_knockout
        """
        # Pixel-level L2 as a lightweight proxy (no LPIPS model needed here)
        diff_steered = (steered_image_tensor - baseline_image_tensor).float()
        diff_knockout = (knockout_image_tensor - baseline_image_tensor).float()

        # Handle NaN pixels (from VAE decode instability)
        diff_steered = torch.nan_to_num(diff_steered, nan=0.0)
        diff_knockout = torch.nan_to_num(diff_knockout, nan=0.0)

        l2_steered = diff_steered.norm().item()
        l2_knockout = diff_knockout.norm().item()

        if l2_steered < 1e-8:
            return {"aks": None, "l2_steered": l2_steered, "l2_knockout": l2_knockout}

        aks_raw = 1.0 - (l2_knockout / l2_steered)
        # Clamp to [0, 1]: negative means knockout caused MORE change (pathological)
        aks = max(0.0, min(1.0, aks_raw))

        return {
            "aks": round(aks, 6),
            "aks_raw": round(aks_raw, 6),  # un-clamped for diagnostics
            "l2_steered": round(l2_steered, 6),
            "l2_knockout": round(l2_knockout, 6),
        }


class MetricEvaluator:
    """
    Evaluates generated output quality and directional CLIP similarity.
    Calculates Delta-CLIP (StyleGAN-NADA directional score) and LPIPS image preservation.
    """
    def __init__(self, device="cuda"):
        self.device = device

        # Load CLIP
        print(f"Loading Evaluator CLIP Model to {device}...")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Load LPIPS for structural preservation checks
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
        self.lpips.eval()

    @torch.no_grad()
    def calculate_delta_clip(self, baseline_img, steered_img, prompt, target_concept, injection_attribute):
        """
        Directional CLIP measurement (StyleGAN-NADA / DiffusionCLIP formulation).

        Text pair construction:
          source_text = full original prompt (e.g., "a majestic tiger on a mountain")
          target_text = prompt with attribute injected into the target concept
                        (e.g., "a majestic red tiger on a mountain")

        This ensures the text direction captures the SEMANTIC EDIT in full
        prompt context, not just the isolated word-to-word direction.

        Computes:
          text_delta  = CLIP(target_text) - CLIP(source_text)
          image_delta = CLIP(steered_img)  - CLIP(baseline_img)
          score       = cosine(text_delta, image_delta)

        A positive score means the image changed in the same direction as the
        text edit. Higher is better.

        Also returns absolute CLIP-T scores (image-text alignment) for both
        the baseline and steered images against their respective prompts.

        Args:
            baseline_img: PIL image from baseline (no intervention) pass
            steered_img: PIL image from intervention pass
            prompt: Full original text prompt
            target_concept: The concept being modified (e.g., "tiger")
            injection_attribute: The attribute injected (e.g., "red")

        Returns:
            (delta_clip_score, base_clip, steered_clip)
        """
        # Construct proper text pairs using full prompt context
        source_text = prompt
        # Insert attribute before the target concept in the full prompt
        if target_concept in prompt:
            target_text = prompt.replace(
                target_concept,
                f"{injection_attribute} {target_concept}"
            )
        else:
            # Fallback: prepend attribute to prompt
            target_text = f"{injection_attribute} {prompt}"

        # Process Images
        inputs_img = self.processor(
            images=[baseline_img, steered_img], return_tensors="pt"
        ).to(self.device)

        # Process Texts — source and target prompts
        inputs_txt = self.processor(
            text=[source_text, target_text], return_tensors="pt", padding=True
        ).to(self.device)

        # Unified Forward Pass → projected 2D embeddings (Batch, 512)
        outputs = self.clip(
            input_ids=inputs_txt.input_ids,
            attention_mask=inputs_txt.attention_mask,
            pixel_values=inputs_img.pixel_values
        )

        img_embeds = F.normalize(outputs.image_embeds, p=2, dim=-1)
        txt_embeds = F.normalize(outputs.text_embeds, p=2, dim=-1)

        # Absolute CLIP-T scores (image-text alignment)
        base_clip = F.cosine_similarity(
            img_embeds[0].unsqueeze(0), txt_embeds[0].unsqueeze(0)
        ).item()
        steered_clip = F.cosine_similarity(
            img_embeds[1].unsqueeze(0), txt_embeds[1].unsqueeze(0)
        ).item()

        # Directional deltas
        img_delta = F.normalize(img_embeds[1] - img_embeds[0], p=2, dim=-1)
        txt_delta = F.normalize(txt_embeds[1] - txt_embeds[0], p=2, dim=-1)

        delta_clip_score = torch.dot(img_delta, txt_delta).item()

        return delta_clip_score, base_clip, steered_clip

    @torch.no_grad()
    def calculate_structure_preservation(self, baseline_tensor, steered_tensor):
        """
        LPIPS (Learned Perceptual Image Patch Similarity).
        Lower is better (means structure was preserved outside the intended edit).
        Expects tensors in [-1, 1], shape [B, C, H, W]
        """
        lpips_score = self.lpips(baseline_tensor, steered_tensor)
        return lpips_score.item()

    # ------------------------------------------------------------------
    # Attention Faithfulness (IoU-based)
    # ------------------------------------------------------------------
    @staticmethod
    def attention_iou(attention_heatmap: torch.Tensor,
                      reference_mask: torch.Tensor,
                      threshold_percentile: float = 75.0) -> float:
        """
        Compute Intersection-over-Union between a soft attention heatmap and
        a binary reference mask.  Used to measure whether the model's internal
        cross-attention actually focuses on the *right spatial region*.

        The attention heatmap is binarised at the given percentile threshold
        before IoU is computed.

        Args:
            attention_heatmap: (H, W) float tensor of attention scores (any range).
            reference_mask:    (H, W) binary tensor (1 = region of interest).
            threshold_percentile: Percentile above which attention is treated as
                                  "active" (default: 75th percentile).

        Returns:
            IoU score in [0, 1].  1.0 = perfect spatial alignment.
        """
        assert attention_heatmap.ndim == 2 and reference_mask.ndim == 2, \
            "Both inputs must be 2-D (H, W)"

        # Resize reference mask to match attention heatmap resolution
        if attention_heatmap.shape != reference_mask.shape:
            reference_mask = F.interpolate(
                reference_mask.unsqueeze(0).unsqueeze(0).float(),
                size=attention_heatmap.shape,
                mode='nearest'
            ).squeeze(0).squeeze(0)

        # Binarise the soft attention map at the chosen percentile
        threshold = torch.quantile(
            attention_heatmap.float().flatten(),
            threshold_percentile / 100.0
        )
        pred_mask = (attention_heatmap >= threshold).float()
        ref_mask = (reference_mask > 0.5).float()

        intersection = (pred_mask * ref_mask).sum()
        union = ((pred_mask + ref_mask) > 0).float().sum()

        if union < 1e-8:
            return 0.0

        return (intersection / union).item()

    @staticmethod
    def attention_precision_recall(attention_heatmap: torch.Tensor,
                                   reference_mask: torch.Tensor,
                                   threshold_percentile: float = 75.0):
        """
        Compute precision and recall of the binarised attention heatmap
        relative to a reference segmentation mask.

        Precision: fraction of high-attention pixels that fall inside the mask.
        Recall:    fraction of mask pixels that receive high attention.

        Returns:
            (precision, recall) tuple of floats.
        """
        assert attention_heatmap.ndim == 2 and reference_mask.ndim == 2

        if attention_heatmap.shape != reference_mask.shape:
            reference_mask = F.interpolate(
                reference_mask.unsqueeze(0).unsqueeze(0).float(),
                size=attention_heatmap.shape,
                mode='nearest'
            ).squeeze(0).squeeze(0)

        threshold = torch.quantile(
            attention_heatmap.float().flatten(),
            threshold_percentile / 100.0
        )
        pred_mask = (attention_heatmap >= threshold).float()
        ref_mask = (reference_mask > 0.5).float()

        true_positive = (pred_mask * ref_mask).sum()
        precision = (true_positive / pred_mask.sum()).item() if pred_mask.sum() > 0 else 0.0
        recall = (true_positive / ref_mask.sum()).item() if ref_mask.sum() > 0 else 0.0

        return precision, recall
