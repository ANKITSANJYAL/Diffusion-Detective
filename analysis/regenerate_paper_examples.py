#!/usr/bin/env python3
"""
Regenerate specific examples for paper qualitative figures.

Produces baseline + steered + heatmap + diff image for each selected case.
All examples use the same seed/config as the ablation run for exact reproduction.

Usage:
    conda activate diff_cvpr
    python analysis/regenerate_paper_examples.py
"""
import sys, os, json, torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.app.pipeline import InterpretableSDPipeline
from experiments.src.metrics.quantitative import AttentionFaithfulness

OUT_DIR = Path("analysis/paper_qualitative")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Selected examples ────────────────────────────────────────────
# Each entry: (sample_idx, prompt, target, attribute, strength, step_start, step_end, label, group)
EXAMPLES = [
    # === SUCCESSES ===
    # S1: Bear honey jar — shadowy (high SF-IoU = 0.71, ADC = -0.99)
    (22, "A bear shaped full jar of honey sits on a table",
     "bear", "shadowy", 0.75, 50, 35,
     "success_bear_shadowy", "success"),

    # S2: Street sign — glass (ADC = -0.999, ACS = -0.57)
    (399, "A hard to miss street sign set between two traffic lights.",
     "street", "glass", 2.0, 45, 5,
     "success_street_glass", "success"),

    # S3: Sheep herd — golden (Color cluster, ADC = -0.9996, SF = 0.47)
    (88, "A herd of sheep walking across a snow covered field.",
     "herd", "golden", 2.0, 35, 20,
     "success_sheep_golden", "success"),

    # S4: People — metallic (Material cluster, ADC = -0.9997, SF = 0.35)
    (573, "Oddly dressed people standing by each other posing towards camera.",
     "people", "metallic", 2.0, 35, 20,
     "success_people_metallic", "success"),

    # S5: Foods — ancient (Style cluster, ADC = -0.9999)
    (183, "Many different foods on dishes on a table",
     "foods", "ancient", 2.0, 20, 5,
     "success_foods_ancient", "success"),

    # === FAILURES ===
    # F1: Zebras — cyberpunk (positive ADC = +0.9999!)
    (605, "A group of zebras drinking water from a pond next to trees.",
     "group", "cyberpunk", 2.0, 50, 35,
     "failure_zebras_cyberpunk", "failure"),

    # F2: Orange fire truck — fiery (ACS = +0.75, attention INCREASED)
    (388, "an orange fire truck parked in a wharehouse",
     "orange", "fiery", 2.0, 35, 20,
     "failure_firetruck_fiery", "failure"),

    # F3: Kitchen appliances — shadowy (SF-IoU = 0.00 despite LPIPS = 0.15)
    (625, "A group of kitchen appliances on a metal table",
     "group", "shadowy", 0.75, 50, 35,
     "failure_kitchen_shadowy", "failure"),

    # F4: People on buggy (first row in dataset, moderate failure at high strength)
    (83, "A person sitting at a table with two pizzas on pizza pans.",
     "person", "cyberpunk", 1.0, 50, 35,
     "failure_pizza_cyberpunk", "failure"),

    # F5: Lot of people — fiery (positive ADC = +0.97)
    (635, "A lot of people that are having some fun.",
     "lot", "fiery", 1.5, 50, 35,
     "failure_people_fiery", "failure"),
]


def make_diff_image(baseline: Image.Image, steered: Image.Image) -> Image.Image:
    """Create a colorized pixel-difference image."""
    b = np.array(baseline).astype(float)
    s = np.array(steered).astype(float)
    diff = np.abs(s - b).mean(axis=-1)  # grayscale diff
    diff = diff / max(diff.max(), 1e-6)  # normalize to [0, 1]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    colored = (cm.inferno(diff)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(colored)


def make_panel(baseline, steered, heatmap_img, diff_img, label, metrics_text):
    """Create a 4-panel figure: baseline | steered | heatmap | diff"""
    W, H = baseline.size
    panel_w = W
    gap = 10
    total_w = panel_w * 4 + gap * 3
    header = 80
    footer = 100
    total_h = H + header + footer

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))

    # Paste images
    for i, (img, sub_label) in enumerate([
        (baseline, "Baseline"),
        (steered, "Steered"),
        (heatmap_img, "Attention Heatmap"),
        (diff_img, "Pixel Difference"),
    ]):
        x = i * (panel_w + gap)
        resized = img.resize((panel_w, H), Image.LANCZOS)
        canvas.paste(resized, (x, header))

    # Draw labels
    draw = ImageDraw.Draw(canvas)
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_metrics = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font_title = font_sub = font_metrics = ImageFont.load_default()

    # Title
    draw.text((10, 5), label, fill=(0, 0, 0), font=font_title)

    # Sub-labels
    for i, sub in enumerate(["Baseline", "Steered", "Attention Heatmap", "Pixel Difference"]):
        x = i * (panel_w + gap) + panel_w // 2 - 50
        draw.text((x, header - 22), sub, fill=(80, 80, 80), font=font_sub)

    # Metrics at bottom
    draw.text((10, H + header + 10), metrics_text, fill=(0, 0, 0), font=font_metrics)

    return canvas


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load pipeline
    pipe = InterpretableSDPipeline(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        device=device,
        torch_dtype=torch.float16,
    )

    # Load dataset to get the correct prompt for each sample_idx
    # We need to map sample_idx back to the actual prompt
    # (the examples above already have the prompts from the JSONL)

    for sample_idx, prompt, target, attribute, strength, step_start, step_end, label, group in EXAMPLES:
        print(f"\n{'='*60}")
        print(f"Generating: {label}")
        print(f"  Prompt: \"{prompt}\"")
        print(f"  {target} → {attribute}  (s={strength}, w={step_start}-{step_end})")
        print(f"{'='*60}")

        out_path = OUT_DIR / group / label
        out_path.mkdir(parents=True, exist_ok=True)

        # Generate baseline
        baseline_cache = pipe.generate_baseline(
            prompt=prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            seed=42,
        )
        baseline_img = baseline_cache["baseline_image"]

        # Generate intervention
        steered_img, logs, meta = pipe.generate_intervention(
            baseline_cache=baseline_cache,
            target_concept=target,
            injection_attribute=attribute,
            intervention_strength=strength,
            intervention_step_start=step_start,
            intervention_step_end=step_end,
        )

        attn_store = pipe.attention_store

        # Compute metrics for verification
        acs_data = AttentionFaithfulness.attention_concentration_score(
            baseline_data=attn_store.baseline_data,
            intervention_data=attn_store.high_fidelity_data,
            target_concept=target,
            step_start=step_start,
            step_end=step_end,
        )
        adc = AttentionFaithfulness.attention_delta_correlation(
            intervention_data=attn_store.high_fidelity_data,
            target_concept=target,
        )

        print(f"  ACS = {acs_data['acs']}, ADC = {adc}")

        # Save individual images
        baseline_img.save(out_path / "baseline.png")
        steered_img.save(out_path / "steered.png")

        # Generate heatmap
        try:
            tokens = baseline_cache["tokens"]
            t_idx = None
            for idx, tok in enumerate(tokens):
                tok_clean = tok.strip().lower()
                if tok_clean == target.lower() or target.lower().startswith(tok_clean):
                    t_idx = idx
                    break

            if t_idx is not None and attn_store.attention_maps:
                from torchvision.transforms import ToTensor
                base_t = ToTensor()(baseline_img)
                steer_t = ToTensor()(steered_img)
                sf_data = AttentionFaithfulness.spatial_faithfulness(
                    attention_maps=attn_store.attention_maps,
                    target_token_idx=t_idx,
                    baseline_image_tensor=base_t,
                    steered_image_tensor=steer_t,
                )
                print(f"  SF-IoU = {sf_data['sf_iou']}")

            # Build heatmap from attention maps
            heatmap_img = _build_heatmap(attn_store.attention_maps, t_idx, baseline_img.size)
        except Exception as e:
            print(f"  Heatmap error: {e}")
            heatmap_img = Image.new("RGB", baseline_img.size, (128, 128, 128))

        heatmap_img.save(out_path / "heatmap.png")

        # Diff image
        diff_img = make_diff_image(baseline_img, steered_img)
        diff_img.save(out_path / "diff.png")

        # Composite panel
        metrics_text = (
            f"ACS = {acs_data['acs']:.4f}  |  ADC = {adc:.4f}  |  "
            f"SF-IoU = {sf_data.get('sf_iou', 'N/A')}  |  "
            f"Strength = {strength}  |  Window = {step_start}→{step_end}  |  "
            f"Target: '{target}' → '{attribute}'"
        )
        panel = make_panel(baseline_img, steered_img, heatmap_img, diff_img, label, metrics_text)
        panel.save(out_path / "panel.png", quality=95)
        print(f"  Saved to {out_path}/")

    # Create a combined grid of all successes and failures
    print("\n\nBuilding combined grids...")
    for group in ["success", "failure"]:
        panels = []
        group_dir = OUT_DIR / group
        if not group_dir.exists():
            continue
        for subdir in sorted(group_dir.iterdir()):
            panel_path = subdir / "panel.png"
            if panel_path.exists():
                panels.append(Image.open(panel_path))

        if panels:
            # Stack vertically
            max_w = max(p.width for p in panels)
            total_h = sum(p.height for p in panels) + 20 * (len(panels) - 1)
            grid = Image.new("RGB", (max_w, total_h), (255, 255, 255))
            y = 0
            for p in panels:
                grid.paste(p, (0, y))
                y += p.height + 20
            grid.save(OUT_DIR / f"grid_{group}.png", quality=95)
            print(f"  Saved {OUT_DIR / f'grid_{group}.png'}")

    print("\nDone!")


def _build_heatmap(attention_maps, target_token_idx, image_size):
    """Build an attention heatmap overlay from attention maps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    if not attention_maps or target_token_idx is None:
        return Image.new("RGB", image_size, (128, 128, 128))

    spatial_maps = []
    for step_key, attn_list in attention_maps.items():
        for attn in attn_list:
            if attn.dim() == 3:
                avg = attn.mean(dim=0)
            elif attn.dim() == 4:
                avg = attn.mean(dim=1).mean(dim=0)
            else:
                continue
            if target_token_idx < avg.shape[-1]:
                spatial_attn = avg[:, target_token_idx]
                side = int(spatial_attn.shape[0] ** 0.5)
                if side * side == spatial_attn.shape[0]:
                    spatial_maps.append(spatial_attn.reshape(side, side))

    if not spatial_maps:
        return Image.new("RGB", image_size, (128, 128, 128))

    # Average all maps at max resolution
    max_side = max(m.shape[0] for m in spatial_maps)
    resized = []
    for m in spatial_maps:
        if m.shape[0] != max_side:
            m = torch.nn.functional.interpolate(
                m.unsqueeze(0).unsqueeze(0).float(),
                size=(max_side, max_side), mode="bilinear", align_corners=False
            ).squeeze()
        resized.append(m)

    avg_map = torch.stack(resized).mean(dim=0).numpy()
    avg_map = (avg_map - avg_map.min()) / (avg_map.max() - avg_map.min() + 1e-8)

    # Colorize and resize to image dimensions
    colored = (cm.jet(avg_map)[:, :, :3] * 255).astype(np.uint8)
    heatmap = Image.fromarray(colored).resize(image_size, Image.LANCZOS)

    return heatmap


if __name__ == "__main__":
    main()
