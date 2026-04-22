#!/usr/bin/env python3
"""
Quick validation that all new metrics (P-ADC, L-ADC, SF-IoU-HR, AKS)
compute correctly on a single example.

Usage:
    conda activate diff_cvpr
    python experiments/validate_new_metrics.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend"))

import torch
from torchvision.transforms.functional import to_tensor
from app.pipeline import InterpretableSDPipeline
from experiments.src.metrics.quantitative import AttentionFaithfulness

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── Load pipeline ─────────────────────────────────────────────
pipe = InterpretableSDPipeline(
    model_id="stabilityai/stable-diffusion-xl-base-1.0",
    device=device,
    torch_dtype=torch.float16,
)

prompt = "A cat sitting on a wooden table"
target = "cat"
attribute = "golden"
strength = 1.0
step_start, step_end = 50, 35

# ── Baseline ──────────────────────────────────────────────────
print("\n1. Generating baseline...")
baseline_cache = pipe.generate_baseline(
    prompt=prompt, num_inference_steps=50, guidance_scale=7.5, seed=42,
)
baseline_img = baseline_cache["baseline_image"]
attn_store = pipe.attention_store

bl_traj_len = len(attn_store.baseline_latent_trajectory)
print(f"   baseline_data keys: {len(attn_store.baseline_data)}")
print(f"   baseline_latent_trajectory: {bl_traj_len} snapshots")
assert bl_traj_len > 0, "FAIL: No baseline latent trajectory captured!"

# ── Intervention ──────────────────────────────────────────────
print("\n2. Generating intervention...")
steered_img, logs, meta = pipe.generate_intervention(
    baseline_cache=baseline_cache,
    target_concept=target,
    injection_attribute=attribute,
    intervention_strength=strength,
    intervention_step_start=step_start,
    intervention_step_end=step_end,
)

iv_traj_len = len(attn_store.intervention_latent_trajectory)
print(f"   high_fidelity_data keys: {len(attn_store.high_fidelity_data)}")
print(f"   intervention_latent_trajectory: {iv_traj_len} snapshots")
assert iv_traj_len > 0, "FAIL: No intervention latent trajectory captured!"

# ── Compute ALL metrics ──────────────────────────────────────
print("\n3. Computing metrics...")

# ACS
acs = AttentionFaithfulness.attention_concentration_score(
    attn_store.baseline_data, attn_store.high_fidelity_data,
    target, step_start, step_end,
)
print(f"   ACS = {acs['acs']}  (steps matched: {acs['num_steps_matched']})")

# ADC (original)
adc = AttentionFaithfulness.attention_delta_correlation(
    attn_store.high_fidelity_data, target,
)
print(f"   ADC = {adc}")

# ★ P-ADC (decoupled)
p_adc = AttentionFaithfulness.predictive_attention_delta_correlation(
    attn_store.baseline_data, attn_store.high_fidelity_data, target,
)
print(f"   P-ADC = {p_adc}")

# ★ L-ADC (latent-delta)
l_adc = AttentionFaithfulness.latent_delta_correlation(
    attn_store.baseline_data, attn_store.high_fidelity_data, target,
    attn_store.baseline_latent_trajectory, attn_store.intervention_latent_trajectory,
)
print(f"   L-ADC = {l_adc}")

# SF-IoU + SF-IoU-HR
tokens = baseline_cache["tokens"]
t_idx = None
for idx, tok in enumerate(tokens):
    if target.lower() in tok.strip().lower():
        t_idx = idx
        break
print(f"   Target token index: {t_idx}")

if t_idx is not None:
    sf = AttentionFaithfulness.spatial_faithfulness(
        attn_store.attention_maps, t_idx,
        to_tensor(baseline_img), to_tensor(steered_img),
    )
    print(f"   SF-IoU    = {sf['sf_iou']}")
    print(f"   SF-IoU-HR = {sf['sf_iou_hr']}")

# ── Knockout ──────────────────────────────────────────────────
print("\n4. Generating knockout (causal ablation)...")
knockout_img = pipe.generate_knockout(
    baseline_cache=baseline_cache,
    target_concept=target,
    injection_attribute=attribute,
    intervention_strength=strength,
    intervention_step_start=step_start,
    intervention_step_end=step_end,
)

aks = AttentionFaithfulness.attention_knockout_score(
    to_tensor(baseline_img), to_tensor(steered_img), to_tensor(knockout_img),
)
print(f"   AKS = {aks['aks']}")
print(f"   AKS (raw, unclamped) = {aks['aks_raw']}")
print(f"   L2 steered  = {aks['l2_steered']}")
print(f"   L2 knockout = {aks['l2_knockout']}")
if aks['aks_raw'] is not None and aks['aks_raw'] < 0:
    print(f"   ⚠ WARNING: Negative AKS means knockout INCREASED change")
    print(f"              (expected: knockout should REDUCE change)")
    print(f"              Ratio: L2_ko/L2_steer = {aks['l2_knockout']/aks['l2_steered']:.2f}x")

# ── Save images for visual inspection ─────────────────────────
out_dir = "experiments/validation_output"
os.makedirs(out_dir, exist_ok=True)
baseline_img.save(f"{out_dir}/baseline.png")
steered_img.save(f"{out_dir}/steered.png")
knockout_img.save(f"{out_dir}/knockout.png")

print(f"\n{'='*60}")
print("VALIDATION COMPLETE — All new metrics computed successfully!")
print(f"{'='*60}")
print(f"\nSummary:")
print(f"  ACS      = {acs['acs']}")
print(f"  ADC      = {adc}  (original, tautological)")
print(f"  P-ADC    = {p_adc}  (★ decoupled)")
print(f"  L-ADC    = {l_adc}  (★ latent-based)")
print(f"  SF-IoU   = {sf['sf_iou'] if t_idx else 'N/A'}")
print(f"  SF-IoU-HR= {sf['sf_iou_hr'] if t_idx else 'N/A'}  (★ DAAM-style)")
print(f"  AKS      = {aks['aks']}  (raw: {aks['aks_raw']})  (★ causal knockout)")
print(f"  L2 steer = {aks['l2_steered']},  L2 knock = {aks['l2_knockout']}")
print(f"\nImages saved to {out_dir}/")

# Sanity checks
assert aks['l2_steered'] > 0, "FAIL: Steered image identical to baseline"
if aks['aks_raw'] is not None and aks['aks_raw'] < -0.5:
    print("\n⚠ WARNING: AKS is strongly negative — knockout disrupted generation.")
    print("  This may indicate the knockout processor needs debugging.")
