"""Test attention capture to debug empty attention maps issue."""

import sys
sys.path.insert(0, '/Users/ankitsanjyal/Desktop/Diffusion-Detective/backend')

from app.pipeline import InterpretableSDPipeline
import torch

print("=" * 80)
print("TESTING ATTENTION CAPTURE")
print("=" * 80)

# Initialize pipeline
print("\n1. Initializing pipeline...")
pipe = InterpretableSDPipeline(device="mps")

# Test generation with minimal steps
print("\n2. Running generation with 10 steps...")
natural_img, controlled_img, logs, metadata = pipe.generate(
    prompt="A red apple",
    num_inference_steps=10,
    guidance_scale=7.5,
    intervention_active=True,
    intervention_strength=1.0,
    intervention_step_start=8,
    intervention_step_end=4,
    seed=42
)

print("\n3. Checking attention maps...")
print(f"   Attention store has maps for steps: {list(pipe.attention_store.attention_maps.keys())}")
print(f"   Total number of steps with attention: {len(pipe.attention_store.attention_maps)}")

for step, maps in pipe.attention_store.attention_maps.items():
    print(f"   Step {step}: {len(maps)} attention maps captured")
    if maps:
        print(f"      First map shape: {maps[0].shape}")

print("\n4. Checking logs...")
print(f"   Total logs: {len(logs)}")
for log in logs:
    if isinstance(log, dict):
        intervention_marker = "[INJECTION]" if log.get('intervention_active') else ""
        print(f"   [Step {log['step']}] {log['phase']}: {intervention_marker} {log['message']}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
