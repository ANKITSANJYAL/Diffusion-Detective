#!/usr/bin/env python3
"""
Reproduce images from aggregated_metrics.jsonl.

Given a JSONL file with `reproduction` parameters (model_id, seed, prompt,
target, attribute, strength, window), re-generates the baseline and steered
images on-demand without needing the original PNGs on disk.

Usage:
  # Reproduce a single sample by prompt substring:
  python -m experiments.reproduce --jsonl results/full-cvpr-eval_.../aggregated_metrics.jsonl \
      --match "tiger" --outdir reproduced/

  # Reproduce all samples:
  python -m experiments.reproduce --jsonl results/full-cvpr-eval_.../aggregated_metrics.jsonl \
      --outdir reproduced/ --all
"""
import argparse
import json
import pathlib
import sys
import torch

backend_path = str(pathlib.Path(__file__).parent.parent / "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from app.pipeline import InterpretableSDPipeline


def reproduce_sample(pipeline, record: dict, outdir: pathlib.Path):
    """Reproduce a single sample from its stored reproduction params."""
    repro = record.get("reproduction", {})
    prompt = record["prompt"]
    target = record["target_concept"]
    injection = record["injection_attribute"]

    seed = repro.get("seed", 42)
    guidance = repro.get("guidance_scale", 7.5)
    steps = repro.get("num_inference_steps", 50)
    strength = repro.get("strength", 1.0)
    step_start = repro.get("step_start", 40)
    step_end = repro.get("step_end", 20)

    baseline_img, steered_img, _, _ = pipeline.generate(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        intervention_active=True,
        intervention_strength=strength,
        intervention_step_start=step_start,
        intervention_step_end=step_end,
        seed=seed,
        target_concept=target,
        injection_attribute=injection,
    )

    safe_name = prompt[:40].replace(" ", "_").replace("/", "")
    sample_dir = outdir / safe_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    baseline_img.save(sample_dir / "baseline.png")
    steered_img.save(sample_dir / "steered.png")
    print(f"  ✓ Reproduced: {prompt} → {sample_dir}")


def main():
    parser = argparse.ArgumentParser(description="Reproduce images from JSONL")
    parser.add_argument("--jsonl", required=True, help="Path to aggregated_metrics.jsonl")
    parser.add_argument("--outdir", default="reproduced", help="Output directory")
    parser.add_argument("--match", default=None, help="Only reproduce samples matching this substring")
    parser.add_argument("--all", action="store_true", help="Reproduce all samples")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    jsonl_path = pathlib.Path(args.jsonl)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load records
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records from {jsonl_path}")

    # Filter
    if args.match:
        records = [r for r in records if args.match.lower() in r["prompt"].lower()]
        print(f"Filtered to {len(records)} records matching '{args.match}'")
    elif not args.all:
        print("Specify --all to reproduce everything, or --match <substring> to filter.")
        return

    if not records:
        print("No records to reproduce.")
        return

    # Determine model from first record
    model_id = records[0].get("reproduction", {}).get("model_id", "runwayml/stable-diffusion-v1-5")
    dtype_str = records[0].get("reproduction", {}).get("torch_dtype", "float16")
    torch_dtype = torch.float16 if dtype_str == "float16" else torch.float32

    print(f"Loading model: {model_id}")
    pipeline = InterpretableSDPipeline(model_id=model_id, device=args.device, torch_dtype=torch_dtype)

    for i, record in enumerate(records):
        print(f"\n[{i+1}/{len(records)}] {record['prompt']}")
        reproduce_sample(pipeline, record, outdir)

    print(f"\nDone. {len(records)} samples reproduced to {outdir}")


if __name__ == "__main__":
    main()
