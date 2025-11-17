#!/usr/bin/env python3
"""Smoke test for Stable Diffusion 2.1 using the project's ModelRunner.

This uses `ModelRunner.generate_with_intermediates` to capture intermediate images
and the final decoded image. It is intended to be run on a GPU node.
"""
import argparse
from pathlib import Path
import sys
import pathlib
import torch

# When running the script directly, ensure the repo root is on sys.path so `import src` works
repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.runner import ModelRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", default="A painting of a detective in a foggy alley, cinematic lighting")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--out", default="sd2_1_smoke.png")
    parser.add_argument("--capture-intermediates", action="store_true", help="Save intermediate decoded images")
    args = parser.parse_args()

    if "cuda" in args.device and not torch.cuda.is_available():
        print("CUDA requested but not available. Aborting smoke test.")
        return 2

    runner = ModelRunner(sd_model=args.model, device=args.device)

    capture_indices = None
    if args.capture_intermediates:
        # capture a few representative steps
        capture_indices = [args.steps // 4, args.steps // 2, (3 * args.steps) // 4]

    print(f"Generating on {args.device} with {args.steps} steps; capture={capture_indices}")
    res = runner.generate_with_intermediates(args.prompt, steps=args.steps, guidance=7.5, height=512, width=512, capture_steps=capture_indices)

    final_img = res.get("final_image")
    intermediates = res.get("intermediates", [])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_img.save(str(out_path))
    print(f"Saved final image: {out_path}")

    if args.capture_intermediates and intermediates:
        for idx, img in intermediates:
            step_path = out_path.with_name(f"{out_path.stem}_step{idx}{out_path.suffix}")
            img.save(str(step_path))
            print(f"Saved intermediate image: {step_path}")


if __name__ == "__main__":
    main()
