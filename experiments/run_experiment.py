"""
Main entry point for Diffusion Detective CVPR benchmark suite.
Usage:
  Single GPU:
    python -m experiments.run_experiment --config-name=smoke_test
  Multi-GPU (DDP):
    torchrun --nproc_per_node=2 -m experiments.run_experiment --config-name=smoke_test
"""
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import os

from .src.data import BenchmarkDataset
from .src.engine import ExperimentRunner
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

def setup_distributed():
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return local_rank, world_size
    else:
        # Fallback to single GPU / CPU
        return 0, 1

@hydra.main(version_base="1.3", config_path="configs", config_name="base")
def main(cfg: DictConfig):
    local_rank, world_size = setup_distributed()
    is_main_process = (local_rank == 0)

    if is_main_process:
        print("=" * 60)
        print("Diffusion Detective - CVPR Evaluation Pipeline")
        print("=" * 60)
        print(OmegaConf.to_yaml(cfg))

    # ------------------------------------------------------------------
    # Build dataset on rank-0 first (downloads from HuggingFace if needed),
    # then barrier so other ranks read from the cached copy.
    # ------------------------------------------------------------------
    if is_main_process:
        dataset = BenchmarkDataset(
            dataset_name=cfg.data.dataset_name,
            categories=cfg.data.categories,
            max_samples=cfg.data.get("max_samples"),
        )
        print(f"Dataset Size: {len(dataset)}")

    if world_size > 1:
        dist.barrier()

    if not is_main_process:
        dataset = BenchmarkDataset(
            dataset_name=cfg.data.dataset_name,
            categories=cfg.data.categories,
            max_samples=cfg.data.get("max_samples"),
        )

    # Setup DDP Dataloader
    sampler = DistributedSampler(dataset) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg.data.num_workers,
    )

    # Initialize Engine
    runner = ExperimentRunner(cfg, local_rank=local_rank, world_size=world_size)

    # Run Generation Sweep
    runner.run(dataloader)

    # Cleanup DDP
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
