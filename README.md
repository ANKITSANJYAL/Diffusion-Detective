# Diffusion Detective — Environment & Setup

This repository contains the project proposal and environment setup for the Diffusion Detective project.

This README documents how to create a reproducible conda environment and prepare local models for a 2-GPU cluster with CUDA 11.8.

Prerequisites
- Conda (Anaconda/Miniconda) available on the cluster
- CUDA toolkit 11.8 (module `cuda11.8/toolkit/11.8.0` or system CUDA)
- git and internet access to download models from Hugging Face

1) Create the environment (Python 3.10)

Run in a bash shell:

```bash
module load cuda11.8/toolkit/11.8.0  # if your cluster uses module system
conda env create -f environment.yml
conda activate diffusion-detective
pip install -r requirements.txt
```

2) Verify basic install

```bash
python -c "import torch, diffusers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

3) Recommended model downloads

- Stable Diffusion 2.1 (Diffusers):
  - Use the `diffusers` API to download at runtime or pre-cache with `huggingface_hub`.
- Mistral-7B-Instruct: download via HF and quantize with `bitsandbytes` to 4-bit for multi-GPU usage.

4) Notes for a 2-GPU setup

- Place SD2.1 on GPU0 and the quantized LLM on GPU1, or use `accelerate` to split layers across devices.
- Use `torch.cuda.set_device` in scripts where necessary; prefer `accelerate` for model launching.

5) Optional performance packages

- If available, install `xformers` for faster attention ops: `pip install xformers --prefer-binary`.
- Use `safetensors` for weights where supported.

6) Next steps (development)

- Implement LLM reasoning generator module (Mistral-7B-Instruct 4-bit)
- Implement SD2.1 harness capturing cross-attention and U-Net activations
- Implement small fusion head (CLIP embeddings -> evidence maps)
- Build Gradio demo to visualize stepwise evolution

If you want, I can now run the environment creation script on the cluster and confirm everything installs cleanly. Reply `go` to proceed.
