#!/usr/bin/env bash
# Setup script for Diffusion-Detective environment
set -euo pipefail

ENV_NAME=diffusion-detective

echo "Creating conda environment '${ENV_NAME}' using environment.yml..."
conda env create -f environment.yml || {
  echo "Conda env create failed - attempting fallback: create with python and pip installs"
  conda create -y -n ${ENV_NAME} python=3.10
  conda activate ${ENV_NAME}
  pip install -r requirements.txt
  exit 0
}

echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

echo "Upgrading pip and installing extras via pip..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Optional: install xformers (speeds attention) if compatible with your CUDA/PyTorch build."
echo "You can try: pip install xformers --prefer-binary"

echo "Setup complete. Next steps (manual):"
echo " 1) Login to Hugging Face: 'huggingface-cli login' and accept model license for SD2.1 if needed."
echo " 2) Download Stable Diffusion 2.1 weights via diffusers or HF Hub."
echo " 3) (Optional) Quantize Mistral-7B-Instruct with bitsandbytes following README instructions."

echo "Example quick tests:"
echo "python -c \"import torch; import diffusers; print(torch.__version__, torch.cuda.is_available())\""
