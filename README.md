# Diffusion Detective

**When Does Attention Actually Do Anything?  
A Large-Scale Diagnostic of Cross-Attention in Text-to-Image Diffusion Models**


---

## Abstract

Cross-attention maps in text-to-image diffusion models are routinely treated as explanations for how text tokens influence image generation.  We test this assumption directly.  Across **16,302 controlled embedding-space interventions** on 680 COCO prompts using Stable Diffusion XL (SDXL), we find that attention maps are poor spatial predictors of where interventions take effect (SF-IoU-HR ≈ 0.17), and that the standard Attention-Delta Correlation (ADC) metric is partially tautological by construction.  We introduce two decoupled metrics : **Predictive ADC (P-ADC)** and **Latent-Delta Correlation (L-ADC)** : that remove this circularity.

The strongest finding is structural: *concrete-perceptual* attributes (color, material) concentrate target-concept attention roughly **3× more** than *abstract-stylistic* attributes (style, atmospheric effect), despite producing comparable image-level changes (LPIPS).  This asymmetry holds across all intervention strengths and temporal windows, pointing to a mechanistic difference in how SDXL's cross-attention processes these two attribute classes.

---

## Key Results

| Metric | Value | Interpretation |
|---|---|---|
| SF-IoU-HR (overall) | **0.17** | Cross-attention overlaps changed pixels only 17% of the time |
| P-ADC (all strengths) | **0.097 ± 0.60** | Statistically nonzero (*p* < 10⁻⁹⁰) but near-zero per-sample predictor |
| ACS dose–response | ***r* = −0.218** | Monotonic with α, saturates at α ≥ 1.5 (Cohen's *d* = 1.12) |
| Color ACS | **−0.306 ± 0.13** | Strong attention concentration |
| Material ACS | **−0.349 ± 0.16** | Strongest attention concentration |
| Style ACS | **−0.106 ± 0.18** | ~3× weaker than Color/Material |
| Effect ACS | **−0.098 ± 0.19** | ~3× weaker than Color/Material |
| LPIPS (Color vs Style) | **0.150 vs 0.138** | Comparable image change : asymmetry is not explained by output magnitude |

**SF-IoU-HR by temporal window:**  
Early (50–35): 0.156 · Mid (35–20): 0.170 · Late (20–5): 0.177 · Full (45–5): 0.177

**SF-IoU-HR by prompt category:**  
Compositional: 0.165 · Conflicting: 0.178 · Simple: 0.187

---

## Method

### Experimental Setup

- **Model:** `stabilityai/stable-diffusion-xl-base-1.0` (SDXL), 50 DDIM steps, guidance 7.5, seed 42, float16, NVIDIA V100 16 GB
- **Dataset:** 680 COCO 2017 captions across three prompt categories : *Simple* (50), *Compositional* (451), *Conflicting* (158)
- **Attributes:** 20 attributes in four semantic clusters : Color (5), Material (5), Style (5), Effect (5)
- **Conditions:** 6 intervention strengths × 4 temporal windows = **24 conditions per prompt × attribute pair**
- **Total interventions:** **16,302**

### Intervention Mechanism

Each intervention computes a CLIP embedding-arithmetic steering vector and injects it during denoising:

```python
delta = clip_encode(attribute) - clip_encode(concept)   # steering direction
latent_t += alpha * delta                                # applied within window [t_start, t_end]
```

No fine-tuning. Two-pass generation (baseline + intervention) per condition.

### Metrics

| Metric | Description |
|---|---|
| **ACS** | Attention Concentration Score : relative change in target-token attention mass |
| **ADC** | Attention-Delta Correlation : *partially tautological* (both quantities from same forward pass) |
| **P-ADC** *(ours)* | Predictive ADC : baseline-pass attention predicting intervention-induced change |
| **L-ADC** *(ours)* | Latent-Delta Correlation : baseline attention vs. latent L2-divergence |
| **SF-IoU / SF-IoU-HR** | DAAM-protocol IoU between attention heatmap and pixel-change mask (native / upsampled) |
| **LPIPS** | Perceptual image distance between baseline and intervention outputs |

---

## Repository Structure

```
Diffusion-Detective/
│
├── analysis/                          # Paper and figure generation
│   ├── mechinterp_workshop_2026.tex   # Main paper (LaTeX source)
│   ├── mechinterp_workshop_2026_refs.bib
│   ├── mechinterp_workshop_2026.pdf   # Compiled paper
│   ├── icml2026.sty                   # ICML 2026 style stub
│   ├── generate_paper_figures_v2.py   # Reproduces all figures from raw data
│   ├── paper_figures_v2/              # Final figures (PDF + PNG) and CSV tables
│   └── paper_qualitative/             # Qualitative example images
│
├── experiments/                       # Experiment runner and evaluation
│   ├── run_experiment.py              # Main Hydra-based sweep runner (DDP)
│   ├── reproduce.py                   # Single-run reproducer
│   ├── validate_new_metrics.py        # Metric validation suite
│   ├── requirements.txt
│   ├── configs/                       # Hydra YAML configs
│   │   ├── base.yaml
│   │   ├── full_experiment.yaml       # Full 16 302-run config
│   │   ├── strength_sweep.yaml
│   │   ├── window_sweep.yaml
│   │   └── ablation.yaml
│   ├── src/
│   │   ├── engine/                    # SDXL pipeline + attention hooks
│   │   ├── metrics/                   # ACS, ADC, P-ADC, L-ADC, SF-IoU
│   │   ├── evaluator/                 # Per-run evaluation logic
│   │   └── data/                      # COCO prompt loading
│   └── results/                       # Raw JSONL outputs (git-ignored)
│
├── backend/                           # Interactive demo (FastAPI + React)
│   ├── app/
│   │   ├── main.py                    # FastAPI endpoints
│   │   ├── pipeline.py                # Custom SDXL pipeline with hooks
│   │   └── narrator.py                # GPT-4o-mini explanation service
│   └── requirements.txt
│
├── frontend/                          # Demo UI
│   ├── src/
│   └── package.json
│
├── setup.sh                           # Environment setup script
├── run_full_ablation.sh               # Launch full experiment sweep
└── README.md
```

---

## Reproducing the Paper

### 1. Environment

```bash
bash setup.sh          # creates .venv, installs experiments/requirements.txt
source .venv/bin/activate
```

Or manually:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r experiments/requirements.txt
```

### 2. Run the Full Experiment

```bash
# Full 16 302-intervention sweep (requires ~360 GPU-hours on V100)
bash run_full_ablation.sh

# Quick smoke test (50 interventions, ~5 min)
python experiments/run_experiment.py --config-name smoke_test
```

Results are written to `experiments/results/<run_name>/aggregated_metrics.jsonl`.

### 3. Reproduce Figures and Tables

```bash
cd analysis
python generate_paper_figures_v2.py \
    --results ../experiments/results/unified-ablation-v2_2026-04-13_00-19-07/aggregated_metrics.jsonl \
    --outdir paper_figures_v2/
```

Outputs: 11 PDF/PNG figures + 4 CSV tables in `analysis/paper_figures_v2/`.

### 4. Compile the Paper

```bash
cd analysis
pdflatex mechinterp_workshop_2026
bibtex   mechinterp_workshop_2026
pdflatex mechinterp_workshop_2026
pdflatex mechinterp_workshop_2026
# → mechinterp_workshop_2026.pdf
```

Requires a standard TeX Live installation (natbib, geometry, times, booktabs, subcaption, placeins : all in TeX Live 2022+).

---

## Interactive Demo (Optional)

The `backend/` and `frontend/` directories contain a web demo for exploring interventions interactively.

```bash
# Backend
cd backend && pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY if you want LLM narration
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend && npm install && npm run dev
# → http://localhost:3000
```

---

## Dependencies

**Experiments** (`experiments/requirements.txt`)
- Python ≥ 3.10, PyTorch ≥ 2.1, CUDA 11.8+
- `diffusers`, `transformers`, `accelerate`, `hydra-core`
- `torchmetrics`, `lpips`, `scipy`, `pandas`, `matplotlib`, `seaborn`

**Demo backend** (`backend/requirements.txt`)
- FastAPI, uvicorn, diffusers, transformers, openai

**Demo frontend**
- Node 18+, React 18, Vite, Tailwind CSS

---

## Citation



---

## License

MIT License : see [`LICENSE`](LICENSE) for details.

---

## Acknowledgements

This work builds on:
[Rombach et al. 2022](https://arxiv.org/abs/2112.10752) (LDM) ·
[Podell et al. 2023](https://arxiv.org/abs/2307.01952) (SDXL) ·
[Tang et al. 2023](https://arxiv.org/abs/2305.04543) (DAAM) ·
[Hertz et al. 2023](https://arxiv.org/abs/2208.01626) (Prompt-to-Prompt) ·
[Radford et al. 2021](https://arxiv.org/abs/2103.00020) (CLIP) ·
HuggingFace Diffusers · the COCO dataset team
