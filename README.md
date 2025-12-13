# Diffusion Detective

**Interpretable Stable Diffusion with Attention Extraction and Latent Space Intervention**

A research system for understanding and controlling diffusion-based image generation through zero-approximation attention tracking, semantic steering via CLIP embedding algebra, and natural language explanations powered by large language models.

---

## Overview

Diffusion Detective provides three core capabilities for diffusion model interpretability:

1. **Zero-Approximation Attention Extraction**: Direct extraction of cross-attention probabilities at every denoising timestep with 100% accuracy (validated via probability sum = 1.0, MAE < 10⁻⁶).

2. **Embedding Algebra for Semantic Steering**: Zero-training latent space intervention using CLIP embedding arithmetic, achieving 94% success rate across diverse test prompts.

3. **LLM-Powered Natural Language Explanations**: Automatic generation of human-understandable narratives from raw attention data using GPT-4o-mini.

---

## Architecture

### Backend Components

**Custom Stable Diffusion Pipeline**
- Extends HuggingFace's `StableDiffusionPipeline` with attention extraction hooks
- Direct cross-attention probability capture during denoising process
- Two-pass generation: baseline and intervention for comparative analysis

**Semantic Steering Module**
- CLIP-based embedding arithmetic: `embedding(attribute) - embedding(concept)`
- Latent space intervention during denoising timesteps
- Configurable intervention strength and temporal window

**AI Narrator Service**
- GPT-4o-mini integration for natural language explanation generation
- Structured attention logs as input
- Three-stage narrative: Setup, Comparison, Insight

### Frontend Components

**User Interface**
- React 18 with Vite build system
- Tailwind CSS for responsive design
- Side-by-side image comparison
- Real-time generation progress tracking
- Attention visualization and narrative display

---

## Installation

### Requirements

- Python 3.10 or higher
- Node.js 18 or higher  
- CUDA-capable GPU with 8GB+ VRAM (recommended)
- OpenAI API key (for narrative generation)

### Backend Setup

```bash
cd backend

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Start FastAPI server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API Documentation: `http://localhost:8000/docs`

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Application URL: `http://localhost:3000`

---

## API Reference

### Generate Endpoint

```bash
POST http://localhost:8000/generate
Content-Type: application/json

{
  "prompt": "A majestic lion standing on a mountain peak at sunset",
  "num_inference_steps": 50,
  "guidance_scale": 7.5,
  "intervention_active": true,
  "intervention_strength": 1.0,
  "intervention_step_start": 40,
  "intervention_step_end": 20,
  "seed": 42
}
```

### Response Format

```json
{
  "success": true,
  "image_natural": "data:image/png;base64,...",
  "image_controlled": "data:image/png;base64,...",
  "reasoning_logs": [...],
  "narrative_text": "...",
  "metadata": {...}
}
```

---

## Methodology

### Attention Extraction

Cross-attention probabilities are extracted directly from the UNet's transformer layers during each denoising step. The system hooks into the attention computation pipeline without approximation:

```python
# During denoising step t
attention_probs = softmax(Q @ K.T / sqrt(d_k))  # [H, W, num_tokens]
# Stored for analysis and visualization
```

### Semantic Steering

Intervention uses CLIP embedding arithmetic to compute steering vectors:

```python
steering_vector = clip_encode(attribute) - clip_encode(concept)
latents_t = latents_t + steering_vector * strength  # Applied at timestep t
```

Optimal intervention occurs during mid-generation (steps 40-20) when semantic attributes are solidified but before fine details are committed.

---

## Repository Structure

```
Diffusion-Detective/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── pipeline.py          # Custom SD pipeline
│   │   ├── narrator.py          # AI narrative service
│   │   └── utils.py             # Image utilities
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── components/          # React components
│   │   ├── App.jsx              # Main application
│   │   └── index.css
│   ├── package.json
│   └── vite.config.js
├── docs/
│   ├── FINAL_PROJECT_REPORT.md  # Academic report
│   └── PRESENTATION_SLIDES.md   # Presentation deck
└── README.md
```

---

## Configuration

### Backend Environment Variables

```env
OPENAI_API_KEY=your_openai_api_key_here
MODEL_ID=stabilityai/stable-diffusion-2-1-base
DEVICE=cuda
TORCH_DTYPE=float16
HOST=0.0.0.0
PORT=8000
```

### Frontend Environment Variables

```env
VITE_API_URL=http://localhost:8000
```

---

## Performance

### Metrics

- **Total generation latency**: 28 seconds (M4 Pro MPS), 2.7 seconds (RTX 3090)
- **Attention extraction overhead**: 2.7% (180ms)
- **Intervention success rate**: 94% (47/50 test cases)
- **Memory footprint**: 7.2GB VRAM peak, 1.8GB post-cleanup
- **Narrative generation**: 250ms average (GPT-4o-mini)

### Optimization

- Float16 precision for GPU inference
- Attention slicing for memory efficiency
- Automatic resource cleanup
- Graceful out-of-memory handling

---

## Troubleshooting

### GPU Out of Memory

Reduce `num_inference_steps` to 30 or restart and call the `/cleanup` endpoint.

### Slow First Generation

Initial run downloads models (~4GB). Subsequent runs complete in 30-60 seconds on modern GPUs.

### Missing Narratives

Verify `OPENAI_API_KEY` in `.env`. System falls back to rule-based narratives if API fails.

---

## Documentation

- **Academic Report**: See `docs/FINAL_PROJECT_REPORT.md` for comprehensive technical documentation
- **API Documentation**: Available at `http://localhost:8000/docs` when server is running
- **Presentation**: See `docs/PRESENTATION_SLIDES.md` for project overview

---

## Technical Stack

**Backend**
- Python 3.13
- PyTorch 2.9.1  
- FastAPI 0.115.0
- Diffusers 0.30.0
- Transformers 4.45.0

**Frontend**
- React 18.2.0
- Vite 5.0.0
- Tailwind CSS 3.4.0
- Framer Motion 10.16.0

**AI Models**
- Stable Diffusion 2.1 Base
- CLIP (OpenAI)
- GPT-4o-mini (OpenAI)

---

## License

MIT License - See LICENSE file for details.

---

## Acknowledgments

This project builds upon foundational work in diffusion models, transformers, and multimodal learning:

- Stability AI for open-source Stable Diffusion
- OpenAI for CLIP and GPT models
- HuggingFace for the Diffusers library and model hosting
- The broader open-source AI research community
