# 🔍 Diffusion Detective

**An Interpretable, Intervene-able Stable Diffusion Interface**

A production-grade system for understanding and manipulating AI image generation through attention extraction and latent steering. Built with zero mock data—every generation runs the full diffusion loop in real-time.

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![React](https://img.shields.io/badge/react-18.2+-blue)

---

## 🎯 Project Vision

Diffusion Detective turns Stable Diffusion into an **investigable system**. Unlike typical image generation tools, this project:

- ✅ **Extracts Attention Maps** at every timestep (no approximations)
- ✅ **Implements Latent Steering** for real interventions (no random seed tricks)
- ✅ **Generates AI Narratives** that explain the generation process
- ✅ **Provides Side-by-Side Comparison** of natural vs. controlled outputs
- ✅ **Runs 100% Live** (no pre-baked images or mock data)

---

## 🏗️ Architecture

### Backend (Python + FastAPI)
- **Custom Pipeline**: `InterpretableSDPipeline` extends HuggingFace's `StableDiffusionPipeline`
- **Attention Hooks**: Custom `AttentionProcessor` intercepts cross-attention probabilities
- **Latent Steering**: Direct manipulation of latent embeddings during denoising
- **AI Narrator**: GPT-4o-mini generates Sherlock Holmes-style investigation reports

### Frontend (React + Vite)
- **Cyberpunk Theme**: Dark mode with neon green accents (#00FF41)
- **Mission Control**: Parameter controls with real-time validation
- **Terminal Log**: Scrolling investigation logs with keyword highlighting
- **Comparison Slider**: Interactive split-view of natural vs. controlled images
- **Timeline**: Visual progress bar showing intervention zones

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- OpenAI API key (for narrative generation)

### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run the server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`
- Docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Configure environment (optional)
cp .env.example .env

# Run development server
npm run dev
```

The app will be available at `http://localhost:3000`

---

## 📡 API Usage

### Generate Image

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A majestic lion standing on a mountain peak at sunset",
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "intervention_active": true,
    "intervention_strength": 1.0,
    "intervention_step_start": 40,
    "intervention_step_end": 20,
    "seed": 42
  }'
```

### Response Structure

```json
{
  "success": true,
  "image_natural": "data:image/png;base64,...",
  "image_controlled": "data:image/png;base64,...",
  "reasoning_logs": [
    "Step 50: Focus on lion, mountain, sunset",
    "Step 40: Applying latent steering (strength: 1.0)",
    ...
  ],
  "narrative_text": "🔍 Elementary, my dear Watson! The model's attention...",
  "metadata": {
    "prompt": "...",
    "num_inference_steps": 50,
    "intervention_active": true,
    ...
  }
}
```

---

## 🧪 How Latent Steering Works

Unlike traditional methods that rely on random seeds or external guidance, **Latent Steering** directly manipulates the diffusion process:

1. **Extract Prompt Embeddings**: Convert the text prompt into a semantic vector
2. **Create Intervention Vector**: Derive a steering direction from the embeddings
3. **Apply During Denoising**: Add the intervention vector to latents at specified timesteps
4. **No Heuristics**: Pure mathematical manipulation of the latent space

```python
# Pseudocode
if intervention_active and intervention_step_end <= current_step <= intervention_step_start:
    latents = latents + intervention_vector * strength
```

This creates **reproducible, interpretable changes** without relying on random chance.

---

## 🎨 Frontend Features

### Control Panel
- Text prompt input
- Inference steps slider (20-100)
- Guidance scale slider (1.0-20.0)
- Intervention toggle with strength control
- Intervention step range configuration
- Optional seed for reproducibility

### Timeline
- Real-time progress visualization
- Intervention zone highlighted in red
- Current step indicator

### Terminal
- Live log streaming
- Keyword highlighting (Focus, Shape, Color, Intervention)
- Auto-scroll with typewriter effect
- Sherlock Holmes-style narrative

### Comparison Slider
- Interactive split-view comparison
- Hover to preview both images
- Download buttons for both outputs
- Metadata display

---

## 📂 Project Structure

```
Diffusion-Detective/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application
│   │   ├── pipeline.py          # Custom SD pipeline
│   │   ├── narrator.py          # AI narrative service
│   │   └── utils.py             # Image utilities
│   ├── requirements.txt
│   ├── .env.example
│   └── README.md
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ControlPanel.jsx
│   │   │   ├── Timeline.jsx
│   │   │   ├── Terminal.jsx
│   │   │   └── ComparisonSlider.jsx
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── README.md
└── README.md                    # This file
```

---

## 🔧 Configuration

### Backend Environment Variables

```env
OPENAI_API_KEY=your_openai_api_key_here
MODEL_ID=runwayml/stable-diffusion-v1-5
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

## 🎯 Technical Highlights

### No Mock Data
Every image generation runs the full 50-step diffusion loop. Attention maps are extracted in real-time. No pre-computed results.

### Memory Efficient
- Float16 precision on GPU
- Attention slicing enabled
- Automatic memory cleanup
- Graceful OOM handling

### Type-Safe
- Pydantic models for API validation
- Comprehensive error handling
- Clear response schemas

### Production-Ready
- Async FastAPI endpoints
- CORS middleware
- Health check endpoints
- Resource cleanup on shutdown

---

## 🐛 Troubleshooting

### GPU Out of Memory
- Reduce `num_inference_steps` to 30
- Restart and call `/cleanup` endpoint
- Check GPU memory: `nvidia-smi`

### Slow Generation
- First run downloads models (~4GB)
- Subsequent runs should take 30-60s on modern GPUs

### Narrative Not Generating
- Check `OPENAI_API_KEY` in `.env`
- System will fall back to rule-based narrative if API fails

---

## 🚧 Future Enhancements

- [ ] WebSocket streaming for real-time updates
- [ ] Multiple model support (SD v2.1, SDXL)
- [ ] Attention map visualization
- [ ] Custom intervention vector upload
- [ ] Batch generation
- [ ] LoRA integration
- [ ] ControlNet support

---

## 📄 License

MIT License - See LICENSE file for details

---

## 👨‍💻 Author

Built by a Senior AI Engineer specializing in interpretable AI systems.

**Tech Stack:**
- Backend: Python, FastAPI, PyTorch, Diffusers
- Frontend: React, Vite, Tailwind, Framer Motion
- AI: Stable Diffusion v1.5, GPT-4o-mini

---

## 🙏 Acknowledgments

- HuggingFace for the Diffusers library
- RunwayML for Stable Diffusion v1.5
- OpenAI for GPT-4o-mini
- The open-source community

---

**"Elementary, my dear Watson! The game is afoot!"** 🕵️
