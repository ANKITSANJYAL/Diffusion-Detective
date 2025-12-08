# Diffusion Detective Backend

A production-grade interpretable Stable Diffusion API with attention extraction and latent steering.

## Setup

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Configure Environment**
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

3. **Run the Server**
```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or run directly:
```bash
python app/main.py
```

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Generate Image
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A majestic lion in a forest",
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "intervention_active": true,
    "intervention_strength": 1.0,
    "intervention_step_start": 40,
    "intervention_step_end": 20,
    "seed": 42
  }'
```

## Architecture

- **pipeline.py**: Custom Stable Diffusion pipeline with attention hooks and latent steering
- **narrator.py**: AI narrator service using GPT-4o-mini for generating investigation reports
- **utils.py**: Image processing utilities
- **main.py**: FastAPI application with REST endpoints

## Features

✅ Real-time attention extraction during generation
✅ Latent steering intervention (no heuristics)
✅ Sherlock Holmes-style narrative generation
✅ Memory-efficient GPU management
✅ Production-grade error handling
