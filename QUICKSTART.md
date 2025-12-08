# 🚀 Quick Start Guide

Get Diffusion Detective running in 5 minutes!

## Prerequisites

- ✅ Python 3.10+
- ✅ Node.js 18+
- ✅ CUDA GPU with 8GB+ VRAM (recommended)
- ✅ OpenAI API Key

---

## Method 1: Automated Setup (Recommended)

```bash
# Clone or navigate to the project
cd Diffusion-Detective

# Run the setup script
./setup.sh

# Follow the instructions to:
# 1. Add your OpenAI API key to backend/.env
# 2. Start the backend
# 3. Start the frontend
```

---

## Method 2: Manual Setup

### Step 1: Backend Setup

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
nano .env  # Add your OPENAI_API_KEY

# Start server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Backend will be at:** http://localhost:8000
**API Docs:** http://localhost:8000/docs

### Step 2: Frontend Setup (New Terminal)

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

**Frontend will be at:** http://localhost:3000

---

## First Generation

1. Open http://localhost:3000
2. Enter a prompt: `"A majestic lion on a mountain peak at sunset"`
3. Toggle **Latent Steering Intervention** ON
4. Set strength to 1.0
5. Click **RUN ANALYSIS**
6. Wait ~60 seconds (first run downloads models)
7. Compare the natural vs. controlled images!

---

## Test the API Directly

```bash
curl -X POST http://localhost:8000/health

curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cyberpunk city at night",
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "intervention_active": true,
    "intervention_strength": 1.0,
    "seed": 42
  }'
```

---

## Troubleshooting

### "Import torch could not be resolved"
This is just a VS Code warning. The code will run fine when executed.

### "CUDA out of memory"
Reduce inference steps to 30 or restart the backend.

### "OpenAI API error"
The system will fall back to rule-based narratives if the API fails.

### Backend won't start
Make sure you're in the virtual environment:
```bash
source backend/venv/bin/activate
```

### Frontend shows "Failed to fetch"
Make sure the backend is running on port 8000.

---

## What's Next?

- 📖 Read the full [README.md](README.md)
- 🔧 Experiment with different prompts
- 🎨 Try different intervention strengths (0.0 - 2.0)
- 📊 Watch the attention analysis in real-time
- 🕵️ Read the detective's investigation reports

---

**Happy Investigating! 🔍**
