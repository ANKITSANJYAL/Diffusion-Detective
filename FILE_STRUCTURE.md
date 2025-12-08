# 📁 Diffusion Detective - Complete File Structure

```
Diffusion-Detective/
│
├── 📄 README.md                    # Main project documentation
├── 📄 QUICKSTART.md                # 5-minute setup guide
├── 📄 DEVELOPMENT.md               # Developer's handbook
├── 📄 TESTING.md                   # Testing procedures
├── 📄 PROJECT_SUMMARY.md           # Executive summary
├── 📄 LICENSE                      # MIT License
├── 🔧 setup.sh                     # Automated setup script (executable)
├── 📄 .gitignore                   # Git ignore patterns
│
├── 🐍 backend/                     # Python FastAPI Backend
│   ├── 📦 app/                     # Application package
│   │   ├── 📄 __init__.py          # Package initialization
│   │   ├── 🚀 main.py              # FastAPI application (routes, lifecycle)
│   │   ├── 🧠 pipeline.py          # Custom Stable Diffusion pipeline
│   │   ├── 📖 narrator.py          # AI narrative generation service
│   │   └── 🛠️ utils.py              # Image processing utilities
│   │
│   ├── 📄 requirements.txt         # Python dependencies
│   ├── 📄 .env.example             # Environment variables template
│   ├── 📄 .gitignore               # Backend-specific ignores
│   └── 📄 README.md                # Backend documentation
│
└── ⚛️ frontend/                    # React + Vite Frontend
    ├── 📦 src/                     # Source code
    │   ├── 🎨 main.jsx             # React entry point
    │   ├── 🏠 App.jsx              # Main application component
    │   ├── 🎭 index.css            # Global styles + Tailwind
    │   │
    │   └── 📦 components/          # React components
    │       ├── 🎛️ ControlPanel.jsx  # Parameter controls
    │       ├── 📊 Timeline.jsx      # Progress visualization
    │       ├── 💻 Terminal.jsx      # Log display with effects
    │       └── 🖼️ ComparisonSlider.jsx # Image comparison slider
    │
    ├── 📄 index.html               # HTML template
    ├── 📄 package.json             # Node.js dependencies
    ├── 📄 vite.config.js           # Vite configuration
    ├── 📄 tailwind.config.js       # Tailwind CSS configuration
    ├── 📄 postcss.config.js        # PostCSS configuration
    ├── 📄 jsconfig.json            # JavaScript configuration
    ├── 📄 .env.example             # Frontend environment template
    ├── 📄 .gitignore               # Frontend-specific ignores
    └── 📄 README.md                # Frontend documentation
```

---

## 📊 File Statistics

### Backend (Python)
- **Total Files:** 10
- **Python Files:** 4 (main.py, pipeline.py, narrator.py, utils.py)
- **Config Files:** 3 (requirements.txt, .env.example, .gitignore)
- **Documentation:** 1 (README.md)

### Frontend (React)
- **Total Files:** 14
- **JavaScript/JSX Files:** 7 (main.jsx, App.jsx, 4 components)
- **Config Files:** 6 (package.json, vite.config.js, tailwind.config.js, etc.)
- **Documentation:** 1 (README.md)

### Root Level
- **Documentation:** 5 (README.md, QUICKSTART.md, DEVELOPMENT.md, TESTING.md, PROJECT_SUMMARY.md)
- **Scripts:** 1 (setup.sh)
- **License:** 1 (LICENSE)

### Total Project
- **Total Files:** ~35 (excluding node_modules, venv, .git)
- **Lines of Code:** ~3,500+ (estimated)
- **Documentation:** ~5,000+ words

---

## 🔍 File Descriptions

### Root Documentation

| File | Purpose | Size |
|------|---------|------|
| README.md | Main project overview, architecture, quick start | Large |
| QUICKSTART.md | 5-minute setup guide for new users | Small |
| DEVELOPMENT.md | Developer guide with architecture details | Large |
| TESTING.md | Testing procedures and validation | Medium |
| PROJECT_SUMMARY.md | Executive summary and achievements | Large |
| LICENSE | MIT License text | Small |

### Backend Files

#### Core Application (app/)

| File | Purpose | LOC | Key Classes/Functions |
|------|---------|-----|----------------------|
| main.py | FastAPI server with routes | ~280 | GenerationRequest, GenerationResponse, generate_image() |
| pipeline.py | Custom SD pipeline | ~420 | InterpretableSDPipeline, CustomAttentionProcessor, AttentionStore |
| narrator.py | AI narrative generation | ~180 | NarratorService, generate_narrative() |
| utils.py | Image processing | ~110 | pil_to_base64(), base64_to_pil() |

#### Configuration

| File | Purpose |
|------|---------|
| requirements.txt | Python packages (FastAPI, PyTorch, Diffusers, etc.) |
| .env.example | Environment variables template |

### Frontend Files

#### Source Code (src/)

| File | Purpose | LOC | Key Components |
|------|---------|-----|---------------|
| main.jsx | React entry point | ~10 | ReactDOM.render |
| App.jsx | Main app component | ~180 | State management, API calls |
| index.css | Global styles | ~100 | Tailwind utilities, custom classes |

#### Components (src/components/)

| File | Purpose | LOC | Props |
|------|---------|-----|-------|
| ControlPanel.jsx | Parameter inputs | ~230 | onGenerate, isGenerating |
| Timeline.jsx | Progress bar | ~90 | progress, totalSteps, interventionRange |
| Terminal.jsx | Log display | ~110 | logs, isActive |
| ComparisonSlider.jsx | Image comparison | ~130 | naturalImage, controlledImage, metadata |

#### Configuration

| File | Purpose |
|------|---------|
| package.json | Dependencies (React, Vite, Tailwind, Framer Motion) |
| vite.config.js | Vite bundler configuration |
| tailwind.config.js | Custom colors, animations, fonts |
| postcss.config.js | PostCSS plugins |

---

## 🎯 Critical Files

These files are the core of the system:

1. **backend/app/pipeline.py** - The brain of the operation
   - Custom attention extraction
   - Latent steering implementation
   - Real-time log generation

2. **backend/app/main.py** - The API gateway
   - RESTful endpoints
   - Request/response handling
   - Error management

3. **frontend/src/App.jsx** - The UI orchestrator
   - State management
   - API integration
   - Component coordination

4. **frontend/src/components/ComparisonSlider.jsx** - The showcase
   - Interactive image comparison
   - Results presentation

---

## 🚦 File Dependencies

### Backend Dependencies

```
main.py
├── pipeline.py (InterpretableSDPipeline)
├── narrator.py (NarratorService)
└── utils.py (pil_to_base64)
```

### Frontend Dependencies

```
App.jsx
├── ControlPanel.jsx
├── Timeline.jsx
├── Terminal.jsx
└── ComparisonSlider.jsx
```

---

## 📦 External Dependencies

### Backend (requirements.txt)

```
fastapi             # Web framework
uvicorn            # ASGI server
torch              # Deep learning
diffusers          # Stable Diffusion
transformers       # NLP models
openai             # GPT-4o-mini
pillow             # Image processing
```

### Frontend (package.json)

```
react              # UI framework
framer-motion      # Animations
react-compare-image # Slider component
axios              # HTTP client
tailwindcss        # Styling
vite               # Build tool
```

---

## 🔒 Important Files for Production

### Backend

- ✅ `.env` - Must configure OPENAI_API_KEY
- ✅ `requirements.txt` - Exact versions for reproducibility

### Frontend

- ✅ `.env` - Must configure VITE_API_URL for production
- ✅ `dist/` - Built files for deployment (created by `npm run build`)

---

## 🗂️ Files by Category

### Code Files (20)
```
Python:     4 files  (~900 LOC)
JavaScript: 7 files  (~750 LOC)
CSS:        1 file   (~100 LOC)
HTML:       1 file   (~20 LOC)
```

### Config Files (9)
```
Python:     requirements.txt, .env.example
Node:       package.json, vite.config.js, tailwind.config.js, 
            postcss.config.js, jsconfig.json
Git:        .gitignore (x2)
```

### Documentation (7)
```
Main:       README.md
Guides:     QUICKSTART.md, DEVELOPMENT.md, TESTING.md
Other:      PROJECT_SUMMARY.md, LICENSE, backend/README.md, 
            frontend/README.md
```

### Scripts (1)
```
Setup:      setup.sh
```

---

## 📏 Code Metrics

| Metric | Backend | Frontend | Total |
|--------|---------|----------|-------|
| Files | 10 | 14 | 24 |
| LOC (Code) | ~900 | ~750 | ~1,650 |
| LOC (Config) | ~50 | ~200 | ~250 |
| LOC (Docs) | ~300 | ~200 | ~500 |
| **Total** | **~1,250** | **~1,150** | **~2,400** |

*Note: Excluding dependencies (node_modules, venv)*

---

## 🎨 Technology Stack by File

### Backend Stack
```
main.py           → FastAPI, Pydantic, Uvicorn
pipeline.py       → PyTorch, Diffusers, Transformers
narrator.py       → OpenAI API
utils.py          → Pillow, NumPy
```

### Frontend Stack
```
App.jsx           → React, Axios, Framer Motion
ControlPanel.jsx  → React, Framer Motion
Timeline.jsx      → React, Framer Motion
Terminal.jsx      → React, Framer Motion
ComparisonSlider.jsx → React, react-compare-image
index.css         → Tailwind CSS
```

---

## 🔧 Generated Files (Not in Repo)

These files are created during setup/runtime:

### Backend
```
venv/              # Virtual environment (10,000+ files)
__pycache__/       # Python bytecode
.env               # Environment variables (from .env.example)
```

### Frontend
```
node_modules/      # Dependencies (50,000+ files)
dist/              # Production build
.env               # Environment variables (from .env.example)
```

---

## 🌳 Complete Tree Visualization

```
Diffusion-Detective/
│
├── 📚 Documentation Layer
│   ├── README.md (Main)
│   ├── QUICKSTART.md (Setup)
│   ├── DEVELOPMENT.md (Dev Guide)
│   ├── TESTING.md (Testing)
│   └── PROJECT_SUMMARY.md (Summary)
│
├── 🐍 Backend Layer (Python + FastAPI)
│   └── app/
│       ├── Core: main.py
│       ├── AI: pipeline.py, narrator.py
│       └── Utils: utils.py
│
├── ⚛️ Frontend Layer (React + Vite)
│   └── src/
│       ├── Core: main.jsx, App.jsx
│       ├── Styles: index.css
│       └── components/
│           ├── Input: ControlPanel.jsx
│           ├── Viz: Timeline.jsx, Terminal.jsx
│           └── Output: ComparisonSlider.jsx
│
└── 🔧 Configuration Layer
    ├── Setup: setup.sh
    ├── License: LICENSE
    └── Env: .env.example (x2)
```

---

**Total Project Complexity: Medium-High**
**Maintainability: High**
**Documentation Coverage: Excellent**

🔍 *"Every file serves a purpose. No file is wasted."* 🕵️
