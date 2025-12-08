# 🛠️ Development Guide

A comprehensive guide for developers working on Diffusion Detective.

---

## Development Environment Setup

### Backend Development

```bash
cd backend

# Activate virtual environment
source venv/bin/activate

# Install dev dependencies
pip install black flake8 pytest pytest-asyncio

# Run with auto-reload
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or use the direct method
python app/main.py
```

### Frontend Development

```bash
cd frontend

# Install dependencies
npm install

# Run dev server with hot reload
npm run dev

# Run linter
npm run lint

# Build for production
npm run build

# Preview production build
npm run preview
```

---

## Code Architecture

### Backend Architecture

```
backend/app/
├── __init__.py          # Package initialization
├── main.py              # FastAPI app, routes, lifecycle
├── pipeline.py          # Custom SD pipeline with hooks
├── narrator.py          # AI narrative generation
└── utils.py             # Image processing utilities
```

#### Key Classes

**InterpretableSDPipeline**
- Extends HuggingFace's StableDiffusionPipeline
- Implements attention extraction via custom processors
- Performs latent steering during denoising loop
- Manages GPU memory efficiently

**AttentionStore**
- Stores attention maps per timestep
- Analyzes attention patterns
- Generates reasoning logs

**CustomAttentionProcessor**
- Intercepts attention computation
- Stores cross-attention probabilities
- Compatible with Diffusers architecture

**NarratorService**
- Interfaces with OpenAI API
- Generates Sherlock Holmes-style narratives
- Falls back to rule-based generation

### Frontend Architecture

```
frontend/src/
├── main.jsx                    # React entry point
├── App.jsx                     # Main application component
├── index.css                   # Global styles + Tailwind
└── components/
    ├── ControlPanel.jsx        # Parameter controls
    ├── Timeline.jsx            # Progress visualization
    ├── Terminal.jsx            # Log display
    └── ComparisonSlider.jsx    # Image comparison
```

#### Component Hierarchy

```
App
├── ControlPanel (handles form state)
├── Timeline (shows progress during generation)
├── Terminal (displays streaming logs)
└── ComparisonSlider (shows results)
```

#### State Management

- Local state with `useState`
- Props passed down to children
- Callback functions for events
- No global state management (intentionally simple)

---

## Adding New Features

### Backend: Add a New Endpoint

```python
# backend/app/main.py

@app.post("/new-endpoint", tags=["Custom"])
async def new_endpoint(param: str):
    """
    Description of what this endpoint does.
    """
    try:
        # Your logic here
        return {"result": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Backend: Modify Pipeline

```python
# backend/app/pipeline.py

def generate(self, ...):
    # Add your custom logic
    # Example: Custom scheduler
    from diffusers import DDIMScheduler
    self.pipeline.scheduler = DDIMScheduler.from_config(
        self.pipeline.scheduler.config
    )
    
    # Continue with generation...
```

### Frontend: Add a New Component

```jsx
// frontend/src/components/NewComponent.jsx

import { motion } from 'framer-motion'

const NewComponent = ({ data }) => {
  return (
    <motion.div
      className="terminal-window"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      <h3 className="text-lg font-bold text-neon-green">
        New Feature
      </h3>
      {/* Your content */}
    </motion.div>
  )
}

export default NewComponent
```

Then import in `App.jsx`:
```jsx
import NewComponent from './components/NewComponent'

// In the JSX:
<NewComponent data={someData} />
```

---

## Styling Guidelines

### CSS Classes (Tailwind)

Use these predefined classes for consistency:

- `terminal-window` - Dark panel with neon border
- `cyber-button` - Neon green button with hover effects
- `neon-glow` - Text shadow effect
- `scanline` - Adds scanning effect overlay

### Color Palette

```javascript
// tailwind.config.js
colors: {
  'cyber-black': '#050505',    // Background
  'cyber-dark': '#0a0a0a',     // Panel background
  'cyber-gray': '#1a1a1a',     // Darker panels
  'neon-green': '#00FF41',     // Primary accent
  'neon-red': '#FF0055',       // Danger/intervention
  'neon-blue': '#00D9FF',      // Info
  'neon-purple': '#BD00FF',    // Secondary
}
```

### Animation Standards

Use Framer Motion for all animations:

```jsx
<motion.div
  initial={{ opacity: 0, y: 20 }}
  animate={{ opacity: 1, y: 0 }}
  transition={{ duration: 0.6 }}
>
  Content
</motion.div>
```

---

## API Design Patterns

### Request/Response Models

Always use Pydantic models:

```python
from pydantic import BaseModel, Field

class MyRequest(BaseModel):
    param: str = Field(..., description="Parameter description")
    optional: int = Field(default=10, ge=1, le=100)

class MyResponse(BaseModel):
    success: bool
    data: dict
```

### Error Handling

```python
from fastapi import HTTPException

try:
    # Your code
    pass
except SpecificError as e:
    raise HTTPException(status_code=400, detail=f"Specific error: {e}")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
```

---

## Performance Optimization

### Backend

1. **Memory Management**
   ```python
   # Clear CUDA cache after heavy operations
   torch.cuda.empty_cache()
   
   # Use float16 for faster computation
   pipeline.to(dtype=torch.float16)
   
   # Enable attention slicing
   pipeline.enable_attention_slicing()
   ```

2. **Async Operations**
   ```python
   @app.post("/async-endpoint")
   async def async_operation():
       # Use async/await for I/O operations
       result = await some_async_function()
       return result
   ```

### Frontend

1. **Lazy Loading**
   ```jsx
   const HeavyComponent = lazy(() => import('./HeavyComponent'))
   
   <Suspense fallback={<Loading />}>
     <HeavyComponent />
   </Suspense>
   ```

2. **Memoization**
   ```jsx
   const ExpensiveComponent = memo(({ data }) => {
     // Expensive rendering logic
   })
   ```

---

## Debugging Tips

### Backend Debugging

1. **Enable verbose logging**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Inspect attention maps**
   ```python
   # In pipeline.py, add:
   print(f"Attention shape: {attention_probs.shape}")
   print(f"Attention values: {attention_probs.mean()}")
   ```

3. **GPU memory tracking**
   ```python
   print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
   ```

### Frontend Debugging

1. **React DevTools**
   - Install React Developer Tools extension
   - Inspect component state and props

2. **Network inspection**
   - Open browser DevTools (F12)
   - Network tab → Filter by XHR
   - Check API request/response

3. **Console logging**
   ```jsx
   console.log('Current state:', { isGenerating, results })
   ```

---

## Git Workflow

### Branch Strategy

```bash
main           # Production-ready code
├── develop    # Integration branch
    ├── feature/attention-viz
    ├── feature/websocket-stream
    └── bugfix/memory-leak
```

### Commit Messages

Follow conventional commits:

```
feat: add attention map visualization
fix: resolve GPU memory leak
docs: update API documentation
style: format code with black
refactor: simplify attention processing
test: add pipeline unit tests
```

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tested locally
- [ ] All tests pass
- [ ] Added new tests

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
```

---

## Environment Variables

### Backend (.env)

```env
# Required
OPENAI_API_KEY=sk-...

# Optional (with defaults)
MODEL_ID=runwayml/stable-diffusion-v1-5
DEVICE=cuda
TORCH_DTYPE=float16
HOST=0.0.0.0
PORT=8000

# For development
DEBUG=true
LOG_LEVEL=DEBUG
```

### Frontend (.env)

```env
# API endpoint
VITE_API_URL=http://localhost:8000

# For production
# VITE_API_URL=https://your-backend.com
```

---

## Deployment Considerations

### Backend Deployment

**Requirements:**
- Python 3.10+
- CUDA-capable GPU
- 16GB+ RAM
- 20GB+ storage (for models)

**Dockerfile example:**
```dockerfile
FROM python:3.10-slim

# Install CUDA dependencies
RUN apt-get update && apt-get install -y \
    cuda-toolkit-11-8

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Frontend Deployment

**Build:**
```bash
npm run build
# Output: dist/
```

**Serve with Nginx:**
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    root /var/www/diffusion-detective;
    index index.html;
    
    location / {
        try_files $uri $uri/ /index.html;
    }
    
    location /api {
        proxy_pass http://localhost:8000;
    }
}
```

---

## Future Roadmap

### Planned Features

1. **WebSocket Streaming**
   - Real-time step-by-step updates
   - Live attention map streaming

2. **Attention Visualization**
   - Heatmap overlay on generated images
   - Token-level attention analysis

3. **Multi-Model Support**
   - SD v2.1
   - SDXL
   - Custom models

4. **Advanced Interventions**
   - Custom vector upload
   - Region-specific steering
   - Temporal intervention profiles

5. **Batch Processing**
   - Generate multiple images at once
   - Compare different parameters

6. **Export & Analysis**
   - Export attention data as JSON
   - Jupyter notebook integration
   - Statistical analysis tools

---

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

**Code Standards:**
- Python: PEP 8, formatted with Black
- JavaScript: ESLint rules, Prettier formatting
- Documentation: Clear docstrings and comments

---

## Resources

- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Tailwind CSS](https://tailwindcss.com/)
- [Framer Motion](https://www.framer.com/motion/)

---

**Happy Coding! 🚀**
