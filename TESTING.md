# 🧪 Testing Guide

## Backend Testing

### 1. Health Check

Test if the backend is running:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "cuda_available": true,
  "model_loaded": false,
  "narrator_available": true
}
```

### 2. Basic Generation (Without Intervention)

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A red apple on a wooden table",
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "intervention_active": false,
    "seed": 42
  }' \
  --max-time 300
```

This should return a JSON with:
- `success: true`
- `image_natural` (base64 string)
- `image_controlled` (base64 string)
- `reasoning_logs` (array of strings)
- `narrative_text` (string)

### 3. Generation with Intervention

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A majestic lion in a forest",
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "intervention_active": true,
    "intervention_strength": 1.5,
    "intervention_step_start": 40,
    "intervention_step_end": 20,
    "seed": 42
  }' \
  --max-time 300
```

### 4. Test Memory Cleanup

```bash
curl -X POST http://localhost:8000/cleanup
```

Expected response:
```json
{
  "message": "Resources cleaned up successfully"
}
```

---

## Frontend Testing

### 1. Local Development

```bash
cd frontend
npm run dev
```

Open http://localhost:3000 and verify:
- ✅ Dark cyberpunk theme loads
- ✅ Control panel is visible
- ✅ All inputs are functional
- ✅ No console errors

### 2. Component Testing

#### Test Control Panel
1. Enter a prompt
2. Adjust sliders
3. Toggle intervention ON/OFF
4. Verify all values update correctly

#### Test Generation Flow
1. Enter: "A sunset over ocean waves"
2. Set steps: 30
3. Enable intervention
4. Click "RUN ANALYSIS"
5. Verify:
   - Timeline appears
   - Terminal logs stream
   - Images load after completion
   - Comparison slider works

#### Test Error Handling
1. Stop the backend
2. Try to generate
3. Verify error message appears

---

## Integration Testing

### Full Pipeline Test

1. Start backend: `python -m uvicorn app.main:app --reload`
2. Start frontend: `npm run dev`
3. Open http://localhost:3000
4. Run generation with these parameters:
   - Prompt: "A cyberpunk detective in neon-lit streets"
   - Steps: 40
   - Guidance: 8.0
   - Intervention: ON
   - Strength: 1.2
   - Start: 35, End: 15
   - Seed: 12345

5. Verify:
   - Generation completes without errors
   - Both images are different
   - Logs show intervention messages
   - Narrative is generated
   - Slider comparison works
   - Images can be downloaded

---

## Performance Testing

### GPU Memory Test

Monitor GPU usage during generation:

```bash
# In another terminal
watch -n 1 nvidia-smi
```

Expected:
- Peak usage: ~6-8GB for SD v1.5
- Memory released after generation

### Generation Speed Test

Time a generation:

```bash
time curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Test image",
    "num_inference_steps": 30,
    "seed": 42
  }' \
  --max-time 300
```

Expected:
- First run: 60-90s (downloads models)
- Subsequent runs: 20-40s (depending on GPU)

---

## Common Issues & Solutions

### Issue: "CUDA out of memory"
**Solution:** Reduce `num_inference_steps` to 25 or call `/cleanup` endpoint

### Issue: Backend crashes on startup
**Solution:** 
- Check Python version (3.10+)
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`

### Issue: Frontend can't connect to backend
**Solution:**
- Verify backend is running on port 8000
- Check CORS settings in `backend/app/main.py`
- Update `VITE_API_URL` in `frontend/.env`

### Issue: Images not loading
**Solution:**
- Check browser console for errors
- Verify base64 data is present in API response
- Check image size (should be ~500KB each)

### Issue: Narrative not generating
**Solution:**
- Verify `OPENAI_API_KEY` in `backend/.env`
- Check API quota: https://platform.openai.com/usage
- System will fall back to rule-based narrative if API fails

---

## Validation Checklist

Before considering the system production-ready:

- [ ] Backend health endpoint responds
- [ ] Generation completes without errors (30 steps)
- [ ] Generation completes without errors (50 steps)
- [ ] Intervention produces different results
- [ ] Attention logs are generated
- [ ] Narrative is generated (or fallback works)
- [ ] Images are comparable in slider
- [ ] Images can be downloaded
- [ ] Memory cleanup works
- [ ] Frontend loads without console errors
- [ ] All sliders and inputs work
- [ ] Terminal logs stream correctly
- [ ] Timeline shows intervention zone
- [ ] Error messages display properly

---

## Automated Testing (Future)

Future additions could include:

```python
# backend/tests/test_pipeline.py
def test_basic_generation():
    pipeline = InterpretableSDPipeline()
    natural, controlled, logs, metadata = pipeline.generate(
        prompt="test",
        num_inference_steps=10
    )
    assert natural is not None
    assert controlled is not None
    assert len(logs) > 0
```

```javascript
// frontend/src/tests/App.test.jsx
import { render, screen } from '@testing-library/react'
import App from './App'

test('renders control panel', () => {
  render(<App />)
  expect(screen.getByText(/Mission Control/i)).toBeInTheDocument()
})
```

---

**Testing is investigating! 🔍**
