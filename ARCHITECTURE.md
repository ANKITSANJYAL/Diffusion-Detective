# 🔍 Diffusion Detective - Visual Architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                         🔍 DIFFUSION DETECTIVE 🔍                            ║
║                                                                              ║
║             An Interpretable & Intervene-able AI Image Generator             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            USER INTERFACE                               │
│                        (React + Tailwind CSS)                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │   Control    │  │   Timeline   │  │   Terminal   │  │ Comparison │ │
│  │    Panel     │  │   Progress   │  │     Logs     │  │   Slider   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘ │
│                                                                         │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                           HTTP/JSON
                                 │
┌────────────────────────────────▼────────────────────────────────────────┐
│                         FASTAPI BACKEND                                 │
│                      (Python 3.10+ Async)                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │   Routing    │  │  Validation  │  │    Error     │  │  Response  │ │
│  │   (main.py)  │  │  (Pydantic)  │  │   Handling   │  │  Builder   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘ │
│                                                                         │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                            Invokes
                                 │
┌────────────────────────────────▼────────────────────────────────────────┐
│                    INTERPRETABLE SD PIPELINE                            │
│                        (pipeline.py)                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────┐      │
│  │                   StableDiffusionPipeline                     │      │
│  │                 (HuggingFace Diffusers)                       │      │
│  └────────────────────────┬──────────────────────────────────────┘      │
│                           │                                             │
│         ┌─────────────────┼─────────────────┐                          │
│         │                 │                 │                          │
│    ┌────▼─────┐    ┌─────▼──────┐   ┌─────▼──────┐                   │
│    │  UNet    │    │ Text       │   │    VAE     │                   │
│    │ (Noise)  │    │ Encoder    │   │ (Decoder)  │                   │
│    └────┬─────┘    └─────┬──────┘   └────────────┘                   │
│         │                 │                                             │
│    Custom Processor   Embeddings                                       │
│         │                 │                                             │
│    ┌────▼─────────────────▼──────┐                                     │
│    │   AttentionStore            │                                     │
│    │   (Captures attention maps) │                                     │
│    └─────────────────────────────┘                                     │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────┐      │
│  │              LATENT STEERING INTERVENTION                    │      │
│  │  if intervention_active and step_in_range:                   │      │
│  │      latents += intervention_vector * strength               │      │
│  └─────────────────────────────────────────────────────────────┘      │
│                                                                         │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                          Generates
                                 │
                    ┌────────────┴────────────┐
                    │                         │
              ┌─────▼──────┐           ┌─────▼──────┐
              │  Natural   │           │ Controlled │
              │   Image    │           │   Image    │
              └─────┬──────┘           └─────┬──────┘
                    │                         │
                    └────────────┬────────────┘
                                 │
                          Sends to
                                 │
┌────────────────────────────────▼────────────────────────────────────────┐
│                      NARRATOR SERVICE                                   │
│                        (narrator.py)                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                    OpenAI GPT-4o-mini                             │ │
│  │                                                                    │ │
│  │  System: You are Sherlock Holmes investigating AI...              │ │
│  │  User: Analyze these attention logs...                            │ │
│  │                                                                    │ │
│  │  → "Elementary, my dear Watson! The model focused on..."          │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                    Fallback Generator                             │ │
│  │                 (Rule-based narratives)                           │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow

```
┌──────────┐
│   USER   │
└────┬─────┘
     │ 1. Enters Prompt
     │    "A lion on a mountain"
     │
     ▼
┌────────────────┐
│ ControlPanel   │
└────┬───────────┘
     │ 2. Submit Parameters
     │    {prompt, steps, intervention}
     │
     ▼
┌────────────────┐
│   API Call     │
│   POST /generate
└────┬───────────┘
     │ 3. HTTP Request
     │
     ▼
┌────────────────────┐
│  FastAPI Backend   │
│  - Validate input  │
│  - Initialize pipe │
└────┬───────────────┘
     │ 4. Generate Images
     │
     ▼
┌─────────────────────────────────────┐
│     InterpretableSDPipeline         │
│                                     │
│  ┌───────────────────────────────┐ │
│  │ Step 50: Natural Generation   │ │
│  │  - No intervention            │ │
│  │  - Pure diffusion             │ │
│  └───────────────────────────────┘ │
│                                     │
│  ┌───────────────────────────────┐ │
│  │ Step 50-1: Controlled Gen     │ │
│  │  - Extract attention          │ │
│  │  - Apply intervention         │ │
│  │  - Log reasoning              │ │
│  └───────────────────────────────┘ │
│                                     │
│  Output:                            │
│  • natural_image.png                │
│  • controlled_image.png             │
│  • reasoning_logs[]                 │
└─────┬───────────────────────────────┘
      │ 5. Images + Logs
      │
      ▼
┌────────────────┐
│ Narrator       │
│ - Analyze logs │
│ - Call GPT     │
│ - Generate     │
│   narrative    │
└────┬───────────┘
     │ 6. Complete Response
     │    {images, logs, narrative}
     │
     ▼
┌────────────────┐
│  Frontend      │
│  - Display     │
│  - Compare     │
│  - Download    │
└────┬───────────┘
     │
     ▼
┌──────────┐
│   USER   │
│  Enjoys! │
└──────────┘
```

---

## 🧠 Attention Extraction Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Text Encoder                             │
│  "A majestic lion on a mountain peak"                       │
│                          │                                  │
│                          ▼                                  │
│  [emb_1, emb_2, emb_3, ..., emb_77]  (Token embeddings)   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    Sent to UNet
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                      UNet Layer                             │
│                                                             │
│  ┌────────────────────────────────────────────────────┐    │
│  │         Cross-Attention Module                     │    │
│  │                                                     │    │
│  │  Query  (Q): Spatial features [H×W, dim]           │    │
│  │  Key    (K): Text embeddings  [77, dim]            │    │
│  │  Value  (V): Text embeddings  [77, dim]            │    │
│  │                                                     │    │
│  │  Attention = softmax(Q @ K^T / sqrt(dim))          │    │
│  │            = [H×W, 77]  (Attention map!)           │    │
│  │                    │                                │    │
│  │                    ▼                                │    │
│  │          ┌──────────────────┐                      │    │
│  │          │ CustomProcessor  │                      │    │
│  │          │ INTERCEPTS HERE! │                      │    │
│  │          └────────┬─────────┘                      │    │
│  │                   │                                 │    │
│  │                   ▼                                 │    │
│  │          ┌──────────────────┐                      │    │
│  │          │ AttentionStore   │                      │    │
│  │          │ .add_attention() │                      │    │
│  │          └──────────────────┘                      │    │
│  │                                                     │    │
│  │  Output = Attention @ V                            │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                           │
                    Continues in UNet
                           │
                           ▼
                    Final Image
```

---

## 🎛️ Latent Steering Mechanism

```
┌─────────────────────────────────────────────────────────────┐
│              Diffusion Denoising Loop                       │
│              (50 steps: 981 → 1)                            │
└─────────────────────────────────────────────────────────────┘

  Step 50  ──→  Step 45  ──→  Step 40  ──→  Step 35  ──→  Step 30
    │             │             │             │             │
    │             │             ▼             ▼             ▼
    │             │        ┌─────────────────────────────┐
    │             │        │  INTERVENTION ZONE          │
    │             │        │  (Steps 40-20)              │
    │             │        │                             │
    │             │        │  if intervention_active:    │
    │             │        │    latents += vector * 0.1  │
    │             │        │                             │
    │             │        │  Logs: "Applying steering"  │
    │             │        └─────────────────────────────┘
    │             │             │             │             │
    ▼             ▼             ▼             ▼             ▼
  Pure        Pure        Steered       Steered       Steered
   Path        Path         Path          Path          Path
    │             │             │             │             │
    ▼             ▼             ▼             ▼             ▼

  Step 25  ──→  Step 20  ──→  Step 15  ──→  Step 10  ──→  Step 1
    │             │             │             │             │
    ▼             ▼             │             │             │
  Steered     Steered           ▼             ▼             ▼
   Path        Path          Pure          Pure          Pure
    │             │            Path          Path          Path
    └─────────────┴─────────────┴─────────────┴─────────────┘
                                 │
                                 ▼
                         ┌──────────────┐
                         │ Final Latent │
                         └──────┬───────┘
                                │
                          VAE Decode
                                │
                                ▼
                         ┌──────────────┐
                         │ Final Image  │
                         │ (Controlled) │
                         └──────────────┘
```

**Key Insight:** Intervention modifies the latent trajectory, creating a
different path through the noise space without changing random seeds!

---

## 🎨 UI Component Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                         App.jsx                             │
│                      (Root Component)                       │
│                                                             │
│  State:                                                     │
│  • isGenerating                                             │
│  • results                                                  │
│  • logs                                                     │
│  • error                                                    │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │                    Header                             │ │
│  │  "🔍 DIFFUSION DETECTIVE"                            │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │              ControlPanel.jsx                         │ │
│  │  ┌─────────────────────────────────────────────┐     │ │
│  │  │ Prompt Input                                │     │ │
│  │  ├─────────────────────────────────────────────┤     │ │
│  │  │ Inference Steps Slider        [20 ──── 100]│     │ │
│  │  ├─────────────────────────────────────────────┤     │ │
│  │  │ Guidance Scale Slider         [1.0 ── 20.0]│     │ │
│  │  ├─────────────────────────────────────────────┤     │ │
│  │  │ 🧪 Intervention Toggle        [ON] / OFF   │     │ │
│  │  │   • Strength Slider           [0.0 ── 2.0] │     │ │
│  │  │   • Start Step                [40]          │     │ │
│  │  │   • End Step                  [20]          │     │ │
│  │  ├─────────────────────────────────────────────┤     │ │
│  │  │ Seed Input (optional)         [42]          │     │ │
│  │  ├─────────────────────────────────────────────┤     │ │
│  │  │     [🚀 RUN ANALYSIS]                       │     │ │
│  │  └─────────────────────────────────────────────┘     │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │               Timeline.jsx                            │ │
│  │  ┌─────────────────────────────────────────────┐     │ │
│  │  │ Progress: 65%                               │     │ │
│  │  │ ▓▓▓▓▓▓▓░░░░░░░   [INTERVENTION ZONE]       │     │ │
│  │  │ Step 35 / 50                                │     │ │
│  │  └─────────────────────────────────────────────┘     │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │               Terminal.jsx                            │ │
│  │  ┌─────────────────────────────────────────────┐     │ │
│  │  │ 💻 INVESTIGATION LOG                        │     │ │
│  │  ├─────────────────────────────────────────────┤     │ │
│  │  │ 🔍 Initializing...                          │     │ │
│  │  │ 📝 Prompt: "A lion..."                      │     │ │
│  │  │ ⏳ Generating images...                     │     │ │
│  │  │ Step 35: Focus on lion, mountain            │     │ │
│  │  │ Step 30: Applying latent steering           │     │ │
│  │  │ Step 25: Focus on sunset, peak              │     │ │
│  │  │ ✅ Generation complete!                     │     │ │
│  │  │ 🕵️ Detective's Report: "Elementary..."     │     │ │
│  │  │ ▋                                           │     │ │
│  │  └─────────────────────────────────────────────┘     │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │           ComparisonSlider.jsx                        │ │
│  │  ┌─────────────────────────────────────────────┐     │ │
│  │  │ 🔬 COMPARATIVE ANALYSIS                     │     │ │
│  │  ├─────────────────────────────────────────────┤     │ │
│  │  │                                             │     │ │
│  │  │   NATURAL    │     CONTROLLED               │     │ │
│  │  │   ┌────────┐ │ ┌────────┐                  │     │ │
│  │  │   │ Image  │ │ │ Image  │                  │     │ │
│  │  │   │        │ │ │        │                  │     │ │
│  │  │   └────────┘ │ └────────┘                  │     │ │
│  │  │              ▲ Drag Slider                  │     │ │
│  │  │                                             │     │ │
│  │  ├─────────────────────────────────────────────┤     │ │
│  │  │ [⬇️ Download Natural] [⬇️ Download Ctrl]   │     │ │
│  │  └─────────────────────────────────────────────┘     │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Technology Stack Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                       │
│  • React 18 (UI components)                                 │
│  • Framer Motion (animations)                               │
│  • Tailwind CSS (styling)                                   │
│  • react-compare-image (slider)                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                    HTTP/JSON
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    APPLICATION LAYER                        │
│  • FastAPI (web framework)                                  │
│  • Pydantic (validation)                                    │
│  • Uvicorn (ASGI server)                                    │
│  • python-dotenv (config)                                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                   Direct Calls
                         │
┌────────────────────────▼────────────────────────────────────┐
│                        AI LAYER                             │
│  • PyTorch (deep learning)                                  │
│  • Diffusers (SD pipeline)                                  │
│  • Transformers (CLIP, GPT)                                 │
│  • OpenAI API (GPT-4o-mini)                                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                  GPU Acceleration
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    HARDWARE LAYER                           │
│  • CUDA (GPU compute)                                       │
│  • cuDNN (optimizations)                                    │
│  • NVIDIA GPU (8GB+ VRAM)                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Request-Response Cycle

```
┌──────────┐                                           ┌──────────┐
│  BROWSER │                                           │  SERVER  │
└────┬─────┘                                           └────┬─────┘
     │                                                      │
     │  1. POST /generate                                  │
     │     {prompt, steps, intervention_active, ...}       │
     ├────────────────────────────────────────────────────>│
     │                                                      │
     │                                        2. Validate   │
     │                                           Input      │
     │                                             │        │
     │                                        3. Generate   │
     │                                           Natural    │
     │                                           Image      │
     │                                             │        │
     │                                        4. Generate   │
     │                                           Controlled │
     │                                           Image      │
     │                                             │        │
     │                                        5. Extract    │
     │                                           Attention  │
     │                                           Logs       │
     │                                             │        │
     │                                        6. Generate   │
     │                                           Narrative  │
     │                                             │        │
     │  7. Response                                ▼        │
     │     {                                                │
     │       success: true,                                 │
     │       image_natural: "data:image/png;base64,...",    │
     │       image_controlled: "data:image/png;base64,...", │
     │       reasoning_logs: [...],                         │
     │       narrative_text: "Elementary, ...",             │
     │       metadata: {...}                                │
     │     }                                                │
     │<─────────────────────────────────────────────────────┤
     │                                                      │
     │  8. Update UI                                        │
     │     • Display images                                 │
     │     • Show logs                                      │
     │     • Render narrative                               │
     │                                                      │
     ▼                                                      ▼
```

---

## 🎯 Key Innovation Points

```
┌────────────────────────────────────────────────────────────────┐
│  1. CUSTOM ATTENTION PROCESSOR                                 │
│     ┌────────────────────────────────────────────────────┐    │
│     │ class CustomAttentionProcessor:                    │    │
│     │     def __call__(self, attn, hidden_states, ...):  │    │
│     │         # Compute attention                        │    │
│     │         attention_probs = softmax(scores)          │    │
│     │         # INTERCEPT HERE!                          │    │
│     │         self.store.add_attention_map(...)          │    │
│     │         return output                              │    │
│     └────────────────────────────────────────────────────┘    │
│  ✅ No heuristics - real attention extraction                 │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  2. LATENT STEERING (NOT RANDOM SEEDS!)                        │
│     ┌────────────────────────────────────────────────────┐    │
│     │ # In denoising loop                                │    │
│     │ for i, t in enumerate(timesteps):                  │    │
│     │     if intervention_active and in_range(step):     │    │
│     │         # Direct latent manipulation               │    │
│     │         latents += intervention_vector * 0.1       │    │
│     │     # Continue denoising with modified latents     │    │
│     │     noise_pred = unet(latents, t, ...)             │    │
│     └────────────────────────────────────────────────────┘    │
│  ✅ Mathematical steering - reproducible interventions         │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  3. AI NARRATIVE GENERATION                                    │
│     ┌────────────────────────────────────────────────────┐    │
│     │ System: "You are Sherlock Holmes..."               │    │
│     │ User: "Analyze these attention logs..."            │    │
│     │ GPT-4o-mini: "Elementary, my dear Watson! ..."     │    │
│     └────────────────────────────────────────────────────┘    │
│  ✅ Engaging narratives - makes AI interpretable & fun         │
└────────────────────────────────────────────────────────────────┘
```

---

## 🏆 Achievement Unlocked

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║               🏆 PRODUCTION-GRADE AI SYSTEM 🏆               ║
║                                                              ║
║  ✅ Real-time attention extraction                           ║
║  ✅ Latent steering intervention                             ║
║  ✅ AI-generated narratives                                  ║
║  ✅ Interactive UI with animations                           ║
║  ✅ Comprehensive documentation                              ║
║  ✅ Memory-efficient GPU management                          ║
║  ✅ Type-safe APIs                                           ║
║  ✅ Error handling                                           ║
║                                                              ║
║            "Elementary, Watson! Case closed!"                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

**Built with 💚 by a Senior AI Engineer**
**December 7, 2025**
