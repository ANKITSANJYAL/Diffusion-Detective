"""
Diffusion Detective FastAPI Application
Main API server for interpretable Stable Diffusion generation.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import torch
import os
import json
from dotenv import load_dotenv

from .pipeline import InterpretableSDPipeline
from .narrator import NarratorService
from .utils import pil_to_base64

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Diffusion Detective API",
    description="An interpretable, intervene-able Stable Diffusion interface",
    version="1.0.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (lazy loaded)
pipeline: Optional[InterpretableSDPipeline] = None
narrator: Optional[NarratorService] = None


# ==================== Request/Response Models ====================

class GenerationRequest(BaseModel):
    """Request model for image generation (v2.0 with semantic steering)."""
    
    prompt: str = Field(..., description="Text prompt for image generation")
    num_inference_steps: int = Field(
        default=50,
        ge=20,
        le=100,
        description="Number of denoising steps"
    )
    guidance_scale: float = Field(
        default=7.5,
        ge=1.0,
        le=20.0,
        description="Classifier-free guidance scale"
    )
    intervention_active: bool = Field(
        default=False,
        description="Whether to apply latent steering intervention"
    )
    intervention_strength: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Strength of latent intervention"
    )
    intervention_step_start: int = Field(
        default=40,
        ge=0,
        le=100,
        description="Step to start intervention (higher = earlier in process)"
    )
    intervention_step_end: int = Field(
        default=20,
        ge=0,
        le=100,
        description="Step to end intervention"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )
    
    # v2.0: Semantic intervention parameters
    target_concept: Optional[str] = Field(
        default=None,
        description="Target concept to modify (e.g., 'tiger', 'mountain')"
    )
    injection_attribute: Optional[str] = Field(
        default=None,
        description="Attribute to inject (e.g., 'blue', 'neon', 'robot')"
    )
    auto_detect_concepts: bool = Field(
        default=True,
        description="Auto-detect and track key concepts from prompt"
    )


class GenerationResponse(BaseModel):
    """Response model for image generation."""
    
    success: bool
    image_baseline: str = Field(..., description="Base64 encoded baseline image (no intervention)")
    image_intervened: str = Field(..., description="Base64 encoded intervened image (with intervention)")
    reasoning_logs: List[dict] = Field(..., description="Structured reasoning logs")
    narrative_text: str = Field(..., description="Sherlock Holmes-style investigation narrative")
    metadata: dict = Field(..., description="Generation metadata")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    cuda_available: bool
    model_loaded: bool
    narrator_available: bool


# ==================== Helper Functions ====================

def get_pipeline() -> InterpretableSDPipeline:
    """Get or initialize the pipeline."""
    global pipeline
    
    if pipeline is None:
        model_id = os.getenv("MODEL_ID", "runwayml/stable-diffusion-v1-5")
        
        # Check environment first, then auto-detect device: CUDA > MPS > CPU
        device = os.getenv("DEVICE")
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        # Set dtype based on device
        dtype_str = os.getenv("TORCH_DTYPE")
        if dtype_str is None:
            # Auto-select dtype based on device
            dtype_str = "float16" if device == "cuda" else "float32"
        
        torch_dtype = torch.float16 if dtype_str == "float16" else torch.float32
        
        print(f"Using device: {device} with dtype: {torch_dtype}")
        
        pipeline = InterpretableSDPipeline(
            model_id=model_id,
            device=device,
            torch_dtype=torch_dtype
        )
    
    return pipeline


def get_narrator() -> NarratorService:
    """Get or initialize the narrator service."""
    global narrator
    
    if narrator is None:
        narrator = NarratorService()
    
    return narrator


def reset_pipeline():
    """Force reset the pipeline (useful for device changes)."""
    global pipeline
    
    if pipeline is not None:
        print("🔄 Resetting pipeline...")
        pipeline.cleanup()
        pipeline = None
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        print("✓ Pipeline reset complete")


# ==================== API Endpoints ====================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Diffusion Detective API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    model_loaded = pipeline is not None
    narrator_available = narrator is not None and narrator.client is not None
    
    return HealthResponse(
        status="healthy",
        cuda_available=cuda_available or mps_available,  # Show True if any GPU available
        model_loaded=model_loaded,
        narrator_available=narrator_available
    )


@app.post("/reset", tags=["Admin"])
async def reset_pipeline_endpoint():
    """Force reset the pipeline to apply new device settings."""
    reset_pipeline()
    return {"status": "success", "message": "Pipeline reset. Will reload on next generation."}


@app.post("/generate_simple", response_model=GenerationResponse, tags=["Generation"])
async def generate_image_simple(request: GenerationRequest):
    """
    Generate images without streaming (backward compatibility).
    Returns complete result after generation finishes.
    """
    try:
        pipe = get_pipeline()
        narr = get_narrator()
        
        if request.intervention_active and request.intervention_step_start < request.intervention_step_end:
            raise HTTPException(
                status_code=400,
                detail="intervention_step_start must be >= intervention_step_end"
            )
        
        def progress_callback(step: int, log: str):
            print(f"[Step {step}] {log}")
        
        natural_image, controlled_image, reasoning_logs, metadata = pipe.generate(
            prompt=request.prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            intervention_active=request.intervention_active,
            intervention_strength=request.intervention_strength,
            intervention_step_start=request.intervention_step_start,
            intervention_step_end=request.intervention_step_end,
            seed=request.seed,
            callback=progress_callback,
            target_concept=request.target_concept,
            injection_attribute=request.injection_attribute,
            auto_detect_concepts=request.auto_detect_concepts
        )
        
        image_baseline_b64 = pil_to_base64(natural_image, format="PNG")
        image_intervened_b64 = pil_to_base64(controlled_image, format="PNG")
        
        # 🧠 GENERATE LLM-POWERED STEP-BY-STEP REASONING
        # Get high-fidelity data packet from attention store
        high_fidelity_data = pipe.attention_store.high_fidelity_data
        
        # Prepare intervention info for LLM
        intervention_info = None
        if request.intervention_active and request.target_concept and request.injection_attribute:
            intervention_info = {
                'target': request.target_concept,
                'attribute': request.injection_attribute,
                'step_start': request.intervention_step_start,
                'step_end': request.intervention_step_end,
                'strength': request.intervention_strength
            }
        
        # Generate intelligent step-by-step reasoning (now returns grouped structure)
        llm_step_logs = narr.generate_step_by_step_reasoning(
            prompt=request.prompt,
            data_packet=high_fidelity_data,
            intervention_info=intervention_info
        )
        
        # LLM now returns dicts with range, type, message, stats
        # Pass them directly to frontend with llm_generated flag
        llm_reasoning_logs = []
        for log_entry in llm_step_logs:
            if isinstance(log_entry, dict):
                llm_reasoning_logs.append({
                    'range': log_entry.get('range', 'Unknown'),
                    'type': log_entry.get('type', 'normal'),
                    'message': log_entry.get('message', ''),
                    'stats': log_entry.get('stats', {}),
                    'intervention_active': request.intervention_active,
                    'llm_generated': True
                })
            else:
                # Fallback for old string format
                llm_reasoning_logs.append({
                    'range': 'Unknown',
                    'type': 'normal',
                    'message': str(log_entry),
                    'stats': {},
                    'intervention_active': request.intervention_active,
                    'llm_generated': True
                })
        
        # Generate narrative (still uses old narrative generator)
        formatted_logs = []
        for log in reasoning_logs:
            if isinstance(log, dict):
                formatted_logs.append(f"Step {log['step']} ({log['phase']}): {log['message']}")
            else:
                formatted_logs.append(str(log))
        
        narrative_text = narr.generate_narrative(
            prompt=request.prompt,
            reasoning_logs=formatted_logs,
            intervention_active=request.intervention_active,
            intervention_strength=request.intervention_strength
        )
        
        return GenerationResponse(
            success=True,
            image_baseline=image_baseline_b64,
            image_intervened=image_intervened_b64,
            reasoning_logs=llm_reasoning_logs,  # Use LLM-generated logs instead
            narrative_text=narrative_text,
            metadata=metadata
        )
    
    except torch.cuda.OutOfMemoryError:
        raise HTTPException(status_code=503, detail="GPU out of memory")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/generate", tags=["Generation"])
async def generate_image_stream(request: GenerationRequest):
    """
    Generate images with Server-Sent Events (SSE) streaming for real-time progress.
    
    This endpoint streams progress updates during:
    1. Denoising steps (0-100%)
    2. Decoding phase (98%)
    3. Narrative generation (99%)
    4. Final result (100%)
    
    Returns a stream of JSON objects with progress updates and final result.
    """
    
    async def event_generator():
        try:
            # Initialize pipeline and narrator
            pipe = get_pipeline()
            narr = get_narrator()
            
            # Validate intervention range
            if request.intervention_active and request.intervention_step_start < request.intervention_step_end:
                yield f"data: {json.dumps({'error': 'intervention_step_start must be >= intervention_step_end'})}\n\n"
                return
            
            # Progress tracking with streaming
            def progress_callback(step: int, log):
                if isinstance(log, dict):
                    # Post-processing status (step = -1)
                    status_update = {
                        "type": "status",
                        "status": log.get("status", "processing"),
                        "message": log.get("message", "Processing..."),
                        "progress": log.get("progress", 0)
                    }
                else:
                    # Regular step progress
                    status_update = {
                        "type": "progress",
                        "step": step,
                        "message": str(log),
                        "progress": int((step / request.num_inference_steps) * 97)  # 0-97%
                    }
                
                # This will be picked up by the async generator
                return status_update
            
            # Store progress updates
            progress_queue = []
            
            def callback_with_queue(step, log):
                update = progress_callback(step, log)
                progress_queue.append(update)
            
            # Start generation
            yield f"data: {json.dumps({'type': 'start', 'message': 'Starting generation...'})}\n\n"
            
            # Generate images (v2.0 with semantic steering)
            natural_image, controlled_image, reasoning_logs, metadata = pipe.generate(
                prompt=request.prompt,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                intervention_active=request.intervention_active,
                intervention_strength=request.intervention_strength,
                intervention_step_start=request.intervention_step_start,
                intervention_step_end=request.intervention_step_end,
                seed=request.seed,
                callback=callback_with_queue,
                # v2.0 parameters
                target_concept=request.target_concept,
                injection_attribute=request.injection_attribute,
                auto_detect_concepts=request.auto_detect_concepts
            )
            
            # Send any queued progress updates
            for update in progress_queue:
                yield f"data: {json.dumps(update)}\n\n"
            
            # Narrative generation phase
            yield f"data: {json.dumps({'type': 'status', 'status': 'analyzing', 'message': '🕵️ Consulting Forensic Narrator...', 'progress': 99})}\n\n"
            
            # Convert images to base64
            image_baseline_b64 = pil_to_base64(natural_image, format="PNG")
            image_intervened_b64 = pil_to_base64(controlled_image, format="PNG")
            
            # Format logs for narrative (convert structured logs to readable text)
            formatted_logs = []
            for log in reasoning_logs:
                if isinstance(log, dict):
                    formatted_logs.append(f"Step {log['step']} ({log['phase']}): {log['message']}")
                else:
                    formatted_logs.append(str(log))
            
            # Generate narrative with clean logs
            narrative_text = narr.generate_narrative(
                prompt=request.prompt,
                reasoning_logs=formatted_logs,
                intervention_active=request.intervention_active,
                intervention_strength=request.intervention_strength
            )
            
            # Send final result
            final_response = {
                "type": "complete",
                "success": True,
                "image_baseline": image_baseline_b64,
                "image_intervened": image_intervened_b64,
                "reasoning_logs": reasoning_logs,
                "narrative_text": narrative_text,
                "metadata": metadata,
                "progress": 100
            }
            
            yield f"data: {json.dumps(final_response)}\n\n"
        
        except torch.cuda.OutOfMemoryError:
            yield f"data: {json.dumps({'type': 'error', 'message': 'GPU out of memory. Try reducing num_inference_steps.'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Generation failed: {str(e)}'})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/cleanup", tags=["Maintenance"])
async def cleanup_resources():
    """
    Cleanup GPU memory and reset pipeline.
    Use this if you encounter memory issues.
    """
    global pipeline
    
    if pipeline is not None:
        pipeline.cleanup()
        pipeline = None
    
    # Clear cache for both CUDA and MPS
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    return {"message": "Resources cleaned up successfully"}


# ==================== Startup/Shutdown Events ====================

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    print("🚀 Diffusion Detective API starting up...")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif torch.backends.mps.is_available():
        print("Using Apple Silicon MPS acceleration")
    else:
        print("Using CPU (this will be slow)")
    
    # Optionally preload the model
    # get_pipeline()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    print("🛑 Diffusion Detective API shutting down...")
    
    global pipeline
    if pipeline is not None:
        pipeline.cleanup()


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True
    )
