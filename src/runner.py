
"""Core runner utilities: load models, run generation, capture intermediate images and latents.

This runner provides a convenience method `generate_with_intermediates` which runs the
diffusers pipeline step-by-step and decodes intermediate latents to PIL images for visualization.
The implementation uses the pipeline's scheduler and U-Net directly to avoid re-implementing denoising.
"""
from typing import List, Dict, Any, Optional, Tuple
import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from PIL import Image
import math


class ModelRunner:
    def __init__(self, sd_model: str = "stabilityai/stable-diffusion-2-1", device: str = "cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # load pipeline lazily to allow CPU-only environments
        self.sd_model = sd_model
        self.pipeline: Optional[StableDiffusionPipeline] = None
        # CLIP for embeddings
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def _ensure_pipeline(self):
        if self.pipeline is None:
            # Use fp16 when running on CUDA
            kwargs = {}
            if str(self.device).startswith("cuda"):
                kwargs["torch_dtype"] = torch.float16
                kwargs["revision"] = "fp16"
            self.pipeline = StableDiffusionPipeline.from_pretrained(self.sd_model, safety_checker=None, **kwargs).to(self.device)

    def generate(self, prompt: str, steps: int = 20, guidance: float = 7.5, height: int = 768, width: int = 768) -> Tuple[Image.Image, Dict[str, Any]]:
        """Simple generate wrapper that returns final image and metadata."""
        self._ensure_pipeline()
        out = self.pipeline(prompt, num_inference_steps=steps, guidance_scale=guidance, height=height, width=width)
        return out.images[0], {}

    def generate_with_intermediates(self, prompt: str, steps: int = 20, guidance: float = 7.5, height: int = 512, width: int = 512, capture_steps: Optional[List[int]] = None, step_weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """Run the diffusion sampling loop and capture intermediate decoded images at requested timesteps.

        Returns a dict with keys:
          - 'final_image': PIL image
          - 'intermediates': list of (step_index, PIL image)
          - 'metadata': additional info

        Note: capture_steps expects step indices in [0, steps], where 0 is the initial noisy latent and steps is final.
        """
        self._ensure_pipeline()
        pipe = self.pipeline

        # prepare prompt embedding
        do_classifier_free_guidance = guidance > 1.0

        # prepare latent shape
        height, width = int(height), int(width)
        generator = None

        # use pipeline components
        text_inputs = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_input_ids = text_inputs.input_ids.to(self.device)

        # get text embeddings
        with torch.no_grad():
            text_embeddings = pipe.text_encoder(text_input_ids)[0]

        # prepare unconditional embeddings for classifier-free guidance
        if do_classifier_free_guidance:
            uncond_input = pipe.tokenizer([""] * 1, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
            uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(self.device))[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)

        # scheduler
        scheduler = pipe.scheduler
        scheduler.set_timesteps(steps)

        # prepare latents (sample directly in latent space).
        # The VAE expects image-space tensors for encode(); here we need initial latent noise
        # with shape (batch, unet_in_channels, H/8, W/8). Use scheduler.init_noise_sigma to scale.
        unet_in_ch = pipe.unet.config.in_channels if hasattr(pipe.unet, 'config') else pipe.unet.in_channels
        model_dtype = next(pipe.unet.parameters()).dtype
        latent_shape = (1, unet_in_ch, height // 8, width // 8)
        init_noise = torch.randn(latent_shape, device=self.device, dtype=model_dtype)
        init_sigma = getattr(scheduler, 'init_noise_sigma', 1.0)
        latents = init_noise * init_sigma

        intermediates = []
        capture_set = set(capture_steps) if capture_steps else set()

        # prepare guidance schedule from step_weights if provided
        guidance_schedule = None
        if step_weights:
            # map reasoning steps over denoising timesteps evenly
            n_steps_weights = len(step_weights)
            timesteps = scheduler.timesteps
            guidance_schedule = [1.0] * len(timesteps)
            for si, w in enumerate(step_weights):
                # assign window
                start = int(len(timesteps) * si / n_steps_weights)
                end = int(len(timesteps) * (si + 1) / n_steps_weights)
                for ti in range(start, end):
                    guidance_schedule[ti] = w

        for i, t in enumerate(scheduler.timesteps):
            # predict noise residual
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            with torch.no_grad():
                noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                # apply dynamic guidance multiplier if present
                g = guidance
                if guidance_schedule is not None:
                    # find index i in timesteps (current loop index)
                    g = guidance * float(guidance_schedule[i])
                noise_pred = noise_pred_uncond + g * (noise_pred_text - noise_pred_uncond)

            latents = scheduler.step(noise_pred, t, latents).prev_sample

            # capture after step
            if i in capture_set:
                with torch.no_grad():
                    # decode latents to image via VAE decode
                    # ensure scaling factor has same dtype/device
                    sf = getattr(pipe.vae.config, 'scaling_factor', 1.0)
                    sf_t = torch.tensor(sf, dtype=latents.dtype, device=self.device)
                    image = pipe.vae.decode(latents / sf_t).sample
                    image = (image / 2 + 0.5).clamp(0, 1)
                    image = (image * 255).round().type(torch.uint8)
                    image = image.permute(0, 2, 3, 1)[0].cpu().numpy()
                    intermediates.append((i, Image.fromarray(image)))

        # final decode
        with torch.no_grad():
            sf = getattr(pipe.vae.config, 'scaling_factor', 1.0)
            sf_t = torch.tensor(sf, dtype=latents.dtype, device=self.device)
            image = pipe.vae.decode(latents / sf_t).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = (image * 255).round().type(torch.uint8)
            image = image.permute(0, 2, 3, 1)[0].cpu().numpy()
            final_img = Image.fromarray(image)

        return {"final_image": final_img, "intermediates": intermediates, "metadata": {}}

    def embed_clip(self, images: List[Any]):
        inputs = self.clip_processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeds = self.clip.get_image_features(**inputs)
            embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
        return embeds.cpu().numpy()


