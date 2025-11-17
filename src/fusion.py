"""Fusion utilities: map reasoning steps (text) to spatial evidence maps using CLIP patch embeddings.

This implementation computes patch-level similarity between CLIP vision patch embeddings
and the CLIP text embedding, then upsamples the resulting patch grid to the requested output size.
It aims to produce a spatial heatmap where each spatial cell reflects the affinity of the
reasoning step text with the local image patch.
"""
from typing import List, Optional
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from PIL import Image
import math
import torch.nn.functional as F


class FusionMapper:
    def __init__(self, device: str = "cuda:0", clip_model: str = "openai/clip-vit-large-patch14"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # load CLIP model and processor
        self.clip = CLIPModel.from_pretrained(clip_model).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(clip_model)

    def _vision_patch_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return patch embeddings from CLIP vision model (no pooling).

        Args:
            pixel_values: tensor shape (B, C, H, W)
        Returns:
            patch_feats: (B, num_patches, proj_dim)  -- projected patch features in CLIP space
        """
        # Get raw hidden states from vision model
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        # last_hidden_state shape = (B, seq_len, hidden)
        last_hidden = vision_outputs.last_hidden_state
        # project with visual projection to CLIP embedding space
        # visual_projection is a parameter tensor of shape (hidden, projection_dim)
        proj = getattr(self.clip, "visual_projection", None)
        if proj is None:
            # fallback: use mean pooling and repeat
            pooled = last_hidden.mean(dim=1, keepdim=True)  # (B,1,hidden)
            proj = torch.eye(pooled.shape[-1], device=self.device)
            patch_proj = pooled @ proj
            return patch_proj

        # project all tokens (including CLS); drop CLS for spatial grid
        patch_proj = last_hidden @ proj
        # remove CLS token (first token) to get only patch tokens
        if patch_proj.shape[1] > 1:
            patch_proj = patch_proj[:, 1:, :]
        return patch_proj

    def _text_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # use CLIP text_model + projection to get text embedding in the same space
        text_outputs = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        # take first token (CLS) representation and project
        last_hidden = text_outputs.last_hidden_state
        cls_rep = last_hidden[:, 0:1, :]
        text_proj = cls_rep @ getattr(self.clip, "text_projection", torch.eye(last_hidden.size(-1), device=self.device))
        # shape (B, 1, proj_dim)
        return text_proj.squeeze(1)

    def text_to_heatmap(self, image: Image.Image, step_text: str, output_size=(64, 64)) -> np.ndarray:
        """Compute a spatial heatmap for a single reasoning step text and an image.

        Returns a numpy float32 array in range [0,1] of shape output_size.
        """
        inputs = self.processor(text=[step_text], images=image, return_tensors="pt", padding=True).to(self.device)
        pixel_values = inputs["pixel_values"]  # (1, C, H, W)

        with torch.no_grad():
            patch_feats = self._vision_patch_embeddings(pixel_values)  # (1, num_patches, proj_dim)
            txt_feat = self._text_embedding(inputs["input_ids"], inputs.get("attention_mask", None))  # (1, proj_dim)

            # normalize
            patch_norm = patch_feats / (patch_feats.norm(dim=-1, keepdim=True) + 1e-8)
            txt_norm = txt_feat / (txt_feat.norm(dim=-1, keepdim=True) + 1e-8)

            # cosine similarity per patch: (1, num_patches)
            sim = (patch_norm @ txt_norm.unsqueeze(-1)).squeeze(-1)
            sim = sim.clamp(-1, 1)

            # convert patches -> spatial grid
            num_patches = sim.shape[1]
            grid_size = int(math.sqrt(num_patches)) if int(math.sqrt(num_patches)) ** 2 == num_patches else None
            if grid_size is None:
                # fallback: treat as 1D and tile
                heat = sim.reshape(1, 1, 1, num_patches)
                heatmap = F.interpolate(heat, size=output_size, mode='bilinear', align_corners=False).squeeze().cpu().numpy()
            else:
                heat = sim.reshape(1, 1, grid_size, grid_size)
                # upsample to desired output
                heatmap = F.interpolate(heat, size=output_size, mode='bilinear', align_corners=False).squeeze().cpu().numpy()

            # rescale to [0,1]
            minv, maxv = float(heatmap.min()), float(heatmap.max())
            if maxv - minv > 1e-6:
                heatmap = (heatmap - minv) / (maxv - minv)
            else:
                heatmap = heatmap * 0.0

        return heatmap.astype(np.float32)

