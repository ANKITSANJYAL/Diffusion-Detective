"""
Utility functions for image processing and encoding.
"""

import base64
from io import BytesIO
from PIL import Image
from typing import Union
import numpy as np


def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Convert PIL Image to base64 string.
    
    Args:
        image: PIL Image object
        format: Image format (PNG, JPEG, etc.)
    
    Returns:
        Base64 encoded string
    """
    buffered = BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"


def base64_to_pil(base64_string: str) -> Image.Image:
    """
    Convert base64 string to PIL Image.
    
    Args:
        base64_string: Base64 encoded image string
    
    Returns:
        PIL Image object
    """
    # Remove data URL prefix if present
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image


def numpy_to_pil(array: np.ndarray) -> Image.Image:
    """
    Convert numpy array to PIL Image.
    
    Args:
        array: Numpy array (H, W, C) with values in [0, 1] or [0, 255]
    
    Returns:
        PIL Image object
    """
    if array.max() <= 1.0:
        array = (array * 255).astype(np.uint8)
    
    return Image.fromarray(array)


def ensure_rgb(image: Image.Image) -> Image.Image:
    """
    Ensure image is in RGB mode.
    
    Args:
        image: PIL Image
    
    Returns:
        RGB PIL Image
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def resize_image(
    image: Image.Image,
    max_size: int = 512,
    maintain_aspect: bool = True
) -> Image.Image:
    """
    Resize image to maximum dimension.
    
    Args:
        image: PIL Image
        max_size: Maximum dimension size
        maintain_aspect: Whether to maintain aspect ratio
    
    Returns:
        Resized PIL Image
    """
    if maintain_aspect:
        # Calculate new dimensions maintaining aspect ratio
        width, height = image.size
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    else:
        return image.resize((max_size, max_size), Image.Resampling.LANCZOS)
