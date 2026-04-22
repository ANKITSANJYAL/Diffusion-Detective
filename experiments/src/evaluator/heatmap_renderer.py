import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class HeatmapRenderer:
    """
    Python-native attention heatmap renderer.
    Bypasses the React frontend to generate visual heatmaps for the LLaVA Judge.
    """
    @staticmethod
    def render_and_save(attention_list: list, image_path: str, output_path: str, alpha: float = 0.6):
        """
        Takes raw 1D attention logs (length 256 for 16x16), reshapes them, applies a
        spectral colormap, and alpha-blends exactly over the generated image.
        """
        try:
            # 1. Load base image
            base_img = Image.open(image_path).convert("RGBA")
            width, height = base_img.size
            
            # 2. Reshape and normalize attention array
            # Assuming standard 16x16 Stable Diffusion Cross-Attention resolution
            grid_size = int(np.sqrt(len(attention_list)))
            attn_grid = np.array(attention_list).reshape((grid_size, grid_size))
            
            # Normalize strictly to [0, 1]
            if attn_grid.max() > 0:
                attn_grid = attn_grid / attn_grid.max()
                
            # 3. Apply Colormap
            # Using 'jet' to match traditional CVPR standard saliency maps
            colormap = plt.get_cmap('jet')
            heatmap_rgba = colormap(attn_grid) # returns [16, 16, 4]
            
            # 4. Convert to Image and resize to target resolution (512x512)
            # Use BILINEAR or BICUBIC to smooth the 16x16 blocks
            heatmap_img = Image.fromarray((heatmap_rgba * 255).astype(np.uint8), "RGBA")
            heatmap_img = heatmap_img.resize((width, height), resample=Image.Resampling.BICUBIC)
            
            # 5. Alpha Blend
            # We want the heatmap to be semi-transparent over the original image
            # Create an alpha mask based on the intensity (so low attention = highly transparent)
            alpha_mask = Image.fromarray((attn_grid * alpha * 255).astype(np.uint8), "L")
            alpha_mask = alpha_mask.resize((width, height), resample=Image.Resampling.BICUBIC)
            heatmap_img.putalpha(alpha_mask)
            
            blended = Image.alpha_composite(base_img, heatmap_img)
            
            # Save to disk as RGB
            blended.convert("RGB").save(output_path)
            return True
            
        except Exception as e:
            print(f"Heatmap Rendering Failed: {e}")
            return False
