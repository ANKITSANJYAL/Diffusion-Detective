"""Gradio demo: prompt -> (LLM reasoning) -> SD generation with intermediate images -> per-step heatmaps."""
import gradio as gr
from src.runner import ModelRunner
from src.fusion import FusionMapper
from src.reasoner import Reasoner
from PIL import Image
import numpy as np


DEVICE = "cuda:0"
runner = ModelRunner(device=DEVICE)
fusion = FusionMapper(device=DEVICE)
reasoner = Reasoner(model_name=None, device=DEVICE)  # set model_name to enable LLM


def generate(prompt: str, steps: int = 20, n_reasoning: int = 4):
    # 1) generate reasoning steps (LLM or heuristic)
    reasoning_steps = reasoner.generate_steps(prompt, n_steps=n_reasoning)

    # 2) compute step weights and run SD with a guidance schedule
    step_weights = reasoner.compute_weights(reasoning_steps)
    capture_indices = [int(steps * i / (n_reasoning + 1)) for i in range(1, n_reasoning + 1)]
    gen = runner.generate_with_intermediates(prompt, steps=steps, guidance=7.5, height=512, width=512, capture_steps=capture_indices, step_weights=step_weights)
    final_img = gen['final_image']
    intermediates = gen['intermediates']

    # 3) compute heatmaps per reasoning step for the final image (fast) and for each intermediate
    heatmaps = []
    heatmap_imgs = []
    for step_text in reasoning_steps:
        hm = fusion.text_to_heatmap(final_img, step_text, output_size=(64, 64))
        heatmaps.append(hm)
        hm_img = Image.fromarray((hm * 255).astype(np.uint8)).resize(final_img.size)
        heatmap_imgs.append(hm_img)

    # prepare output display: final image, reasoning text, and list of heatmaps and intermediate images
    reasoning_display = "\n".join([f"{i+1}. {s}  (w={step_weights[i]:.2f})" for i, s in enumerate(reasoning_steps)])

    # combine intermediates and heatmaps for gallery output: show final with each heatmap overlayed
    gallery = []
    # show intermediates (if any)
    for (idx, img) in intermediates:
        gallery.append(img)
    # then show heatmap overlays on final image
    for hm in heatmap_imgs:
        gallery.append(hm)

    return final_img, reasoning_display, gallery


with gr.Blocks() as demo:
    gr.Markdown("## Diffusion Detective — Demo")
    with gr.Row():
        inp = gr.Textbox(label="Prompt", placeholder="A detective in a foggy alley, cinematic lighting")
        out_img = gr.Image(label="Generated Image")
    steps_txt = gr.Textbox(label="Reasoning Steps")
    out_gallery = gr.Gallery(label="Intermediates & Evidence Maps").style(grid=[2])
    btn = gr.Button("Generate")
    btn.click(generate, inputs=[inp], outputs=[out_img, steps_txt, out_gallery])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
