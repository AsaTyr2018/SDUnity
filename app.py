import gradio as gr
from PIL import Image
import numpy as np

def generate_image(prompt, negative_prompt):
    # Placeholder: returns random noise image
    arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(arr)

with gr.Blocks() as demo:
    gr.Markdown("# SDUnity - Baseline")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt")
        negative_prompt = gr.Textbox(label="Negative Prompt")
    generate_btn = gr.Button("Generate")
    output = gr.Image()
    generate_btn.click(generate_image, inputs=[prompt, negative_prompt], outputs=output)

if __name__ == "__main__":
    demo.launch()
