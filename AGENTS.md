# Project Name: SDUnity
Type: Web-based Stable Diffusion Interface
Framework: Gradio (Python)

## Objective
The goal of SDUnity is to create a modern, self-hosted web application that acts as a user-friendly interface for generating and managing AI-generated images using Stable Diffusion. The project will be built using Gradio as the front-end framework for simplicity and rapid development, enabling smooth interaction with the model backend.

# Core Components
## THE Generator
The centerpiece of the application.
Text-to-Image generation using Stable Diffusion.

### Support for:
Prompt + Negative Prompt input fields.
Real-time seed control.
Aspect ratio and steps/sampling settings.
A “Smooth Preview” mode that shows image transitions between steps (like a mini-timelapse).

__Optional: Image2Image and Inpainting/Outpainting support (future phase).__

## Model Selector
Choose from multiple model backends at runtime:
Supported formats: Stable Diffusion 1.5, SDXL, PonyXL (initially).
Additional models can be added via config.

## LoRA Library
Unified management for all LoRA models:
Structured browser for LoRA files (grouped by category or style).
Auto-detection of new LoRAs in a designated folder.
Optional tags, metadata, and preview images.
Direct injection of selected LoRAs into the prompt.

## WebGallery
A browsable gallery of all generated images:

Displays images along with:
Prompt
Model
LoRA used
Generation date & time

Sorting & filtering by model, LoRA, date, and tags.
Bulk download/export options.
Optional: Create themed "Collections" or "Albums".

## Optional (Future Roadmap Ideas)
User system (with local login/authentication and usage tracking).
API access for external tools to call the generator.
Dynamic LoRA merging/blending.
Integration with your FiftyOne-based deduplication pipeline.
Animated LoRA preview (show effect across same prompt).
Multi-user collaboration gallery ("Community Mode").

Naming Note
“SDUnity” emphasizes the unity of features — generator, models, LoRAs, and gallery — under one cohesive UI.

