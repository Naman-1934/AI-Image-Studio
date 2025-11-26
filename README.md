# AI-Powered Text-to-Image Generator

This project implements a **text-to-image generation system** using a **pre-trained Stable Diffusion v1.5 model** (`runwayml/stable-diffusion-v1-5`) via Hugging Face Diffusers.

---

## Features

- Open-source, pre-trained text-to-image model (Stable Diffusion v1.5)
- Runs locally with **GPU (CUDA)** or **CPU** fallback
- Streamlit web UI:
  - Prompt input
  - Style selection: *Photorealistic, Artistic, Cartoon, Custom*
  - Adjustable parameters (steps, guidance scale, size, number of images, seed)
  - View and download images (PNG & JPEG)
  - Basic progress bar and estimated completion time
- Prompt engineering & negative prompts for better quality
- Simple content filtering for unsafe prompts
- Images saved with:
  - Watermark `"AI Generated"`
  - Per-run `metadata.json` (prompt, timestamp, parameters, file paths)
 
## Model Configuration

All model-related parameters — such as inference settings, safety rules, and hardware logic —
are documented inside:

📄 'config/model_config.yaml'

This ensures transparency and reproducibility when reviewing or extending the project.

---

## Project Structure

- `app.py` – Streamlit web interface
- `generator.py` – Pre-trained model loading, image generation, watermarking, saving, metadata
- `safety.py` – Content filtering, style prompt engineering, negative prompts
- `requirements.txt` – Python dependencies
- `generated/` – Auto-created folder with timestamped subfolders of images and metadata

---

## Setup & Installation

1. Clone the repo:

```bash
git clone https://github.com/<your-username>/ai-image-generator.git
cd ai-image-generator
