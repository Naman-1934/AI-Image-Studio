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

## Hardware Requirements

This project runs fully locally using PyTorch and the pre-trained Stable Diffusion v1.5 model.

### Recommended (GPU – Best Experience)
- NVIDIA GPU with **6 GB+ VRAM** (e.g., RTX 2060 or better)
- 16 GB system RAM
- Generation time: **~4–8 seconds** per 512×512 image (depending on GPU and settings)

### Minimum (CPU-Only)
- Quad-core CPU (modern Intel/AMD)
- **8 GB RAM** (more recommended)
- Generation time: **~30–90 seconds** per 512×512 image

### Apple Silicon (M1/M2/M3)
- macOS with PyTorch MPS support
- Generation time: typically between CPU and mid-range GPU

The app automatically selects the best available device in this order:

1. 'cuda' (NVIDIA GPU with CUDA)
2. 'mps' (Apple Silicon GPU)
3. 'cpu' (fallback)

## Hardware Requirements

This project runs fully locally using PyTorch and the pre-trained Stable Diffusion v1.5 model.

### Recommended (GPU – Best Experience)
- NVIDIA GPU with **6 GB+ VRAM** (e.g., RTX 2060 or better)
- 16 GB system RAM
- Generation time: **~4–8 seconds** per 512×512 image (depending on GPU and settings)

### Minimum (CPU-Only)
- Quad-core CPU (modern Intel/AMD)
- **8 GB RAM** (more recommended)
- Generation time: **~30–90 seconds** per 512×512 image

### Apple Silicon (M1/M2/M3)
- macOS with PyTorch MPS support
- Generation time: typically between CPU and mid-range GPU

The app automatically selects the best available device in this order:

1. 'cuda' (NVIDIA GPU with CUDA)
2. 'mps' (Apple Silicon GPU)
3. 'cpu' (fallback)

## Usage Instructions (with Example Prompts)

### 1. Start the App

From the project root:

'''bash
streamlit run app.py

## Prompt Engineering Tips & Best Practices

The system applies simple prompt engineering automatically based on the selected style, but good prompts still matter.

### General Tips

- Be **specific**: describe subject, environment, lighting, and style.
  - Example: `a cozy reading corner with plants, warm lighting, soft shadows, 4k`
- Mention the **camera or medium** for photorealistic results:
  - `photographed on a 35mm lens, shallow depth of field`
- Add **quality terms**:
  - `highly detailed, 4k, sharp focus, cinematic lighting`

### Style-Specific Suggestions

- **Photorealistic**
  - `ultra realistic, 4k, high resolution, professional photography, highly detailed, sharp focus`
- **Artistic**
  - `digital art, concept art, artstation, cinematic lighting, painterly style`
- **Cartoon**
  - `cartoon style, 2d illustration, bold lines, flat colors, clean outline`

These style-specific suffixes are automatically added in code via `build_prompt()`.

### Using Negative Prompts

Negative prompts help remove common artifacts:

Recommended defaults:
- 'blurry, low quality, pixelated, deformed, disfigured, extra fingers, watermark, text, logo'

You can add your own negative terms on top of the defaults in the UI.  
The final negative prompt is constructed in code via `build_negative_prompt()`.

## Limitations & Future Improvements

### Current Limitations

- **Generation Time**
  - CPU-only setups are significantly slower than GPUs (30–90s vs 4–8s per 512×512 image).
- **Memory Usage**
  - Higher resolutions and more steps increase VRAM and RAM usage.
  - Very high resolutions may not fit on low-VRAM GPUs.
- **Safety Filter**
  - Uses a keyword-based content filter.
  - Can miss subtle unsafe content or give false positives on edge cases.
- **Model Generality**
  - Uses the base Stable Diffusion v1.5 model.
  - Not fine-tuned for domain-specific images (e.g., medical, product catalog, etc.).

### Planned / Potential Future Improvements

- **Model Fine-Tuning**
  - Fine-tune on custom datasets for specific domains (e.g., products, brand styles).
- **Style Transfer & Img2Img**
  - Add support to upload an image and generate variations or stylized versions.
- **Advanced Safety**
  - Integrate NSFW / toxic content classifiers instead of simple keyword matching.
- **Async / Streaming UI**
  - Show per-image progress and allow canceling long runs.
- **Multi-Model Support**
  - Allow choosing different open-source models from the UI (e.g., SDXL, other checkpoints).


