# generator.py

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont

from safety import build_prompt, build_negative_prompt


class TextToImageGenerator:
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        output_dir: str = "generated",
        device: Optional[str] = None,
    ):
        """
        Text-to-image generator using a **pre-trained** Stable Diffusion model.

        Parameters
        ----------
        model_id : str
            Hugging Face model ID of a pre-trained text-to-image model.
            Default: "runwayml/stable-diffusion-v1-5" (pre-trained Stable Diffusion v1.5).
        output_dir : str
            Directory where generated images + metadata will be saved.
        device : str or None
            "cuda", "cpu", "mps" or None for auto-detect.
        """
        self.model_id = model_id
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # ---- Device selection (GPU preferred, then MPS, then CPU) ----
        if device is not None:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon
            else:
                self.device = "cpu"

        # Use float16 on GPU/MPS (more efficient), float32 on CPU
        if self.device in ["cuda", "mps"]:
            dtype = torch.float16
        else:
            dtype = torch.float32

        # ---- Load PRE-TRAINED Stable Diffusion pipeline ----
        # This downloads / loads pre-trained weights from Hugging Face Hub.
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            safety_checker=None,  # we use our own simple filtering in safety.py
        )

        # Move model to the selected device
        self.pipe = self.pipe.to(self.device)

    def _add_watermark(self, image: Image.Image, text: str = "AI Generated") -> Image.Image:
        """ Add a small semi-transparent watermark to the bottom-right corner.
        Compatible with Pillow 10.x (textsize removed). """
        img = image.copy().convert("RGBA")
        width, height = img.size

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        font = ImageFont.load_default()

        # Measure text using textbbox instead of textsize
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        text_padding = 10
        x = width - text_w - text_padding
        y = height - text_h - text_padding

        # Draw semi-transparent background rectangle
        rect_x0 = x - 5
        rect_y0 = y - 2
        rect_x1 = x + text_w + 5
        rect_y1 = y + text_h + 2
        draw.rectangle((rect_x0, rect_y0, rect_x1, rect_y1), fill=(0, 0, 0, 120))

        # Draw text in white
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 230))

        watermarked = Image.alpha_composite(img, overlay)
        return watermarked.convert("RGB")


    def _create_run_dir(self) -> str:
        """
        Create a new directory for each generation run with timestamp.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.output_dir, ts)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def generate_images(
        self,
        prompt: str,
        num_images: int = 1,
        style: str = "Photorealistic",
        negative_prompt: str = "",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
        base_filename: Optional[str] = None,
    ) -> Tuple[List[Image.Image], Dict]:
        """
        Generate images from a text prompt using the pre-trained Stable Diffusion model.

        Returns
        -------
        images : list of PIL.Image.Image
        metadata : dict
            Metadata for the entire run, including per-image info and file paths.
        """
        # Make CPU settings more conservative
        if self.device == "cpu":
            num_inference_steps = min(num_inference_steps, 30)
            width = min(width, 512)
            height = min(height, 512)

        if num_images < 1:
            num_images = 1
        if num_images > 6:
            num_images = 6  # avoid extreme values

        # Build prompts (prompt engineering + negative prompt)
        full_prompt = build_prompt(prompt, style)
        full_negative = build_negative_prompt(negative_prompt)

        # Run-specific directory
        run_dir = self._create_run_dir()
        safe_base_name = (
            base_filename.strip().replace(" ", "_").replace("/", "_")
            if base_filename
            else "image"
        )

        all_image_infos: List[Dict] = []
        all_images: List[Image.Image] = []

        for idx in range(num_images):
            if seed is not None:
                this_seed = seed + idx
                generator = torch.Generator(device=self.device).manual_seed(this_seed)
            else:
                generator = None
                this_seed = None

            start_time = time.time()
            result = self.pipe(
                prompt=full_prompt,
                negative_prompt=full_negative,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                num_images_per_prompt=1,
                generator=generator,
            )
            elapsed = time.time() - start_time

            image = result.images[0]
            image = self._add_watermark(image, text="AI Generated")

            # Save PNG and JPEG
            png_name = f"{safe_base_name}_{idx+1}.png"
            jpg_name = f"{safe_base_name}_{idx+1}.jpg"
            png_path = os.path.join(run_dir, png_name)
            jpg_path = os.path.join(run_dir, jpg_name)

            image.save(png_path, format="PNG")
            image.save(jpg_path, format="JPEG", quality=95)

            image_info = {
                "index": idx + 1,
                "prompt": prompt,
                "full_prompt": full_prompt,
                "negative_prompt": full_negative,
                "style": style,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "width": width,
                "height": height,
                "seed": this_seed,
                "png_path": png_path,
                "jpg_path": jpg_path,
                "generation_time_sec": elapsed,
                "timestamp": datetime.now().isoformat(),
            }

            all_images.append(image)
            all_image_infos.append(image_info)

        metadata: Dict = {
            "model_id": self.model_id,
            "device": self.device,
            "created_at": datetime.now().isoformat(),
            "prompt": prompt,
            "style": style,
            "num_images": num_images,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "width": width,
            "height": height,
            "seed_base": seed,
            "images": all_image_infos,
            "run_dir": run_dir,
        }

        meta_path = os.path.join(run_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return all_images, metadata
