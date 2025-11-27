# app.py

import os
import time
import streamlit as st

from generator import TextToImageGenerator
from safety import check_prompt_safety


# ---------------- Page Config & Styles ---------------- #
st.set_page_config(
    page_title="AI Image Studio",
    page_icon="🎨",
    layout="wide"
)

st.markdown("""
<style>
    .hero {
        border-radius: 1rem;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        background: linear-gradient(135deg, #4f46e5, #111827);
        color: #ffffff;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #e5e7eb;
    }
    .pill {
        display: inline-block;
        padding: 0.15rem 0.7rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.15);
        font-size: 0.75rem;
        margin-right: 0.4rem;
        margin-top: 0.4rem;
    }
    .card {
        padding: 1.2rem;
        border-radius: 0.9rem;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        box-shadow: 0px 3px 18px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #6b7280;
    }
    .stButton>button {
        border-radius: 999px !important;
        height: 3rem;
        font-weight: 600;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- Model Loader ---------------- #
@st.cache_resource(show_spinner=True)
def load_generator():
    return TextToImageGenerator()

gen = load_generator()


# ---------------- App UI ---------------- #
def main():
    # Header
    st.markdown("""
    <div class="hero">
        <div class="hero-title">AI Image Studio</div>
        <div class="hero-subtitle">Create stunning images from simple text descriptions using AI.</div>
        <span class="pill">Text → Image</span>
        <span class="pill">Smart Filtering</span>
        <span class="pill">Watermarked</span>
    </div>
    """, unsafe_allow_html=True)

    # Two Equal Columns Layout
    left, right = st.columns(2, gap="large")

    # LEFT SIDE — PROMPT
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Describe Your Image</div>', unsafe_allow_html=True)

        prompt = st.text_area(
            "Prompt",
            height=160,
            placeholder="e.g. a futuristic city with neon lights and flying cars",
        )

        st.markdown('<div class="section-title">Avoid (optional)</div>', unsafe_allow_html=True)

        negative_prompt = st.text_area(
            "Negative Prompt",
            height=80,
            placeholder="e.g. blurry, text, watermark...",
        )

        generate = st.button("🚀 Generate Images", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # RIGHT SIDE — SETTINGS
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Settings</div>', unsafe_allow_html=True)

        style = st.selectbox("Style", ["Photorealistic", "Artistic", "Cartoon", "Custom"])
        num_images = st.slider("Number of Images", 1, 4, 1)
        guidance_scale = st.slider("Guidance Scale", 1.0, 15.0, 7.5, 0.5)
        num_steps = st.slider("Steps", 10, 60, 30)

        w, h = st.columns(2)
        with w:
            width = st.number_input("Width", 256, 1024, 512, step=64)
        with h:
            height = st.number_input("Height", 256, 1024, 512, step=64)

        seed = st.number_input("Seed (0 = Random)", min_value=0, value=0)
        base_filename = st.text_input("File Name Prefix", "image")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- IMAGE GENERATION ---------------- #
    if generate:
        if not prompt.strip():
            st.error("Please enter a text prompt.")
            st.stop()

        safe, reason = check_prompt_safety(prompt)
        if not safe:
            st.error(f"🚫 Content Blocked: {reason}")
            st.stop()

        with st.spinner("Generating... 🔥"):
            start = time.time()
            images, metadata = gen.generate_images(
                prompt=prompt,
                num_images=num_images,
                style=style,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                width=width,
                height=height,
                seed=seed if seed != 0 else None,
                base_filename=base_filename,
            )
            duration = time.time() - start

        st.success(f"Generated {num_images} image(s) in {duration:.1f} seconds ✔")

        # Display Images + Download Buttons
        for idx, (img, info) in enumerate(zip(images, metadata["images"]), start=1):
            st.image(img, caption=f"Image {idx}", use_column_width=True)

            colA, colB = st.columns(2)
            with colA:
                st.download_button(
                    label="⬇ Download PNG",
                    data=open(info["png_path"], "rb").read(),
                    file_name=os.path.basename(info["png_path"]),
                    mime="image/png",
                    key=f"png{idx}"
                )
            with colB:
                st.download_button(
                    label="⬇ Download JPG",
                    data=open(info["jpg_path"], "rb").read(),
                    file_name=os.path.basename(info["jpg_path"]),
                    mime="image/jpeg",
                    key=f"jpg{idx}"
                )

    # Footer
    st.markdown('<div class="footer">© 2025 – AI Image Studio by Naman</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
