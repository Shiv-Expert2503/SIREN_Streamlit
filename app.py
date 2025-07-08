import streamlit as st
from utils import load_image, run_siren_model, upscale_image, psnr, heat_map, crop_zoom, sharpen_image, zoomable_image
import torch
from model import Siren
import os 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import base64
import streamlit.components.v1 as components
import traceback

st.set_page_config(page_title="SIREN Demo", layout="centered")

st.title("📸 SIREN - Image as a Neural Function")

option = st.sidebar.selectbox(
    "Choose a feature:",
    ("Documentation","Grayscale Reconstruction", "Color Reconstruction", "Upsampling", "Compression")
)

if option == "Documentation":
    with st.sidebar.expander("ℹ️ About This App"):
        st.markdown("""
        This tool visualizes how **SIREN** can represent and reconstruct images using only neural networks.

        - Built using PyTorch + Streamlit
        - Fully interpretable and interactive
        - PSNR, Heatmaps, Hover Zoom included!
        """)

    st.title("📖 SIREN - Documentation")

    st.markdown("""
    ### What is SIREN?
    **SIREN (Sinusoidal Representation Networks)** is a neural architecture that learns a continuous function to represent signals. It uses periodic activation functions (sine/cosine) which are especially good at learning high-frequency details.

    ---

    ### Modules in This Demo
    - **Grayscale Reconstruction** – Reconstructs grayscale image using SIREN
    - **Color Reconstruction** – Same concept applied on RGB images
    - **Upsampling** – SIREN-based continuous representation used to zoom/upsample images at 2×, 4×, 8×, etc.
    - **Heatmap Visualizations** – Color-coded difference map showing where reconstructions deviate from ground truth
    - **Zoom & Hover Lens** – Explore images with zoom interaction like Amazon product views

    ---

    ### Metric Used → PSNR
    - PSNR stands for **Peak Signal-to-Noise Ratio**.
    - A higher PSNR means the reconstructed image is closer to the original.
    - **Above 30 dB: Generally considered good quality.**
    - **Above 40 dB** is considered excellent quality in image processing and the difference is often imperceptible to the human eye.

    ---

    ### Heatmap Visualization
    - Shows **pixel-level differences** using a `jet` colormap.
    - Red → high error, Blue → low error.
    - Helps **diagnose** where my model is underperforming visually.
    - In RGB images, Black areas indicate no difference, while brighter areas show larger discrepancies.

    ---

    ### Tech Stack
    - **Frontend**: Streamlit
    - **Backend**: PyTorch
    - **Image Handling**: NumPy, PIL
    - **Visualization**: Matplotlib, OpenCV

    ---

    ### Notes
    - Training done from scratch.
    - Models are light and efficient for inference.
    - Zooming and UI are optimized to avoid GPU overload.
    - Custom Activation function used for better convergence.
    - Likewise Custom Weight Initialization for better performance.
    - He/Xavier are not used here as they are Relu/tanh specific.
    - Adding $\\omega_0$ solves the problem of diminishing gradient or vanishing gradient, as 
    """, unsafe_allow_html=True)

    st.latex(r"y = \sin(\omega_0 x)")
    st.latex(r"\frac{dy}{dx} = \omega_0 \cos(\omega_0 x)")
    st.latex(r"\frac{d^2y}{dx^2} = -\omega_0^2 \sin(\omega_0 x)")

    st.markdown("""
    - So If $\\omega_0$ is large (like 30), the magnitude of the gradient (derivative) is also scaled up by $\\omega_0$.
    ---
    - **[SIREN Image Training Notebook](https://www.kaggle.com/code/shivansh2503/image-compression)**
    
    - Built with ❤️ by shiv_expert
    """, unsafe_allow_html=True)

    st.success("Navigate from the sidebar to try each module live with demo images!")


# elif option == "Grayscale Reconstruction":
#     st.subheader("🖤 Grayscale Image Reconstruction")
#     st.markdown("""
#     In this section, my SIREN model learns to represent a grayscale image.
#     The model takes `(x, y)` coordinates as input and predicts the grayscale pixel intensity at that location.
#     Observe how closely the reconstructed image matches the original, even after learning from scratch!
#     """)
#     img = load_image("data/apple.jpg", mode='L')
#     # model = torch.load("models/model_grayscale.pth", map_location='cpu', weights_only=False)
#     model = Siren(inputs=2, hidden_features=256, hidden_layers=3, output_number=1)
#     model.load_state_dict(torch.load("models/model_grayscale_1.pth", map_location='cpu'))
#     with st.spinner("Reconstructing..."):
#         output = run_siren_model(model, img, grayscale=True)
#     # st.image([img, output], caption=["Original", "Reconstructed"], width=300)
#     col1, col2 = st.columns(2)

#     with col1:
#         st.markdown("**Original Image**")
#         zoomable_image(img, width=300)

#     with col2:
#         st.markdown("**Reconstructed Image**")
#         zoomable_image(output, width=300)

#     st.subheader("📊 Image Comparison")
#     st.markdown("""
#     These metrics quantify the similarity between the original and the SIREN-reconstructed image.
#     """)

#     st.markdown("#### **Peak Signal-to-Noise Ratio (PSNR)**")
#     st.info(f"""
#     **PSNR: {psnr(np.array(img), np.array(output)):.2f} dB**

#     PSNR is a common metric to measure image quality. A higher PSNR indicates that the
#     reconstructed image is very close to the original.
#     * **Above 30 dB:** Generally considered good quality.
#     * **Above 40 dB:** Often visually indistinguishable from the original for most human eyes.
#     My current PSNR of **{psnr(np.array(img), np.array(output)):.2f} dB** indicates an excellent reconstruction!
#     """)
#     # st.text(f"PSNR: {psnr(np.array(img), np.array(output)):.2f} dB")

#     st.markdown("#### **Heat Map of Differences**")
#     st.markdown("""
#     This heatmap visually highlights the pixel-wise discrepancies between the **Original Image**
#     and the **Reconstructed Image**.
#     * **Darker areas (or cooler colors like blue):** Indicate very small or no difference (pixels are almost identical).
#     * **Brighter areas (or warmer colors like red):** Indicate larger differences.
#     * So here you can see the inside of the apple is reconstructed very well (blue + light red) signifying that errors are minimal and distributed, not concentrated in visible artifacts., while the outside has some minor differences (Glowing red).
#     """)
#     st.image(heat_map(np.array(img), np.array(output)), caption="Heat Map of Differences", width=300)

elif option == "Grayscale Reconstruction":
    st.subheader("🖤 Grayscale Image Reconstruction")
    st.markdown("""
    In this section, a SIREN model learns to represent a grayscale image.
    The model takes `(x, y)` coordinates as input and predicts the grayscale pixel intensity at that location.
    Observe how closely the reconstructed image matches the original, even after learning from scratch!
    """)

    # --- Start of the main TRY block for debugging ---
    try:
        img = load_image("data/apple.jpg", mode='L')
        # Ensure your Siren model parameters (in_features, out_features) match your trained model
        model = Siren(inputs=2, hidden_features=256, hidden_layers=3, output_number=1)
        model.load_state_dict(torch.load("models/model_grayscale_1.pth", map_location='cpu'))

        with st.spinner("Reconstructing..."):
            # This is where the core reconstruction happens
            try:
                output = run_siren_model(model, img, grayscale=True)
            except RuntimeError as e:
                st.error("⚠️ Ran out of memory during reconstruction. Try a smaller image or model.")
                st.markdown(f"Error details: `{e}`")
                print(f"Error during reconstruction: {e}")
                
            except Exception as e:
                st.error("⚠️ An unexpected error occurred during reconstruction!")
                st.error(f"Error Details: {e}")
                st.code(traceback.format_exc(), language='python') # Display full traceback in the Streamlit UI
                print(f"\n--- ERROR IN GRAYSCALE RECONSTRUCTION ---")
                print(f"Error message: {e}")
                print(traceback.format_exc())
                

        col1, col2 = st.columns(2)
        try:
            with col1:
                st.markdown("**Original Image**")
                zoomable_image(img, width=300) # This calls a utility function

            with col2:
                st.markdown("**Reconstructed Image**")
                zoomable_image(output, width=300) # This calls a utility function
        except Exception as e:
            st.image([img, output], caption=["Original", "Reconstructed"], width=300)

        # --- Comment out the PSNR and Heatmap sections for the first test ---
        # If the app now works (doesn't crash) without these, then uncomment them one by one.
        # If it still crashes, the error is somewhere in the code ABOVE this comment block.

        # st.subheader("📊 Image Comparison")
        # st.markdown("""
        # These metrics quantify the similarity between the original and the SIREN-reconstructed image.
        # """)
        #
        # st.markdown("#### **Peak Signal-to-Noise Ratio (PSNR)**")
        # st.info(f"""
        # **PSNR: {psnr(np.array(img), np.array(output)):.2f} dB**
        #
        # PSNR is a common metric to measure image quality. A higher PSNR indicates that the
        # reconstructed image is very close to the original.
        # * **Above 30 dB:** Generally considered good quality.
        # * **Above 40 dB:** Often visually indistinguishable from the original for most human eyes.
        # My current PSNR of **{psnr(np.array(img), np.array(output)):.2f} dB** indicates an excellent reconstruction!
        # """)
        #
        # st.markdown("#### **Heat Map of Differences**")
        # st.markdown("""
        # This heatmap visually highlights the pixel-wise discrepancies between the **Original Image**
        # and the **Reconstructed Image**.
        # * **Darker areas (or cooler colors like blue):** Indicate very small or no difference (pixels are almost identical).
        # * **Brighter areas (or warmer colors like red):** Indicate larger differences.
        # * So here you can see the inside of the apple is reconstructed very well (blue + light red) signifying that errors are minimal and distributed, not concentrated in visible artifacts., while the outside has some minor differences (Glowing red).
        # """)
        # st.image(heat_map(np.array(img), np.array(output)), caption="Heat Map of Differences", width=300)

    # --- Catch any exception that occurs within the try block ---
    except Exception as e:
        st.error(f"⚠️ An unexpected error occurred during Grayscale Reconstruction!")
        st.error(f"Error Details: {e}")
        st.code(traceback.format_exc(), language='python') # Display full traceback in the Streamlit UI

        # Also print to the underlying Streamlit Cloud logs (Manage App -> Logs)
        print(f"\n--- ERROR IN GRAYSCALE RECONSTRUCTION ---")
        print(f"Error message: {e}")
        print(traceback.format_exc())
        print(f"-----------------------------------------\n")



elif option == "Color Reconstruction":
    st.subheader("🌈 Color Image Reconstruction")
    st.markdown("""
    Similar to grayscale, this section demonstrates how my SIREN learned to represent
    a full-color (RGB) image. My model now predicts 3 color channels (Red, Green, Blue)
    for each `(x, y)` coordinate.
    """)
    img = load_image("data/pika128.jpg", mode='RGB')
    model = Siren(inputs=2, hidden_features=256, hidden_layers=3, output_number=3)
    model.load_state_dict(torch.load("models/colored.pth", map_location='cpu'))
    # model = torch.load("models/siren_rgb.pth", map_location='cpu')
    with st.spinner("Upscaling..."):
        output = run_siren_model(model, img, grayscale=False)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Image**")
        zoomable_image(img, width=300)

    with col2:
        st.markdown("**Reconstructed Image**")
        zoomable_image(output, width=300)
    # st.image([img, output], caption=["Original", "Reconstructed"], width=300)

    st.subheader("📊 Image Comparison")
    st.markdown("#### **PSNR**")
    st.info(f"PSNR: {psnr(np.array(img), np.array(output)):.2f} dB - Another excellent reconstruction!")

    st.markdown("#### **Heat Map of Differences**")
    st.markdown("""
    In this heatmap you can interpret it similarly to the grayscale heatmap:
    darker means less difference, brighter/warmer means more difference.
    Here since there are no major artifacts, the heatmap is mostly on the darker side, with no bright areas.
    This indicates that the SIREN model has successfully captured the color distribution and details of the original image.
    """)
    st.image(heat_map(np.array(img), np.array(output), grayscale=False), caption="Heat Map of Differences", width=300)
    # output.save("data/reconstructed_image_pika.jpg")

    st.subheader("🔍 Zoom-in Effect")
    st.markdown("""
    Observe the difference in sharpness and detail between zooming into the original pixel-based
    image versus the continuous SIREN-reconstructed image. SIRENs often maintain better clarity.
    """)
    st.image(["data/pika_rgb_zoom.gif","data/zoom_reconstructed.gif"], caption=["Zooming into the Original", "Zooming into the Reconstructed"], width=300)
    
    st.subheader("🔍 Zoomed Comparison (Center Region)")
    st.markdown("""
    A direct side-by-side comparison of a cropped central region further highlights
    the reconstruction quality.
    """)
    original_zoom = crop_zoom(img)
    recon_zoom = crop_zoom(img=output)
    st.image([original_zoom, recon_zoom], caption=["Original Zoom", "Reconstructed Zoom"], width=300)


elif option == "Upsampling":
    st.subheader("🔼 Upsample Low-Resolution Image")
    st.markdown("""
    One of the most exciting applications of SIRENs is **super-resolution**.
    Because the model learns a *continuous function*, it can be queried at an arbitrarily
    higher density of coordinates than the original image, effectively generating a higher
    resolution version without typical pixelation or blurring artifacts common in
    traditional upscaling methods.
    """)
    scale = st.selectbox("Select upsampling factor", [2, 4, 8, 16])
    # img = load_image("data/low_res.png", mode='RGB')
    img = load_image("data/pika128.jpg", mode='RGB')
    # model = torch.load("models/siren_4x.pth", map_location='cpu')  # Use different model per scale if needed
    model = Siren(inputs=2, hidden_features=256, hidden_layers=3, output_number=3)

    model.load_state_dict(torch.load("models/colored.pth", map_location='cpu'))
    # output = upscale_image(model, img, scale=scale)
    # st.image([img, output], caption=["Low Res", f"Upscaled {scale}×"], width=300)
    try:
        with st.spinner("Upscaling..."):
            output = upscale_image(model, img, scale=scale)
        st.image([img, output], caption=["Low Res", f"Upscaled {scale}×"], width=300)

        st.subheader("🔍 Zoomed Comparison (Center Region)")
        st.markdown("""
        Observe how the SIREN-upscaled image maintains details and smooth edges
        even at high zoom levels, compared to simple pixel scaling.
        """)
        original_zoom = crop_zoom(img)
        recon_zoom = crop_zoom(img=output, upsample_factor=scale)

        st.image([original_zoom, recon_zoom], caption=["Original Zoom", "Upscaled Zoom"], width=300)


    except RuntimeError as e:
        st.error("⚠️ Ran out of memory during upsampling. Try a smaller scale or image.")
        st.markdown("""
        This typically happens when attempting very large upscaling factors on high-resolution images,
        especially on free-tier GPUs with limited memory. Try a smaller upscaling factor or a smaller source image.
        The system logs show: `""" + str(e) + "`")
        print(e)

        print(e)

elif option == "Compression":
    st.subheader("🗜️ Image Compression via SIREN")
    st.markdown("""
    In the context of SIRENs, the 'compressed' image is not a file derived from the image itself,
    but rather the **trained neural network model weights**. The idea is that if the model is small,
    it compactly represents the image. Here, we compare the size of the saved SIREN model
    to a standard JPEG-compressed version of the reconstructed image.
    """)

    st.info("""
        **When JPEG might become smaller:**
        For smaller images, the fixed number of parameters (the 'overhead') in a SIREN model
        can often be larger than what highly-optimized standard compression algorithms like JPEG achieve.
        SIRENs often show compression benefits for very large, high-resolution images or when
        their continuous, resolution-independent nature is the primary benefit.
        """)
# elif option == "Compression":
#     st.subheader("🗜️ Image Compression via SIREN")
#     img = load_image("data/apple.jpg", mode='RGB')

#     model = Siren(inputs=2, hidden_features=256, hidden_layers=3, output_number=1)
#     model.load_state_dict(torch.load("models/model_grayscale_1.pth", map_location='cpu'))
#     # output = run_siren_model(model, img, grayscale=True)
#     # st.image([img, output], caption=["Original", "SIREN Compressed (Reconstructed)"], width=300)
#     siren_model_size_bytes = os.path.getsize("models/model_grayscale_1.pth")
#     # print(f"Size of SIREN model weights: {siren_model_size_bytes / 1024:.2f} KB")
#     st.text(f"Size of SIREN model weights: {siren_model_size_bytes / 1024:.2f} KB")

#     jpeg_quality = 95 # High quality JPEG
#     output = run_siren_model(model, img, grayscale=True)

#     jpeg_compare_path = 'data/reconstructed_image_compare.jpg'

#     output.save(jpeg_compare_path, quality=jpeg_quality)
#     jpeg_size_bytes = os.path.getsize(jpeg_compare_path)
#     st.text(f"Size of JPEG image: {jpeg_size_bytes / 1024:.2f} KB")
#     # print(f"Reconstructed image saved as JPEG to: {jpeg_compare_path} (Quality: {jpeg_quality})")
#     # print()
#     if siren_model_size_bytes < jpeg_size_bytes:
#         st.text(f"SIREN model is smaller than JPEG by {(jpeg_size_bytes - siren_model_size_bytes) / 1024:.2f} KB!")
#         st.text(f"Compression Ratio (JPEG/SIREN): {jpeg_size_bytes / siren_model_size_bytes:.2f}x")
#         # print(f"SIREN model is smaller than JPEG by {(jpeg_size_bytes - siren_model_size_bytes) / 1024:.2f} KB!")
#         # print(f"Compression Ratio (JPEG/SIREN): {jpeg_size_bytes / siren_model_size_bytes:.2f}x")
#     elif siren_model_size_bytes > jpeg_size_bytes:
#         st.text(f"JPEG is smaller than SIREN model by {(siren_model_size_bytes - jpeg_size_bytes) / 1024:.2f} KB.")
#         st.text(f"Compression Ratio (SIREN/JPEG): {siren_model_size_bytes / jpeg_size_bytes:.2f}x")

#     else:
#         st.text("SIREN model size is equal to JPEG size.")
        




#streamlit run app.py
#conda activate siren-env

#Docs
#Wordings and limitations
# PSNR display
#Deploy
