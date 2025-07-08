import math
from PIL import Image
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import base64
import streamlit.components.v1 as components

def preprocess(img):


    # Converting grayscale image into pixels
    image_np = np.array(img)

    #normalising the image
    if len(image_np.shape) == 2:
        # Grayscale
        pixels = (image_np / 255.0) * 2 - 1
        pixels_reshape = pixels.reshape(-1, 1)
        H, W = image_np.shape
    else:
        # Color
        pixels = (image_np / 255.0) * 2 - 1
        H, W, C = image_np.shape
        pixels_reshape = pixels.reshape(-1, C)

    x = np.linspace(-1, 1, W)
    y = np.linspace(-1, 1, H)

    xx, yy = np.meshgrid(x,y, indexing="xy")

    coords = np.stack([xx, yy],axis=-1).reshape(-1, 2)

    X = torch.tensor(coords, dtype = torch.float32) # Coordinates flatten -> Given as input to siren
    Y = torch.tensor(pixels_reshape, dtype = torch.float32) # GT used for MSE and Back propogation MSE(Y', Y)
    return X, Y, H, W

def batched_predict(model, X, batch_size=10000):
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            out = model(batch)
            outputs.append(out.cpu())
    return torch.cat(outputs, dim=0)

def sharpen_image(image_array, sharpen_strength=1.5):
    """
    Sharpens an image array using a basic sharpening kernel.
    Args:
        image_array (np.ndarray): The input image array (H, W, 3) and uint8 dtype.
        sharpen_strength (float): Controls the intensity of sharpening.
                                  A common value is between 1.0 and 3.0.
    Returns:
        np.ndarray: The sharpened image array.
    """
    # Define a basic sharpening kernel
    # This kernel emphasizes the center pixel relative to its neighbors
    # kernel = np.array([[-1, -1, -1],
    #                    [-1,  9, -1],
    #                    [-1, -1, -1]]) # Simple sharpening kernel, sum is 1

    # Alternatively, for a stronger effect:
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]]) # Common sharpening kernel, sum is 1

    # Apply the kernel to the image
    # We use cv2.filter2D for convolution
    # -1 indicates that the output image will have the same depth as the input
    sharpened_image = cv2.filter2D(image_array, -1, kernel)

    # Note: For Unsharp Masking, a more robust approach:
    # blurred = cv2.GaussianBlur(image_array, (0,0), 3) # blur with 3 sigma
    # sharpened = cv2.addWeighted(image_array, 1.5, blurred, -0.5, 0) # Adjust weights for strength
    # sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return sharpened_image

def load_image(path, mode='RGB'):
    img = Image.open(path).convert(mode)
    return img

def run_siren_model(model, img, grayscale=True):
    # Preprocess (convert to normalized tensor)
    X, Y, H, W = preprocess(img)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(X)

    # Reconstruct image
    output_image = output.reshape(H, W, 3) if output.shape[1] > 1 else output.reshape(H, W)

    output_image = ((output_image + 1) / 2.0) * 255
    output_image = output_image.cpu().numpy() 
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)

    # Return final PIL image or tensor
    if len(output_image.shape) == 2:
        # image_array = Image.fromarray(output_image, mode="L")
        sharpened_upsampled_array = sharpen_image(output_image, sharpen_strength=1.5)
        return Image.fromarray(sharpened_upsampled_array, mode="L")

    
    else:
        # image_array = Image.fromarray(output_image, mode="L")
        # sharpened_upsampled_array = sharpen_image(output_image, sharpen_strength=1.5)
        # return Image.fromarray(sharpened_upsampled_array, mode="RGB")
        return Image.fromarray(output_image, mode="RGB")


def upscale_image(model, img, scale=4):
    # For colored images
    X, Y, H, W = preprocess(img)

    model.eval()
    upsample_factor = scale


    upsampled_height = math.ceil(H * upsample_factor)
    upsampled_width = math.ceil(W * upsample_factor)


    x_coords_up = torch.linspace(-1, 1, upsampled_width)
    y_coords_up = torch.linspace(-1, 1, upsampled_height)

    X_upsampled = torch.stack(torch.meshgrid(x_coords_up, y_coords_up, indexing='xy'), dim=-1)

    X_upsampled = X_upsampled.reshape(-1, 2) # Flatten to (num_pixels, 2)

    with torch.no_grad(): # No need for gradients during inference
        upsampled_outputs = batched_predict(model, X_upsampled, batch_size=8192).numpy()
        # upsampled_outputs = model(X_upsampled).cpu().numpy()

    upsampled_image_array = upsampled_outputs.reshape(upsampled_height, upsampled_width, 3) # Assuming 3 channels for color
    upsampled_image_array = np.clip(upsampled_image_array, 0, 1) # Clip values to [0, 1] range
    upsampled_image_array = (upsampled_image_array * 255).astype(np.uint8) # Scale to [0, 255] and convert to uint8

    # --- 7. Convert to PIL Image and Save/Display ---
    # upsampled_pil_image = Image.fromarray(upsampled_image_array)
    sharpened_upsampled_array = sharpen_image(upsampled_image_array, sharpen_strength=1.5)
    sharpened_upsampled_image = Image.fromarray(sharpened_upsampled_array)

    return sharpened_upsampled_image

def psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def heat_map(original, reconstructed, grayscale=True):
    diff_map = np.abs(np.array(reconstructed) - np.array(original))

    if diff_map.ndim == 3 and diff_map.shape[2] == 3:
        # Per-pixel magnitude of difference
        diff_map = np.linalg.norm(diff_map, axis=2)

    diff_map_norm = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)

    if grayscale:
        colormap = plt.get_cmap('jet')
        heatmap = colormap(diff_map_norm)

        # Convert to RGB image for Streamlit
        heatmap_img = (heatmap[:, :, :3] * 255).astype(np.uint8)

    else:
        heatmap_img = (diff_map_norm * 255).astype(np.uint8)
    return heatmap_img


def crop_zoom(img, upsample_factor=1, zoom_factor=4, crop_size=64):
    """
    Crops a central square region and resizes it (zooms in).
    """
    img_np = np.array(img)
    h, w = img_np.shape[:2]
    
    cx, cy = w // 2, h // 2
    true_crop_size = crop_size * upsample_factor
    half = true_crop_size // 2
    # Crop center
    cropped = img_np[cy - half: cy + half, cx - half: cx + half]
    pil_crop = Image.fromarray(cropped)

    # Resize for zoom effect
    zoomed_size = crop_size * zoom_factor  # Always zoom to same visual scale
    zoomed = pil_crop.resize((zoomed_size, zoomed_size), resample=Image.NEAREST)

        
    return zoomed

# def zoomable_image(image_path, width=400):
#     with open(image_path, "rb") as f:
#         data = f.read()
#         encoded = f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"
        
#     st.components.v1.html(f"""
#         <style>
#         .zoom-img-container {{
#             position: relative;
#             width: {width}px;
#             overflow: hidden;
#         }}
#         .zoom-img-container img {{
#             transition: transform 0.2s ease;
#             width: 100%;
#         }}
#         .zoom-img-container:hover img {{
#             transform: scale(2.5);  /* Adjust zoom level here */
#         }}
#         </style>
#         <div class="zoom-img-container">
#             <img src="{encoded}" />
#         </div>
#     """, height=width)


def zoomable_image(image_path_or_pil, width=300):
    import io
    if isinstance(image_path_or_pil, str):  # Path
        with open(image_path_or_pil, "rb") as f:
            data = f.read()
    else:  # PIL Image
        buf = io.BytesIO()
        image_path_or_pil.save(buf, format="PNG")
        data = buf.getvalue()

    encoded = base64.b64encode(data).decode()
    components.html(f"""
        <style>
        .zoom-img-container {{
            position: relative;
            width: {width}px;
            overflow: hidden;
        }}
        .zoom-img-container img {{
            transition: transform 0.3s ease;
            width: 100%;
            border-radius: 10px;
        }}
        .zoom-img-container:hover img {{
            transform: scale(2.5);
        }}
        </style>
        <div class="zoom-img-container">
            <img src="data:image/png;base64,{encoded}" />
        </div>
    """, height=width)

#implicit image reconstruction or lossless compression