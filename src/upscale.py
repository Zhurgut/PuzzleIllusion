
import sys
import torchvision.transforms.functional as tv_functional

# Trick basicsr into finding the moved function
sys.modules['torchvision.transforms.functional_tensor'] = tv_functional

import cv2
import torchvision
import numpy as np
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import os
import urllib
import fitz


script_dir = os.path.dirname(os.path.abspath(__file__))


def upscale(img_path, out_filename="upscaled.png"):

    # =====================================================================
    # STEP 2: Configure the Real-ESRGAN Model
    # =====================================================================
    # This matches the structure of the standard "RealESRGAN_x4plus" network
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    
    model_dir = os.path.join(script_dir, "..", "esrgan")
    model_path = os.path.join(model_dir, "RealESRGAN_x4plus.pth")
    model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'

    # Automatically grab the weights if they are missing
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        urllib.request.urlretrieve(model_url, model_path)

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=400,         # Slices image into 400x400px chunks. Essential to prevent CUDA OOM errors.
        tile_pad=10,      # Slightly overlaps tiles so you don't get visible seams or cut lines.
        half=True         # Runs inference in FP16 to drastically speed up processing
    )

    # =====================================================================
    # STEP 3: Convert Formats and Execute Upscale
    # =====================================================================
    # Convert PIL Image (RGB) to NumPy Array (BGR) for OpenCV compatibility


    img = torchvision.io.read_image(img_path)
    img = img.permute(1, 2, 0)
    
    img_array = np.array(img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    print("Starting Real-ESRGAN 4x upscale...")
    # The network sharpens edges and generates micro-textures here
    output_bgr, _ = upsampler.enhance(img_bgr, outscale=4)

    # Convert back from BGR to standard RGB PIL Image
    output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
    final_upscaled_image = Image.fromarray(output_rgb)

    # Save your final A4-ready background plate
    final_upscaled_image.save(os.path.join(script_dir, "..", "out", out_filename))


def add_cut_lines(img_path, w, h, print1_or_2 = 1, out_filename="with_lines.png"):
    img = cv2.imread(img_path)
    pdf_path = os.path.join(script_dir, "..", "puzzles", f"{w}x{h}", f"print{print1_or_2}.pdf")
    H, W, C = img.shape
    pdf = fitz.open(pdf_path)[0]
    pdf_w, pdf_h = pdf.rect.width, pdf.rect.height

    print(H / pdf_h, " ==? ", W / pdf_w)
    
    zoom = H / pdf_h
    mat = fitz.Matrix(zoom, zoom)

    pix = pdf.get_pixmap(matrix=mat, alpha=False)
    puzzle_layout_img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.h, pix.w, 3))
    puzzle_layout_img = cv2.cvtColor(puzzle_layout_img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(puzzle_layout_img, 70, 255, cv2.THRESH_BINARY_INV)

    img[mask == 255] = 0

    out_img = img.copy()
    cv2.copyTo(src=puzzle_layout_img, mask=mask, dst=out_img)


    cv2.imwrite(os.path.join(script_dir, "..", "out", out_filename), img)


upscale("selection/puzzle_alpcow_7x5_28.png", out_filename="cow75up.png")
add_cut_lines("out/cow75up.png", 7, 5, out_filename="cow75wl.png")