from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
import torch
import torchvision
import numpy as np
import helpers
import torch.nn.functional as F

model_id = "stabilityai/stable-diffusion-3.5-medium"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16
)

pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id, 
    transformer=model_nf4,
    torch_dtype=torch.bfloat16
)
pipeline.enable_model_cpu_offload()



def optimize(target_latents, nr_steps=1500, lr=0.01):
    B, C, H, W = target_latents.shape
    pixels = torch.randn(B, 3, 8*H, 8*W, device="cuda")
    pixels.requires_grad = True
    target_latents = target_latents.detach()
    target_latents.requires_grad = False
    pipeline.vae.eval()

    optimizer = torch.optim.Adam([pixels], lr=lr)

    for i in range(nr_steps):
        
        with torch.no_grad():
            input_pixels = pixels.clamp(-1, 1) 

        optimizer.zero_grad()
        
        latents = helpers.encode(pixels, pipeline)
        
        loss_mse = F.mse_loss(latents, target_latents)
        
        # Total Variation Loss: Keeps the image smooth/natural (removes noise)
        loss_tv = torch.sum(torch.abs(input_pixels[:, :, :, :-1] - input_pixels[:, :, :, 1:])) + torch.sum(torch.abs(input_pixels[:, :, :-1, :] - input_pixels[:, :, 1:, :]))
        
        total_loss = loss_mse + (1e-2 * loss_tv)

        total_loss.backward()
        optimizer.step()

        if i%10 == 0:
            print(f"{i}: {total_loss.item()}")

    with torch.no_grad():
        
        out = pipeline.image_processor.postprocess(pixels, output_type="pil")[0]
        out.save("test/optpixels.png")

        return out

img = torchvision.io.read_image("goodies/img1_0.png") * (1/255)
img = pipeline.image_processor.preprocess(img, 512, 512)
target = helpers.encode(img, pipeline)

optimize(target)