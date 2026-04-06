
import torch
import torchvision
from transformers import T5EncoderModel, BitsAndBytesConfig
from diffusers import StableDiffusion3Pipeline

import math

model_id = "stabilityai/stable-diffusion-3.5-medium"

# 1. Define the quantization config for the T5 text encoder
# Using 8-bit (int8) is a great balance; use load_in_4bit=True for even more savings.
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# 2. Load the T5 encoder separately (it's 'text_encoder_3' in SD3.5)
text_encoder_3 = T5EncoderModel.from_pretrained(
    model_id,
    subfolder="text_encoder_3",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
)

# 3. Load the pipeline, passing in your quantized encoder
# We set text_encoder_3 to the one we just loaded
pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    text_encoder_3=text_encoder_3,
    torch_dtype=torch.bfloat16,
)

# Optional: Further memory safety for 16GB cards
pipeline.enable_model_cpu_offload()


def get_noise_pred(
        prompt, 
        latents,
        time,
        guidance_scale=7,
):
    with torch.no_grad():

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
        )

        pipeline.scheduler.set_timesteps(500)
        pipeline.scheduler.set_begin_index(0)
        i = round(time*499)

        t = pipeline.scheduler.timesteps[i:i+1]
        s = pipeline.scheduler.sigmas[i]


        noise_pred = pipeline.transformer(
            hidden_states=latents,
            timestep=t.to("cuda"),
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]

        # perform guidance
        if guidance_scale > 1:
            noise_pred_uncond = pipeline.transformer(
                hidden_states=latents,
                timestep=t.to("cuda"),
                encoder_hidden_states=negative_prompt_embeds,
                pooled_projections=negative_pooled_prompt_embeds,
                return_dict=False,
            )[0]

            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
        
        return noise_pred


def encode_prompts(prompt, negative_prompt=""):
    with torch.no_grad():
        prompt_embeds = pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
            negative_prompt_3=negative_prompt,
        )
    
    for p in prompt_embeds:
        p.requires_grad_(False)
    
    return prompt_embeds


def encode(pixels):
    img = pixels.to(torch.bfloat16).to("cuda")
    enc = pipeline.vae.encode(img).latent_dist.mean
    latent = (enc - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
    return latent

def decode(latents):
    latent_unscaled = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    dec = pipeline.vae.decode(latent_unscaled, return_dict=False)[0]
    return dec

def latents_roundtrip(latents, permutex, permutey):
    dec = decode(latents)
    permuted = dec[:, :, permutey, permutex]
    enc = pipeline.vae.encode(permuted).latent_dist.mean
    latents2 = (enc - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
    return latents2

def latent_to_pil(latents):
    with torch.no_grad():
        dec = decode(latents)
        out = pipeline.image_processor.postprocess(dec, output_type="pil")[0]
        return out


def get_noise_pred2(latents, prompt_embeds, t, guidance_scale):
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = prompt_embeds

    noise_pred = pipeline.transformer(
        hidden_states=latents,
        timestep=t.to("cuda"),
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
        return_dict=False,
    )[0]

    if guidance_scale > 1:
        noise_pred_uncond = pipeline.transformer(
            hidden_states=latents,
            timestep=t.to("cuda"),
            encoder_hidden_states=negative_prompt_embeds,
            pooled_projections=negative_pooled_prompt_embeds,
            return_dict=False,
        )[0]

        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
    
    return noise_pred

def align_to_64(width, height):
    if width % 64 == 0 and height % 64 == 0:
        return width, height
    
    # if sizes are not aligned, make it as big as possible
    max_scale_factor = math.sqrt(1024 * 1024 / (width * height))
    nw = max_scale_factor * width
    nh = max_scale_factor * height
    nw = math.floor(nw) // 64 * 64
    nh = math.floor(nh) // 64 * 64
    return nw, nh

def prepare_scheduler(num_inference_steps, next_index):
    pipeline.scheduler.set_timesteps(num_inference_steps)
    pipeline.scheduler.set_begin_index(next_index)