
import torch
from transformers import T5EncoderModel, BitsAndBytesConfig
from diffusers import StableDiffusion3Pipeline
import os

import math

model_id = "stabilityai/stable-diffusion-3.5-medium"

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

text_encoder_3 = T5EncoderModel.from_pretrained(
    model_id,
    subfolder="text_encoder_3",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
)

pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    text_encoder_3=text_encoder_3,
    torch_dtype=torch.bfloat16,
)

pipeline.enable_model_cpu_offload()


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

def encode2(decoded):
    enc = pipeline.vae.encode(decoded).latent_dist.mean
    latents = (enc - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
    return latents

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


def get_noise_pred(latents, all_prompt_embeds, t, guidance_scale):
    (prompt_embeds, 
     negative_prompt_embeds, 
     pooled_prompt_embeds, 
     negative_pooled_prompt_embeds) = all_prompt_embeds

    # batch sizes must match
    assert latents.shape[0] == prompt_embeds.shape[0]

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

def prepare_scheduler(num_inference_steps, begin_index):
    pipeline.scheduler.set_timesteps(num_inference_steps)
    pipeline.scheduler.set_begin_index(begin_index)

def prepare_linear_schedule(nr_steps, start=1):
    
    pipeline.scheduler.sigmas = torch.linspace(start, 0, nr_steps+1)
    pipeline.scheduler.timesteps = 1000 * pipeline.scheduler.sigmas[0:-1]

    pipeline.scheduler._step_index = None
    pipeline.scheduler._begin_index = None
    pipeline.scheduler.set_begin_index(0)

def get_src_path():
    return os.path.dirname(os.path.abspath(__file__))
