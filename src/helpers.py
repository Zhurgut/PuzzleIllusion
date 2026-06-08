
import torch
from transformers import T5EncoderModel, BitsAndBytesConfig
from diffusers import StableDiffusion3Pipeline
import os
import numpy as np

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


def inverse_permutation(permutex, permutey):
    assert permutex.shape == permutey.shape
    H, W = permutex.shape

    rangey, rangex = torch.meshgrid(
        torch.arange(0, H),
        torch.arange(0, W),
        indexing = "ij"
    )

    inv_permutex = torch.empty_like(permutex)
    inv_permutey = torch.empty_like(permutey)

    inv_permutex[permutey, permutex] = rangex
    inv_permutey[permutey, permutex] = rangey

    assert torch.all(rangex == permutex[inv_permutey, inv_permutex])
    assert torch.all(rangey == permutey[inv_permutey, inv_permutex])

    return inv_permutex, inv_permutey


def get_src_path():
    return os.path.dirname(os.path.abspath(__file__))


def load_latent_transform_data(puzzle_w, puzzle_h):

    dir_path = get_src_path()
    puzzle_path = os.path.join(dir_path, "..", f"puzzles/{puzzle_w}x{puzzle_h}")

    datax = np.loadtxt(os.path.join(puzzle_path, "perm_x.csv"), delimiter=',')
    datay = np.loadtxt(os.path.join(puzzle_path, "perm_y.csv"), delimiter=',')
    permutex = (torch.from_numpy(datax) - datax.min()).long()
    permutey = (torch.from_numpy(datay) - datay.min()).long()

    invpermutex, invpermutey = inverse_permutation(permutex, permutey)

    rot_map1 = torch.from_numpy(np.loadtxt(os.path.join(puzzle_path, "rot1.csv"), delimiter=',')).long().to("cuda")
    rot_map2 = torch.from_numpy(np.loadtxt(os.path.join(puzzle_path, "rot2.csv"), delimiter=',')).long().to("cuda")

    rot90fn = torch.nn.Linear(144, 16, bias=False)
    rot90fn.load_state_dict(
        torch.load(os.path.join(dir_path, "..", "latent_transforms", "rot90.pt"), weights_only=False)
    )
    rot180fn = torch.nn.Linear(144, 16, bias=False)
    rot180fn.load_state_dict(
        torch.load(os.path.join(dir_path, "..", "latent_transforms", "rot180.pt"), weights_only=False)
    )
    rot270fn = torch.nn.Linear(144, 16, bias=False)
    rot270fn.load_state_dict(
        torch.load(os.path.join(dir_path, "..", "latent_transforms", "rot270.pt"), weights_only=False)
    )

    rot90fn.to("cuda")
    rot180fn.to("cuda")
    rot270fn.to("cuda")

    latent_transform_data = permutey, permutex, invpermutey, invpermutex, rot_map1, rot_map2, rot90fn, rot180fn, rot270fn

    return latent_transform_data

def latent_img_to_in_samples(img):
    return torch.nn.functional.unfold(img, 3, padding=1).permute((0, 2, 1)).reshape(-1, 144)

def apply_view_to_latents(latents, permutex, permutey):
    BS, C, H, W = latents.shape
    expanded = latents.unsqueeze(3).unsqueeze(5).expand(BS, C, H, 8, W, 8).reshape(BS, C, 8*H, 8*W)
    transformed = expanded[:, :, permutey, permutex].float()
    pooled = torch.nn.functional.avg_pool2d(transformed, kernel_size=8, stride=8)
    return pooled.to(latents.dtype)


def latent_transform(latents, latent_transform_data):
    permutey, permutex, invpermutey, invpermutex, rot_map1, rot_map2, rot90fn, rot180fn, rot270fn = latent_transform_data
    BS, C, H, W = latents.shape
    samples = latent_img_to_in_samples(latents).float()
    rotated90 = rot90fn(samples).reshape(BS, -1, 16).permute(0, 2, 1).reshape(BS, C, H, W)
    rotated180 = rot180fn(samples).reshape(BS, -1, 16).permute(0, 2, 1).reshape(BS, C, H, W)
    rotated270 = rot270fn(samples).reshape(BS, -1, 16).permute(0, 2, 1).reshape(BS, C, H, W)
    rotated = latents * (rot_map1 == 0) + rotated90 * (rot_map1 == 1) + rotated180 * (rot_map1 == 2)  + rotated270 * (rot_map1 == 3) 
    expanded = rotated.unsqueeze(3).unsqueeze(5).expand(BS, C, H, 8, W, 8).reshape(BS, C, 8*H, 8*W)
    transformed = expanded[:, :, permutey, permutex]
    pooled = torch.nn.functional.avg_pool2d(transformed, kernel_size=8, stride=8)
    assert pooled.shape == latents.shape

    return pooled.to(torch.bfloat16)


def latent_inv_transform(latents, latent_transform_data):
    permutey, permutex, invpermutey, invpermutex, rot_map1, rot_map2, rot90fn, rot180fn, rot270fn = latent_transform_data
    BS, C, H, W = latents.shape
    samples = latent_img_to_in_samples(latents).float()
    rotated90 = rot90fn(samples).reshape(BS, -1, 16).permute(0, 2, 1).reshape(BS, C, H, W)
    rotated180 = rot180fn(samples).reshape(BS, -1, 16).permute(0, 2, 1).reshape(BS, C, H, W)
    rotated270 = rot270fn(samples).reshape(BS, -1, 16).permute(0, 2, 1).reshape(BS, C, H, W)
    rotated = latents * (rot_map2 == 0) + rotated90 * (rot_map2 == 1) + rotated180 * (rot_map2 == 2)  + rotated270 * (rot_map2 == 3) 
    expanded = rotated.unsqueeze(3).unsqueeze(5).expand(BS, C, H, 8, W, 8).reshape(BS, C, 8*H, 8*W)
    transformed = expanded[:, :, invpermutey, invpermutex]
    pooled = torch.nn.functional.avg_pool2d(transformed, kernel_size=8, stride=8)
    assert pooled.shape == latents.shape

    return pooled.to(torch.bfloat16)