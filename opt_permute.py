from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
import torch
import torchvision
import numpy as np
import inspect

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

def permute_latent(latents, permutex, permutey):
    bs, c, h, w = latents.shape
    pixels = torch.repeat_interleave(latents, 8, dim=-1)
    pixels = torch.repeat_interleave(pixels, 8, dim=-2)
    permuted = pixels[:, :, permutey, permutex]
    permuted_latents = torch.nn.functional.avg_pool2d(permuted, 8, stride=8)
    return permuted_latents


def optimize(
    prompt1: str,
    prompt2: str,
    permutex, permutey,
    num_inference_steps: int = 32,
    guidance_scale = 7.0,
    weight1 = 0.5,
    hint_weight = 0.4,
    hint_until = 0.75, # for how long in the process to steer using the hint, as percentage
):
    with torch.no_grad():

        prompt_embeds1, negative_prompt_embeds1, pooled_prompt_embeds1, negative_pooled_prompt_embeds1 = pipeline.encode_prompt(
            prompt=prompt1,
            prompt_2=prompt1,
            prompt_3=prompt1,
        )

        prompt_embeds2, negative_prompt_embeds2, pooled_prompt_embeds2, negative_pooled_prompt_embeds2 = pipeline.encode_prompt(
            prompt=prompt2,
            prompt_2=prompt2,
            prompt_3=prompt2,
        )

        height, width = permutex.shape

        # 4. Prepare latent variables
        num_channels_latents = pipeline.transformer.config.in_channels
        latents = pipeline.prepare_latents(
            1,
            num_channels_latents,
            height,
            width,
            prompt_embeds1.dtype,
            "cuda",
            None,
            None,
        )

        invpermutex, invpermutey = inverse_permutation(permutex, permutey)

        pipeline.scheduler.set_timesteps(num_inference_steps)
        pipeline.scheduler.set_begin_index(0)

        
        # 7. Denoising loop
        with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i in range(len(pipeline.scheduler.timesteps)):

                t = pipeline.scheduler.timesteps[i:i+1]
                s = pipeline.scheduler.sigmas[i]

                # noise prediction for standard image
                noise_pred1 = pipeline.transformer(
                    hidden_states=latents,
                    timestep=t.to("cuda"),
                    encoder_hidden_states=prompt_embeds1,
                    pooled_projections=pooled_prompt_embeds1,
                    return_dict=False,
                )[0]

                if guidance_scale > 1:
                    noise_pred_uncond1 = pipeline.transformer(
                        hidden_states=latents,
                        timestep=t.to("cuda"),
                        encoder_hidden_states=negative_prompt_embeds1,
                        pooled_projections=negative_pooled_prompt_embeds1,
                        return_dict=False,
                    )[0]

                    noise_pred1 = noise_pred_uncond1 + guidance_scale * (noise_pred1 - noise_pred_uncond1)
                

                # noise prediction for permuted image
                permuted_latents = permute_latent(latents, permutex, permutey)

                noise_pred2 = pipeline.transformer(
                    hidden_states=permuted_latents,
                    timestep=t.to("cuda"),
                    encoder_hidden_states=prompt_embeds2,
                    pooled_projections=pooled_prompt_embeds2,
                    return_dict=False,
                )[0]

                if guidance_scale > 1:
                    noise_pred_uncond2 = pipeline.transformer(
                        hidden_states=permuted_latents,
                        timestep=t.to("cuda"),
                        encoder_hidden_states=negative_prompt_embeds2,
                        pooled_projections=negative_pooled_prompt_embeds2,
                        return_dict=False,
                    )[0]

                    noise_pred2 = noise_pred_uncond2 + guidance_scale * (noise_pred2 - noise_pred_uncond2)

                noise_pred2 = permute_latent(noise_pred2, invpermutex, invpermutey)

                noise_pred = weight1 * noise_pred1 + (1-weight1) * noise_pred2

                # compute the previous noisy sample x_t -> x_t-1
                latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # call the callback, if provided
                if i == len(pipeline.scheduler.timesteps) - 1 or ((i + 1) > 0 and (i + 1) % pipeline.scheduler.order == 0):
                    progress_bar.update()

        latents = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor

        imgs = []

        dec = pipeline.vae.decode(latents, return_dict=False)[0]
        dec2 = dec[:, :, permutey, permutex]
        imgs.append(pipeline.image_processor.postprocess(dec, output_type="pil")[0])

        permuted_latents = permute_latent(latents, permutex, permutey)
        dec3 = pipeline.vae.decode(permuted_latents, return_dict=False)[0]
        dec4 = dec3[:, :, invpermutey, invpermutex]
        imgs.append(pipeline.image_processor.postprocess(dec3, output_type="pil")[0])

        imgs.append(pipeline.image_processor.postprocess(0.5 * (dec + dec4), output_type="pil")[0])
        imgs.append(pipeline.image_processor.postprocess(0.5 * (dec2 + dec3), output_type="pil")[0])

        # Offload all models
        pipeline.maybe_free_model_hooks()

        return imgs



datax = np.loadtxt("perm2_x.csv", delimiter=',')
datay = np.loadtxt("perm2_y.csv", delimiter=',')

permutex = torch.from_numpy(datax).long() - 1
permutey = torch.from_numpy(datay).long() - 1

for attempt in range(10):
    imgs = optimize(
        "a charcoal sketch of a duck",
        "a charcoal sketch of a bunny",
        permutex, 
        permutey,
        num_inference_steps=50,
        weight1 = 0.5,
        guidance_scale=7.0,
    )
        
    for i in range(len(imgs)):
        imgs[i].save(f"out/img{attempt}_{i}.png")