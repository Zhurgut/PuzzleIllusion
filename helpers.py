
import torch
import torchvision

def get_noise_pred(
        prompt, 
        latents,
        time,
        pipeline,
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


def encode(pixels, pipeline):
    img = pixels.to(torch.bfloat16).to("cuda")
    enc = pipeline.vae.encode(img).latent_dist.mean
    latent = (enc - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
    return latent

def decode(latents, pipeline):
    latent_unscaled = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    dec = pipeline.vae.decode(latent_unscaled, return_dict=False)[0]
    return dec

def latents_roundtrip(latents, permutex, permutey, pipeline):
    dec = decode(latents, pipeline)
    permuted = dec[:, :, permutey, permutex]
    enc = pipeline.vae.encode(permuted).latent_dist.mean
    latents2 = (enc - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
    return latents2


def latent_to_pil(latents, pipeline):
    with torch.no_grad():
        dec = decode(latents, pipeline)
        out = pipeline.image_processor.postprocess(dec, output_type="pil")[0]
        return out


def get_noise_pred2(latents, prompt_embeds, t, pipeline, guidance_scale):
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