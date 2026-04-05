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



def generate(
    prompt: str,
    target_img_path, 
    num_inference_steps: int = 32,
    guidance_scale = 7.0,
    hint_weight = 0.6,
    hint_until = 0.75,
):
    
    with torch.no_grad():
        
        hint_img = torchvision.io.read_image(target_img_path) * (1/255)
        c, height, width = hint_img.shape
        hint_img = pipeline.image_processor.preprocess(hint_img, height, width).to(torch.bfloat16).to("cuda")
        enc = pipeline.vae.encode(hint_img).latent_dist.mean
        hint_latents = (enc - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
        )

        # 4. Prepare latent variables
        num_channels_latents = pipeline.transformer.config.in_channels
        latents = pipeline.prepare_latents(
            1,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            "cuda",
            None,
            None,
        )

        pipeline.scheduler.set_timesteps(num_inference_steps)
        pipeline.scheduler.set_begin_index(0)

        # 7. Denoising loop
        with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i in range(len(pipeline.scheduler.timesteps)):

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

                hint_noise_pred = (latents - hint_latents) / (s + 0.01)


                if (1-s) < hint_until:
                    x = 1 - (1-s) / hint_until
                    w_snr = x**2 / (x**2 + (1-x)**2)
                    w = hint_weight * w_snr
                    final_noise_pred = w * hint_noise_pred + (1-w) * noise_pred
                else:
                    final_noise_pred = noise_pred
                
                
                # compute the previous noisy sample x_t -> x_t-1
                latents = pipeline.scheduler.step(final_noise_pred, t, latents, return_dict=False)[0]
                

                # call the callback, if provided
                if i == len(pipeline.scheduler.timesteps) - 1 or ((i + 1) > 0 and (i + 1) % pipeline.scheduler.order == 0):
                    progress_bar.update()

            enc = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor

            dec = pipeline.vae.decode(enc, return_dict=False)[0]
            image = pipeline.image_processor.postprocess(dec, output_type="pil")[0]

            return [image]


# for weight in [0.4, 0.5, 0.6, 0.7, 0.8]:
#     for until in [0.4, 0.5, 0.6, 0.7, 0.8]:

img = generate(
    "tropical island",
    "goodies/donut512.png",
    # hint_weight=0.65,
    # hint_until=0.55,
    num_inference_steps=50
)[0]

img.save(f"out/similar.png")


