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


def optimize(
    prompt: str,
    height: int = pipeline.default_sample_size * pipeline.vae_scale_factor,
    width: int = pipeline.default_sample_size * pipeline.vae_scale_factor,
    num_inference_steps: int = 32,
    guidance_scale = 7.0
):
    with torch.no_grad():

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


                # compute the previous noisy sample x_t -> x_t-1
                # x_t = (1-t)*x_0 + t*epsilon
                latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]


                # call the callback, if provided
                if i == len(pipeline.scheduler.timesteps) - 1 or ((i + 1) > 0 and (i + 1) % pipeline.scheduler.order == 0):
                    progress_bar.update()

        latents = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor

        dec = pipeline.vae.decode(latents, return_dict=False)[0]
        image = pipeline.image_processor.postprocess(dec, output_type="pil")[0]

        return image



# img = optimize(
#     "delicious chocolate donut in the ocean",
#     256, 256
# )

# img.save("out.png")