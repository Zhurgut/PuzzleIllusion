
import torch

import helpers
pipeline = helpers.pipeline



# expects the scheduler to be set
def optimize(
    latents, prompt_embeds, guidance_scale
):
    nr_steps = len(pipeline.scheduler.timesteps)
    # 7. Denoising loop
    with pipeline.progress_bar(total=nr_steps) as progress_bar:
        for i in range(nr_steps):

            t = pipeline.scheduler.timesteps[i:i+1]

            noise_pred = helpers.get_noise_pred(latents, prompt_embeds, t, guidance_scale)

            # compute the previous noisy sample x_t -> x_t-1
            # x_t = (1-t)*x_0 + t*epsilon
            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # call the callback, if provided
            if i == len(pipeline.scheduler.timesteps) - 1 or ((i + 1) > 0 and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()

    return latents

    

def generate(
    prompt: str,
    height: int = pipeline.default_sample_size * pipeline.vae_scale_factor,
    width: int = pipeline.default_sample_size * pipeline.vae_scale_factor,
    num_inference_steps: int = 50,
    guidance_scale = 7.0,
    negative_prompt="",
):
    with torch.no_grad():

        prompt_embeds = helpers.encode_prompts(prompt, negative_prompt=negative_prompt)

        # 4. Prepare latent variables
        num_channels_latents = pipeline.transformer.config.in_channels
        latents = pipeline.prepare_latents(
            1,
            num_channels_latents,
            height,
            width,
            prompt_embeds[0].dtype,
            "cuda",
            None,
            None,
        )

        helpers.prepare_scheduler(num_inference_steps, 0)

        finished_latents = optimize(latents, prompt_embeds, guidance_scale)

        image = helpers.latent_to_pil(finished_latents)

        return image
