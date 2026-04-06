
import torch

import helpers
pipeline = helpers.pipeline


def optimize(
    latents, prompt_embeds, num_inference_steps, guidance_scale, begin_index=0, nr_steps=None
):

    helpers.prepare_scheduler(num_inference_steps, begin_index)

    nr_steps_to_take = num_inference_steps - begin_index if nr_steps is None else min(nr_steps, num_inference_steps - begin_index)
    assert begin_index + nr_steps_to_take <= num_inference_steps

    # 7. Denoising loop
    with pipeline.progress_bar(total=nr_steps_to_take) as progress_bar:
        for i in range(begin_index, begin_index + nr_steps_to_take):

            t = pipeline.scheduler.timesteps[i:i+1]

            noise_pred = helpers.get_noise_pred2(latents, prompt_embeds, t, guidance_scale)

            # compute the previous noisy sample x_t -> x_t-1
            # x_t = (1-t)*x_0 + t*epsilon
            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # call the callback, if provided
            if i == len(pipeline.scheduler.timesteps) - 1 or ((i + 1) > 0 and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()

    if begin_index + nr_steps_to_take == num_inference_steps:
        return latents, None
    else:
        return latents, begin_index + nr_steps_to_take
    

def generate(
    prompt: str,
    height: int = pipeline.default_sample_size * pipeline.vae_scale_factor,
    width: int = pipeline.default_sample_size * pipeline.vae_scale_factor,
    num_inference_steps: int = 50,
    guidance_scale = 7.0
):
    with torch.no_grad():

        prompt_embeds = helpers.encode_prompts(prompt)

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

        finished_latents, next_t = optimize(latents, prompt_embeds, num_inference_steps, guidance_scale)

        image = helpers.latent_to_pil(finished_latents)

        return image


# for i in range(10):
# img = generate(
#     "forest in the style of studio ghibli, anime style",
#     512, 512,
#     num_inference_steps=10,
# )

# img.save("out/image.png")