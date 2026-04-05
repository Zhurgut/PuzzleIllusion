
import torch
import torchvision

import helpers
pipeline = helpers.pipeline


def generate(
    prompt: str,
    target_img_path, 
    num_inference_steps: int = 32,
    guidance_scale = 7.0,
    hint_weight = 0.6,
    hint_until = 0.75,
):
        
    hint_img = torchvision.io.read_image(target_img_path) * (1/255)
    c, height, width = hint_img.shape
    width, height = helpers.align_to_64(width, height)

    hint_pixels = pipeline.image_processor.preprocess(hint_img, height, width)
    hint_latents = helpers.encode(hint_pixels)

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

    pipeline.scheduler.set_timesteps(num_inference_steps)
    pipeline.scheduler.set_begin_index(0)

    # 7. Denoising loop
    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
        for i in range(len(pipeline.scheduler.timesteps)):

            t = pipeline.scheduler.timesteps[i:i+1]
            s = pipeline.scheduler.sigmas[i]

            with torch.no_grad():
                noise_pred = helpers.get_noise_pred2(latents, prompt_embeds, t, guidance_scale)

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


        image = helpers.latent_to_pil(latents)

        return image


# for weight in [0.4, 0.5, 0.6, 0.7, 0.8]:
#     for until in [0.4, 0.5, 0.6, 0.7, 0.8]:

img = generate(
    "tropical island",
    "goodies/donut512.png",
    # hint_weight=0.65,
    # hint_until=0.55,
    num_inference_steps=50
)

img.save(f"out/similar.png")


