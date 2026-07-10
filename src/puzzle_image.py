import torch
import helpers
import image
import math

pipeline = helpers.pipeline

def estimate_clean(latents, s, noise_preds, LT_data):
        
        perm_data, _, _, _, _, _, _ = LT_data
        permutey, permutex, invpermutey, invpermutex = perm_data

        # the raw clean image predictions (with artifacts)
        target_latents = latents - s * noise_preds

        target = helpers.decode(target_latents)
        permuted_target = torch.cat([
            target[1:2, :, invpermutey, invpermutex],
            target[0:1, :, permutey, permutex]
        ])

        # averaging in pixel space!
        common_target_latents = helpers.encode2(0.5 * (target + permuted_target))
        
        # if s > 0.2:
        residuals = target_latents - helpers.encode2(target)

        res1 = 0.6 * (residuals[0:1] + helpers.latent_inv_transform(residuals[1:2], LT_data))
        res2 = 0.6 * (residuals[1:2] + helpers.latent_transform(residuals[0:1], LT_data))
        
        # equation 8
        res = torch.cat([res1, res2])

        common_target_latents = common_target_latents + res

        return common_target_latents


def optimize(
    latents, 
    prompt_embeds,
    num_inference_steps,
    guidance_scale,
    LT_data,
    refine_seperately,
    refine_seperately_amount,
    time_travel, time_travel_steps, time_travel_gamma,
):
    
    perm_data, _, _, _, _, _, _ = LT_data
    permutey, permutex, invpermutey, invpermutex = perm_data

    time_traveled = [False for i in range(num_inference_steps)]
    i = 0

    helpers.prepare_scheduler(num_inference_steps, i)
    s = pipeline.scheduler.sigmas
    total_steps = num_inference_steps + time_travel * time_travel_steps * sum((0.2 < s) * (s < 0.8))
   
    # 7. Denoising loop
    with pipeline.progress_bar(total=total_steps.item()) as progress_bar:
        while i < num_inference_steps:

            helpers.prepare_scheduler(num_inference_steps, i)

            t = pipeline.scheduler.timesteps[i:i+1]
            s = pipeline.scheduler.sigmas[i]

            noise_preds = helpers.get_noise_pred(latents, prompt_embeds, t, guidance_scale)

            if refine_seperately and s < refine_seperately_amount:
                latents = pipeline.scheduler.step(noise_preds, t, latents, return_dict=False)[0]
                
                i += 1
                progress_bar.update()
                continue

            if time_travel and 0.2 < s < 0.8 and not time_traveled[i]:
                time_traveled[i] = True
                x0 = latents - s * noise_preds
                noise = (latents - (1-s)*x0) / s
                noise2 = torch.randn_like(noise)
                noise2[1:2] = helpers.latent_transform(noise2[0:1], LT_data)
                new_noise = time_travel_gamma * noise + math.sqrt(1 - time_travel_gamma ** 2) * noise2
                i = i-time_travel_steps
                prev_s = pipeline.scheduler.sigmas[i]
                latents = (1 - prev_s) * x0 + prev_s * new_noise
                continue
            
            
            z_hat = estimate_clean(latents, s, noise_preds, LT_data)
            
            final_noise_pred = (latents - z_hat) / s
            
            latents = pipeline.scheduler.step(final_noise_pred, t, latents, return_dict=False)[0]

            i += 1
            progress_bar.update()
            
            
    imgs = []

    decoded = helpers.decode(latents)

    decoded0 = decoded[0:1, :, :, :]
    decoded1 = decoded[1:2, :, :, :]

    imgs.append(pipeline.image_processor.postprocess(decoded0, output_type="pil")[0])
    imgs.append(pipeline.image_processor.postprocess(decoded0[:, :, permutey, permutex], output_type="pil")[0])
    imgs.append(pipeline.image_processor.postprocess(decoded1, output_type="pil")[0])
    imgs.append(pipeline.image_processor.postprocess(decoded1[:, :, invpermutey, invpermutex], output_type="pil")[0])
    
    return imgs
    

def generate(
    puzzle_w, puzzle_h,
    prompt1: str,
    prompt2: str,
    num_inference_steps = 32,
    guidance_scale = 7.0,
    negative_prompt1="",
    negative_prompt2="",
    refine_seperately=False,
    refine_seperately_amount=0.3,
    time_travel=True,
    time_travel_steps=1,
    time_travel_gamma=0.85, # bigger gamma -> less new noise
    seed=None
):
    with torch.no_grad():

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        prompt_embeds = helpers.encode_prompts([prompt1, prompt2], [negative_prompt1, negative_prompt2])

        LT_data = helpers.load_latent_transform_data(puzzle_w, puzzle_h)
        perm_data, _, _, _, _, _, _ = LT_data
        permutey, permutex, invpermutey, invpermutex = perm_data

        height, width = permutex.shape

        # 4. Prepare latent variables
        num_channels_latents = pipeline.transformer.config.in_channels
        latents = pipeline.prepare_latents(
            2,
            num_channels_latents,
            height,
            width,
            prompt_embeds[0].dtype,
            "cuda",
            None,
            None,
        )

        return optimize(
            latents, prompt_embeds,
            num_inference_steps, guidance_scale,
            LT_data,
            refine_seperately, refine_seperately_amount,
            time_travel, time_travel_steps, time_travel_gamma,
        )


# s = 0
# for i in range(10):
#     imgs = generate(
#         8,8,
        # "painting of a fantasy castle with lots of trees",
        # "painting of a large fish close up, in an underwater landscape, with plants and corals.",
        # "oil painting of a large fish, close up, swimming in an underwater landscape, with plants, algae and corals",
        # "painting of a donut on a white plate. the painting has a thin white border",
        # "painting of a mostly white coffe mug on a wooden kitchen table in front of a darker background. the painting has a thin white border",
        # "abstract oil painting of the face of an old man with a beard",
        # "abstract oil painting of a woman's face, with prominent eyes and red lips",
        # "a painting of houseplants in the style of studio ghibli, anime style",
        # "abstract illustration of marilyn monroe",
        # "a detailed color pencil drawing of exotic houseplants",
        # "abstract color pencil drawing of marilyn monroe",
        # "watercolor of a duck in a lake",
        # "watercolor of a bunny with some grass and flowers around it",
        # "watercolor of a duck, there are waterlillies and reeds",
        # "watercolor of a bunny in the grass",
        # "ink drawing of a robot",
        # "ink drawing of a birthday cake",
        # "painting of a duck in the style of Picasso's cubism",
        # "painting of a bunny in the style of Picasso's cubism",
        # "oil painting of a bowl of fruit on a kitchen table",# table with fruit bowl and decorations", 
        # "oil painting of a deer",
        # "painting of a bottle of wine",
        # "minimalist art of a pole dancer",
        # "alpine landscape",
        # "a large intricate beautiful colorful flower",
        # "nice sports car in the streets of tokyo, anime style, studio ghibli",
        # "illustration of the countryside by night during full moon, in a field next to a small house, anime style",
        # "close-up photo of a duck in a forest, there is moss and brown pine needles on the floor",
        # "close-up photo of a bunny in a pine forest, there are pine cones, moss and mushrooms",
        # "a quick abstract charcoal sketch of a duck, with rough, approximate strokes",
        # "a quick abstract charcoal sketch of a bunny, with rough, approximate strokes",
        # negative_prompt1="realism, photography",
        # negative_prompt2="realism, photography",
    #     num_inference_steps=50,
        
    # )

    # for j in range(len(imgs)):
    #     imgs[j].save(f"out/puzzle{s}.png")
    #     s += 1