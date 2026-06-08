import torch
import helpers

pipeline = helpers.pipeline


def optimize(
    latents, 
    prompt_embeds,
    num_inference_steps,
    guidance_scale,
    LT_data
):
    
    permutey, permutex, invpermutey, invpermutex, _,_,_,_,_ = LT_data

    latents[1:2, :, :, :] = helpers.latent_transform(latents[0:1, :, :, :], LT_data)

    helpers.prepare_scheduler(num_inference_steps, 0)

    # 7. Denoising loop
    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
        for i in range(num_inference_steps-1):

            t = pipeline.scheduler.timesteps[i:i+1]
            s = pipeline.scheduler.sigmas[i]

            f_l1 = helpers.latent_transform(latents[0:1, :, :, :], LT_data)
            f_l2 = helpers.latent_inv_transform(latents[1:2, :, :, :], LT_data)

            delta = 0.5

            avg_latents = (1-delta) * latents + delta * (torch.cat([f_l2, f_l1]))

            noise_preds = helpers.get_noise_pred(avg_latents, prompt_embeds, t, guidance_scale)

            

            # noise_pred1, noise_pred2 = noise_preds.chunk(2)
            # noise_pred2 = helpers.latent_inv_transform(noise_pred2, LT_data)

            # final_noise_pred = 0.5 * (noise_pred1 + noise_pred2 + (latents - helpers.latent_inv_transform(latents2, LT_data)))
            # # final_noise_pred = 0.5 * (noise_pred1 + noise_pred2)
        
            latents = pipeline.scheduler.step(noise_preds, t, latents, return_dict=False)[0]

            if i == len(pipeline.scheduler.timesteps) - 1 or ((i + 1) > 0 and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()
        

    i = num_inference_steps - 1
    
    t = pipeline.scheduler.timesteps[i:i+1]
    s = pipeline.scheduler.sigmas[i]

    noise_preds = helpers.get_noise_pred(latents, prompt_embeds, t, guidance_scale)

    future_latents = pipeline.scheduler.step(noise_preds, t, latents, return_dict=False)[0]

    fut2 = helpers.latents_roundtrip(future_latents[0:1, :, :, :], permutex, permutey)
    fut1 = helpers.latents_roundtrip(future_latents[1:2, :, :, :], invpermutex, invpermutey)

    latents = 0.5 * (future_latents + torch.cat([fut1, fut2]))

    if i == len(pipeline.scheduler.timesteps) - 1 or ((i + 1) > 0 and (i + 1) % pipeline.scheduler.order == 0):
        progress_bar.update()



    imgs = []

    decoded = helpers.decode(latents)
    decoded2 = decoded[0:1, :, permutey, permutex]
    decoded1 = decoded[1:2, :, invpermutey, invpermutex]

    # tfd = helpers.latent_transform(latents, LT_data)
    # tfd_decoded = helpers.decode(tfd)
    # tfd_decoded1 = tfd_decoded[:, :, invpermutey, invpermutex]

    imgs.append(pipeline.image_processor.postprocess(0.5 * (decoded[0:1, :, :, :] + decoded1), output_type="pil")[0])
    imgs.append(pipeline.image_processor.postprocess(0.5 * (decoded2 + decoded[1:2, :, :, :]), output_type="pil")[0])
    imgs.append(pipeline.image_processor.postprocess(decoded[0:1, :, :, :], output_type="pil")[0])
    imgs.append(pipeline.image_processor.postprocess(decoded[1:2, :, :, :], output_type="pil")[0])
    imgs.append(pipeline.image_processor.postprocess(decoded1, output_type="pil")[0])
    imgs.append(pipeline.image_processor.postprocess(decoded2, output_type="pil")[0])
    
    return imgs
    

def generate(
    puzzle_w, puzzle_h,
    prompt1: str,
    prompt2: str,
    num_inference_steps = 32,
    guidance_scale = 7.0,
    negative_prompt1="",
    negative_prompt2="",
):
    with torch.no_grad():

        prompt_embeds = helpers.encode_prompts([prompt1, prompt2], [negative_prompt1, negative_prompt2])

        LT_data = helpers.load_latent_transform_data(puzzle_w, puzzle_h)
        permutex = LT_data[0]

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
            LT_data
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