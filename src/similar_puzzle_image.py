import torch
import torchvision
import numpy as np
import helpers
import os
pipeline = helpers.pipeline



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


def generate(
    puzzle_w, puzzle_h,
    prompt: str,
    target_img_path, 
    num_inference_steps = 50,
    guidance_scale = 7.0,
    hint_weight = 0.7,
    hint_until = 0.6, # for how long in the process to steer using the hint, as percentage
    negative_prompt="",
    
):
    with torch.no_grad():

        prompt_embeds = helpers.encode_prompts(prompt, negative_prompt)

        puzzle_path = os.path.join(os.path.dirname(__file__), f"../puzzles/{puzzle_w}x{puzzle_h}")

        datax = np.loadtxt(os.path.join(puzzle_path, "perm_x.csv"), delimiter=',')
        datay = np.loadtxt(os.path.join(puzzle_path, "perm_y.csv"), delimiter=',')
        permutex = (torch.from_numpy(datax) - datax.min()).long()
        permutey = (torch.from_numpy(datay) - datay.min()).long()

        height, width = permutex.shape

        hint_img = torchvision.io.read_image(target_img_path) * (1/255)
        pixels = pipeline.image_processor.preprocess(hint_img, height, width)
        permuted_pixels = pixels[:, :, permutey, permutex]
        hint_latents = helpers.encode(permuted_pixels)

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

        invpermutex, invpermutey = inverse_permutation(permutex, permutey)

        pipeline.scheduler.set_timesteps(num_inference_steps)
        pipeline.scheduler.set_begin_index(0)
        
        # 7. Denoising loop
        with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i in range(len(pipeline.scheduler.timesteps)):

                t = pipeline.scheduler.timesteps[i:i+1]
                s = pipeline.scheduler.sigmas[i]

                noise_pred = helpers.get_noise_pred(latents, prompt_embeds, t, guidance_scale)
                
                hint_noise_pred = (latents - hint_latents) / s

                if (1-s) < hint_until:
                    w = s * hint_weight
                    final_noise_pred = w * hint_noise_pred + (1-w) * noise_pred
                else:
                    final_noise_pred = noise_pred


                latents = pipeline.scheduler.step(final_noise_pred, t, latents, return_dict=False)[0]

                # call the callback, if provided
                if i == len(pipeline.scheduler.timesteps) - 1 or ((i + 1) > 0 and (i + 1) % pipeline.scheduler.order == 0):
                    progress_bar.update()


        dec1 = helpers.decode(latents)
        dec1_permuted = dec1[:, :, invpermutey, invpermutex]

        imgs = []

        imgs.append(pipeline.image_processor.postprocess(dec1, output_type="pil")[0])
        imgs.append(pipeline.image_processor.postprocess(dec1_permuted, output_type="pil")[0])
        

        return imgs



# imgs = generate(
#     10,10,
#     "a rock formation in the alps, among small pine trees and bushes, birds flying in the air. there is a little hut",
#     "assets/matt.png",
#     "assets/steve.png",
#     num_inference_steps=50,
#     hint_weight=1.0,
#     hint_until=0.7,
#     negative_prompt="pixel art"
# )

# out_path = puzzle_path = os.path.join(os.path.dirname(__file__), f"../out")

# for i in range(len(imgs)):
#     imgs[i].save(os.path.join(out_path, f"similar_puzzle{i}.png"))