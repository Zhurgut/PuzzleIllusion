import torch
import numpy as np
import helpers
import image

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



def permute_latent(latents, permutex, permutey):
    bs, c, h, w = latents.shape
    pixels = torch.repeat_interleave(latents, 8, dim=-1)
    pixels = torch.repeat_interleave(pixels, 8, dim=-2)
    permuted = pixels[:, :, permutey, permutex]
    permuted_latents = torch.nn.functional.avg_pool2d(permuted, 8, stride=8)
    return permuted_latents



def optimize(
    latents, 
    prompt_embeds,
    num_inference_steps,
    guidance_scale,
    permutex, permutey, invpermutex, invpermutey
):

    pipeline.scheduler.set_timesteps(num_inference_steps)
    pipeline.scheduler.set_begin_index(0)

    def map(x, a, b, A, B):
        return (x-a) / (b-a) * (B-A) + A

    # 7. Denoising loop
    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
        for i in range(len(pipeline.scheduler.timesteps)):

            t = pipeline.scheduler.timesteps[i:i+1]
            s = pipeline.scheduler.sigmas[i]

        
            print()
            future_latents = image.optimize(latents, prompt_embeds, num_inference_steps, guidance_scale, begin_index=i)

            
            future_img1, future_img2 = helpers.decode(future_latents).chunk(2)
            
            hint_for_img2 = future_img1[:, :, permutey, permutex]
            hint_for_img1 = future_img2[:, :, invpermutey, invpermutex]
            hint_latents = helpers.encode2(torch.cat([hint_for_img1, hint_for_img2]))

            if i % 5 == 0:
                pipeline.image_processor.postprocess(future_img1, output_type="pil")[0].save(f"prog1/img{i}.png")
                pipeline.image_processor.postprocess(future_img2, output_type="pil")[0].save(f"prog2/img{i}.png")

                pipeline.image_processor.postprocess(hint_for_img1, output_type="pil")[0].save(f"prog1/omg{i}.png")
                pipeline.image_processor.postprocess(hint_for_img2, output_type="pil")[0].save(f"prog2/omg{i}.png")
            
            # w = 0.1
            # latents = latents + w * (hint_latents - future_latents)
            noise_preds = helpers.get_noise_pred(latents, prompt_embeds, t, guidance_scale)
        
            hint_noise_preds = (latents - hint_latents) / s # s != 0 here always

            w_snr = s**2 / (s**2 + (1-s)**2)
            w = map(w_snr, 1, 0, 0.7, 0.5)
            # w = 0.7
            final_noise_preds = (1-w) * noise_preds + w * hint_noise_preds
            # final_noise_preds = hint_noise_preds

            

            helpers.prepare_scheduler(num_inference_steps, i)

            latents = pipeline.scheduler.step(final_noise_preds, t, latents, return_dict=False)[0]


            
            if i == len(pipeline.scheduler.timesteps) - 1 or ((i + 1) > 0 and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()

    imgs = []

    dec1, dec2 = helpers.decode(future_latents).chunk(2)

    dec1_permuted = dec1[:, :, permutey, permutex]
    dec2_invpermuted = dec2[:, :, invpermutey, invpermutex]

    imgs.append(pipeline.image_processor.postprocess(0.5 * (dec1 + dec2_invpermuted), output_type="pil")[0])
    imgs.append(pipeline.image_processor.postprocess(0.5 * (dec2 + dec1_permuted), output_type="pil")[0])
    

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

        datax = np.loadtxt(f"puzzles/{puzzle_w}x{puzzle_h}/perm_x.csv", delimiter=',')
        datay = np.loadtxt(f"puzzles/{puzzle_w}x{puzzle_h}/perm_y.csv", delimiter=',')
        permutex = (torch.from_numpy(datax) - datax.min()).long()
        permutey = (torch.from_numpy(datay) - datay.min()).long()

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

        invpermutex, invpermutey = inverse_permutation(permutex, permutey)
        
        return optimize(
            latents, prompt_embeds,
            num_inference_steps, guidance_scale,
            permutex.to("cuda"), 
            permutey.to("cuda"), 
            invpermutex.to("cuda"), 
            invpermutey.to("cuda")
        )


s = 0
for i in range(10):
    imgs = generate(
        6,6, 
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
        # "painting of a duck in the style of Picasso's cubism",
        # "painting of a bunny in the style of Picasso's cubism",
        # "oil painting of a kitchen",
        # "oil painting of a deer",
        "close-up photo of a duck in a forest, there is moss and brown pine needles on the floor",
        "close-up photo of a bunny in a pine forest, there are pine cones, moss and mushrooms",
        # "a quick abstract charcoal sketch of a duck, with rough, approximate strokes",
        # "a quick abstract charcoal sketch of a bunny, with rough, approximate strokes",
        # negative_prompt1="realism, photography",
        # negative_prompt2="realism, photography",
        num_inference_steps=40,
        
    )

    # imgs = generate(
    #     "a water color drawing of a duck",
    #     "a water color drawing of a bunny",
    #     
    #     num_inference_steps=50,
    # )

    for j in range(len(imgs)):
        imgs[j].save(f"out/puzzle{s}.png")
        s += 1