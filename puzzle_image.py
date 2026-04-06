from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
import torch
import torchvision
import numpy as np
import helpers
import math
import generate_image

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



def _generate(
    prompt1: str,
    prompt2: str,
    num_inference_steps,
    guidance_scale,
    nr_steps_to_look_ahead,
    
):

    prompt_embeds1 = helpers.encode_prompts(prompt1)
    prompt_embeds2 = helpers.encode_prompts(prompt2)

    datax = np.loadtxt("out/perm_x.csv", delimiter=',')
    datay = np.loadtxt("out/perm_y.csv", delimiter=',')
    permutex = (torch.from_numpy(datax) - datax.min()).long()
    permutey = (torch.from_numpy(datay) - datay.min()).long()

    height, width = permutex.shape

    # 4. Prepare latent variables
    num_channels_latents = pipeline.transformer.config.in_channels
    latents1 = pipeline.prepare_latents(
        1,
        num_channels_latents,
        height,
        width,
        prompt_embeds1[0].dtype,
        "cuda",
        None,
        None,
    )

    latents2 = permute_latent(latents1, permutex, permutey)

    invpermutex, invpermutey = inverse_permutation(permutex, permutey)

    pipeline.scheduler.set_timesteps(num_inference_steps)
    pipeline.scheduler.set_begin_index(0)
    
    # 7. Denoising loop
    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
        for i in range(len(pipeline.scheduler.timesteps)):

            t = pipeline.scheduler.timesteps[i:i+1]
            s = pipeline.scheduler.sigmas[i]

            noise_pred1 = helpers.get_noise_pred2(latents1, prompt_embeds1, t, guidance_scale)
            noise_pred2 = helpers.get_noise_pred2(latents2, prompt_embeds2, t, guidance_scale)

            # hint_for_img2 = latents1 - s * noise_pred1
            # hint_for_img1 = latents2 - s * noise_pred2

            future_latents1, next_t_idx = generate_image.optimize(latents1, prompt_embeds1, num_inference_steps, guidance_scale, begin_index=i, nr_steps=nr_steps_to_look_ahead)
            future_latents2, next_t_idx = generate_image.optimize(latents2, prompt_embeds1, num_inference_steps, guidance_scale, begin_index=i, nr_steps=nr_steps_to_look_ahead)

            if next_t_idx is not None:
                timestep = pipeline.scheduler.timesteps[next_t_idx:next_t_idx+1]
                sigma = pipeline.scheduler.sigmas[next_t_idx]

                hint_noise_pred1 = helpers.get_noise_pred2(future_latents1, prompt_embeds1, timestep, guidance_scale)
                hint_noise_pred2 = helpers.get_noise_pred2(future_latents2, prompt_embeds2, timestep, guidance_scale)
                
                future_latents1 = future_latents1 - sigma * hint_noise_pred1
                future_latents2 = future_latents2 - sigma * hint_noise_pred2


            hint_for_img2 = helpers.latents_roundtrip(future_latents1, permutex, permutey)
            hint_for_img1 = helpers.latents_roundtrip(future_latents2, invpermutex, invpermutey)
            
        
            hint_noise_pred1 = (latents1 - hint_for_img1) / (s + 0.01)
            hint_noise_pred2 = (latents2 - hint_for_img2) / (s + 0.01)

            
            # w = 0 if (1-s) > hint_until else 0.5 * hint_weight * math.sqrt(map(1-s, 0, hint_until, 4, 0))
            w = 0.5
            final_noise_pred1 = (1-w) * noise_pred1 + w * hint_noise_pred1
            final_noise_pred2 = (1-w) * noise_pred2 + w * hint_noise_pred2
            # dif1 = hint_noise_pred1 - noise_pred1
            # dif2 = hint_noise_pred2 - noise_pred2
            # lf1 = torch.nn.functional.avg_pool2d(dif1, 3, stride=1, padding=1)
            # lf2 = torch.nn.functional.avg_pool2d(dif2, 3, stride=1, padding=1)
            # lf1 = torchvision.transforms.functional.gaussian_blur(dif1, (5,5), sigma=0.7)
            # lf2 = torchvision.transforms.functional.gaussian_blur(dif2, (5,5), sigma=0.7)
            # TODO other idea, temporal averaging, might also wash out any high frequency stuff
            # hint_noise_pred1 = 0.5 * hint_noise_pred1 + 0.5 * prev_hint_pred
            # prev_hint_pred = hint_noise_pred1

            # final_noise_pred1 = noise_pred1 + w * lf1
            # final_noise_pred2 = noise_pred2 + w * lf2

            helpers.prepare_scheduler(num_inference_steps, i)

            latents1 = pipeline.scheduler.step(final_noise_pred1, t, latents1, return_dict=False)[0]
            pipeline.scheduler._step_index -= 1
            latents2 = pipeline.scheduler.step(final_noise_pred2, t, latents2, return_dict=False)[0]

            # call the callback, if provided
            if i == len(pipeline.scheduler.timesteps) - 1 or ((i + 1) > 0 and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()

    imgs = []

    dec1 = helpers.decode(latents1)
    dec2 = helpers.decode(latents2)

    dec1_permuted = dec1[:, :, permutey, permutex]
    dec2_invpermuted = dec2[:, :, invpermutey, invpermutex]

    imgs.append(pipeline.image_processor.postprocess(0.5 * (dec1 + dec2_invpermuted), output_type="pil")[0])
    imgs.append(pipeline.image_processor.postprocess(0.5 * (dec2 + dec1_permuted), output_type="pil")[0])
    

    return imgs
    

def generate(
    prompt1: str,
    prompt2: str,
    num_inference_steps = 50,
    guidance_scale = 7.0,
    nr_steps_to_look_ahead = 10,
):
    with torch.no_grad():
        return _generate(prompt1, prompt2, num_inference_steps, guidance_scale, nr_steps_to_look_ahead)


imgs = generate(
    # "oil painting of a fantasy castle with lots of trees",
    # "abstract oil painting of a large fish close up, in an underwater landscape, with plants, algae and corals",
    # "oil painting of a large fish, close up, swimming in an underwater landscape, with plants, algae and corals",
    # "abstract painting of the face of an old man with a beard",
    # "abstract painting of the face of a beautiful young woman",
    # "watercolor painting of a duck",
    # "watercolor painting of a bunny",
    # "abstract oil painting of the face of an old man with a beard",
    # "abstract oil painting of a woman's face, with prominent eyes and red lips",
    "a painting of houseplants in the style of studio ghibli, anime style",
    "abstract illustration of marilyn monroe",
    num_inference_steps=30,
    nr_steps_to_look_ahead=20,
)

# imgs = generate(
#     "a water color drawing of a duck",
#     "a water color drawing of a bunny",
#     
#     num_inference_steps=50,
# )

for i in range(len(imgs)):
    imgs[i].save(f"out/puzzle{i}.png")