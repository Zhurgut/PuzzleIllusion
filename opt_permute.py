from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
import torch
import torchvision
import numpy as np
import helpers

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
    prompt1: str,
    prompt2: str,
    permutex, permutey,
    num_inference_steps: int = 50,
    guidance_scale = 7.0,
    hint_weight = 0.6,
    hint_until = 0.94,
    
):
    with torch.no_grad():

        prompt_embeds1 = pipeline.encode_prompt(
            prompt=prompt1,
            prompt_2=prompt1,
            prompt_3=prompt1,
        )

        prompt_embeds2 = pipeline.encode_prompt(
            prompt=prompt2,
            prompt_2=prompt2,
            prompt_3=prompt2,
        )

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

        def map(x, a, b, A, B):
            return (x-a) / (b-a) * (B-A) + A
        
        # 7. Denoising loop
        with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i in range(len(pipeline.scheduler.timesteps)):

                t = pipeline.scheduler.timesteps[i:i+1]
                s = pipeline.scheduler.sigmas[i]

                noise_pred1 = helpers.get_noise_pred2(latents1, prompt_embeds1, t, pipeline, guidance_scale)
                noise_pred2 = helpers.get_noise_pred2(latents2, prompt_embeds2, t, pipeline, guidance_scale)

                hint_for_img2 = latents1 - s * noise_pred1
                hint_for_img2 = helpers.latents_roundtrip(hint_for_img2, permutex, permutey, pipeline)

                hint_for_img1 = latents2 - s * noise_pred2
                hint_for_img1 = helpers.latents_roundtrip(hint_for_img1, invpermutex, invpermutey, pipeline)
                
                hint_noise_pred1 = (latents1 - hint_for_img1) / (s + 0.01)
                hint_noise_pred2 = (latents2 - hint_for_img2) / (s + 0.01)

                
                w = 0 if (1-s) > hint_until else 0.5 * hint_weight * np.sqrt(map(1-s, 0, hint_until, 4, 0))
                final_noise_pred1 = (1-w) * noise_pred1 + w * hint_noise_pred1
                final_noise_pred2 = (1-w) * noise_pred2 + w * hint_noise_pred2

                latents1 = pipeline.scheduler.step(final_noise_pred1, t, latents1, return_dict=False)[0]
                pipeline.scheduler._step_index -= 1
                latents2 = pipeline.scheduler.step(final_noise_pred2, t, latents2, return_dict=False)[0]

                # call the callback, if provided
                if i == len(pipeline.scheduler.timesteps) - 1 or ((i + 1) > 0 and (i + 1) % pipeline.scheduler.order == 0):
                    progress_bar.update()


        dec1 = helpers.decode(latents1, pipeline)
        dec2 = helpers.decode(latents2, pipeline)

        dec1_permuted = dec1[:, :, permutey, permutex]
        dec2_invpermuted = dec2[:, :, invpermutey, invpermutex]

        imgs = []

        imgs.append(pipeline.image_processor.postprocess(dec1, output_type="pil")[0])
        imgs.append(pipeline.image_processor.postprocess(dec2_invpermuted, output_type="pil")[0])
        imgs.append(pipeline.image_processor.postprocess(0.5 * (dec1 + dec2_invpermuted), output_type="pil")[0])

        imgs.append(pipeline.image_processor.postprocess(dec2, output_type="pil")[0])
        imgs.append(pipeline.image_processor.postprocess(dec1_permuted, output_type="pil")[0])
        imgs.append(pipeline.image_processor.postprocess(0.5 * (dec2 + dec1_permuted), output_type="pil")[0])

        return imgs



# datax = np.loadtxt("perm2_x.csv", delimiter=',')
# datay = np.loadtxt("perm2_y.csv", delimiter=',')

# permutex = torch.from_numpy(datax).long() - 1
# permutey = torch.from_numpy(datay).long() - 1

# for attempt in range(10):
#     imgs = optimize(
#         # "a charcoal sketch of a duck",
#         # "a charcoal sketch of a bunny",
#         "abstract painting of the face of an old man with a beard",
#         "abstract painting of the face of a beautiful young woman",
#         permutex, 
#         permutey,
#         num_inference_steps=50,
#     )
        
#     for i in range(len(imgs)):
#         imgs[i].save(f"out/img{attempt}_{i}.png")