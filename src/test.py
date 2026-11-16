import torch
import helpers
import image

pipeline = helpers.pipeline
    

def test_transform_errors(
    puzzle_w, puzzle_h,
    prompt1: str,
    prompt2: str,
    num_inference_steps = 32,
    guidance_scale = 7.0,
    seed=None
):
    with torch.no_grad():

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        prompt_embeds = helpers.encode_prompts([prompt1, prompt2], ["", ""])

        LT_data = helpers.load_latent_transform_data(puzzle_w, puzzle_h)
        permutey, permutex, invpermutey, invpermutex, _,_,_,_,_ = LT_data

        height, width = permutex.shape

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

        dec = helpers.decode(latents)
        enc = helpers.encode2(dec)
        res = latents - enc

        print("encode-decode error: ", res.norm())

        res_transformed   = helpers.apply_view_to_latents(res, permutex, permutey)
        res_untransformed = helpers.apply_view_to_latents(res_transformed, invpermutex, invpermutey)

        with_transformed_res = enc + res_untransformed
        print("with view applied to res: ", (latents - with_transformed_res).norm())

        res_transformed   = helpers.latent_transform(res, LT_data)
        res_untransformed = helpers.latent_inv_transform(res_transformed, LT_data)

        with_transformed_res = enc + res_untransformed
        print("with linear rotation maps applied to res: ", (latents - with_transformed_res).norm())

        transformed = helpers.latent_transform(latents, LT_data)
        untransformed = helpers.latent_inv_transform(transformed, LT_data)

        print("with linear rotation maps: ", (latents - untransformed).norm())
        

test_transform_errors(4, 4, "", "")



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