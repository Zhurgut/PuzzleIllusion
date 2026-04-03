from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
import torch
import torch.nn.functional as F
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
pipeline.vae.eval()
pipeline.transformer.eval()




def image_latents_from_rotations(img_path, pipeline):
    with torch.no_grad():
        img = torchvision.io.read_image(img_path) * (1/255)
        img = pipeline.image_processor.preprocess(img, 512, 512)

        img90 = torch.rot90(img, k=1, dims=(3,2))
        img180 = torch.rot90(img, k=2, dims=(3,2))
        img270 = torch.rot90(img, k=3, dims=(3,2))

        # saveimg90 = pipeline.image_processor.postprocess(img90, output_type="pil")[0]
        # saveimg90.save("rot90.png")

        lats = [helpers.encode(im, pipeline) for im in [img, img90, img180, img270]]
        from_rotations_latents = [
            lats[0], 
            torch.rot90(lats[1], k=-1, dims=(3,2)),
            torch.rot90(lats[2], k=-2, dims=(3,2)),
            torch.rot90(lats[3], k=-3, dims=(3,2)),
        ]

        # from_rot90 = decode(from_rotations_latents[1], pipeline)
        # fr90 = pipeline.image_processor.postprocess(from_rot90, output_type="pil")[0]
        # fr90.save("rot90_fromlatents.png")

        return from_rotations_latents


def rot_latents_matrix(l, l90, X, Y):
    bs, c, h, w = l.shape
    cs = torch.randint(high=w, size=(h,))
    for r in range(h):
        X[r, :] = l[0, :, r, cs[r]]
        Y[r, :] = l90[0, :, r, cs[r]]

    dif = l - l90
    l2_error = torch.einsum("bchw, bchw -> bhw", dif, dif)[0, :, :]
    for r in range(h):
        i = torch.argmax(l2_error[r, :]).item()
        X[h+r, :] = l[0, :, r, i]
        Y[h+r, :] = l90[0, :, r, i]


def rot_latents_dataset90(img_path, pipeline):
    l0, l90, l180, l270 = image_latents_from_rotations(img_path, pipeline)
    bs, C, H, W = l0.shape
    X = torch.zeros(8*H, 16)
    Y = torch.zeros(8*H, 16)
    rot_latents_matrix(l0, l90, X[0:2*H, :], Y[0:2*H, :])
    rot_latents_matrix(l90, l180, X[2*H:4*H, :], Y[2*H:4*H, :])
    rot_latents_matrix(l180, l270, X[4*H:6*H, :], Y[4*H:6*H, :])
    rot_latents_matrix(l270, l0, X[6*H:8*H, :], Y[6*H:8*H, :])

    return X, Y

def rot_latents_dataset180(img_path, pipeline):
    l0, l90, l180, l270 = image_latents_from_rotations(img_path, pipeline)
    bs, C, H, W = l0.shape
    X = torch.zeros(8*H, 16)
    Y = torch.zeros(8*H, 16)
    rot_latents_matrix(l0, l180, X[0:2*H, :], Y[0:2*H, :])
    rot_latents_matrix(l90, l270, X[2*H:4*H, :], Y[2*H:4*H, :])
    rot_latents_matrix(l180, l0, X[4*H:6*H, :], Y[4*H:6*H, :])
    rot_latents_matrix(l270, l90, X[6*H:8*H, :], Y[6*H:8*H, :])

    return X, Y

def rot_latents_dataset270(img_path, pipeline):
    l0, l90, l180, l270 = image_latents_from_rotations(img_path, pipeline)
    bs, C, H, W = l0.shape
    X = torch.zeros(8*H, 16)
    Y = torch.zeros(8*H, 16)
    rot_latents_matrix(l0, l270, X[0:2*H, :], Y[0:2*H, :])
    rot_latents_matrix(l90, l0, X[2*H:4*H, :], Y[2*H:4*H, :])
    rot_latents_matrix(l180, l90, X[4*H:6*H, :], Y[4*H:6*H, :])
    rot_latents_matrix(l270, l180, X[6*H:8*H, :], Y[6*H:8*H, :])

    return X, Y



def find_matrix():
    imgs = [
        "dataset/i0.png", "dataset/i1.png",
        "dataset/i3.png", "dataset/i13.png",
        "dataset/i14.png", "dataset/i15.png",
        "dataset/i16.png", "dataset/i20.png",
        "dataset/i21.png", "dataset/i22.png",
        "dataset/i23.png", "dataset/i24.png",
        "dataset/i25.png", "dataset/i26.png",
    ]

    datasets90 = [rot_latents_dataset90(img, pipeline) for img in imgs]
    xs, ys = zip(*datasets90)
    xs, ys = list(xs), list(ys)

    X90 = torch.concat(xs)
    Y90 = torch.concat(ys)

    datasets180 = [rot_latents_dataset90(img, pipeline) for img in imgs]
    xs, ys = zip(*datasets180)
    xs, ys = list(xs), list(ys)

    X180 = torch.concat(xs)
    Y180 = torch.concat(ys)

    datasets270 = [rot_latents_dataset90(img, pipeline) for img in imgs]
    xs, ys = zip(*datasets270)
    xs, ys = list(xs), list(ys)

    X270 = torch.concat(xs)
    Y270 = torch.concat(ys)

    M = torch.linalg.lstsq(X90, Y90).solution.T
    identity = torch.eye(16, 16)

    def loss(M, X, Y):
        return F.mse_loss(torch.einsum("ij, kj -> ki", M, X), Y)
        

    loss(M, X90, Y90)
    loss(torch.eye(16, 16), X90, Y90)

    # matrix = M.clone().detach() + 0.1 * torch.randn(16, 16)
    matrix = torch.rand(16, 16)
    matrix.requires_grad = True

    optimizer = torch.optim.Adam([matrix], lr=1e-3)
    for i in range(50000):
        optimizer.zero_grad()
        L90 = loss(matrix, X90, Y90)
        # L180 = loss(matrix @ matrix, X180, Y180)
        # L270 = loss(matrix @ matrix @ matrix, X270, Y270)
        id_loss = F.mse_loss(matrix @ matrix @ matrix @ matrix, identity)
        L = L90 + id_loss + 0.2 * F.l1_loss(matrix, torch.zeros_like(matrix))
        L.backward()
        optimizer.step()

        
        print(f"loss: {L.item()}, id: {id_loss.item()}")
    
    torch.set_printoptions(precision=3, linewidth=200, sci_mode=False)
    print(matrix)

    torch.save(matrix, "rot90.pt")
    
    with torch.no_grad():
        l, l90, l180, l270 = image_latents_from_rotations("dataset/i0.png", pipeline)

        helpers.latent_to_pil(l, pipeline).save("test/owl_original.png")
        helpers.latent_to_pil(l90, pipeline).save("test/owl90_orig.png")
        helpers.latent_to_pil(l180, pipeline).save("test/owl180_orig.png")
        helpers.latent_to_pil(l270, pipeline).save("test/owl270_orig.png")

        helpers.latent_to_pil(torch.einsum("ic, bchw -> bihw", M, l270.to("cpu", dtype=torch.float)).to("cuda", dtype=torch.bfloat16), pipeline).save("test/owl270_rot.png")
        helpers.latent_to_pil(torch.einsum("ic, bchw -> bihw", M.T, l270.to("cpu", dtype=torch.float)).to("cuda", dtype=torch.bfloat16), pipeline).save("test/owl270_rot(t).png")

        helpers.latent_to_pil(torch.einsum("ic, bchw -> bihw", M @ M, l180.to("cpu", dtype=torch.float)).to("cuda", dtype=torch.bfloat16), pipeline).save("test/owl180_rot.png")
        helpers.latent_to_pil(torch.einsum("ic, bchw -> bihw", M @ M @ M, l90.to("cpu", dtype=torch.float)).to("cuda", dtype=torch.bfloat16), pipeline).save("test/owl90_rot.png")

        M = matrix
        
        helpers.latent_to_pil(torch.einsum("ic, bchw -> bihw", M, l270.to("cpu", dtype=torch.float)).to("cuda", dtype=torch.bfloat16), pipeline).save("test/owl270_rot_learned.png")
        helpers.latent_to_pil(torch.einsum("ic, bchw -> bihw", M @ M, l180.to("cpu", dtype=torch.float)).to("cuda", dtype=torch.bfloat16), pipeline).save("test/owl180_rot_learned.png")
        helpers.latent_to_pil(torch.einsum("ic, bchw -> bihw", M @ M @ M, l90.to("cpu", dtype=torch.float)).to("cuda", dtype=torch.bfloat16), pipeline).save("test/owl90_rot_learned.png")




find_matrix()