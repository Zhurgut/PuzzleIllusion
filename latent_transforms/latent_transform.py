import torch
import torch.nn.functional as F
import os
import torchvision
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src"))
import helpers

pipeline = helpers.pipeline
pipeline.text_encoder.to("cpu")
pipeline.text_encoder_2.to("cpu")
pipeline.text_encoder_3.to("cpu")
import gc
gc.collect()
torch.cuda.empty_cache()

def decode(Z):
    BS, C, H, W= Z.shape
    X = torch.empty(BS, 3, 8*H, 8*W, device=Z.device)
    t = BS // 8 * 8

    for i in range(0, t, 8):
        X[i:i+8, :, :, :] = helpers.decode(Z[i:i+8, :, :, :])
    X[t:BS, :, :, :] = helpers.decode(Z[t:BS, :, :, :])

    return X

def encode(X):
    BS, C, H, W= X.shape
    Z = torch.empty(BS, 16, H // 8, W // 8, device=X.device, dtype=torch.bfloat16)
    t = BS // 8 * 8

    for i in range(0, t, 8):
        Z[i:i+8, :, :, :] = helpers.encode(X[i:i+8, :, :, :])
    Z[t:BS, :, :, :] = helpers.encode(X[t:BS, :, :, :])

    return Z


def latent_img_to_in_samples(img):
    return F.unfold(img, 3, padding=1).permute((0, 2, 1)).reshape(-1, 144)

def generate_dataset(nr_samples):
    with torch.no_grad():
        img_size = 64
        nr_images = nr_samples // (img_size ** 2) + 1
        noise = 3 * torch.randn(nr_images, 16, img_size, img_size, device="cuda", dtype=torch.bfloat16)
        decoded = decode(noise)
        denoised    = encode(decoded)
        denoised90  = torch.rot90(encode(torch.rot90(decoded, 1, (3, 2))), -1, (3, 2))
        denoised180 = torch.rot90(encode(torch.rot90(decoded, 2, (3, 2))), -2, (3, 2))
        denoised270 = torch.rot90(encode(torch.rot90(decoded, 3, (3, 2))), -3, (3, 2))

        x = latent_img_to_in_samples(denoised)
        x90 = latent_img_to_in_samples(denoised90)
        x180 = latent_img_to_in_samples(denoised180)
        x270 = latent_img_to_in_samples(denoised270)

        y = denoised.permute(0, 2, 3, 1).reshape((-1, 16))
        y90 = denoised90.permute(0, 2, 3, 1).reshape((-1, 16))
        y180 = denoised180.permute(0, 2, 3, 1).reshape((-1, 16))
        y270 = denoised270.permute(0, 2, 3, 1).reshape((-1, 16))

        X = torch.cat([x, x90, x180, x270]).to(torch.float32)
        Y90 = torch.cat([y90, y180, y270, y]).to(torch.float32)
        Y180 = torch.cat([y180, y270, y, y90]).to(torch.float32)
        Y270 = torch.cat([y270, y, y90, y180]).to(torch.float32)

        return X, Y90, Y180, Y270
    


def train(nr_samples, lr=2e-3, var_preservation=1.0):
    X, Y90, Y180, Y270 = generate_dataset(nr_samples)
    BS, IN = X.shape

    dir_path = os.path.join(helpers.get_src_path(), "..")

    test_img = torchvision.io.read_image(os.path.join(dir_path, "assets/donut.png")) * (1/255)
    test_img = pipeline.image_processor.preprocess(test_img)
    test_img = torch.rot90(test_img, -1, (3, 2))

    test_target = torch.rot90(test_img, 1, (3, 2))
    pipeline.image_processor.postprocess(test_target, output_type="pil")[0].save(os.path.join(dir_path, "out/LT_target.png"))

    test_latents = torch.rot90(helpers.encode(test_img), 1, (3, 2))
    BS, C, H, W = test_latents.shape
    helpers.latent_to_pil(test_latents).save(os.path.join(dir_path, "out/LT_identity.png"))

    r = 1
    for Y in [Y90, Y180, Y270]:
        BS, OUT = Y.shape

        model = torch.nn.Linear(IN, OUT, bias=False, dtype=X.dtype, device=X.device)

        opt = torch.optim.AdamW(model.parameters() , lr=lr)
        mse = torch.nn.MSELoss()

        nr_epochs = 1500
        for i in range(nr_epochs):
            opt.zero_grad()

            wt = model.weight
            cov = wt @ wt.t()
            var_loss = mse(cov, torch.eye(16, device=wt.device, dtype=wt.dtype))
            loss = mse(model(X), Y) + var_preservation * var_loss
            loss.backward()
            opt.step()

            if i % (nr_epochs // 20) == 0:
                print(i, ", ", var_loss.item(), ", ", loss.item())

        if i == 1:
            test_linear = model(latent_img_to_in_samples(test_latents.float())).reshape(BS, H, W, C).permute((0, 3, 1, 2))
            helpers.latent_to_pil(test_linear.to(torch.bfloat16)).save(os.path.join(dir_path, "out/LT_linear.png"))
        
        torch.save(model.state_dict, os.path.join(dir_path, f"latent_transforms/rot{r*90}.pt"))

        r += 1
