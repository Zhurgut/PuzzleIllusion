import torch
import numpy as np
import helpers
import image
import os
import torchvision

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

def generate_dataset(nr_samples):
    with torch.no_grad():
        img_size = 64
        nr_images = nr_samples // (img_size ** 2) + 1
        noise = torch.randn(nr_images, 16, img_size, img_size, device="cuda", dtype=torch.bfloat16)
        decoded = decode(noise)
        denoised    = encode(decoded)
        denoised90  = torch.rot90(encode(torch.rot90(decoded, 1, (3, 2))), -1, (3, 2))
        denoised180 = torch.rot90(encode(torch.rot90(decoded, 2, (3, 2))), -2, (3, 2))
        denoised270 = torch.rot90(encode(torch.rot90(decoded, 3, (3, 2))), -3, (3, 2))

        x = denoised.permute(0, 2, 3, 1).reshape((-1, 16))
        x90 = denoised90.permute(0, 2, 3, 1).reshape((-1, 16))
        x180 = denoised180.permute(0, 2, 3, 1).reshape((-1, 16))
        x270 = denoised270.permute(0, 2, 3, 1).reshape((-1, 16))

        X = torch.cat([x, x90, x180, x270])
        Y = torch.cat([x90, x180, x270, x])

        return noise.to(torch.float32), X.to(torch.float32), Y.to(torch.float32)

def train(nr_samples, lr=2e-3):
    noise, X, Y = generate_dataset(nr_samples)

    dir_path = os.path.join(helpers.get_src_path(), "..")

    test_img = torchvision.io.read_image(os.path.join(dir_path, "assets/donut.png")) * (1/255)
    test_img = pipeline.image_processor.preprocess(test_img)
    test_img = torch.rot90(test_img, -1, (3, 2))

    test_target = torch.rot90(test_img, 1, (3, 2))
    pipeline.image_processor.postprocess(test_target, output_type="pil")[0].save(os.path.join(dir_path, "out/LT_target.png"))

    test_latents = torch.rot90(helpers.encode(test_img), 1, (3, 2))
    BS, C, H, W = test_latents.shape
    helpers.latent_to_pil(test_latents).save(os.path.join(dir_path, "out/LT_identity.png"))

    linear = torch.nn.Linear(16, 16, bias=False, dtype=X.dtype, device=X.device)

    opt = torch.optim.AdamW(linear.parameters() , lr=lr)
    mse = torch.nn.MSELoss()

    nr_epochs = 1000
    for i in range(nr_epochs):
        opt.zero_grad()
        loss = mse(linear(X), Y)
        loss.backward()
        opt.step()

        if i % (nr_epochs // 20) == 0:
            print(i, ", ", loss.item())

    test_linear = linear(test_latents.float().permute((0, 2, 3, 1)).reshape(-1, 16)).reshape(BS, H, W, C).permute((0, 3, 1, 2))
    helpers.latent_to_pil(test_linear.to(torch.bfloat16)).save(os.path.join(dir_path, "out/LT_linear.png"))


    # model = torch.nn.Sequential(
    #     torch.nn.Linear(16, 512, dtype=X.dtype, device=X.device),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(512, 16, dtype=X.dtype, device=X.device)
    # )

    # opt = torch.optim.AdamW(model.parameters() , lr=1e-2)
    # mse = torch.nn.MSELoss()

    # nr_epochs = 1000
    # for i in range(nr_epochs):
    #     opt.zero_grad()
    #     loss = mse(model(X), Y)
    #     loss.backward()
    #     opt.step()

    #     if i % (nr_epochs // 20) == 0:
    #         print(i, ", ", loss.item())

    # test_model = model(test_latents.permute((0, 2, 3, 1)).reshape(-1, 16)).reshape(BS, H, W, C).permute((0, 3, 1, 2))
    # helpers.latent_to_pil(test_model).save(os.path.join(dir_path, "out/LT_model.png"))

