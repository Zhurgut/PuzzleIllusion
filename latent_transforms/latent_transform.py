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


KERNEL_SIZE = 3
PADDING = 1
IMG_DIM = 64 # latent images in the dataset have size 64 x 64 pixels 

def model_in_dim():
    return 16 * (KERNEL_SIZE ** 2)

def model_out_dim():
    return 16


def latent_img_to_in_samples(img):
    BS, C, H, W = img.shape # BS_IN, 16, H, W
    samples = torch.nn.functional.unfold(img, KERNEL_SIZE, padding=PADDING).permute((0, 2, 1))
    BS, N, D = samples.shape
    assert D == model_in_dim()
    return samples.reshape(BS*N, D)

def latent_img_to_out_samples(img):
    BS, C, H, W = img.shape # BS_IN, 16, H, W
    samples = torch.nn.functional.unfold(img, 1, padding=0).permute((0, 2, 1))
    BS, N, D = samples.shape
    assert D == model_out_dim()
    return samples.reshape(BS*N, D)


def out_samples_to_latent_img(samples, BS, H=IMG_DIM, W=IMG_DIM):
    BS_OUT, D = samples.shape # BS_OUT, 16
    assert D == model_out_dim()
    return samples.reshape(BS, -1, 16).permute(0, 2, 1).reshape(BS, 16, H, W)


def decode(latent_batch):
    Z = latent_batch
    BS, C, H, W = Z.shape # BS, 16, 64, 64
    X = torch.empty(BS, 3, 8*H, 8*W, device=Z.device)

    assert BS % 8 == 0
    for i in range(0, BS, 8):
        X[i:i+8, :, :, :] = helpers.decode(Z[i:i+8, :, :, :])

    return X

def encode(X):
    BS, C, H, W = X.shape # BS, 3, 512, 512
    Z = torch.empty(BS, 16, IMG_DIM, IMG_DIM, device=X.device, dtype=torch.bfloat16)

    assert BS % 8 == 0
    for i in range(0, BS, 8):
        Z[i:i+8, :, :, :] = helpers.encode(X[i:i+8, :, :, :])

    return Z

def rotate90(img, r):
    return torch.rot90(img, r, (3, 2))

def apply_ideal_model(latents, rot):
    return encode(rotate90(decode(latents), rot))

def apply_model(latents, model, rot):
    if rot==0:
        return latents

    samples = latent_img_to_in_samples(latents)
    out = model(samples)
    out_img = out_samples_to_latent_img(out, latents.shape[0])
    return rotate90(out_img, rot)


def generate_dataset(min_nr_samples):
    with torch.no_grad():
        nr_images = min_nr_samples // (IMG_DIM ** 2)
        BS = (nr_images // 8 + 1) * 8
        noise = 1 * torch.randn(BS, 16, IMG_DIM, IMG_DIM, device="cuda", dtype=torch.bfloat16)
        decoded = decode(noise)
        denoised = encode(decoded)
        assert denoised.shape == (BS, 16, IMG_DIM, IMG_DIM)

        X = denoised.float()

        Y = X
        Y90 = apply_ideal_model(denoised, 1).float()
        Y180 = apply_ideal_model(denoised, 2).float()
        Y270 = apply_ideal_model(denoised, 3).float()

        return BS, X, Y, Y90, Y180, Y270


def save_test_image(dir_path, model90, model_name, model180, model270, inv_weight):
    with torch.no_grad():
        test_img = torchvision.io.read_image(os.path.join(dir_path, "assets/donut.png")) * (1/255)
        og_test_img = pipeline.image_processor.preprocess(test_img)
        test_img = rotate90(og_test_img, -1)
        test_latents = helpers.encode(test_img)
        assert test_latents.shape == (1, 16, IMG_DIM, IMG_DIM)

        # the target
        test_target = apply_ideal_model(test_latents.repeat(8, 1, 1, 1), 1)[0:1]
        helpers.latent_to_pil(test_target).save(os.path.join(dir_path, "latent_transforms/images/target.png"))

        # if we do nothing, i.e. apply the identity
        identity = rotate90(test_latents, 1)
        helpers.latent_to_pil(identity).save(os.path.join(dir_path, "latent_transforms/images/identity.png"))

        # if we apply our model
        model_out = apply_model(test_latents.float(), model90, 1)
        helpers.latent_to_pil(model_out.to(torch.bfloat16)).save(os.path.join(dir_path, f"latent_transforms/images/{model_name}_{KERNEL_SIZE}_{PADDING}_{inv_weight}.png"))

        test_img = rotate90(og_test_img, -2)
        test_latents = helpers.encode(test_img)
        model_out = apply_model(test_latents.float(), model180, 2)
        helpers.latent_to_pil(model_out.to(torch.bfloat16)).save(os.path.join(dir_path, f"latent_transforms/images/{model_name}_{KERNEL_SIZE}_{PADDING}_{inv_weight}(180).png"))

        test_img = rotate90(og_test_img, 1)
        test_latents = helpers.encode(test_img) 
        model_out = apply_model(test_latents.float(), model270, 3)
        helpers.latent_to_pil(model_out.to(torch.bfloat16)).save(os.path.join(dir_path, f"latent_transforms/images/{model_name}_{KERNEL_SIZE}_{PADDING}_{inv_weight}(270).png"))

        test_latents = helpers.encode(og_test_img) 
        helpers.latent_to_pil(inverted(model90, 1, model270, 3, test_latents.float()).to(torch.bfloat16)).save(os.path.join(dir_path, f"latent_transforms/images/{model_name}_{KERNEL_SIZE}_{PADDING}_{inv_weight}(90+270).png"))



def inverted(model, rot1, inv_model, rot2, X):
    out1 = apply_model(X, model, rot1)
    out2 = apply_model(out1, inv_model, rot2)
    return out2


def smoothstep(x):
    return 3 * x ** 2 - 2 * x ** 3


def train_and_save(nr_samples, model_fn, lr=2e-3, nr_epochs=2500, inv_weight=1):
    
    BS, X, Y, Y90, Y180, Y270 = generate_dataset(nr_samples)

    model90  = model_fn(X)
    model180 = model_fn(X)
    model270 = model_fn(X)

    dir_path = os.path.join(helpers.get_src_path(), "..")

    opt = torch.optim.AdamW(list(model90.parameters()) + list(model180.parameters()) + list(model270.parameters()), lr=lr)
    mse = torch.nn.MSELoss()

    for i in range(nr_epochs+1):
        opt.zero_grad()

        loss1 = mse(apply_model(X, model90, 1), Y90) + mse(apply_model(X, model180, 2), Y180) + mse(apply_model(X, model270, 3), Y270)
        loss2 = mse(inverted(model90, 1, model270, 3, X), Y) + mse(inverted(model180, 2, model180, 2, X), Y) + mse(inverted(model270, 3, model90, 1, X), Y)

        loss = loss1 + inv_weight * smoothstep(i / nr_epochs) * loss2
        # loss = loss1 + 10 * loss2
        
        loss.backward()
        opt.step()

        if i % (nr_epochs // 20) == 0:
            print(i, "\t", loss1.item(), "\t", loss2.item(), "\t", (inverted(model90, 1, model270, 3, X) - Y).abs().max().item())


    save_test_image(dir_path, model90, model_fn.__name__, model180, model270, inv_weight)
    
    torch.save(model90.state_dict(), os.path.join(dir_path, f"latent_transforms/{model_fn.__name__}90_{KERNEL_SIZE}_{PADDING}_{inv_weight}.pt"))
    torch.save(model180.state_dict(), os.path.join(dir_path, f"latent_transforms/{model_fn.__name__}180_{KERNEL_SIZE}_{PADDING}_{inv_weight}.pt"))
    torch.save(model270.state_dict(), os.path.join(dir_path, f"latent_transforms/{model_fn.__name__}270_{KERNEL_SIZE}_{PADDING}_{inv_weight}.pt"))



def linear(X):
    return torch.nn.Linear(model_in_dim(), model_out_dim(), bias=False, dtype=X.dtype, device=X.device)

def linear_with_bias(X):
    return torch.nn.Linear(model_in_dim(), model_out_dim(), dtype=X.dtype, device=X.device)

def mlp(X):
    return torch.nn.Sequential(
        torch.nn.Linear(model_in_dim(), 512, dtype=X.dtype, device=X.device),
        torch.nn.ReLU(),
        torch.nn.Linear(512, model_out_dim(), dtype=X.dtype, device=X.device),
    )

# train_and_save(50000, linear, lr=2e-3, nr_epochs=2000, inv_weight=2)
# train_and_save(50000, mlp, lr=1e-3, nr_epochs=1000, inv_weight=2)


