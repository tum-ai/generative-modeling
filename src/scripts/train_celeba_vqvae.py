import os
import random
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
import wandb

from generative_modeling.variational.celeba_vqvae import CelebAVQVAE
from generative_modeling.losses import sobel_loss_2d, get_pgan_discriminator, adversarial_loss, get_facenet_model, perceptual_loss, get_lpips_model, lpips_loss

CONFIG = {
    "num_embeddings": 1024,
    "embedding_dim": 512, # latent map is 4x4, so total embedding dim is 4*4*256 = 4096
    "image_size": 128,
    "hidden_dims": [32, 64, 128, 256, 512],
    "epochs": 20,
    "batch_size": 256,
    "lr": 1e-3,
    "commitment_cost": 0.25,
    "mse_weight": 1.0,
    "sobel_weight": 0.0,
    "sobel_loss_type": "L2",
    "adv_weight": 0.0,
    "adv_eval_samples": 128,  # Number of random samples to evaluate adversarial loss on each iteration
    "perceptual_weight": 0.0,
    "lpips_weight": 0.0,
    "celeb_path": "./data/",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "project_name": "gm-variational",
    "run_name": "celeba-vqvae-1024-512",
    "seed": 42,
    "num_workers": 8,
}
CONFIG["checkpoint_dir"] = "out/checkpoints/" + CONFIG["run_name"]

DISCRIMINATOR = None
FACENET = None
LPIPS_MODEL = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloaders(config):
    celeb_transform = transforms.Compose([
        transforms.Resize(config["image_size"], antialias=True),
        transforms.CenterCrop(config["image_size"]),
        transforms.ToTensor()
    ])

    os.makedirs(config["celeb_path"], exist_ok=True)

    train_dataset = CelebA(config["celeb_path"], transform=celeb_transform, download=False, split='train')
    test_dataset = CelebA(config["celeb_path"], transform=celeb_transform, download=False, split='valid')

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    return train_loader, test_loader


def get_discriminator(config):
    global DISCRIMINATOR
    if DISCRIMINATOR is None and config["adv_weight"] > 0:
        print("Loading PGAN discriminator...")
        DISCRIMINATOR = get_pgan_discriminator(use_gpu=(config["device"] == "cuda"))
        DISCRIMINATOR = DISCRIMINATOR.to(config["device"])
    return DISCRIMINATOR


def get_facenet(config):
    global FACENET
    if FACENET is None and config["perceptual_weight"] > 0:
        print("Loading FaceNet...")
        FACENET = get_facenet_model(pretrained='vggface2', use_gpu=(config["device"] == "cuda"))
        FACENET = FACENET.to(config["device"])
    return FACENET


def get_lpips(config):
    global LPIPS_MODEL
    if LPIPS_MODEL is None and config["lpips_weight"] > 0:
        print("Loading LPIPS model...")
        LPIPS_MODEL = get_lpips_model(net='vgg', use_gpu=(config["device"] == "cuda"))
        LPIPS_MODEL = LPIPS_MODEL.to(config["device"])
    return LPIPS_MODEL


def loss_function(recon_x, x, vq_loss, config, discriminator=None, facenet=None, lpips_model=None):
    mse = F.mse_loss(recon_x, x, reduction='mean')
    sobel_loss = sobel_loss_2d(recon_x, x, loss_type=config["sobel_loss_type"])
    recon_loss = config["mse_weight"] * mse + config["sobel_weight"] * sobel_loss

    adv_loss = torch.tensor(0.0, device=x.device)
    if config["adv_weight"] > 0 and discriminator is not None:
        adv_loss, _ = adversarial_loss(recon_x, discriminator)

    perc_loss = torch.tensor(0.0, device=x.device)
    if config["perceptual_weight"] > 0 and facenet is not None:
        perc_loss, _ = perceptual_loss(recon_x, x, facenet)

    lpips_loss_val = torch.tensor(0.0, device=x.device)
    if config["lpips_weight"] > 0 and lpips_model is not None:
        lpips_loss_val, _ = lpips_loss(recon_x, x, lpips_model)

    loss = recon_loss + vq_loss + config["adv_weight"] * adv_loss + config["perceptual_weight"] * perc_loss + config["lpips_weight"] * lpips_loss_val
    return loss, mse, vq_loss, sobel_loss, adv_loss, perc_loss, lpips_loss_val


def train(model, train_loader, optimizer, epoch, config, discriminator=None, facenet=None, lpips_model=None):
    model.train()
    train_loss = 0
    total_mse = 0
    total_vq = 0
    total_sobel = 0
    total_adv = 0
    total_perc = 0
    total_lpips = 0
    total_perplexity = 0
    total_adv_eval = 0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} [Train]")
    for batch_idx, (data, _) in pbar:
        data = data.to(config["device"])
        optimizer.zero_grad()

        recon_batch, vq_loss, perplexity = model(data)

        loss, mse, vq, sobel, adv, perc, lpips_val = loss_function(
            recon_batch, data, vq_loss, config, discriminator, facenet, lpips_model
        )

        # Evaluate adversarial loss on samples from prior (to smooth latent space)
        adv_eval_loss = torch.tensor(0.0, device=config["device"])
        if discriminator is not None and config["adv_weight"] > 0 and config["adv_eval_samples"] > 0:
            # Sample from prior and generate
            prior_samples = model.sample(config["adv_eval_samples"], config["device"])
            adv_eval_loss, _ = adversarial_loss(prior_samples, discriminator)
            # Add weighted adversarial loss to smooth latent space
            loss = loss + config["adv_weight"] * adv_eval_loss

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total_mse += mse.item()
        total_vq += vq.item()
        total_sobel += sobel.item()
        total_adv += adv.item()
        total_perc += perc.item()
        total_lpips += lpips_val.item()
        total_perplexity += perplexity.item()
        total_adv_eval += adv_eval_loss.item()

        if batch_idx % 100 == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "vq": f"{vq.item():.4f}", "perc": f"{perplexity.item():.2f}"})

            wandb.log({
                "batch_loss": loss.item(),
                "batch_mse": mse.item(),
                "batch_vq_loss": vq.item(),
                "batch_sobel": sobel.item(),
                "batch_adv": adv.item(),
                "batch_perc": perc.item(),
                "batch_lpips": lpips_val.item(),
                "batch_perplexity": perplexity.item(),
                "batch_adv_eval": adv_eval_loss.item(),
                "epoch": epoch
            })

    avg_loss = train_loss / len(train_loader)
    avg_mse = total_mse / len(train_loader)
    avg_vq = total_vq / len(train_loader)
    avg_sobel = total_sobel / len(train_loader)
    avg_adv = total_adv / len(train_loader)
    avg_perc = total_perc / len(train_loader)
    avg_lpips = total_lpips / len(train_loader)
    avg_perplexity = total_perplexity / len(train_loader)
    avg_adv_eval = total_adv_eval / len(train_loader)

    return avg_loss, avg_mse, avg_vq, avg_sobel, avg_adv, avg_perc, avg_lpips, avg_perplexity, avg_adv_eval


def test(model, test_loader, epoch, config, discriminator=None, facenet=None, lpips_model=None):
    model.eval()
    test_loss = 0
    test_mse = 0
    test_vq = 0
    test_sobel = 0
    test_adv = 0
    test_perc = 0
    test_lpips = 0
    test_perplexity = 0
    test_adv_eval = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Epoch {epoch} [Test ]")
        for i, (data, _) in pbar:
            data = data.to(config["device"])
            recon_batch, vq_loss, perplexity = model(data)

            loss, mse, vq, sobel, adv, perc, lpips_val = loss_function(
                recon_batch, data, vq_loss, config, discriminator, facenet, lpips_model
            )
            test_loss += loss.item()
            test_mse += mse.item()
            test_vq += vq.item()
            test_sobel += sobel.item()
            test_adv += adv.item()
            test_perc += perc.item()
            test_lpips += lpips_val.item()
            test_perplexity += perplexity.item()

            # Evaluate adversarial loss on samples from prior (for monitoring only)
            adv_eval_loss = torch.tensor(0.0, device=config["device"])
            if discriminator is not None and config["adv_weight"] > 0 and config["adv_eval_samples"] > 0:
                prior_samples = model.sample(config["adv_eval_samples"], config["device"])
                adv_eval_loss, _ = adversarial_loss(prior_samples, discriminator)
            test_adv_eval += adv_eval_loss.item()

            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch[:n]])
                grid = make_grid(comparison, nrow=n)
                wandb.log({"reconstructions": wandb.Image(grid, caption=f"Epoch {epoch}")})

    avg_test_loss = test_loss / len(test_loader)
    avg_test_mse = test_mse / len(test_loader)
    avg_test_vq = test_vq / len(test_loader)
    avg_test_sobel = test_sobel / len(test_loader)
    avg_test_adv = test_adv / len(test_loader)
    avg_test_perc = test_perc / len(test_loader)
    avg_test_lpips = test_lpips / len(test_loader)
    avg_test_perplexity = test_perplexity / len(test_loader)
    avg_test_adv_eval = test_adv_eval / len(test_loader)

    return avg_test_loss, avg_test_mse, avg_test_vq, avg_test_sobel, avg_test_adv, avg_test_perc, avg_test_lpips, avg_test_perplexity, avg_test_adv_eval


def main():
    set_seed(CONFIG["seed"])

    wandb.init(
        project=CONFIG["project_name"],
        name=CONFIG["run_name"],
        config=CONFIG
    )

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    model = CelebAVQVAE(
        num_embeddings=CONFIG["num_embeddings"],
        embedding_dim=CONFIG["embedding_dim"],
        image_size=CONFIG["image_size"],
        hidden_dims=CONFIG["hidden_dims"]
    ).to(CONFIG["device"])

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

    discriminator = get_discriminator(CONFIG)
    facenet = get_facenet(CONFIG)
    lpips_model = get_lpips(CONFIG)

    train_loader, test_loader = get_dataloaders(CONFIG)

    print(f'Starting training for {CONFIG["epochs"]} epochs on {CONFIG["device"]}...')
    if CONFIG["adv_weight"] > 0:
        print(f'Adversarial loss enabled with weight {CONFIG["adv_weight"]}')
    if CONFIG["perceptual_weight"] > 0:
        print(f'Perceptual loss enabled with weight {CONFIG["perceptual_weight"]}')
    if CONFIG["lpips_weight"] > 0:
        print(f'LPIPS loss enabled with weight {CONFIG["lpips_weight"]}')

    for epoch in range(1, CONFIG["epochs"] + 1):
        avg_train_loss, avg_train_mse, avg_train_vq, avg_train_sobel, avg_train_adv, avg_train_perc, avg_train_lpips, avg_train_perplexity, avg_train_adv_eval = train(
            model, train_loader, optimizer, epoch, CONFIG, discriminator, facenet, lpips_model
        )
        avg_test_loss, avg_test_mse, avg_test_vq, avg_test_sobel, avg_test_adv, avg_test_perc, avg_test_lpips, avg_test_perplexity, avg_test_adv_eval = test(
            model, test_loader, epoch, CONFIG, discriminator, facenet, lpips_model
        )

        print(f'====> Epoch: {epoch} Train loss: {avg_train_loss:.4f}, Test loss: {avg_test_loss:.4f}, VQ: {avg_train_vq:.4f}, Perplexity: {avg_train_perplexity:.2f}')

        wandb.log({
            "epoch_train_loss": avg_train_loss,
            "epoch_train_mse": avg_train_mse,
            "epoch_train_vq_loss": avg_train_vq,
            "epoch_train_sobel": avg_train_sobel,
            "epoch_train_adv": avg_train_adv,
            "epoch_train_perc": avg_train_perc,
            "epoch_train_lpips": avg_train_lpips,
            "epoch_train_perplexity": avg_train_perplexity,
            "epoch_train_adv_eval": avg_train_adv_eval,
            "epoch_test_loss": avg_test_loss,
            "epoch_test_mse": avg_test_mse,
            "epoch_test_vq_loss": avg_test_vq,
            "epoch_test_sobel": avg_test_sobel,
            "epoch_test_adv": avg_test_adv,
            "epoch_test_perc": avg_test_perc,
            "epoch_test_lpips": avg_test_lpips,
            "epoch_test_perplexity": avg_test_perplexity,
            "epoch_test_adv_eval": avg_test_adv_eval,
            "epoch": epoch
        })

        checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], f"vqvae_model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        with torch.no_grad():
            samples = model.sample(64, CONFIG["device"])
            sample_grid = make_grid(samples, nrow=8)
            wandb.log({"samples": wandb.Image(sample_grid, caption=f"Epoch {epoch}")})

    wandb.finish()


if __name__ == "__main__":
    main()
