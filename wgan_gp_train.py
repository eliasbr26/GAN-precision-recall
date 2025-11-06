
"""
- Uses Adam (betas=(0.0, 0.9)) as per WGAN-GP
- Flattens MNIST images to 784-D vectors and saves sample grids

Run:
  python wgan_gp_train.py --epochs 20 --batch_size 128 --lr 1e-4 --n_critic 5 --lambda_gp 10
"""

import os
import argparse
import random
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils

from model import Generator, Discriminator_W

# ---------------------------
# Utils
# ---------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def weights_init_linear(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


def mirror_to_activated_discriminator(C_state_dict, d_input_dim=784, device=None):
    """
    Build the activated Discriminator and load weights from the WGAN critic (same fc layers).
    Returns the model placed on `device` (if provided).
    """
    D_act = Discriminator_W(d_input_dim=d_input_dim)
    D_act.load_state_dict(C_state_dict, strict=True)
    if device is not None:
        D_act = D_act.to(device)
    return D_act

# ---------------------------
# Gradient Penalty (vector inputs)
# ---------------------------

def gradient_penalty(critic, real, fake, device):
    bsz = real.size(0)
    eps = torch.rand(bsz, 1, device=device)
    eps = eps.expand_as(real)
    x_hat = eps * real + (1 - eps) * fake
    x_hat.requires_grad_(True)

    scores = critic(x_hat)
    grad_outputs = torch.ones_like(scores, device=device)
    grads = torch.autograd.grad(
        outputs=scores,
        inputs=x_hat,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grads = grads.view(bsz, -1)
    gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
    return gp

# ---------------------------
# Training
# ---------------------------

def train(args):
    set_seed(args.seed)
    device = get_device()

    # Data: normalized to [-1,1], then flattened
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)

    z_dim = args.z_dim  # should match Generator's expected input size (100 by default)
    if z_dim != 100:
        print(f"Warning: model.Generator expects z_dim=100; overriding to 100 (was {z_dim}).")
        z_dim = 100

    G = Generator(g_output_dim=784).to(device)
    C = Discriminator_W(d_input_dim=784).to(device)

    G.apply(weights_init_linear)
    C.apply(weights_init_linear)

    # Optimizers
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_C = torch.optim.Adam(C.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # Output dirs
    run_dir = os.path.join(args.out_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    samples_dir = os.path.join(run_dir, "samples")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    fixed_z = torch.randn(64, z_dim, device=device)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        for real, _ in loader:
            real = real.to(device).view(real.size(0), -1)  # flatten to (B, 784)

            # ----- Train Critic n_critic steps -----
            for _ in range(args.n_critic):
                opt_C.zero_grad(set_to_none=True)

                z = torch.randn(real.size(0), z_dim, device=device)
                with torch.no_grad():
                    fake = G(z)

                real_score = C(real)
                fake_score = C(fake)

                gp = args.lambda_gp * gradient_penalty(C, real, fake, device)
                loss_C = fake_score.mean() - real_score.mean() + gp
                loss_C.backward()
                opt_C.step()

            # ----- Train Generator -----
            opt_G.zero_grad(set_to_none=True)
            z = torch.randn(real.size(0), z_dim, device=device)
            gen = G(z)
            gen_score = C(gen)
            loss_G = -gen_score.mean()
            loss_G.backward()
            opt_G.step()

            # Logging
            if global_step % args.log_every == 0:
                print(f"Epoch {epoch:03d}/{args.epochs}  Step {global_step:06d}  "
                      f"LossC {loss_C.item():.4f}  LossG {loss_G.item():.4f}  GP {gp.item():.4f}")

            # Save samples
            if global_step % args.sample_every == 0:
                with torch.no_grad():
                    fake_fixed = G(fixed_z).view(-1, 1, 28, 28).cpu()
                    grid = vutils.make_grid(fake_fixed, nrow=8, normalize=True, value_range=(-1, 1))
                    vutils.save_image(grid, os.path.join(samples_dir, f"step_{global_step:06d}.png"))

            # Save checkpoints
            if global_step % args.ckpt_every == 0 and global_step > 0:
                D_act = mirror_to_activated_discriminator(C.state_dict(), d_input_dim=784)
                torch.save(
                    {
                        "G": G.state_dict(),
                        "D": D_act.state_dict(),
                        "args": vars(args),
                        "step": global_step,
                        "epoch": epoch,
                    },
                    os.path.join(ckpt_dir, f"ckpt_{global_step:06d}.pt"),)
                global_step += 1

        # End-of-epoch sample
        with torch.no_grad():
            fake_fixed = G(fixed_z).view(-1, 1, 28, 28).cpu()
            grid = vutils.make_grid(fake_fixed, nrow=8, normalize=True, value_range=(-1, 1))
            vutils.save_image(grid, os.path.join(samples_dir, f"epoch_{epoch:03d}.png"))

    # Final save
    D_act = mirror_to_activated_discriminator(C.state_dict(), d_input_dim=784)
    torch.save({"G": G.state_dict(), "D": D_act.state_dict()}, os.path.join(ckpt_dir, "final.pt"))
    print(f"Done. Samples -> {samples_dir} | Checkpoints (G + activated D) -> {ckpt_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Train WGAN-GP on MNIST with user's model.py")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./runs/wgan_gp_mnist_model", help="output dir")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--z_dim", type=int, default=100, help="latent dim; model.Generator is fixed at 100")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta1", type=float, default=0.0)
    p.add_argument("--beta2", type=float, default=0.9)
    p.add_argument("--lambda_gp", type=float, default=10.0)
    p.add_argument("--n_critic", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--sample_every", type=int, default=500)
    p.add_argument("--ckpt_every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
