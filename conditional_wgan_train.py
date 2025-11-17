import os
import argparse
import random
from datetime import datetime

import torch
from torch import device
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils

from model import Generator, Discriminator_W
from GMM import GaussianMixture
from data_utils import load_data


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def weights_init_linear(m):
    if isinstance(m, (nn.Linear,)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


@torch.no_grad()
def sample_grid(G, n=64, z_dim=100, GM=None, device='cpu', conditional=True):
    z, labels = GM.sample(n)
    if conditional:
        z = conditional_z(z, labels, num_classes=10)
    fake = G(z).view(-1, 1, 28, 28).cpu()
    grid = vutils.make_grid(fake, nrow=8, normalize=True, value_range=(-1, 1))
    return grid


def mirror_to_activated_discriminator(state_dict, d_input_dim=784):
    """
    Builds a Discriminator_W (critic, no sigmoid) and loads weights,
    returning a nn.Module to save alongside G for inference/compat.
    """
    D_act = Discriminator_W(d_input_dim=d_input_dim)
    D_act.load_state_dict(state_dict, strict=True)
    return D_act

def conditional_z(z, labels, num_classes=10):
    """
    Remove the last 'num_classes' dimensions of z and concatenate one-hot labels.
    """
    z = z[:, :-num_classes]
    one_hot = nn.functional.one_hot(labels, num_classes=num_classes).float()
    one_hot = one_hot.to(z.device)
    return torch.cat([z, one_hot], dim=1)


# ---------------------------
#    WGAN-GP training core
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
    )[0]  # shape [B, 784]

    grads = grads.view(bsz, -1)
    gp = (grads.norm(2, dim=1) - 1.0).pow(2).mean()
    return gp


def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    loader = load_data(args.data_dir, args.batch_size, conditional=args.conditional_GAN)

    # Models
    z_dim = args.z_dim
    if z_dim != 100:
        print(f"Warning: model.Generator expects z_dim=100; overriding to 100 (was {z_dim}).")
        z_dim = 100

    G = Generator(g_output_dim=784).to(device)
    C = Discriminator_W(d_input_dim=784).to(device)
    if args.use_GM:
        GM = GaussianMixture(n_components=args.n_components, c=args.c, sigma=args.sigma, device=device)
    elif args.conditional_GAN:
        GM = GaussianMixture(n_components=10, c=0, sigma=1.0, device=device) 
    else:
        GM = GaussianMixture(n_components=1, c=0, sigma=1.0, device=device)  # Standard normal

    G.apply(weights_init_linear)
    C.apply(weights_init_linear)

    # Optimizers (match first script: separate LRs + betas (0.5, 0.9))
    opt_G = torch.optim.Adam(G.parameters(), lr=args.g_lr, betas=(args.beta1, args.beta2))
    opt_C = torch.optim.Adam(C.parameters(), lr=args.d_lr, betas=(args.beta1, args.beta2))

    # Output dirs
    run_dir = os.path.join(args.out_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    samples_dir = os.path.join(run_dir, "samples")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    fixed_z, _ = GM.sample(64)
    global_step = 0

    G.train()
    C.train()

    if not args.conditional_GAN:
        args.n_inner_loop = 1  # No need for multiple loops in unconditional case

    for epoch in range(1, args.epochs + 1):
        for _ in range(len(loader)//GM.n_components):

            # ============================================
            # 1. CRITIC TRAINING (args.n_critic steps)
            # ============================================
            for _ in range(args.n_critic):

                critic_loss_accum = 0.0

                # ---------------------------
                # GM.n_components inner loop
                # ---------------------------
                for _comp in range(args.n_inner_loop):

                    # ---- Sample real (NEW batch every time!) ----
                    real, real_labels = loader.get_batch()
                    real = real.view(real.size(0), -1).to(device)
                    real_labels = real_labels.to(device)
                    batch_size = real.size(0)

                    # ---- Sample latent ----
                    if args.conditional_GAN:
                        z, _ = GM.sample(batch_size, cluster_ids=real_labels)
                    else:
                        z, _ = GM.sample(batch_size)

                    z = z.to(device)
                    # ---- Conditional z adjustment ----
                    if args.conditional_GAN:
                        z = conditional_z(z, real_labels, num_classes=10)
                    # ---- Generate fake ----
                    fake = G(z).detach()

                    # ---- Critic scores ----
                    C_real = C(real, real_labels).mean()
                    C_fake = C(fake, real_labels).mean()

                    wasserstein = C_real - C_fake

                    # ---- Gradient penalty ----
                    critic_fn = lambda x: C(x, real_labels)
                    gp = args.lambda_gp * gradient_penalty(critic_fn, real, fake, device)

                    critic_loss_accum += (-wasserstein + gp)

                # ---- average and backprop critic ----
                critic_loss = critic_loss_accum / GM.n_components

                opt_C.zero_grad(set_to_none=True)
                critic_loss.backward()
                opt_C.step()

            # ============================================
            # 2. GENERATOR UPDATE (balanced GM loop)
            # ============================================

            gen_loss_accum = 0.0

            for _ in range(args.n_inner_loop):

                # ---- Random class label ∈ {0,...,9} ----
                if args.conditional_GAN:
                    k = torch.randint(GM.n_components, (1,), device=device).item()
                    cluster_ids = torch.full((batch_size,), k, dtype=torch.long, device=device)
                    z, _ = GM.sample(batch_size, cluster_ids=cluster_ids)
                    z = conditional_z(z, cluster_ids, num_classes=10)
                    fake = G(z)
                    gen_loss_accum += -C(fake, cluster_ids).mean()

                else:
                    # unconditional model
                    z, _ = GM.sample(batch_size)
                    fake = G(z)
                    gen_loss_accum += -C(fake).mean()

            # ---- average over GM components ----
            G_loss = gen_loss_accum / GM.n_components

            opt_G.zero_grad(set_to_none=True)
            G_loss.backward()
            opt_G.step()

            # Logs
            if global_step % args.log_every == 0:
                print(f"Epoch {epoch:03d}/{args.epochs}  Step {global_step:06d}  "
                      f"C_loss={critic_loss.item():.4f}  G_loss={G_loss.item():.4f}  "
                      f"W={wasserstein.item():.4f}  GP={gp.item():.4f}")

            # Sample grid
            if global_step % args.sample_every == 0:
                with torch.no_grad():
                    grid = sample_grid(G, n=64, z_dim=z_dim, device=device, GM=GM, conditional=args.conditional_GAN)
                    vutils.save_image(grid, os.path.join(samples_dir, f"step_{global_step:06d}.png"))

            # Checkpoint (periodic)
            if global_step == 0:
                    GM.save(os.path.join(ckpt_dir, "gmm_final.pt"))

            if global_step % args.ckpt_every == 0 and global_step > 0:
                D_act = mirror_to_activated_discriminator(C.state_dict(), d_input_dim=784)
                torch.save(
                    {
                        "G": G.state_dict(),
                        "D": D_act.state_dict(),
                        "epoch": epoch,
                        "step": global_step,
                    },
                    os.path.join(ckpt_dir, f"ckpt_{global_step:06d}.pt"),
                )
            # global_step increment moved below to run every batch
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
    p.add_argument("--out_dir", type=str, default="./runs/conditional_gm_wgan_gp_mnist_model", help="output dir")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--z_dim", type=int, default=100, help="latent dim; model.Generator is fixed at 100")
    p.add_argument("--g_lr", type=float, default=5e-4)
    p.add_argument("--d_lr", type=float, default=3e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.9)
    p.add_argument("--lambda_gp", type=float, default=10.0)
    p.add_argument("--n_critic", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--sample_every", type=int, default=500)
    p.add_argument("--ckpt_every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_GM", type=bool, default=False, help="Whether to use Gaussian Mixture for latent sampling.")
    p.add_argument("--n_components", type=int, default=10, help="Number of Gaussian components in GMM.")
    p.add_argument("--c", type=float, default=1, help="Range for Gaussian component means.")
    p.add_argument("--sigma", type=float, default=0.2, help="Standard deviation for Gaussian components.")
    p.add_argument("--conditional_GAN", type=bool, default=True, help="Whether to use a conditional GAN setup.")
    p.add_argument("--n_inner_loop", type=int, default=5, help="Number of inner loop iterations per GM component.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
