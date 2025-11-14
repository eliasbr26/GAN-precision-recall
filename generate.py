import os
import re
import argparse

import torch
import torchvision.utils as vutils

from model import Generator, Discriminator_W
from utils import generate_samples_with_full_DRS, generate_samples_with_DRS

def weight_file_paths(gen_name="G_WGAN-GP_130.pth", disc_name="D_WGAN-GP_130.pth", folder="checkpoints"):
    g_path = os.path.join(folder, gen_name)
    d_path = os.path.join(folder, disc_name)
    return g_path, d_path



def get_device(arg=None):
    if arg:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_weights(model, path, device="cpu", strict=True, pick=None):

    obj = torch.load(path, map_location=device)

    # Extract candidate state_dict
    state = obj
    if isinstance(obj, dict) and pick and pick in obj and isinstance(obj[pick], dict):
        state = obj[pick]
    elif isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        state = obj["state_dict"]
    elif isinstance(obj, dict) and any(k in obj for k in ("G", "C", "D")) and pick is None:
        for k in ("G", "C", "D"):
            if k in obj and isinstance(obj[k], dict):
                state = obj[k]
                break

    # Normalize DataParallel prefixes
    dst_keys = model.state_dict().keys()
    wants_module = any(str(k).startswith("module.") for k in dst_keys)
    has_module = any(isinstance(k, str) and k.startswith("module.") for k in state.keys())

    if has_module and not wants_module:
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    elif not has_module and wants_module:
        state = {f"module.{k}": v for k, v in state.items()}

    model.load_state_dict(state, strict=strict)
    return model


def parse_step(path):
    name = os.path.basename(path)
    m = re.search(r"_(\d{4,})\.(pth|pt)$", name)
    return int(m.group(1)) if m else None




def main():
    parser = argparse.ArgumentParser(description="Generate samples with DRS using saved WGAN-GP weights")
    parser.add_argument("--folder", type=str, default="checkpoints",
                        help="Folder to auto-discover weights (defaults to ./checkpoints)")
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--ckpt_path", type=str, default=None, help="Path to a single checkpoint dict that contains 'G' and 'C'")
    src.add_argument("--g_path", type=str, default=None, help="Path to Generator state_dict (.pth/.pt)")
    parser.add_argument("--c_path", type=str, default=None, help="Path to Critic (Discriminator_W) state_dict (.pth/.pt)")
    parser.add_argument("--device", type=str, default=None, help="cpu | cuda | mps (auto if omitted)")
    parser.add_argument("--out_dir", type=str, default="samples", help="Output folder for PNGs")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of accepted samples to generate")
    parser.add_argument("--batch_size", type=int, default=128, help="Proposals per iteration for DRS")
    parser.add_argument("--tau", type=float, default=10, help="DRS threshold; higher = more selective")
    parser.add_argument("--z_dim", type=int, default=100, help="Latent dimension (Generator expects 100)")
    parser.add_argument("--grid", action="store_true", help="Also save a grid preview image")
    args = parser.parse_args()

    # Default to known filenames in ./checkpoints when no paths are provided
    if not args.ckpt_path and args.g_path is None and args.c_path is None:
        args.g_path, args.c_path = weight_file_paths(folder=args.folder or "checkpoints")
        print(f"[default] G: {args.g_path}\n[default] D: {args.c_path}")
        # Optional existence check for clearer error messages
        for p in (args.g_path, args.c_path):
            if not os.path.isfile(p):
                raise SystemExit(f"[error] Expected weight file not found: {p}")

    device = get_device(args.device)
    print(f"[device] {device}")

    # Build models
    mnist_dim = 784
    G = Generator(g_output_dim=mnist_dim).to(device)
    C = Discriminator_W(d_input_dim=mnist_dim).to(device)

    # Load weights
    if args.ckpt_path:
        print(f"[load] checkpoint: {args.ckpt_path}")
        load_weights(G, args.ckpt_path, device=device, pick="G")
        load_weights(C, args.ckpt_path, device=device, pick="C")
    else:
        if args.g_path is None or args.c_path is None:
            raise ValueError("When using --g_path, you must also provide --c_path.")
        print(f"[load] G: {args.g_path}")
        print(f"[load] C: {args.c_path}")
        load_weights(G, args.g_path, device=device)
        load_weights(C, args.c_path, device=device)

    # Generate with DRS (using logits from Discriminator_W)
    print(f"[DRS] Generating {args.num_samples} samples (tau={args.tau}, batch_size={args.batch_size})")
    target_percentile = int(args.tau * 100)
    samples = generate_samples_with_full_DRS(
        G,
        C,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )

    # Save
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    for idx in range(samples.size(0)):
        vutils.save_image(
            samples[idx].view(1, 28, 28),
            os.path.join(out_dir, f"{idx:05d}.png"),
            normalize=True,
        )

    if args.grid:
        grid = vutils.make_grid(samples.view(-1, 1, 28, 28), nrow=16, normalize=True, value_range=(-1, 1))
        vutils.save_image(grid, os.path.join(out_dir, "grid.png"))
        print(f"[save] grid -> {os.path.join(out_dir, 'grid.png')}")

    print(f"[done] Saved {samples.size(0)} images to {out_dir}")


if __name__ == "__main__":
    main()

