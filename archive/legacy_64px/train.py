"""Train the WikiArt DCGAN and save losses, checkpoints, and image grids."""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from gan import Discriminator, Generator, init_weights, noise_batch


def make_dataset(root, seed, holdout_fraction=0.2):
    transform = transforms.Compose([
        transforms.Resize(72), transforms.CenterCrop(64),
        transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    dataset = datasets.ImageFolder(root, transform=transform)
    if len(dataset) < 100:
        raise ValueError("At least 100 downloaded images are required")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(dataset))
    count = int(len(dataset) * holdout_fraction)
    return dataset, Subset(dataset, order[count:].tolist()), Subset(dataset, order[:count].tolist())


def save_grid(generator, fixed_noise, path):
    generator.eval()
    with torch.inference_mode():
        images = generator(fixed_noise).cpu()
    save_image(images, path, nrow=8, normalize=True, value_range=(-1, 1))
    generator.train()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=100)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--lr-g", type=float, default=0.0002)
    p.add_argument("--lr-d", type=float, default=0.0001)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=0, help="For a quick smoke test only")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full, train, holdout = make_dataset(args.data, args.seed)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=device.type == "cuda",
                        drop_last=True, persistent_workers=args.workers > 0)
    generator = Generator(args.latent_dim, args.width).to(device).apply(init_weights)
    discriminator = Discriminator(args.width).to(device).apply(init_weights)
    optim_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.0, 0.9))
    optim_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.0, 0.9))
    fixed_noise = noise_batch(64, args.latent_dim, device)
    sample_real = torch.stack([train[i][0] for i in range(64)])
    save_image(sample_real, args.output / "real_grid.png", nrow=8,
               normalize=True, value_range=(-1, 1))
    config = vars(args).copy()
    config.update(device=str(device), dataset_size=len(full), train_size=len(train),
                  holdout_size=len(holdout), classes=full.classes,
                  torch_version=torch.__version__)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    start = time.time()
    with (args.output / "losses.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "step", "loss_d", "loss_g"])
        step = 0
        for epoch in range(1, args.epochs + 1):
            d_total = g_total = batches = 0
            for real, _ in loader:
                real = real.to(device, non_blocking=True)
                count = real.size(0)
                # Spectral-normalized discriminator with the hinge objective.
                optim_d.zero_grad(set_to_none=True)
                real_logits = discriminator(real)
                fake = generator(noise_batch(count, args.latent_dim, device))
                fake_logits = discriminator(fake.detach())
                loss_d = torch.relu(1 - real_logits).mean() + torch.relu(1 + fake_logits).mean()
                loss_d.backward()
                optim_d.step()
                # Generator maximizes discriminator scores for synthetic images.
                optim_g.zero_grad(set_to_none=True)
                fake_logits = discriminator(fake)
                loss_g = -fake_logits.mean()
                loss_g.backward()
                optim_g.step()
                step += 1
                batches += 1
                d_total += loss_d.item()
                g_total += loss_g.item()
                writer.writerow([epoch, step, loss_d.item(), loss_g.item()])
                if step % 20 == 0:
                    print(f"epoch {epoch}/{args.epochs} step {step}: D={loss_d.item():.3f} G={loss_g.item():.3f}", flush=True)
                if args.max_steps and step >= args.max_steps:
                    break
            stream.flush()
            save_grid(generator, fixed_noise, args.output / f"samples_epoch_{epoch:02d}.png")
            checkpoint = {"generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
                          "optim_g": optim_g.state_dict(), "optim_d": optim_d.state_dict(),
                          "epoch": epoch, "step": step, "config": config}
            torch.save(checkpoint, args.output / "checkpoint.pt")
            print(f"epoch {epoch} complete: D={d_total/batches:.3f} G={g_total/batches:.3f}; "
                  f"elapsed={(time.time()-start)/60:.1f} min", flush=True)
            if args.max_steps and step >= args.max_steps:
                break


if __name__ == "__main__":
    main()
