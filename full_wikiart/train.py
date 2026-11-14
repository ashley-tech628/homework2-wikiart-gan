"""Train a full-dataset, style-conditional painting GAN with EMA checkpoints."""

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from model import Discriminator, Generator, initialize


def make_dataset(root: Path, seed: int, image_size: int,
                 holdout_fraction: float = 0.1, min_images: int = 50000):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    full = datasets.ImageFolder(root, transform=transform)
    if len(full) < min_images:
        raise ValueError(f"Expected the full WikiArt dataset, found only {len(full)} images")
    order = np.random.default_rng(seed).permutation(len(full))
    holdout_size = round(len(full) * holdout_fraction)
    return full, Subset(full, order[holdout_size:].tolist()), Subset(full, order[:holdout_size].tolist())


def translate(images: torch.Tensor, pixels: int = 2) -> torch.Tensor:
    if pixels <= 0:
        return images
    dy = random.randint(-pixels, pixels)
    dx = random.randint(-pixels, pixels)
    padded = F.pad(images, (pixels, pixels, pixels, pixels), mode="reflect")
    top, left = pixels + dy, pixels + dx
    return padded[:, :, top:top + images.size(2), left:left + images.size(3)]


def update_ema(ema: nn.Module, current: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for target, source in zip(ema.parameters(), current.parameters()):
            target.lerp_(source, 1 - decay)
        for target, source in zip(ema.buffers(), current.buffers()):
            target.copy_(source)


def save_grid(generator: nn.Module, noise: torch.Tensor, labels: torch.Tensor,
              destination: Path) -> None:
    generator.eval()
    with torch.inference_mode():
        generated = generator(noise, labels).float().cpu()
    save_image(generated, destination, nrow=8, normalize=True, value_range=(-1, 1))


def save_checkpoint(path: Path, generator: nn.Module, discriminator: nn.Module,
                    ema: nn.Module, optim_g, optim_d, scaler, epoch: int,
                    step: int, config: dict) -> None:
    state = {"generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
             "ema": ema.state_dict(), "optim_g": optim_g.state_dict(),
             "optim_d": optim_d.state_dict(), "scaler": scaler.state_dict(),
             "epoch": epoch, "step": step, "config": config}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=128, choices=[64, 128])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr-g", type=float, default=0.0001)
    parser.add_argument("--lr-d", type=float, default=0.0002)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-hours", type=float, default=10.5)
    parser.add_argument("--max-steps", type=int, default=0, help="Short verification run only")
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--allow-small-data", action="store_true", help="Verification run only")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required. Enable a Kaggle GPU accelerator.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    full, train, holdout = make_dataset(args.data, args.seed, args.image_size,
                                        min_images=100 if args.allow_small_data else 50000)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True,
                        drop_last=True, persistent_workers=args.workers > 0)
    generator = Generator(len(full.classes), args.image_size, args.latent_dim,
                          args.base_channels).to(device)
    discriminator = Discriminator(len(full.classes), args.image_size,
                                  args.base_channels).to(device)
    generator.apply(initialize)
    discriminator.apply(initialize)
    ema = copy.deepcopy(generator).eval().requires_grad_(False)
    optim_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.0, 0.9))
    optim_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.0, 0.9))
    scaler = torch.amp.GradScaler("cuda")
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(dataset_size=len(full), train_size=len(train), holdout_size=len(holdout),
                  classes=full.classes, torch_version=str(torch.__version__),
                  gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    start_epoch, step = 1, 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        previous = state["config"]
        for key in ("classes", "dataset_size", "image_size", "latent_dim", "base_channels", "seed"):
            if previous[key] != config[key]:
                raise ValueError(f"Cannot resume: {key} changed")
        generator.load_state_dict(state["generator"])
        discriminator.load_state_dict(state["discriminator"])
        ema.load_state_dict(state["ema"])
        optim_g.load_state_dict(state["optim_g"])
        optim_d.load_state_dict(state["optim_d"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, step = state["epoch"] + 1, state["step"]
        print(f"Resumed after epoch {state['epoch']} step {step}", flush=True)
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    forward_g = (nn.DataParallel(generator) if args.multi_gpu and torch.cuda.device_count() > 1
                 else generator)
    forward_d = (nn.DataParallel(discriminator) if args.multi_gpu and torch.cuda.device_count() > 1
                 else discriminator)
    fixed_noise = torch.randn(64, args.latent_dim, device=device)
    fixed_labels = torch.linspace(0, len(full.classes) - 1, 8, device=device).long().repeat_interleave(8)
    if start_epoch == 1:
        real_grid = torch.stack([train[i][0] for i in range(64)])
        save_image(real_grid, args.output / "real_grid.png", nrow=8,
                   normalize=True, value_range=(-1, 1))
    loss_path = args.output / "losses.csv"
    with loss_path.open("a" if start_epoch > 1 else "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if start_epoch == 1:
            writer.writerow(["epoch", "step", "loss_d", "loss_g"])
        begin = time.time()
        for epoch in range(start_epoch, args.epochs + 1):
            torch.manual_seed(args.seed + epoch)
            torch.cuda.manual_seed_all(args.seed + epoch)
            generator.train()
            discriminator.train()
            d_total = g_total = batches = 0
            for real, labels in loader:
                real = real.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                count = real.size(0)
                optim_d.zero_grad(set_to_none=True)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    fake = forward_g(torch.randn(count, args.latent_dim, device=device), labels)
                with torch.autocast("cuda", dtype=torch.float16):
                    real_scores = forward_d(translate(real), labels)
                    fake_scores = forward_d(translate(fake.detach()), labels)
                    loss_d = F.relu(1 - real_scores).mean() + F.relu(1 + fake_scores).mean()
                scaler.scale(loss_d).backward()
                scaler.step(optim_d)

                for parameter in discriminator.parameters():
                    parameter.requires_grad_(False)
                optim_g.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    fake = forward_g(torch.randn(count, args.latent_dim, device=device), labels)
                    loss_g = -forward_d(translate(fake), labels).mean()
                scaler.scale(loss_g).backward()
                scaler.step(optim_g)
                scaler.update()
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)
                update_ema(ema, generator, args.ema_decay)
                step += 1
                batches += 1
                d_total += loss_d.item()
                g_total += loss_g.item()
                writer.writerow([epoch, step, loss_d.item(), loss_g.item()])
                if step % 100 == 0:
                    print(f"epoch {epoch}/{args.epochs} step {step}: "
                          f"D={loss_d.item():.3f} G={loss_g.item():.3f}", flush=True)
                if args.max_steps and step >= args.max_steps:
                    break
            if batches == 0:
                raise RuntimeError("Training produced no batches; reduce --batch-size")
            stream.flush()
            save_grid(ema, fixed_noise, fixed_labels, args.output / f"samples_epoch_{epoch:03d}.png")
            save_checkpoint(args.output / "checkpoint.pt", generator, discriminator,
                            ema, optim_g, optim_d, scaler, epoch, step, config)
            print(f"epoch {epoch} complete: D={d_total/batches:.3f} "
                  f"G={g_total/batches:.3f}; elapsed={(time.time()-begin)/3600:.2f} h", flush=True)
            if (args.max_steps and step >= args.max_steps or
                    time.time() - begin >= args.max_hours * 3600):
                print("Stopped after a saved epoch; resume from checkpoint.pt if needed.", flush=True)
                break


if __name__ == "__main__":
    main()
