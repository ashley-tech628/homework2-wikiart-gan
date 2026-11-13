"""Generate a grid of paintings from the trained generator weights."""

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from gan import Generator, noise_batch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=64)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.model, map_location=device, weights_only=True)
    generator = Generator(payload["latent_dim"], payload["width"]).to(device)
    generator.load_state_dict(payload["generator"])
    generator.eval()
    torch.manual_seed(args.seed)
    with torch.inference_mode():
        images = generator(noise_batch(args.count, payload["latent_dim"], device)).cpu()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(images, args.output, nrow=8, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    main()
