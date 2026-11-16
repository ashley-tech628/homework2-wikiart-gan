"""Compute Ignite Inception Score and FID against held-out WikiArt images."""

import argparse
import json
from pathlib import Path

import torch
from ignite.metrics import FID, InceptionScore
from ignite.metrics.gan.utils import InceptionModel
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize

from gan import Generator, noise_batch
from train import make_dataset


def inception_input(images):
    # Both the real and generated images originate in [-1, 1].
    images = ((images + 1) / 2).clamp(0, 1)
    return resize(images, [299, 299], antialias=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-images", type=int, default=300)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = state["config"]
    generator = Generator(config["latent_dim"], config["width"]).to(device)
    generator.load_state_dict(state["generator"])
    generator.eval()
    _, _, holdout = make_dataset(args.data, config["seed"])
    count = min(args.max_images, len(holdout))
    loader = DataLoader(torch.utils.data.Subset(holdout, range(count)),
                        batch_size=args.batch_size, shuffle=False, num_workers=0)
    # Ignite's blog defaults to 1000-dimensional class probabilities for FID.
    # Pool3 features are the conventional, more interpretable FID variant.
    fid = FID(num_features=2048,
              feature_extractor=InceptionModel(return_features=True, device=device),
              device=device)
    fid_blog = FID(device=device)
    inception_score = InceptionScore(device=device)
    fid.reset()
    fid_blog.reset()
    inception_score.reset()
    torch.manual_seed(config["seed"] + 1)
    with torch.inference_mode():
        for i, (real, _) in enumerate(loader, 1):
            real = real.to(device)
            fake = generator(noise_batch(real.size(0), config["latent_dim"], device))
            real = inception_input(real)
            fake = inception_input(fake)
            fid.update((fake, real))
            fid_blog.update((fake, real))
            inception_score.update(fake)
            print(f"Evaluated {min(i * args.batch_size, count)}/{count} image pairs", flush=True)
    result = {"fid_pool3": float(fid.compute()),
              "fid_ignite_blog_default": float(fid_blog.compute()),
              "inception_score": float(inception_score.compute()),
              "real_images": count, "generated_images": count,
              "checkpoint_epoch": state["epoch"], "metric_library": "pytorch-ignite",
              "resize": "bilinear antialias to 299x299", "real_split": "held-out 20%"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
