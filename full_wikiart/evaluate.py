"""Evaluate the EMA generator with held-out real paintings using Ignite IS/FID."""

import argparse
import json
from pathlib import Path

import torch
from ignite.metrics import FID, InceptionScore
from ignite.metrics.gan.utils import InceptionModel
from torch.utils.data import DataLoader, Subset
from torchvision.transforms.functional import resize

from model import Generator
from train import make_dataset


def inception_input(images: torch.Tensor) -> torch.Tensor:
    return resize(((images + 1) / 2).clamp(0, 1), [299, 299], antialias=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    generator = Generator(len(config["classes"]), config["image_size"],
                          config["latent_dim"], config["base_channels"]).to(device)
    generator.load_state_dict(state["ema"])
    generator.eval()
    full, _, holdout = make_dataset(args.data, config["seed"], config["image_size"],
                                    min_images=100 if config["allow_small_data"] else 50000)
    if full.classes != config["classes"] or len(full) != config["dataset_size"]:
        raise ValueError("Evaluation dataset does not match training checkpoint")
    count = min(args.max_images, len(holdout))
    if count < 100:
        raise ValueError("At least 100 held-out images are required")
    loader = DataLoader(Subset(holdout, range(count)), batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    fid = FID(num_features=2048,
              feature_extractor=InceptionModel(return_features=True, device=device),
              device=device)
    fid_blog = FID(device=device)
    inception_score = InceptionScore(device=device)
    for metric in (fid, fid_blog, inception_score):
        metric.reset()
    torch.manual_seed(config["seed"] + 1000)
    with torch.inference_mode():
        for i, (real, labels) in enumerate(loader, 1):
            real = real.to(device)
            labels = labels.to(device)
            noise = torch.randn(real.size(0), config["latent_dim"], device=device)
            fake = generator(noise, labels)
            real_for_metric = inception_input(real)
            fake_for_metric = inception_input(fake)
            fid.update((fake_for_metric, real_for_metric))
            fid_blog.update((fake_for_metric, real_for_metric))
            inception_score.update(fake_for_metric)
            print(f"Evaluated {min(i * args.batch_size, count)}/{count} pairs", flush=True)
    result = {"fid_pool3": float(fid.compute()),
              "fid_ignite_blog_default": float(fid_blog.compute()),
              "inception_score": float(inception_score.compute()),
              "real_images": count, "generated_images": count,
              "checkpoint_epoch": state["epoch"], "training_images": config["train_size"],
              "heldout_images": config["holdout_size"], "source_images": config["dataset_size"],
              "image_size": config["image_size"], "styles": len(config["classes"]),
              "generator": "EMA", "metric_library": "pytorch-ignite",
              "resize": "bilinear antialias to 299x299",
              "real_split": "seeded held-out 10%; matching style labels for fakes"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
