"""Generate style-conditioned painting grids from a full WikiArt checkpoint."""

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from model import Generator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--style", type=str, help="One of the style names in config.json")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    generator = Generator(len(config["classes"]), config["image_size"],
                          config["latent_dim"], config["base_channels"]).to(device)
    generator.load_state_dict(state["ema"])
    generator.eval()
    torch.manual_seed(args.seed)
    if args.style:
        if args.style not in config["classes"]:
            parser.error(f"Unknown style {args.style!r}; choose from {config['classes']}")
        labels = torch.full((args.count,), config["classes"].index(args.style),
                            device=device, dtype=torch.long)
    else:
        styles = torch.linspace(0, len(config["classes"]) - 1, 8,
                                device=device).long()
        labels = styles.repeat_interleave((args.count + 7) // 8)[:args.count]
    with torch.inference_mode():
        images = generator(torch.randn(args.count, config["latent_dim"], device=device), labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(images.float().cpu(), args.output, nrow=8,
               normalize=True, value_range=(-1, 1))
    print(f"Saved {args.count} paintings to {args.output}")


if __name__ == "__main__":
    main()
