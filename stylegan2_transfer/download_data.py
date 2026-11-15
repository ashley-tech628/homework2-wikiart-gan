"""Download the assigned complete WikiArt dataset directly into /tmp."""

import argparse
from pathlib import Path

from data import EXTENSIONS, locate_source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        existing = locate_source(args.output)
    except FileNotFoundError:
        existing = None
    if existing is not None:
        count = sum(p.is_file() and p.suffix.lower() in EXTENSIONS
                    for p in existing.rglob("*"))
        styles = sum(p.is_dir() for p in existing.iterdir())
        if count == 81444 and styles == 27:
            print(f"Using existing {count}-image dataset under {existing}", flush=True)
            return
        print(f"Existing download is incomplete ({count} images, {styles} styles); retrying.",
              flush=True)
    import kagglehub

    downloaded = Path(kagglehub.dataset_download(
        "steubk/wikiart", output_dir=str(args.output),
        force_download=existing is not None))
    source = locate_source(args.output)
    print(f"Downloaded dataset to {downloaded}; style folders at {source}", flush=True)


if __name__ == "__main__":
    main()
