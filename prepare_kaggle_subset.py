"""Prepare a seeded image subset from the assigned Kaggle WikiArt upload."""

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageOps


EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def locate_source(input_root):
    matches = [path for path in input_root.rglob("Abstract_Expressionism") if path.is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one mounted WikiArt dataset with an Abstract_Expressionism folder under {input_root}; "
            f"found {len(matches)}. Attach steubk/wikiart in Kaggle Notebook's Add Input menu."
        )
    return matches[0].parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=2390)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    source_root = locate_source(args.input_root)
    candidates = sorted(path for path in source_root.rglob("*")
                        if path.is_file() and path.suffix.lower() in EXTENSIONS)
    if len(candidates) < args.count:
        raise ValueError(f"Found only {len(candidates)} images, requested {args.count}")
    rng = random.Random(args.seed)
    selected = rng.sample(candidates, len(candidates))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "https://www.kaggle.com/datasets/steubk/wikiart",
        "mounted_root": str(source_root), "available_images": len(candidates),
        "selected_images": 0, "seed": args.seed, "files": [], "skipped_files": [],
    }
    for source in selected:
        if manifest["selected_images"] == args.count:
            break
        index = manifest["selected_images"]
        style = source.relative_to(source_root).parts[0]
        destination = args.output / style / f"kaggle_{index:05d}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(source) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image = ImageOps.fit(image, (72, 72), method=Image.Resampling.LANCZOS)
                image.save(destination, "JPEG", quality=92)
        except (OSError, ValueError) as error:
            manifest["skipped_files"].append({"source": source.relative_to(source_root).as_posix(),
                                               "error": str(error)})
            continue
        manifest["files"].append({"source": source.relative_to(source_root).as_posix(),
                                  "prepared": destination.relative_to(args.output).as_posix()})
        manifest["selected_images"] += 1
        if manifest["selected_images"] % 250 == 0:
            print(f"Prepared {manifest['selected_images']}/{args.count} images", flush=True)
    if manifest["selected_images"] != args.count:
        raise RuntimeError(f"Only {manifest['selected_images']} decodable images were found")
    (args.output / "kaggle_provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {args.count} images from {source_root}")


if __name__ == "__main__":
    main()
