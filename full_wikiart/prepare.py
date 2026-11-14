"""Prepare every decodable painting from the assigned steubk/wikiart dataset."""

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps


EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SOURCE_URL = "https://www.kaggle.com/datasets/steubk/wikiart"


def locate_source(input_root: Path) -> Path:
    matches = [p for p in input_root.rglob("Abstract_Expressionism") if p.is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one mounted {SOURCE_URL} dataset below {input_root}; "
            f"found {len(matches)} Abstract_Expressionism directories."
        )
    return matches[0].parent


def prepare(input_root: Path, output: Path, size: int = 128) -> dict:
    source_root = locate_source(input_root)
    candidates = sorted(p for p in source_root.rglob("*")
                        if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not candidates:
        raise RuntimeError(f"No images found under {source_root}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"source_url": SOURCE_URL, "mounted_root": str(source_root),
                "available_images": len(candidates), "image_size": size,
                "prepared_images": 0, "styles": {}, "files": [], "skipped": []}
    style_counts = Counter()
    for index, source in enumerate(candidates):
        relative = source.relative_to(source_root)
        style = relative.parts[0]
        destination = output / style / f"wikiart_{index:06d}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not destination.is_file() or destination.stat().st_size == 0:
                with Image.open(source) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    image = ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)
                    image.save(destination, "JPEG", quality=90, subsampling=0)
        except (OSError, ValueError) as error:
            manifest["skipped"].append({"source": relative.as_posix(), "error": str(error)})
            continue
        style_counts[style] += 1
        manifest["files"].append({"source": relative.as_posix(),
                                  "prepared": destination.relative_to(output).as_posix()})
        manifest["prepared_images"] += 1
        if manifest["prepared_images"] % 5000 == 0:
            print(f"Prepared {manifest['prepared_images']}/{len(candidates)} paintings", flush=True)
    manifest["styles"] = dict(sorted(style_counts.items()))
    (output / "full_provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared all {manifest['prepared_images']} decodable paintings across "
          f"{len(style_counts)} styles; skipped {len(manifest['skipped'])}.", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()
    if args.size not in {64, 128}:
        parser.error("--size must be 64 or 128")
    prepare(args.input_root, args.output, args.size)


if __name__ == "__main__":
    main()
