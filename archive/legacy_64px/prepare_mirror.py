"""Extract image shards from the public HugGAN WikiArt mirror.

The mirror is a fallback for cases where Kaggle's per-file endpoint blocks
bulk requests. It is not claimed to be byte-identical to the assigned Kaggle
upload. The output manifest records provenance for every extracted file.
"""

import argparse
import io
import json
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageOps


INFO_URL = "https://huggingface.co/datasets/huggan/wikiart/raw/main/dataset_infos.json"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shards", nargs="+", type=Path, required=True)
    p.add_argument("--kaggle-dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    info = json.load(urllib.request.urlopen(INFO_URL))
    style_names = next(iter(info.values()))["features"]["style"]["names"]
    manifest = {"mirror": "https://huggingface.co/datasets/huggan/wikiart",
                "kaggle": "https://www.kaggle.com/datasets/steubk/wikiart",
                "mirror_shards": [p.name for p in args.shards], "mirror_count": 0,
                "kaggle_count": 0, "files": []}
    for shard in args.shards:
        parquet = pq.ParquetFile(shard)
        print(f"Reading {shard}: {parquet.metadata.num_rows} rows", flush=True)
        for batch in parquet.iter_batches(batch_size=32, columns=["image", "style"]):
            for row in batch.to_pylist():
                style = style_names[row["style"]]
                index = manifest["mirror_count"]
                relative = Path(style) / f"mirror_{index:05d}.jpg"
                destination = args.output / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with Image.open(io.BytesIO(row["image"]["bytes"])) as image:
                        image = ImageOps.exif_transpose(image).convert("RGB")
                        image = ImageOps.fit(image, (72, 72), method=Image.Resampling.LANCZOS)
                        image.save(destination, "JPEG", quality=92)
                except (OSError, ValueError):
                    continue
                manifest["files"].append({"file": relative.as_posix(), "source": "mirror"})
                manifest["mirror_count"] += 1
    if args.kaggle_dir:
        for source in args.kaggle_dir.rglob("*.jpg"):
            relative = source.relative_to(args.kaggle_dir)
            destination = args.output / relative
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                with Image.open(source) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    image = ImageOps.fit(image, (72, 72), method=Image.Resampling.LANCZOS)
                    image.save(destination, "JPEG", quality=92)
            except (OSError, ValueError):
                continue
            manifest["files"].append({"file": relative.as_posix(), "source": "kaggle"})
            manifest["kaggle_count"] += 1
    (args.output / "provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {manifest['mirror_count']} mirror and {manifest['kaggle_count']} Kaggle images", flush=True)


if __name__ == "__main__":
    main()
