"""Download a reproducible, balanced subset of the assigned Kaggle WikiArt set.

The dataset contains over 80k images. Kaggle's public file-list API and
single-file download endpoint let us avoid pulling the entire 33.7 GB archive.
"""

import argparse
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


BASE = "https://www.kaggle.com/api/v1/datasets"
HEADERS = {"User-Agent": "homework2-wikiart-research/1.0"}


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as response:
        return json.load(response)


def list_files(styles):
    """Fetch metadata until all requested alphabetical style directories pass."""
    found = defaultdict(list)
    token = None
    pages = 0
    while True:
        query = {"pageSize": 200}
        if token:
            query["pageToken"] = token
        url = f"{BASE}/list/steubk/wikiart?{urllib.parse.urlencode(query)}"
        payload = get_json(url)
        files = payload.get("datasetFiles", [])
        pages += 1
        for item in files:
            name = item["name"]
            style = name.split("/", 1)[0]
            if style in styles and name.lower().endswith((".jpg", ".jpeg", ".png")):
                found[style].append({"name": name, "bytes": item.get("totalBytes")})
        if pages % 10 == 0:
            print(f"Indexed {pages * 200} entries; " + ", ".join(f"{k}: {len(found[k])}" for k in styles), flush=True)
        token = payload.get("nextPageToken")
        if not token:
            break
        # API pages are sorted by path. Stop after the last requested directory.
        if files and files[-1]["name"].split("/", 1)[0] > max(styles):
            break
    return found


def download_one(entry, root):
    name = entry["name"]
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (not entry["bytes"] or path.stat().st_size == entry["bytes"]):
        return name, "cached"
    url = f"{BASE}/download/steubk/wikiart?" + urllib.parse.urlencode({"filename": name})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120) as response:
                payload = response.read()
            if entry["bytes"] and len(payload) != entry["bytes"]:
                raise IOError(f"wrong size: {len(payload)} != {entry['bytes']}")
            path.write_bytes(payload)
            return name, "downloaded"
        except (OSError, urllib.error.URLError) as exc:
            if attempt == 4:
                return name, f"failed: {exc}"
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--styles", nargs="+", default=["Abstract_Expressionism", "Baroque", "Impressionism"])
    parser.add_argument("--per-style", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["styles"] != args.styles or manifest["per_style"] != args.per_style:
            raise ValueError("Existing manifest uses different selection settings")
        selected = manifest["files"]
    else:
        found = list_files(set(args.styles))
        rng = random.Random(args.seed)
        selected = []
        for style in args.styles:
            if len(found[style]) < args.per_style:
                raise ValueError(f"{style} contains only {len(found[style])} images")
            selected.extend(rng.sample(found[style], args.per_style))
        manifest = {
            "source": "https://www.kaggle.com/datasets/steubk/wikiart",
            "dataset_version": 1,
            "styles": args.styles,
            "per_style": args.per_style,
            "seed": args.seed,
            "files": selected,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    done = 0
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, item, args.output) for item in selected]
        for future in as_completed(futures):
            name, status = future.result()
            done += 1
            if status.startswith("failed"):
                failures.append((name, status))
            if done % 100 == 0 or done == len(selected):
                print(f"{done}/{len(selected)} files processed; failures={len(failures)}", flush=True)
    if failures:
        (args.output / "download_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise RuntimeError(f"{len(failures)} downloads failed")


if __name__ == "__main__":
    main()
