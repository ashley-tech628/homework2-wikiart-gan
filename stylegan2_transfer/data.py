"""Index the whole assigned dataset; decode original-resolution images on demand."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


def locate_source(root):
    root = Path(root)
    if (root / 'Abstract_Expressionism').is_dir():
        return root
    matches = [p.parent for p in root.rglob('Abstract_Expressionism') if p.is_dir()]
    if len(matches) != 1:
        raise ValueError(f'Expected one WikiArt dataset below {root}; found {len(matches)}')
    return matches[0]


def index_dataset(root, seed=2026, allow_small=False):
    root = locate_source(root)
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob('*')
                   if p.is_file() and p.suffix.lower() in EXTENSIONS)
    styles = sorted({f.split('/')[0] for f in files})
    if not allow_small and (len(files) != 81444 or len(styles) != 27):
        raise ValueError(f'Incomplete source: {len(files)} files, {len(styles)} styles; expected ~81,444 / 27')
    if len(files) < 20:
        raise ValueError('Too few images, even for a local diagnostic')
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files)).tolist()
    # Whole source is used: 90% train, 5% selection validation, 5% final test.
    ntrain = int(len(files) * .9)
    nval = (len(files) - ntrain) // 2
    sizes = [int((root / f).stat().st_size) for f in files]
    fingerprint = hashlib.sha256(json.dumps(list(zip(files, sizes))).encode()).hexdigest()
    return {'source_url': 'https://www.kaggle.com/datasets/steubk/wikiart',
            'root': str(root), 'files': files, 'styles': styles, 'source_images': len(files),
            'train': order[:ntrain], 'validation': order[ntrain:ntrain+nval],
            'test': order[ntrain+nval:], 'seed': seed, 'fingerprint': fingerprint,
            'preprocessing': 'EXIF transpose; RGB; center crop and Lanczos resize on demand to 1024; no upscaling of old 128px prepared dataset'}


class Paintings(Dataset):
    def __init__(self, manifest, indices, size=1024):
        self.root = Path(manifest['root'])
        self.files = manifest['files']
        self.indices = indices
        self.size = size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        path = self.root / self.files[self.indices[i]]
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert('RGB')
                image = ImageOps.fit(image, (self.size, self.size), method=Image.Resampling.LANCZOS)
                array = np.array(image, dtype=np.uint8, copy=True)
        except Exception as e:
            raise RuntimeError(f'Unreadable source image (not silently skipped): {path}: {e}') from e
        return torch.from_numpy(array).permute(2, 0, 1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-small-data', action='store_true')
    a = p.parse_args()
    m = index_dataset(a.source, allow_small=a.allow_small_data)
    # Catch the common mistake of supplying the previous 128px output folder.
    dims = []
    for i in np.linspace(0, len(m['files'])-1, min(32, len(m['files'])), dtype=int):
        with Image.open(Path(m['root']) / m['files'][i]) as im:
            dims.append(min(im.size))
    if not a.allow_small_data and max(dims) <= 128:
        raise ValueError('This is the old resized dataset. Supply the original raw WikiArt download.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(m, indent=2), encoding='utf-8')
    print(f"Indexed {m['source_images']} original images, {len(m['styles'])} styles; train/val/test = {len(m['train'])}/{len(m['validation'])}/{len(m['test'])}", flush=True)
