"""Fixed seeds; native-resolution PNGs; no enhancement or cherry-picked images."""
import argparse
import json
from pathlib import Path
import torch
from torchvision.utils import save_image
from torchvision.transforms.functional import resize
from runtime import load_state, make_models


@torch.no_grad()
def sample(G, out, seed=2026, count=16, psi=1.0, individual=False):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    device = next(G.parameters()).device
    G.eval()
    thumbs = []
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
        for i in range(count):
            torch.manual_seed(seed+i)
            img = G(torch.randn(1, G.z_dim, device=device), torch.zeros(1, G.c_dim, device=device),
                    truncation_psi=psi, noise_mode='const').float().cpu()
            if individual:
                folder = out.parent / (out.stem + '_1024px')
                folder.mkdir(exist_ok=True)
                save_image(img, folder/f'seed_{seed+i:05d}.png', normalize=True, value_range=(-1,1))
            thumbs.append(resize(img, [256,256], antialias=True))
    save_image(torch.cat(thumbs), out, nrow=4, normalize=True, value_range=(-1,1))
    out.with_suffix('.json').write_text(json.dumps({'seeds': list(range(seed,seed+count)),
         'truncation_psi': psi, 'noise_mode': 'const', 'native_resolution': G.img_resolution,
         'grid_tile_resolution':256,'postprocessing':'display resizing only; no enhancement'},indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--psi',type=float,default=1)
    p.add_argument('--count',type=int,default=16)
    a=p.parse_args()
    s=load_state(a.checkpoint)
    G=make_models(s,generator_only=True)
    sample(G,a.output,count=a.count,psi=a.psi,individual=True)
