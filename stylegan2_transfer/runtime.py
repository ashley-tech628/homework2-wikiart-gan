"""Use the pinned NVIDIA implementation, with compiler-free reference ops."""
import copy
import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parent / 'vendor' / 'stylegan2_ada'
sys.path.insert(0, str(VENDOR))
import torch
from training.networks import Generator, Discriminator
from torch_utils.ops import conv2d_gradfix, grid_sample_gradfix

conv2d_gradfix.enabled = False  # Native torch 2.x supports convolution double backward.
grid_sample_gradfix.enabled = True  # NVIDIA StyleGAN3's newer double-backward adapter.


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def make_models(state, device='cuda', generator_only=False):
    cfg = state['model_config']
    G = Generator(**cfg['G']).to(device).eval().requires_grad_(False)
    G.load_state_dict(state['G_ema'] if generator_only else state['G'])
    if generator_only:
        return G
    D = Discriminator(**cfg['D']).to(device).eval().requires_grad_(False)
    D.load_state_dict(state['D'])
    ema = copy.deepcopy(G).eval().requires_grad_(False)
    ema.load_state_dict(state['G_ema'])
    return G, D, ema


def load_state(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    with temp.open('wb') as f:
        torch.save(value, f)
        f.flush()
        import os
        os.fsync(f.fileno())
    temp.replace(path)


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
