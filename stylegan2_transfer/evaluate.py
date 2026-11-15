"""Ignite pool3 FID / probability FID / IS with a fixed, explicit protocol."""
import argparse
import json
from pathlib import Path
import torch
from torchvision.transforms.functional import resize
from ignite.metrics import FID, InceptionScore
from ignite.metrics.gan.utils import InceptionModel
from runtime import make_models, load_state, atomic_save
from data import Paintings


@torch.no_grad()
def evaluate(G, manifest, split, count, cache_dir, seed=32026):
    count = min(count, len(manifest[split]))
    if count < 2:
        raise ValueError('Evaluation needs at least two images')
    device = next(G.parameters()).device
    net = InceptionModel(return_features=False, device=device)
    head = net.model.fc
    net.model.fc = torch.nn.Identity()
    def features(x):
        x = resize(((x.float() + 1) / 2).clamp(0, 1), [299, 299], antialias=True)
        f = net(x)
        return f.cpu(), head(f).cpu()
    cache_dir = Path(cache_dir)
    cache = cache_dir / f"real-{manifest['fingerprint'][:16]}-{split}-{count}-ignite054.pt"
    if cache.exists():
        real = load_state(cache)
    else:
        ds = Paintings(manifest, manifest[split][:count], G.img_resolution)
        loader = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=0)
        fs, ps = [], []
        for x in loader:
            f, p = features(x.to(device).float() / 127.5 - 1)
            fs.append(f); ps.append(p)
        real = {'features': torch.cat(fs), 'probabilities': torch.cat(ps)}
        atomic_save(real, cache)
    fs, ps = [], []
    # Never truncate the distribution for reported IS/FID.
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
        torch.manual_seed(seed)
        G.eval()
        for i in range(count):
            x = G(torch.randn(1, G.z_dim, device=device), torch.zeros(1, G.c_dim, device=device),
                  truncation_psi=1, noise_mode='random')
            f, p = features(x)
            fs.append(f); ps.append(p)
            if (i+1) % 100 == 0 or i+1 == count:
                print(f'{split}: evaluated {i+1}/{count}', flush=True)
    del net, head
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    fake_f, fake_p = torch.cat(fs), torch.cat(ps)
    fid = FID(num_features=2048, feature_extractor=torch.nn.Identity(), device='cpu')
    blog = FID(num_features=1000, feature_extractor=torch.nn.Identity(), device='cpu')
    score = InceptionScore(num_features=1000, feature_extractor=torch.nn.Identity(), device='cpu')
    fid.update((fake_f, real['features']))
    blog.update((fake_p, real['probabilities']))
    score.update(fake_p)
    result = {'fid_pool3': float(fid.compute()), 'fid_ignite_blog_default': float(blog.compute()),
              'inception_score': float(score.compute()), 'real_images': count, 'generated_images': count,
              'split': split, 'seed': seed, 'image_size': G.img_resolution, 'truncation_psi': 1,
              'metric_library': 'pytorch-ignite 0.5.4', 'features': 'torchvision ImageNet Inception-v3 pool3, 2048',
              'preprocess': '[-1,1] -> [0,1]; torchvision bilinear antialias resize 299x299; Ignite InceptionModel transform_input',
              'dataset_fingerprint': manifest['fingerprint'],
              'limitations': 'Finite-sample estimate. Public WikiArt pretraining may overlap all splits. Not evidence of unseen-pretraining generalization; do not compare with different sample counts/protocols.'}
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--split', choices=['validation', 'test'], default='test')
    p.add_argument('--count', type=int, default=1000)
    a = p.parse_args()
    torch.set_num_threads(4)
    s = load_state(a.checkpoint)
    m = json.loads(a.manifest.read_text(encoding='utf-8'))
    if 'dataset_fingerprint' in s and s['dataset_fingerprint'] != m['fingerprint']:
        raise ValueError('Checkpoint/dataset mismatch')
    G = make_models(s, generator_only=True)
    result = evaluate(G, m, a.split, a.count, a.cache)
    result.update(nimg=s['nimg'], pretrained=s.get('pretrained'), checkpoint=str(a.checkpoint),
                  all_training_images_seen=s.get('all_training_images_seen',False))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
