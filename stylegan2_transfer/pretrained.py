"""Download the pinned public WikiArt GAN and convert with NVIDIA's converter."""
import argparse
import copy
import hashlib
import json
import urllib.request
from pathlib import Path
from runtime import plain, atomic_save
import legacy

URL = 'https://archive.org/download/wikiart-stylegan2-conditional-model/WikiArt_Uncond2.pkl'
SHA1 = '48634616ab4b4408b28378b649da67a52c309ea9'
SIZE = 381615715


def digest(path):
    h = hashlib.sha1()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class RestrictedLegacyUnpickler(legacy._LegacyUnpickler):
    """This specific original TF checkpoint contains arrays, never Python code."""
    def find_class(self, module, name):
        if (module, name) == ('dnnlib.tflib.network', 'Network'):
            return legacy._TFNetworkStub
        allowed = {('numpy.core.multiarray', '_reconstruct'), ('numpy', 'ndarray'),
                   ('numpy', 'dtype'), ('numpy.core.multiarray', 'scalar')}
        if (module, name) in allowed:
            return super().find_class(module, name)
        raise ValueError(f'Unexpected pickle global: {module}.{name}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    a = p.parse_args()
    a.cache.mkdir(parents=True, exist_ok=True)
    source = a.cache / 'WikiArt_Uncond2.pkl'
    output = a.cache / 'initial.pt'
    if output.exists():
        from runtime import load_state
        s = load_state(output)
        if s.get('pretrained', {}).get('sha1') != SHA1:
            raise ValueError('Existing initial.pt has a different source')
        print(f'Converted weights already exist: {output}', flush=True)
        return
    if not source.exists() or digest(source) != SHA1:
        temp = source.with_suffix('.download')
        start = temp.stat().st_size if temp.exists() else 0
        req = urllib.request.Request(URL, headers={'Range': f'bytes={start}-'} if start else {})
        with urllib.request.urlopen(req, timeout=120) as response:
            resume = start > 0 and response.status == 206
            received = start if resume else 0
            with temp.open('ab' if resume else 'wb') as f:
                while chunk := response.read(4 * 1024 * 1024):
                    f.write(chunk)
                    received += len(chunk)
                    print(f'Pretrained download: {received / 2**20:.0f}/{SIZE / 2**20:.0f} MiB', flush=True)
        if digest(temp) != SHA1:
            raise RuntimeError('Pretrained checksum mismatch; do not use this file')
        temp.replace(source)
    print('Verified original weight checksum. Converting without TensorFlow...', flush=True)
    with source.open('rb') as f:
        tf_g, tf_d, tf_ema = RestrictedLegacyUnpickler(f).load()
    ema = legacy.convert_tf_generator(tf_ema)
    d = legacy.convert_tf_discriminator(tf_d)
    gc = plain(ema.init_kwargs)
    dc = plain(d.init_kwargs)
    dc.setdefault('block_kwargs', {})['freeze_layers'] = 0
    # Keep the full 1024px graph. Removing upper RGB blocks destroys this model's colors.
    initial = {'format_version': 1, 'model_config': {'G': gc, 'D': dc},
               'G': ema.state_dict(), 'G_ema': ema.state_dict(), 'D': d.state_dict(),
               'pretrained': {'url': URL, 'sha1': SHA1, 'author': 'Peter Baylies / pbaylies',
                              'source': 'WikiArt_Uncond2', 'native_resolution': 1024,
                              'G_initialization': 'public EMA generator',
                              'dataset_overlap': 'Unknown overlap with our train/validation/test split; NOT an unseen-pretraining test.'},
               'nimg': 0, 'step': 0}
    atomic_save(initial, output)
    (a.cache / 'pretrained_provenance.json').write_text(json.dumps(initial['pretrained'], indent=2))
    print(f'Saved {output}', flush=True)


if __name__ == '__main__':
    main()
