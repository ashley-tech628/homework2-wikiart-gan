# Homework 2 — WikiArt painting GAN

This repository contains a reproducible 64 × 64 painting-generation experiment: data preparation, GAN training, sampling, loss curves, Inception Score (IS), two FID configurations, and model weights. The report is intentionally deferred.

## Recorded run

| Item | Value |
| --- | --- |
| Images | 2,390 WikiArt paintings: 126 from the [assigned Kaggle upload](https://www.kaggle.com/datasets/steubk/wikiart), 2,264 from the [HugGAN WikiArt mirror](https://huggingface.co/datasets/huggan/wikiart) |
| Split | 1,912 train / 478 held out; seeded random split |
| Image resolution | RGB, center crop to 64 × 64, scaled to [-1, 1] |
| Model | Upsampling convolutional generator; spectral-normalized convolutional discriminator |
| Objective | Hinge GAN loss, one generator and one discriminator update per batch |
| Optimization | Adam, G learning rate 2e-4, D learning rate 1e-4, betas (0, 0.9) |
| Training | 40 epochs, batch size 64, latent dimension 100, seed 2026 |
| Hardware | NVIDIA RTX 5070 Laptop GPU (8 GB); PyTorch 2.10.0 + CUDA 13.0 |
| Evaluation | 400 generated images against 400 held-out real images, Inception-v3 at 299 × 299 |
| IS | **1.5704** (higher is better) |
| FID, pool3 features | **152.99** (lower is better) |
| FID, Ignite blog default | **0.0820** (lower is better; 1,000-class probability features, a different scale from pool3 FID) |

The Kaggle per-file endpoint returned persistent 404 responses after 126 downloads. We used a public WikiArt mirror to complete this run and recorded image-level provenance in `results/provenance.json`. The mirror is not asserted to be byte-identical to the assigned Kaggle version. This limits direct comparison to experiments using that exact upload.

## Repository contents

- `gan.py`: generator, discriminator, and initialization.
- `download_wikiart.py`: seeded, style-balanced download from the assigned Kaggle dataset.
- `prepare_mirror.py`: extract a public WikiArt mirror into the training directory, with provenance.
- `train.py`: training loop, fixed-noise sample grids, losses, checkpoint.
- `evaluate.py`: IS and FID using [PyTorch-Ignite's GAN evaluation approach](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/).
- `sample.py`: generate new paintings from `model/generator.pt`.
- `plot_losses.py`: make the training-loss figure.
- `results/`: recorded metrics, config, losses, sample grid, plot, provenance.

## Run locally or on DataHub

Python 3.13 is recommended for `pytorch-ignite==0.5.4`. For an NVIDIA GPU, install the CUDA build of PyTorch appropriate for the host using the [official selector](https://pytorch.org/get-started/locally/), then install the remaining requirements. The recorded Windows run used a Python 3.14 environment with `pip --ignore-requires-python` for Ignite; its functions were validated in that environment.

```bash
python -m venv .venv
source .venv/bin/activate                    # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

To use the assigned Kaggle source directly:

```bash
python download_wikiart.py --output data/wikiart --per-style 500 \
  --styles Abstract_Expressionism Baroque Impressionism
```

If the Kaggle endpoint is unavailable, download the first two shards of the [HugGAN mirror](https://huggingface.co/datasets/huggan/wikiart/tree/main/data), install `pyarrow`, and prepare images:

```bash
python -m pip install pyarrow
mkdir -p data/shards
curl -L --fail -o data/shards/train-00000.parquet https://huggingface.co/datasets/huggan/wikiart/resolve/main/data/train-00000-of-00072.parquet
curl -L --fail -o data/shards/train-00001.parquet https://huggingface.co/datasets/huggan/wikiart/resolve/main/data/train-00001-of-00072.parquet
python prepare_mirror.py --shards data/shards/train-00000.parquet data/shards/train-00001.parquet --output data/images
```

To reproduce the recorded **mixed-source** run, add `--kaggle-dir data/wikiart` to `prepare_mirror.py` after downloading the Kaggle files. The exact Kaggle filenames used in this run appear in `results/provenance.json`; its mirrored portion will be the same if the two pinned shard paths remain unchanged.

```bash
python train.py --data data/images --output run --epochs 40 --batch-size 64 --workers 4
python evaluate.py --checkpoint run/checkpoint.pt --data data/images --output run/metrics.json --max-images 400
python plot_losses.py --losses run/losses.csv --output run/loss_curves.png
python sample.py --model model/generator.pt --output run/new_paintings.png --seed 1234
```

`TORCH_HOME` may be set to a writable directory before evaluation to control where the pretrained Inception-v3 weights are cached. Training images and optimizer checkpoints are excluded from Git; the 9.5 MB generator file is included for immediate sampling.

## Interpretation

The generated samples show painterly colors and scene-like shapes, but objects are indistinct and the results lean toward Impressionist textures. IS measures ImageNet class confidence/diversity, which is imperfect for paintings. Pool3 FID and the blog-default Ignite FID use different feature representations and must not be compared numerically. The 400-image evaluation is a small-sample estimate, not a benchmark on the full WikiArt dataset.

## References

1. [Radford, Metz, and Chintala, DCGAN](https://arxiv.org/abs/1511.06434).
2. [Miyato et al., Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957).
3. [Heusel et al., FID and two time-scale GAN training](https://proceedings.neurips.cc/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html).
4. [PyTorch-Ignite, GAN evaluation with FID and IS](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/).
