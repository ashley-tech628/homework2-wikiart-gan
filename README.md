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

The separate DataHub run completed 40 epochs and evaluated 400 held-out images: pool3 FID **147.3517**, Ignite blog-default FID **0.08746**, and IS **1.53699**. Its metrics were copied from the user's terminal output into `results/datahub_metrics.json`; the remaining DataHub artifacts must be downloaded from DataHub before they can be included in this repository.

## Repository contents

- `gan.py`: generator, discriminator, and initialization.
- `download_wikiart.py`: seeded, style-balanced download from the assigned Kaggle dataset.
- `prepare_mirror.py`: extract a public WikiArt mirror into the training directory, with provenance.
- `train.py`: training loop, fixed-noise sample grids, losses, checkpoint.
- `evaluate.py`: IS and FID using [PyTorch-Ignite's GAN evaluation approach](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/).
- `sample.py`: generate new paintings from `model/generator.pt`.
- `plot_losses.py`: make the training-loss figure.
- `prepare_kaggle_subset.py`: prepare a seeded subset from the attached Kaggle WikiArt upload.
- `kaggle_homework2.ipynb`: self-contained Kaggle notebook with the project code embedded.
- `results/`: recorded metrics, config, losses, sample grid, plot, provenance.

## Run on Kaggle with the assigned dataset

The [Kaggle notebook](kaggle_homework2.ipynb) embeds all required Python scripts, so the repository does not need to be cloned inside Kaggle. Create a new Kaggle Notebook and import `kaggle_homework2.ipynb` through the editor's **File > Import Notebook** command. In the **Input** pane, attach [steubk/wikiart](https://www.kaggle.com/datasets/steubk/wikiart). In **Session options**, select a GPU accelerator and enable Internet. Run the notebook from the first cell, or choose **Save Version > Save & Run All** for a saved version with outputs.

The notebook randomly selects 2,390 decodable images from the mounted Kaggle dataset using seed 2026, writes image-level provenance to `data/images/kaggle_provenance.json`, trains for 40 epochs, evaluates 400 held-out images, and writes the generator, metrics, sample grids, losses, and figures under `/kaggle/working/homework2-wikiart-gan/`. This is a new run on the assigned source; its metrics should be recorded separately from the mixed-source local and DataHub runs. Kaggle saves `/kaggle/working` output files with a completed notebook version. The notebook checks CUDA before starting and installs only Ignite if needed, leaving the preinstalled GPU-enabled PyTorch in place.

Before training, the notebook downloads and caches the Inception-v3 evaluation weights. This gives a clear early error when Kaggle Internet is disabled; it does not affect the GAN training weights.

Kaggle may mount datasets under `/kaggle/input/datasets/<owner>/<dataset>/...`. The preparation script searches recursively for the `Abstract_Expressionism` style directory, so it works with this deeper mount layout.

On DataHub, the 40-epoch run currently lives in `/tmp/homework2-run`. Copy the model weights and final artifacts into the persistent project directory before the container ends, then download an archive from JupyterLab:

```bash
cd ~/homework2-wikiart-gan
source /tmp/homework2-venv/bin/activate
mkdir -p model results
python - <<'PY'
import torch
from pathlib import Path
s = torch.load('/tmp/homework2-run/checkpoint.pt', map_location='cpu', weights_only=False)
torch.save({'generator': s['generator'], 'latent_dim': s['config']['latent_dim'],
            'width': s['config']['width'], 'epoch': s['epoch']},
           Path('model/datahub_generator.pt'))
PY
cp /tmp/homework2-run/metrics.json results/datahub_metrics.json
cp /tmp/homework2-run/config.json results/datahub_config.json
cp /tmp/homework2-run/losses.csv results/datahub_losses.csv
cp /tmp/homework2-run/loss_curves.png results/datahub_loss_curves.png
cp /tmp/homework2-run/samples_epoch_40.png results/datahub_generated_grid.png
python -m zipfile -c datahub-results.zip results/datahub_metrics.json results/datahub_config.json results/datahub_losses.csv results/datahub_loss_curves.png results/datahub_generated_grid.png model/datahub_generator.pt
```

## Run locally

Python 3.13 is recommended for `pytorch-ignite==0.5.4`. For an NVIDIA GPU, install the CUDA build of PyTorch appropriate for the host using the [official selector](https://pytorch.org/get-started/locally/), then install the remaining requirements. The recorded Windows run used a Python 3.14 environment with `pip --ignore-requires-python` for Ignite; its functions were validated in that environment.

```bash
python -m venv .venv
source .venv/bin/activate                    # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

### UCSD DataHub

UCSD's `scipy-ml-notebook` GPU image already provides CUDA and PyTorch. **Do not install the full `requirements.txt` into DataHub's home directory:** that can download another multi-gigabyte CUDA/PyTorch stack and exhaust the course storage quota. After unpacking the supplied DataHub ZIP, first check the existing environment:

```bash
python -c "import torch, torchvision, scipy, matplotlib, PIL; print(torch.__version__, torch.cuda.is_available())"
```

`torch.cuda.is_available()` only confirms that a GPU is visible. Before a full run, verify that PyTorch actually has kernels for the allocated card:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list()); print(torch.ones(1, device='cuda'))"
```

The Fall 2026 UCSD image observed on an RTX PRO 6000 Blackwell (`sm_120`) had PyTorch 2.2.1 + CUDA 12.1 and failed this kernel check. That image can run on an older compatible GPU, but [Blackwell requires a newer PyTorch CUDA build](https://discuss.pytorch.org/t/rtx-5050-sm-120-support-timeline-when-in-stable-pytorch/224288/4) (at least PyTorch 2.7 with CUDA 12.8). A compatible build should be installed in a temporary environment or supplied by an updated image; the older system installation should not be replaced in the quota-limited home directory.

If the CUDA operation succeeds, install only Ignite if it is missing:

```bash
python -c "import ignite; print(ignite.__version__)" || python -m pip install --user --no-cache-dir --no-deps pytorch-ignite==0.5.4
```

The supplied DataHub ZIP includes the 2,390 prepared images at `data/images`, so no data download or `pyarrow` install is needed for that run. If an import fails, inspect the installed package versions and paths before replacing the platform's GPU-enabled PyTorch.

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
