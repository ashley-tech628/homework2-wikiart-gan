# High-resolution WikiArt transfer experiment

This directory implements the 1024 × 1024 experiment described in the [report](../report/homework2_wikiart_report.pdf). It uses the NVIDIA StyleGAN2-ADA architecture, initialized from Peter Baylies's public `WikiArt_Uncond2` checkpoint, and then fine-tunes all trainable weights on the full assigned WikiArt upload. The model is **unconditional**; its paintings cannot be requested by style label. The [public checkpoint provenance](UPSTREAM.json) and conversion details are recorded with the code.

The experiment indexes 81,444 paintings across 27 style folders, then assigns 73,299 images to training, 4,072 to validation, and 4,073 to test with a fixed seed. It uses native-resolution images rather than enlarging output from the earlier 128px model. Training reaches 80.003 kimg, enough to traverse every training image once. A validation metric selects the 79.459-kimg EMA checkpoint for the final comparison. The same test subset and generation settings are used for the public starting model and the selected fine-tuned model.

| Test evaluation, 1,000 pairs | Public start | Fine-tuned |
| --- | ---: | ---: |
| Pool3 FID ↓ | 55.97 | 58.12 |
| Inception Score ↑ | 7.019 | 7.080 |

The changes disagree: IS rises slightly while FID worsens. Fixed-seed examples are shown in the [repository overview](../README.md), and the precise JSON metrics, configuration, losses, status, and split manifest are in [`../results/stylegan2_final/`](../results/stylegan2_final/). The original public checkpoint may have seen images in our test split before this project; these scores do not measure transfer to a demonstrably unseen corpus.

The main components are `data.py` (dataset indexing and split manifest), `pretrained.py` (verified public checkpoint conversion), `train.py` (resumable optimization and checkpoint selection), `evaluate.py` (FID and IS), `sample.py` (fixed-seed images), and `package_results.py` (artifact export). The vendored NVIDIA implementation and attribution are kept under `vendor/`.

For the DataHub environment, dependencies, storage layout, resume behavior, and exact commands, see [RUNNING.md](RUNNING.md). That document retains the operational details without making this overview a terminal transcript.
