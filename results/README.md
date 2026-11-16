# Recorded full-data experiments

This folder contains the outputs used in the final [report](../report/homework2_wikiart_report.pdf). No training images or large checkpoints are committed here.

| Experiment | Main artifacts |
| --- | --- |
| Independently trained conditional GAN, 128px | [`full_datahub_metrics.json`](full_datahub_metrics.json), [`full_datahub_config.json`](full_datahub_config.json), [`full_datahub_losses.csv`](full_datahub_losses.csv), [generated grid](full_datahub_generated_grid.png), [loss curves](full_datahub_loss_curves.png), and [`full_datahub_full_provenance.json`](full_datahub_full_provenance.json). |
| Publicly initialized StyleGAN2 and fine-tuning, 1024px | [`stylegan2_final/`](stylegan2_final/) contains the split manifest, source and training settings, before/after test metrics, validation history, losses, fixed-seed images, and run status. |

The scratch and transfer experiments use different holdout partitions, image resolutions, and conditioning. Their metric values should not be interpreted as an isolated effect of architecture. The two StyleGAN2 test JSON files use the same 1,000-pair protocol. The Ignite-default probability-feature FID is on a different scale from pool3 FID.

Earlier subset experiments and their result files are preserved under [`../archive/legacy_64px/`](../archive/legacy_64px/) and are not included in the final report's result table.
