# Homework 2 report (NeurIPS 2026 LaTeX)

The report is authored by Xinying Liu. The first two pages contain the paper body; references begin on page 3. The project uses the official `neurips_2026.sty` in `preprint` mode so the author name appears without claiming conference acceptance.

## Files

- `main.tex`: editable report, equations, tables, captions, and references.
- `neurips_2026.sty`: official NeurIPS 2026 style file.
- `loss_curves.png`: chart from the recorded scratch and transfer training losses.
- `generated_examples.pdf`: vector layout of unmodified generated sample tiles.

## Overleaf

Create a project with **New Project → Upload Project**, select the accompanying ZIP, set `main.tex` as the main file if needed, and compile with XeLaTeX. Download the resulting PDF.

The numerical results are transcribed from the saved DataHub experiment outputs in the GitHub repository. The public StyleGAN2 starting checkpoint is explicitly identified in the text and references. The pretrained model may have seen some evaluation paintings before this project; the report does not claim a disjoint pretraining test.
