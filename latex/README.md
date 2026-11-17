# Homework 2 report (NeurIPS 2026 LaTeX)

The report is authored by Xinying Liu. The first two pages contain the paper body; references begin on page 3. The project uses the official `neurips_2026.sty` in `preprint` mode so the author name appears without claiming conference acceptance. `main.tex` replaces the template's first-page "Preprint." notice with the full, clickable GitHub URL.

## Files

- `main.tex`: editable report text, metric table, captions, references, and the repository URL in the first-page footer.
- `neurips_2026.sty`: official NeurIPS 2026 style file.
- `loss_curves.png`: chart from the recorded scratch and transfer training losses.
- `generated_examples.pdf`: vector layout of 18 fixed, unmodified generated sample tiles (six per model), made by [`../tools/make_generated_examples.py`](../tools/make_generated_examples.py).

## Overleaf

For Overleaf, ZIP the five files in this directory, then choose **New Project → Upload Project**. Set `main.tex` as the main file if needed and compile with XeLaTeX. The repository's [submitted PDF](../report/homework2_wikiart_report.pdf) is already compiled.

The numerical results are transcribed from the saved DataHub experiment outputs in the [GitHub repository](https://github.com/ashley-tech628/homework2-wikiart-gan). The public StyleGAN2 starting checkpoint is explicitly identified in the text and references. The pretrained model may have seen some evaluation paintings before this project; the report does not claim a disjoint pretraining test.
