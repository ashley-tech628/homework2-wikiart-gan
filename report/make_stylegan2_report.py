"""Two-page Homework 2 report for the verified full-WikiArt experiments."""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "stylegan2_final"
OUT = ROOT / "homework2_wikiart_full_report.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)
W, H = 612, 792
X, COL = 83, 446

BODY = ParagraphStyle("body", fontName="Times-Roman", fontSize=9.4,
                      leading=11.2, alignment=TA_JUSTIFY)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.6, leading=10.1)
CAP = ParagraphStyle("caption", parent=SMALL, alignment=TA_LEFT)
REF = ParagraphStyle("reference", parent=SMALL, fontSize=7.7, leading=8.8)


def p(pdf, text, y, style=BODY, gap=6):
    obj = Paragraph(text, style)
    _, height = obj.wrap(COL, 1000)
    obj.drawOn(pdf, X, y - height)
    return y - height - gap


def heading(pdf, n, text, y):
    pdf.setFont("Times-Bold", 10.8)
    pdf.drawString(X, y, f"{n}  {text}")
    return y - 8


def table(pdf, rows, y, widths, size=8.1):
    obj = Table(rows, colWidths=widths)
    obj.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), size),
        ("LEADING", (0, 0), (-1, -1), size + 1.2),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 1),
    ]))
    _, height = obj.wrap(COL, 1000)
    obj.drawOn(pdf, X, y - height)
    return y - height - 6


def footer(pdf, page):
    pdf.setLineWidth(0.4)
    pdf.line(X, 49, X + COL, 49)
    pdf.setFont("Times-Roman", 8)
    pdf.drawCentredString(W / 2, 37, str(page))


manifest = json.loads((DATA / "dataset_manifest.json").read_text())
pre = json.loads((DATA / "pretrained_test_metrics.json").read_text())
fine = json.loads((DATA / "finetuned_test_metrics.json").read_text())
status = json.loads((DATA / "status.json").read_text())
cfg = json.loads((DATA / "config.json").read_text())
assert manifest["source_images"] == 81444 and len(manifest["styles"]) == 27
assert status["status"] == "target_reached" and status["all_training_images_seen"]
assert pre["dataset_fingerprint"] == fine["dataset_fingerprint"] == manifest["fingerprint"]
assert pre["real_images"] == fine["real_images"] == 1000
assert pre["generated_images"] == fine["generated_images"] == 1000

pdf = canvas.Canvas(str(OUT), pagesize=(W, H), pageCompression=1)
pdf.setTitle("Full WikiArt Painting Generation: GAN Baseline and StyleGAN2 Transfer")
pdf.setAuthor("Homework 2")

# Page 1: method, reproducibility, and loss curves.
pdf.setFont("Times-Bold", 16.2)
pdf.drawCentredString(W / 2, 750, "Full WikiArt Painting Generation:")
pdf.drawCentredString(W / 2, 731, "GAN Baseline and StyleGAN2 Transfer")
pdf.setFont("Times-Roman", 9)
pdf.drawCentredString(W / 2, 714, "Homework 2")
pdf.setLineWidth(0.65)
pdf.line(X, 704, X + COL, 704)
y = 692
pdf.setFont("Times-Bold", 9.8)
pdf.drawString(X, y, "Abstract")
y -= 6
y = p(pdf, "We implemented a style-conditional GAN from scratch on the full "
      "81,444-painting WikiArt upload, then fine-tuned an attributed public "
      "WikiArt StyleGAN2-ADA checkpoint at native 1024 x 1024 resolution. "
      "The transfer run saw every image in its 73,299-image training split. "
      "On a 1,000-pair test protocol, its selected generator obtains "
      f"IS {fine['inception_score']:.3f} and pool3 FID {fine['fid_pool3']:.2f}. "
      "The public starting checkpoint has slightly lower FID, so the "
      "fine-tuning does not establish a quality improvement.", y, gap=9)

y = heading(pdf, "1", "Dataset and experimental design", y)
y = p(pdf, "We use the assigned <i>steubk/wikiart</i> Kaggle upload: "
      f"{manifest['source_images']:,} indexed paintings in {len(manifest['styles'])} style folders. "
      "Images are EXIF-corrected, converted to RGB, center-cropped, and "
      "resized with Lanczos. The transfer experiment uses a seeded (2026) "
      f"file-level split of {len(manifest['train']):,} train, "
      f"{len(manifest['validation']):,} validation, and "
      f"{len(manifest['test']):,} test paintings. Only the training split "
      "updates weights; validation selects a checkpoint; test is evaluated "
      "after selection. The from-scratch baseline used a separate 90/10 "
      "split of the same source upload.", y, gap=9)

y = heading(pdf, "2", "Models and training", y)
y = p(pdf, "<b>Independent GAN.</b> A 128 x 128 generator maps Gaussian noise "
      "and a style label through convolutional upsampling blocks. A "
      "spectral-normalized projection discriminator uses a hinge objective. "
      "We trained 40 epochs (45,800 steps), batch 64, Adam rates "
      "1e-4 / 2e-4 (G / D), betas (0, 0.9), and EMA. This experiment "
      "demonstrates a separately implemented GAN; its image quality is "
      "limited by indistinct subjects and repeated motifs.", y, gap=6)
y = p(pdf, "<b>1024px transfer.</b> We used NVIDIA's StyleGAN2-ADA "
      "architecture and loss [1] and initialized both networks from "
      "Peter Baylies's public <i>WikiArt_Uncond2</i> checkpoint [2]; "
      "the generator starts from its EMA weights. Our single-GPU loop "
      "adds deterministic full-data traversal, lazy regularization, "
      "adaptive color/flip augmentation, resume-safe optimizer state, "
      "EMA sampling, and validation-based selection. This model is "
      "unconditional; it does not take a style label.", y, gap=6)
y = table(pdf, [
    ["Transfer setting", "Value", "Transfer setting", "Value"],
    ["Target / achieved", f"{cfg['kimg']:.0f} / {status['nimg']/1000:.3f} kimg", "Batch / microbatch", "16 / 2"],
    ["Resolution", "1024 x 1024", "Learning rate", "1e-4 (G, D)"],
    ["R1 gamma / interval", "10 / 16 steps", "Path length weight", "2"],
    ["EMA half-life", "10 kimg", "ADA target", "0.6"],
    ["Validation", "512 every 10 kimg", "Seed", "2026"],
], y, [110, 100, 133, 103], size=7.8)

y = p(pdf, "The 80,003 kimg run completed one full training pass. Figure 1 "
      "plots non-saturating logistic losses and discriminator scores; "
      "rising generator loss alone does not prove collapse or improvement.", y,
      style=SMALL, gap=3)
chart_h = 170
pdf.drawImage(str(DATA / "loss_curves.png"), X, y - chart_h,
              width=COL, height=chart_h, preserveAspectRatio=False, mask="auto")
y -= chart_h + 3
y = p(pdf, "<b>Figure 1:</b> Full-data StyleGAN2 fine-tuning through 80 kimg. "
      "Raw per-step adversarial losses (left) and discriminator scores (right).",
      y, style=CAP, gap=0)
if y < 57:
    raise RuntimeError(f"Page 1 overflow: {y}")
footer(pdf, 1)
pdf.showPage()

# Page 2: exact test protocol, samples, and limitations.
y = 749
y = heading(pdf, "3", "Evaluation", y)
y = p(pdf, "Following the PyTorch-Ignite FID/IS workflow [3], we compare "
      "1,000 generated images with 1,000 real images from the test split "
      "using Inception-v3 features after antialiased resize to 299 x 299. "
      "Pool3 FID uses 2,048 features; Ignite's blog-default probability "
      "FID is a different quantity and cannot be numerically compared "
      "with pool3 FID. Both transfer rows below use the same seed, test "
      "split, sample count, and preprocessing, with truncation psi = 1.", y,
      gap=5)
y = table(pdf, [
    ["Model / checkpoint", "Pool3 FID (down)", "IS (up)", "Default FID"],
    ["Public pretrained, 0 kimg", f"{pre['fid_pool3']:.2f}", f"{pre['inception_score']:.3f}",
     f"{pre['fid_ignite_blog_default']:.4f}"],
    [f"Selected fine-tuned, {fine['nimg']/1000:.2f} kimg", f"{fine['fid_pool3']:.2f}",
     f"{fine['inception_score']:.3f}", f"{fine['fid_ignite_blog_default']:.4f}"],
], y, [179, 98, 68, 101], size=8.0)
y = p(pdf, "The independently implemented 128px GAN yielded IS 2.405 "
      "and pool3 FID 177.32 on 1,000 pairs, but its different image size, "
      "split, and conditioning protocol prevent a controlled numerical "
      "comparison with the transfer model.", y, style=SMALL, gap=14)

y = heading(pdf, "4", "Generated paintings", y)
grid = 340
pdf.drawImage(str(DATA / "best_grid_psi1.png"), (W - grid) / 2,
              y - grid, width=grid, height=grid, preserveAspectRatio=True,
              mask="auto")
y -= grid + 4
y = p(pdf, "<b>Figure 2:</b> Sixteen selected fine-tuned EMA samples "
      "(seeds 2026-2041, psi = 1); each tile is 256px here and "
      "was generated at native 1024px. The grid is uncurated beyond "
      "fixed seeds and checkpoint selection.", y, style=CAP, gap=12)

y = heading(pdf, "5", "Discussion and limitations", y)
y = p(pdf, "The images retain recognizable portraits, landscapes, and "
      "painterly texture, but the fixed-seed grid changes only modestly "
      "from the public pretrained generator. The selected model's IS "
      f"increased by {fine['inception_score']-pre['inception_score']:.3f}, "
      f"while pool3 FID worsened by {fine['fid_pool3']-pre['fid_pool3']:.2f}; "
      "with 1,000 pairs these are finite-sample estimates, not a "
      "statistically established gain. The public pretraining corpus "
      "may overlap all splits, so the test scores do not measure "
      "generalization to artworks unseen during pretraining. "
      "ImageNet features also imperfectly represent painting quality.", y,
      style=SMALL, gap=8)

pdf.setFont("Times-Bold", 9.2)
pdf.drawString(X, y, "References")
y -= 6
refs = [
    "[1] Karras et al. Training Generative Adversarial Networks with "
    "Limited Data. NeurIPS, 2020. github.com/NVlabs/stylegan2-ada-pytorch.",
    "[2] Baylies, P. WikiArt StyleGAN2 Conditional Model, "
    "WikiArt_Uncond2 checkpoint. archive.org/details/wikiart-stylegan2-conditional-model.",
    "[3] PyTorch-Ignite. GAN Evaluation: FID and Inception Score. "
    "pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/.",
    "[4] Heusel et al. GANs Trained by a Two Time-Scale Update Rule "
    "Converge to a Local Nash Equilibrium. NeurIPS, 2017.",
]
for ref in refs:
    y = p(pdf, ref, y, style=REF, gap=1)
if y < 57:
    raise RuntimeError(f"Page 2 overflow: {y}")
footer(pdf, 2)
pdf.save()
print(OUT, "pages=2", f"page2_end={y:.1f}")
