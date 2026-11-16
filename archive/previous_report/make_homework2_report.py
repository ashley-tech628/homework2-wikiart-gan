"""Render the superseded two-page ReportLab draft from run artifacts.

The paper follows the 2026 author's US Letter text block, 10/11 pt body,
title rules, caption order, and numbered references. It is a course report.
"""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle


ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "results"
NEW = ROOT / "results" / "stylegan2_final"
OUT = ROOT / "archive" / "previous_report" / "old_two_page_report.pdf"
CHART = ROOT / "archive" / "previous_report" / "homework2_combined_losses.png"
OUT.parent.mkdir(parents=True, exist_ok=True)
CHART.parent.mkdir(parents=True, exist_ok=True)

W, H = 612, 792  # US Letter, points
LEFT, WIDTH, BOTTOM, TOP = 108, 396, 72, 720  # NeurIPS 5.5 x 9 in text block
TITLE_COLOR = colors.HexColor("#182B3A")
ACCENT = "#246889"
SECOND = "#BA5D3B"

BODY = ParagraphStyle("body", fontName="Times-Roman", fontSize=10,
                      leading=11, alignment=TA_JUSTIFY)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=9, leading=10.2)
CAPTION = ParagraphStyle("caption", parent=BODY, fontSize=8.7,
                         leading=9.8, alignment=TA_LEFT)
ABSTRACT = ParagraphStyle("abstract", parent=BODY, fontSize=10, leading=11)
REF = ParagraphStyle("ref", parent=BODY, fontSize=8.5, leading=9.4,
                     alignment=TA_LEFT)


def paragraph(pdf, text, y, *, style=BODY, gap=5.5, x=LEFT, width=WIDTH):
    flow = Paragraph(text, style)
    _, height = flow.wrap(width, 1000)
    flow.drawOn(pdf, x, y - height)
    return y - height - gap


def heading(pdf, number, label, y):
    y -= 3
    pdf.setFillColor(TITLE_COLOR)
    pdf.setFont("Times-Bold", 12)
    pdf.drawString(LEFT, y, f"{number}  {label}")
    pdf.setFillColor(colors.black)
    return y - 9


def caption(pdf, label, text, y):
    return paragraph(pdf, f"<b>{label}</b> {text}", y, style=CAPTION, gap=7)


def result_table(pdf, rows, y, widths):
    obj = Table(rows, colWidths=widths)
    obj.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("LEADING", (0, 0), (-1, -1), 9.6),
        ("LINEABOVE", (0, 0), (-1, 0), 0.65, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.45, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.65, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.8),
        ("LEFTPADDING", (0, 0), (-1, -1), 1),
    ]))
    _, height = obj.wrap(WIDTH, 1000)
    obj.drawOn(pdf, LEFT, y - height)
    return y - height - 6


def footer(pdf, number):
    pdf.setFont("Times-Roman", 8.5)
    pdf.drawCentredString(W / 2, 54, str(number))


def tile(pdf, path, src_n, row, col, x, y, size):
    """Clip one source grid tile during page layout without editing the image."""
    pdf.saveState()
    box = pdf.beginPath()
    box.rect(x, y, size, size)
    pdf.clipPath(box, stroke=0)
    pdf.drawImage(str(path), x - col * size, y - (src_n - 1 - row) * size,
                  width=src_n * size, height=src_n * size,
                  preserveAspectRatio=False, mask="auto")
    pdf.restoreState()
    pdf.setLineWidth(.25)
    pdf.setStrokeColor(colors.HexColor("#666666"))
    pdf.rect(x, y, size, size, fill=0, stroke=1)
    pdf.setStrokeColor(colors.black)


def example_panel(pdf, y):
    size = 77
    gutter = 1.5
    x0 = LEFT + (WIDTH - (4 * size + 3 * gutter)) / 2
    specs = [
        ("From scratch, 128px", OLD / "full_datahub_generated_grid.png", 8,
         [(0, 0), (2, 2), (4, 4), (6, 6)]),
        ("Public starting model, 1024px", NEW / "pretrained_before_finetuning_psi1.png", 4,
         [(0, 0), (0, 1), (0, 2), (0, 3)]),
        ("Selected fine-tuned model, 1024px", NEW / "best_grid_psi1.png", 4,
         [(0, 0), (0, 1), (0, 2), (0, 3)]),
    ]
    for label, path, count, positions in specs:
        pdf.setFont("Times-Bold", 8.8)
        pdf.drawString(x0, y, label)
        y -= 5
        bottom = y - size
        for j, (row, col) in enumerate(positions):
            tile(pdf, path, count, row, col, x0 + j * (size + gutter), bottom, size)
        y = bottom - 10
    return y


baseline = json.loads((OLD / "full_datahub_metrics.json").read_text())
bm = json.loads((OLD / "full_datahub_config.json").read_text())
manifest = json.loads((NEW / "dataset_manifest.json").read_text())
cfg = json.loads((NEW / "config.json").read_text())
pre = json.loads((NEW / "pretrained_test_metrics.json").read_text())
fine = json.loads((NEW / "finetuned_test_metrics.json").read_text())
status = json.loads((NEW / "status.json").read_text())
validation = [json.loads(line) for line in (NEW / "validation.jsonl").read_text().splitlines()]
assert manifest["source_images"] == baseline["source_images"] == 81444
assert len(manifest["styles"]) == baseline["styles"] == 27
assert baseline["checkpoint_epoch"] == 40 and status["status"] == "target_reached"
assert status["all_training_images_seen"] and len(manifest["train"]) == 73299
assert pre["dataset_fingerprint"] == fine["dataset_fingerprint"] == manifest["fingerprint"]
assert pre["real_images"] == fine["real_images"] == baseline["real_images"] == 1000
assert pre["generated_images"] == fine["generated_images"] == baseline["generated_images"] == 1000
assert pre["seed"] == fine["seed"] and pre["truncation_psi"] == fine["truncation_psi"] == 1
assert any(v.get("nimg") == fine["nimg"] for v in validation)
assert CHART.exists(), f"Generate the loss figure first: {CHART}"

pdf = canvas.Canvas(str(OUT), pagesize=(W, H), pageCompression=1)
pdf.setTitle("Painting Synthesis on Full WikiArt: From-Scratch and Pretrained GANs")
pdf.setAuthor("Xinying Liu")

# Page 1.
pdf.setStrokeColor(colors.black)
pdf.setLineWidth(4)
pdf.line(LEFT, 731, LEFT + WIDTH, 731)
pdf.setFont("Times-Bold", 17)
pdf.drawCentredString(W / 2, 701, "Painting Synthesis on Full WikiArt:")
pdf.drawCentredString(W / 2, 681, "From-Scratch and Pretrained GANs")
pdf.setFont("Times-Bold", 10)
pdf.drawCentredString(W / 2, 659, "Xinying Liu")
pdf.setFont("Times-Roman", 9)
pdf.drawCentredString(W / 2, 646, "University of California, San Diego  |  ECE 285, Homework 2")
pdf.setLineWidth(1)
pdf.line(LEFT, 634, LEFT + WIDTH, 634)
y = 617
pdf.setFont("Times-Bold", 12)
pdf.drawCentredString(W / 2, y, "Abstract")
y -= 7
y = paragraph(pdf, "Using all 81,444 images in the assigned WikiArt upload, "
    "we implement a style-conditional 128px GAN and fine-tune a publicly "
    "pretrained 1024px StyleGAN2-ADA on the full training split. On 1,000 "
    "test pairs, the selected transfer generator attains IS 7.080 and "
    "pool3 FID 58.12. Its public starting checkpoint yields FID 55.97; "
    "fine-tuning therefore does not improve both metrics. Possible overlap "
    "between pretraining and test images limits interpretation.", y, style=ABSTRACT,
    x=LEFT + 36, width=WIDTH - 72, gap=13)

y = heading(pdf, "1", "Data and evaluation design", y)
y = paragraph(pdf, "The assigned <i>steubk/wikiart</i> archive has 81,444 "
    "paintings in 27 styles. We apply EXIF orientation, RGB conversion, "
    "square center cropping, and Lanczos resizing. Seeded splits yield "
    "73,300/8,144 train/holdout images for the scratch model and "
    "73,299/4,072/4,073 train/validation/test images for transfer. "
    "Every source file is indexed. Transfer validation selects weights; "
    "the test set is used only for final metrics.", y, gap=10)

y = heading(pdf, "2", "GAN models and optimization", y)
y = paragraph(pdf, "<b>From scratch.</b> A 128-dimensional latent vector and "
    "style label drive a convolutional generator; a spectrally normalized "
    "projection discriminator [1] uses a hinge adversarial objective. "
    "Forty epochs (45,800 updates) use batch 64, Adam with "
    "G/D learning rates 1e-4/2e-4 and betas (0, 0.9), and an EMA generator. "
    "This model is our independent GAN implementation.", y, gap=5)
y = paragraph(pdf, "<b>Transfer.</b> We initialize NVIDIA's 1024px "
    "StyleGAN2-ADA generator and discriminator [2] from Peter Baylies's "
    "public WikiArt_Uncond2 checkpoint [3], starting G from EMA weights. "
    "Training uses deterministic traversal, lazy R1 and path-length "
    "penalties, adaptive color/flip augmentation, EMA, and resumable checkpoints. "
    "The model is unconditional. At 80,003 kimg every training image "
    "has been seen; validation selects the checkpoint at "
    f"{fine['nimg']/1000:.3f} kimg.", y, gap=9)

y = paragraph(pdf, "<b>Table 1:</b> Training settings. The two objectives "
    "and their loss magnitudes are not directly comparable.", y,
    style=CAPTION, gap=3)
y = result_table(pdf, [
    ["Setting", "From-scratch GAN", "StyleGAN2 transfer"],
    ["Resolution / batch", "128px / 64", "1024px / 16 (micro 2)"],
    ["Optimization", "Adam; 1e-4 G, 2e-4 D", "Adam; 1e-4 G and D"],
    ["Regularization", "Spectral norm; hinge", "R1=10, PL=2; each 16 steps"],
    ["EMA / seed", "0.999 / 2026", "10-kimg half-life / 2026"],
], y, [105, 130, 161])

chart_h = 119
pdf.drawImage(str(CHART), LEFT, y - chart_h, width=WIDTH, height=chart_h,
              preserveAspectRatio=False, mask="auto")
y -= chart_h + 4
y = caption(pdf, "Figure 1:", "Training losses over the completed runs: "
    "raw updates (faint) and moving averages (solid; 250 and 50 updates, "
    "respectively). Different objectives and axes preclude cross-model "
    "loss comparisons.", y)
if y < BOTTOM:
    raise RuntimeError(f"Page 1 extends below NeurIPS text block: {y:.1f}")
footer(pdf, 1)
pdf.showPage()

# Page 2.
y = TOP - 2
y = heading(pdf, "3", "Quantitative and qualitative results", y)
y = paragraph(pdf, "We follow the PyTorch-Ignite FID/IS pipeline [4, 5]: "
    "1,000 real and 1,000 generated images are antialiased to 299px before "
    "Inception-v3 inference. Pool3 FID uses 2,048 features; the blog-default "
    "FID instead uses class-probability features and has a different scale. "
    "Both StyleGAN2 rows use the same 1,000 test paintings, seed 32026, "
    "preprocessing, and truncation psi=1.", y, gap=6)
y = paragraph(pdf, "<b>Table 2:</b> Complete recorded 1,000-pair metrics. "
    "Lower FID and higher IS are favorable; the 128px baseline uses a "
    "different split, conditioning scheme, and resolution.", y,
    style=CAPTION, gap=3)
y = result_table(pdf, [
    ["Model", "Pool3 FID", "Ignite FID", "IS"],
    ["128px GAN (epoch 40)", f"{baseline['fid_pool3']:.2f}",
     f"{baseline['fid_ignite_blog_default']:.4f}", f"{baseline['inception_score']:.3f}"],
    ["Public StyleGAN2 (0 kimg)", f"{pre['fid_pool3']:.2f}",
     f"{pre['fid_ignite_blog_default']:.4f}", f"{pre['inception_score']:.3f}"],
    [f"Fine-tuned ({fine['nimg']/1000:.2f} kimg)", f"{fine['fid_pool3']:.2f}",
     f"{fine['fid_ignite_blog_default']:.4f}", f"{fine['inception_score']:.3f}"],
], y, [191, 70, 75, 60])
y = paragraph(pdf, "Relative to its public initialization, fine-tuning "
    f"changes test FID by +{fine['fid_pool3']-pre['fid_pool3']:.2f} and IS "
    f"by +{fine['inception_score']-pre['inception_score']:.3f}. "
    "The 512-pair validation FID also rises from "
    f"{validation[0]['fid_pool3']:.2f} at initialization to "
    f"{validation[-1]['fid_pool3']:.2f} at 80 kimg. These finite-sample "
    "estimates do not support an overall metric improvement.", y, gap=7)

y -= 4
y = example_panel(pdf, y)
y = caption(pdf, "Figure 2:", "Fixed generated examples. Top: scratch "
    "GAN (grid positions 0, 18, 36, 54). Middle/bottom: public and "
    "fine-tuned StyleGAN2, with identical seeds 2026-2029 (psi=1). "
    "Tiles are reduced only for display; visual changes are modest.", y)

y = heading(pdf, "4", "Interpretation and limitations", y)
y = paragraph(pdf, "The scratch model captures texture but lacks coherent "
    "subjects. Transfer samples are sharp, yet before/after views change "
    "little. Cross-model FID is confounded by the split, conditioning, and "
    "resolution. Public pretraining may overlap held-out WikiArt, so the "
    "test scores cannot establish unseen-art generalization. ImageNet-based "
    "metrics imperfectly measure art; 1,000-pair estimates are noisy.", y,
    style=SMALL, gap=11)

y -= 3
pdf.setFont("Times-Bold", 11)
pdf.drawString(LEFT, y, "References")
y -= 6
references = [
    "[1] Miyato, T. and Koyama, M. cGANs with Projection Discriminator. "
    "ICLR, 2018.",
    "[2] Karras, T. et al. Training Generative Adversarial Networks with "
    "Limited Data. NeurIPS, 2020.",
    "[3] Baylies, P. WikiArt StyleGAN2 Conditional Model, "
    "WikiArt_Uncond2 weights. Internet Archive, 2020.",
    "[4] Heusel, M. et al. GANs Trained by a Two Time-Scale Update Rule "
    "Converge to a Local Nash Equilibrium. NeurIPS, 2017.",
    "[5] PyTorch-Ignite. GAN Evaluation: FID and IS. "
    "pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/.",
]
for ref in references:
    y = paragraph(pdf, ref, y, style=REF, gap=1.5)
if y < BOTTOM:
    raise RuntimeError(f"Page 2 extends below NeurIPS text block: {y:.1f}")
footer(pdf, 2)
pdf.save()
print(f"{OUT}\npage_1_end=checked page_2_end={y:.1f} pages=2")
