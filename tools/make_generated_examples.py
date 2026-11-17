"""Assemble fixed generated samples for the two-page LaTeX report.

The source PNG grids are not modified. ReportLab places clipped tiles in a
single vector PDF so the figure remains reproducible from saved run artifacts.
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TRANSFER = RESULTS / "stylegan2_final"
OUT = ROOT / "latex" / "generated_examples.pdf"

WIDTH, HEIGHT = 396, 190
TILE, GAP, IMAGE_LEFT = 52, 1.5, 72


def draw_tile(pdf, path: Path, grid_size: int, row: int, col: int,
              x: float, y: float) -> None:
    pdf.saveState()
    clip = pdf.beginPath()
    clip.rect(x, y, TILE, TILE)
    pdf.clipPath(clip, stroke=0)
    pdf.drawImage(
        str(path), x - col * TILE, y - (grid_size - 1 - row) * TILE,
        width=grid_size * TILE, height=grid_size * TILE,
        preserveAspectRatio=False, mask="auto",
    )
    pdf.restoreState()
    pdf.setLineWidth(0.25)
    pdf.setStrokeColor(colors.HexColor("#555555"))
    pdf.rect(x, y, TILE, TILE, fill=0, stroke=1)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(OUT), pagesize=(WIDTH, HEIGHT), pageCompression=1)
    pdf.setTitle("Eighteen fixed painting samples from three GAN checkpoints")
    rows = [
        (["From scratch", "128px"], RESULTS / "full_datahub_generated_grid.png", 8,
         [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (6, 6)]),
        (["Public pretrained", "1024px"], TRANSFER / "pretrained_before_finetuning_psi1.png", 4,
         [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]),
        (["Fine-tuned", "1024px"], TRANSFER / "best_grid_psi1.png", 4,
         [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]),
    ]

    y = HEIGHT - 8
    for label_lines, source, grid_size, positions in rows:
        if not source.is_file():
            raise FileNotFoundError(source)
        pdf.setFont("Times-Bold", 8)
        for k, label in enumerate(label_lines):
            pdf.drawString(5, y - 20 - 10 * k, label)
        bottom = y - TILE
        for j, (row, col) in enumerate(positions):
            draw_tile(pdf, source, grid_size, row, col,
                      IMAGE_LEFT + j * (TILE + GAP), bottom)
        y = bottom - 8
    if y <= 0 or IMAGE_LEFT + 6 * TILE + 5 * GAP > WIDTH:
        raise RuntimeError("Figure exceeds its page box")
    pdf.save()
    print(OUT)


if __name__ == "__main__":
    main()
