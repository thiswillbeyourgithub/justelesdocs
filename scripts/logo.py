# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "pillow"]
# ///
"""Draw `src/logo.svg`: an open scroll bearing a psi, its trailing roll dissolving into pixels.

The mark is the sibling of justelesRCP's `src/logo.svg`, a capsule whose leading
half breaks up into squares. That file was drawn once and its pixels are literal
numbers; here the pixels are DERIVED, so a change to the drawing is one edit and a
re-run rather than a hundred hand-placed rectangles:

1. the mark is a plain solid drawing on a 256 x 256 canvas;
2. `rsvg-convert` rasterises it, one pixel per unit;
3. left of the cut the drawing is kept as vectors (a clip path);
4. right of it the canvas is sampled on a grid, and each cell that lands on ink
   becomes a jittered square of that cell's colour, kept with a probability that
   falls with distance from the cut, plus a few strays beyond the drawing.

The random stream is seeded, so a re-run is byte-identical.
"""

from __future__ import annotations

import random
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import click
from PIL import Image

PSI_DARK = "#4B2A9E"
VIOLET = "#8A5CF0"
VIOLET_DARK = "#6A3FD6"
PAPER = "#F1E4C6"
PAPER_EDGE = "#CDB88C"


def psi(cx: float, cy: float, s: float, colour: str) -> str:
    """A Greek capital psi drawn with strokes, centred on (cx, cy), `s` = 1 for a ~84-unit glyph.

    Sans-serif on purpose: parallel upright arms, a centre stem and a foot bar
    read as a candelabrum. So the arms are one U-shaped cup, the stem runs
    well below it, and nothing sits under the stem.
    """
    w = 12 * s
    return f"""
    <g fill="none" stroke="{colour}" stroke-width="{w:.1f}" stroke-linecap="round">
      <path d="M {cx - 30 * s:.1f} {cy - 36 * s:.1f} C {cx - 30 * s:.1f} {cy + 4 * s:.1f} {cx - 18 * s:.1f} {cy + 12 * s:.1f} {cx:.1f} {cy + 12 * s:.1f} C {cx + 18 * s:.1f} {cy + 12 * s:.1f} {cx + 30 * s:.1f} {cy + 4 * s:.1f} {cx + 30 * s:.1f} {cy - 36 * s:.1f}"/>
      <line x1="{cx:.1f}" y1="{cy - 42 * s:.1f}" x2="{cx:.1f}" y2="{cy + 42 * s:.1f}"/>
    </g>"""


def roll(x: float, y: float, w: float, h: float, vertical: bool) -> str:
    """One rolled end of a scroll: a violet cylinder with a darker cap at each end."""
    r = (w if vertical else h) / 2
    body = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{VIOLET}"/>'
    if vertical:
        caps = (f'<ellipse cx="{x + r}" cy="{y + 4}" rx="{r}" ry="4" fill="{VIOLET_DARK}"/>'
                f'<ellipse cx="{x + r}" cy="{y + h - 4}" rx="{r}" ry="4" fill="{VIOLET_DARK}"/>')
    else:
        caps = (f'<ellipse cx="{x + 4}" cy="{y + r}" rx="4" ry="{r}" fill="{VIOLET_DARK}"/>'
                f'<ellipse cx="{x + w - 4}" cy="{y + r}" rx="4" ry="{r}" fill="{VIOLET_DARK}"/>')
    return body + caps


@dataclass(frozen=True)
class Mark:
    idea: str
    drawing: str      # solid, unrotated, 256 x 256
    cut: float        # x where vectors stop and pixels start
    rotate: float     # degrees, applied to the finished mark
    seed: int
    cell: float = 12.0


LOGO = Mark(
    idea="A landscape scroll held open between two rolls, upright; the right roll trails off as pixels.",
    drawing=(
        f'<rect x="50" y="68" width="156" height="120" fill="{PAPER}" stroke="{PAPER_EDGE}" stroke-width="2"/>'
        + roll(30, 56, 26, 144, vertical=True)
        + roll(200, 56, 26, 144, vertical=True)
        + psi(128, 128, 0.9, PSI_DARK)
    ),
    cut=166, rotate=0, seed=2,
)


def rasterise(svg: str) -> Image.Image:
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "in.svg", Path(tmp) / "out.png"
        src.write_text(svg)
        subprocess.run(["rsvg-convert", "-w", "256", "-h", "256", "-o", str(out), str(src)], check=True)
        return Image.open(out).convert("RGBA")


def pixels(c: Mark, img: Image.Image) -> list[str]:
    """Grid cells right of the cut that land on ink, thinned out with distance."""
    rng = random.Random(c.seed)
    rects, span = [], 256 - c.cut
    y0 = c.cell / 2
    x = c.cut + 1.5
    while x < 256:
        y = y0
        while y < 256:
            r, g, b, a = img.getpixel((min(int(x + c.cell / 2), 255), min(int(y), 255)))
            frac = (x - c.cut) / span
            if a > 200 and rng.random() < 1.25 - 1.5 * frac:
                size = c.cell * rng.uniform(0.72, 1.02) * (1 - 0.35 * frac)
                px, py = x + rng.uniform(-1.2, 1.2) + 2 * frac * c.cell, y - size / 2 + rng.uniform(-1.5, 1.5)
                rects.append(rect(px, py, size, f"#{r:02X}{g:02X}{b:02X}", rng.uniform(-6, 6) * (1 + 2 * frac)))
            y += c.cell + 1.5
        x += c.cell + 1.5
    # Strays: a few small squares flung past the drawing, in colours it already uses.
    colours = sorted({r[r.index('fill="') + 6:r.index('fill="') + 13] for r in rects})
    for _ in range(3):
        size = rng.uniform(4.5, 6.5)
        rects.append(rect(rng.uniform(c.cut + 0.55 * span, 240), rng.uniform(40, 216), size,
                          rng.choice(colours), rng.uniform(-10, 10)))
    return rects


def rect(x: float, y: float, s: float, fill: str, angle: float) -> str:
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{s:.1f}" height="{s:.1f}" rx="1.6" fill="{fill}" '
            f'transform="rotate({angle:.1f} {x + s / 2:.1f} {y + s / 2:.1f})"/>')


def build(c: Mark, name: str = "justelesdocs") -> str:
    header = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"'
    img = rasterise(f"{header} width=\"256\" height=\"256\">{c.drawing}</svg>")
    body = "\n      ".join(pixels(c, img))
    return f"""{header} role="img" aria-label="{name}">
  <title>{name}</title>
  <defs><clipPath id="solid"><rect x="0" y="0" width="{c.cut}" height="256"/></clipPath></defs>
  <g transform="rotate({c.rotate} 128 128)">
    <g clip-path="url(#solid)">{c.drawing}
    </g>
    <!-- Trailing half dissolving into pixels. -->
    <g>
      {body}
    </g>
  </g>
</svg>
"""


@click.command()
@click.option("--out", type=click.Path(path_type=Path), default=Path("src/logo.svg"), show_default=True)
@click.option("--png", type=click.Path(path_type=Path), default=None,
              help="Also render a 256 px preview here, to look at without a browser.")
@click.option("--name", default="justelesdocs", show_default=True,
              help="The site name the SVG's title and aria-label carry (corpus.toml [site] name).")
def main(out: Path, png: Path | None, name: str) -> None:
    out.write_text(build(LOGO, name))
    if png:
        subprocess.run(["rsvg-convert", "-w", "256", "-o", str(png), str(out)], check=True)
    click.echo(f"{out}: {LOGO.idea}")

if __name__ == "__main__":
    main()
