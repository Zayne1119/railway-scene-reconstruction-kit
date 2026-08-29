"""Build privacy-safe GitHub showcase media from the synthetic concept cover.

This script never reads project inputs, customer imagery, point clouds, coordinates,
or production meshes. It only consumes ``docs/media/cover-pointcloud-to-cim.png``.

Optional local dependencies (not required by the reconstruction toolkit itself):

    python -m pip install pillow numpy imageio-ffmpeg
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COVER = ROOT / "docs" / "media" / "cover-pointcloud-to-cim.png"
DEFAULT_GIF = ROOT / "docs" / "media" / "hero-preview.gif"
DEFAULT_STILL = ROOT / "docs" / "media" / "showcase-input-output.png"

BG = (3, 15, 27)
PANEL = (7, 29, 45)
CYAN = (34, 211, 238)
BLUE = (50, 118, 255)
LIME = (183, 255, 74)
WHITE = (239, 247, 252)
MUTED = (150, 174, 190)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = [
        "C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for name in names:
        path = Path(name)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def cover_frame(cover: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_ratio = size[0] / size[1]
    source_ratio = cover.width / cover.height
    if source_ratio > target_ratio:
        width = int(cover.height * target_ratio)
        x = (cover.width - width) // 2
        crop = cover.crop((x, 0, x + width, cover.height))
    else:
        height = int(cover.width / target_ratio)
        y = (cover.height - height) // 2
        crop = cover.crop((0, y, cover.width, y + height))
    return crop.resize(size, Image.Resampling.LANCZOS).convert("RGB")


def ease(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def draw_badge(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    x, y = xy
    label_font = font(18, True)
    bbox = draw.textbbox((0, 0), text, font=label_font)
    width = bbox[2] - bbox[0] + 30
    draw.rounded_rectangle((x, y, x + width, y + 34), radius=17, fill=(12, 42, 59), outline=CYAN)
    draw.text((x + 15, y + 6), text, font=label_font, fill=WHITE)


def draw_title(draw: ImageDraw.ImageDraw, eyebrow: str, title: str, subtitle: str) -> None:
    draw.text((54, 42), eyebrow, font=font(18, True), fill=LIME)
    draw.text((54, 76), title, font=font(44, True), fill=WHITE)
    draw.text((56, 134), subtitle, font=font(21), fill=MUTED)


def phase_for_time(t: float) -> tuple[str, str, str, float]:
    if t < 7.0:
        return (
            "01 / INPUT",
            "POINT OBSERVATIONS",
            "Geometry evidence enters an auditable workspace",
            t / 7.0,
        )
    if t < 14.0:
        return (
            "02 / STRUCTURE",
            "TRACKGRAPH + EVIDENCE",
            "Topology guards reject broken, duplicated or mismatched rails",
            (t - 7.0) / 7.0,
        )
    if t < 23.0:
        return (
            "03 / RECONSTRUCTION",
            "STRUCTURED RAILWAY CIM",
            "Parametric assets retain IDs, provenance and confidence",
            (t - 14.0) / 9.0,
        )
    if t < 30.0:
        return (
            "04 / QUALITY GATE",
            "FIT, TOPOLOGY, MESH QA",
            "Release only after geometry and registry checks pass",
            (t - 23.0) / 7.0,
        )
    return (
        "05 / DELIVERY",
        "WEB / BLENDER / UE",
        "Reusable code in public. Production data stays private.",
        (t - 30.0) / 6.0,
    )


def make_video_frame(base: Image.Image, t: float) -> Image.Image:
    image = base.copy()
    phase, title, subtitle, progress = phase_for_time(t)
    progress = ease(progress)

    # A slow cinematic push-in across the conceptual point-to-model transition.
    zoom = 1.0 + 0.045 * math.sin(min(t / 36.0, 1.0) * math.pi)
    width = int(image.width / zoom)
    height = int(image.height / zoom)
    drift = int((image.width - width) * min(t / 36.0, 1.0))
    image = image.crop((drift, (image.height - height) // 2, drift + width, (image.height + height) // 2))
    image = image.resize(base.size, Image.Resampling.LANCZOS)

    # Dark glass panels keep text readable while preserving the visual transition.
    shade = Image.new("RGBA", image.size, (0, 0, 0, 0))
    shade_draw = ImageDraw.Draw(shade)
    shade_draw.rectangle((0, 0, image.width, 190), fill=(2, 13, 24, 210))
    shade_draw.rectangle((0, image.height - 104, image.width, image.height), fill=(2, 13, 24, 216))
    image = Image.alpha_composite(image.convert("RGBA"), shade)

    # Moving scan line reinforces the conversion metaphor without using real data.
    scan_x = int((0.08 + 0.84 * ((t / 8.0) % 1.0)) * image.width)
    scan = Image.new("RGBA", image.size, (0, 0, 0, 0))
    scan_draw = ImageDraw.Draw(scan)
    scan_draw.rectangle((scan_x - 48, 190, scan_x + 48, image.height - 104), fill=(35, 189, 255, 18))
    scan_draw.line((scan_x, 190, scan_x, image.height - 104), fill=(*CYAN, 190), width=2)
    scan = scan.filter(ImageFilter.GaussianBlur(radius=7))
    image = Image.alpha_composite(image, scan)

    draw = ImageDraw.Draw(image)
    draw_title(draw, phase, title, subtitle)

    if image.width < 900:
        badges = ["ASSET IDs", "TRACKGRAPH", "QA GATES"]
        x = 30
        x_step = 200
    else:
        badges = ["TRACEABLE ASSETS", "TOPOLOGY GUARDS", "QUALITY GATES"]
        x = 54
        x_step = 245
    for badge in badges:
        draw_badge(draw, (x, image.height - 73), badge)
        x += x_step

    # Phase progress indicator.
    if image.width >= 900:
        x0, y0, x1 = image.width - 330, image.height - 61, image.width - 56
        draw.rounded_rectangle((x0, y0, x1, y0 + 8), radius=4, fill=(39, 65, 78))
        draw.rounded_rectangle((x0, y0, x0 + int((x1 - x0) * progress), y0 + 8), radius=4, fill=LIME)
        draw.text((x0, y0 - 29), "EVIDENCE-AWARE PIPELINE", font=font(14, True), fill=MUTED)
    return image.convert("RGB")


def build_gif(base: Image.Image, output: Path, fps: int = 6, seconds: int = 6) -> None:
    frames: list[Image.Image] = []
    gif_base = base.resize((640, 360), Image.Resampling.LANCZOS)
    for index in range(fps * seconds):
        t = index / fps
        # Cover all main phases in the compact loop.
        video_t = (t / seconds) * 36.0
        frame = make_video_frame(gif_base, video_t)
        frames.append(frame.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=True,
        disposal=1,
    )


def build_still(base: Image.Image, output: Path) -> None:
    canvas = Image.new("RGB", (1600, 760), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((60, 38), "FROM POINT OBSERVATIONS TO STRUCTURED CIM", font=font(34, True), fill=WHITE)
    draw.text((60, 86), "Synthetic concept illustration — no production data or site coordinates", font=font(18), fill=MUTED)

    left = base.crop((0, 0, base.width // 2 + 120, base.height)).resize((690, 520), Image.Resampling.LANCZOS)
    right = base.crop((base.width // 2 - 120, 0, base.width, base.height)).resize((690, 520), Image.Resampling.LANCZOS)
    canvas.paste(left, (60, 148))
    canvas.paste(right, (850, 148))
    draw.rounded_rectangle((60, 148, 750, 668), radius=8, outline=CYAN, width=2)
    draw.rounded_rectangle((850, 148, 1540, 668), radius=8, outline=LIME, width=2)
    draw.rectangle((60, 612, 750, 668), fill=(3, 15, 27))
    draw.rectangle((850, 612, 1540, 668), fill=(3, 15, 27))
    draw.text((84, 625), "INPUT  /  POINT OBSERVATIONS + PANORAMA EVIDENCE", font=font(18, True), fill=CYAN)
    draw.text((874, 625), "OUTPUT  /  QUERYABLE ASSETS + QA + DELIVERY", font=font(18, True), fill=LIME)
    draw.line((766, 408, 832, 408), fill=WHITE, width=4)
    draw.polygon([(832, 408), (814, 397), (814, 419)], fill=WHITE)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def build_mp4(base: Image.Image, output: Path, fps: int = 24, seconds: int = 36) -> None:
    try:
        import imageio_ffmpeg
    except ImportError as exc:  # pragma: no cover - optional media tool
        raise SystemExit("MP4 export requires imageio-ffmpeg") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(output),
        base.size,
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        quality=7,
        macro_block_size=2,
        output_params=["-movflags", "+faststart", "-metadata", "comment=synthetic concept illustration"],
    )
    writer.send(None)
    try:
        for index in range(fps * seconds):
            frame = make_video_frame(base, index / fps)
            writer.send(np.asarray(frame, dtype=np.uint8).tobytes())
    finally:
        writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cover", type=Path, default=DEFAULT_COVER)
    parser.add_argument("--gif", type=Path, default=DEFAULT_GIF)
    parser.add_argument("--still", type=Path, default=DEFAULT_STILL)
    parser.add_argument("--mp4", type=Path, help="Optional 36-second MP4 output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cover = Image.open(args.cover).convert("RGB")
    base = cover_frame(cover, (1280, 720))
    build_gif(base, args.gif)
    build_still(base, args.still)
    if args.mp4:
        build_mp4(base, args.mp4)


if __name__ == "__main__":
    main()
