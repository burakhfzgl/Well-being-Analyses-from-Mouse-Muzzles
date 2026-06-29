from pathlib import Path

from PIL import Image, ImageOps, ImageDraw, ImageFilter
import random
import math

from paths import FIGURES_DIR, IMAGES_PERFECT_DIR


def make_image_brick_wall(
    image_dir,
    output_path=None,
    n_images=30,
    tile_size=100,
    gap=18,
    cols=8,
    background=(245, 245, 245),
    seed=45
):
    """Create a reproducible image wall from randomly sampled images."""

    if output_path is None:
        output_path = FIGURES_DIR / "image_brick_wall.png"

    random.seed(seed)

    image_dir = Path(image_dir)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = [
        p for p in image_dir.iterdir()
        if p.suffix.lower() in exts
    ]

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {image_dir}")

    selected = random.sample(
        image_paths,
        min(n_images, len(image_paths))
    )

    rows = math.ceil(len(selected) / cols)

    canvas_w = cols * tile_size + (cols + 1) * gap
    canvas_h = rows * tile_size + (rows + 1) * gap

    canvas = Image.new("RGB", (canvas_w, canvas_h), background)

    for idx, img_path in enumerate(selected):
        row = idx // cols
        col = idx % cols

        x = gap + col * (tile_size + gap)
        y = gap + row * (tile_size + gap)

        # Slight offset gives the wall a less rigid look.
        x += random.randint(-5, 5)
        y += random.randint(-5, 5)

        img = Image.open(img_path).convert("RGB")

        img = ImageOps.fit(
            img,
            (tile_size, tile_size),
            method=Image.Resampling.LANCZOS
        )

        shadow_offset = 8
        shadow = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 90))
        shadow = shadow.filter(ImageFilter.GaussianBlur(8))

        shadow_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        shadow_layer.paste(shadow, (x + shadow_offset, y + shadow_offset))

        canvas = Image.alpha_composite(
            canvas.convert("RGBA"),
            shadow_layer
        ).convert("RGB")

        bordered = ImageOps.expand(img, border=5, fill="white")

        canvas.paste(bordered, (x, y))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    make_image_brick_wall(
        image_dir=IMAGES_PERFECT_DIR,
        output_path=FIGURES_DIR / "image_perfect_brick_wall.png",
        n_images=70,
        tile_size=120,
        gap=16,
        cols=10,
    )