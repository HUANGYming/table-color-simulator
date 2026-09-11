from __future__ import annotations

import argparse
import io
import json
import mimetypes
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter


APP_DIR = Path(__file__).resolve().parent
ORIGINAL_PATH = APP_DIR / "original-hd.jpg"
PPT_MASK_PATH = APP_DIR / "mask.png"
MANUAL_TABLE_PATH = APP_DIR / "manual-table.png"
MANUAL_B_PATH = APP_DIR / "manual-b.png"
BASE_W = 1280
BASE_H = 555
MASK_SCALE = 4

PALETTE = {
    "original": ("Original beige", "#bba286"),
    "green": ("Green", "#4b9b63"),
    "light-green": ("Light green", "#bfe39b"),
    "mint": ("Mint", "#9bdcc5"),
    "pink": ("Pink", "#df6d9c"),
    "light-pink": ("Light pink", "#f7bfd6"),
    "rose": ("Rose", "#d84f72"),
    "sky": ("Sky blue", "#74bbe9"),
    "purple": ("Lavender", "#b998eb"),
    "navy": ("Deep blue", "#163a72"),
    "red": ("Original red", "#8d1c22"),
}

PRESETS = {
    "original": {
        "name": "Original",
        "a": "#bba286",
        "b": "#bba286",
        "c": "#bba286",
        "oa": 0,
        "ob": 0,
        "oc": 0,
    },
}


def sx(x: float, width: int) -> int:
    return round(x * width / BASE_W)


def sy(y: float, height: int) -> int:
    return round(y * height / BASE_H)


def scale_points(points: list[tuple[float, float]], width: int, height: int) -> list[tuple[int, int]]:
    return [(sx(x, width), sy(y, height)) for x, y in points]


def image_pixels(image: Image.Image):
    if hasattr(image, "get_flattened_data"):
        return image.get_flattened_data()
    return image.getdata()


def make_polygon_mask(
    size: tuple[int, int],
    polygons: list[list[tuple[float, float]]],
    feather: float = 0.8,
) -> Image.Image:
    width, height = size
    high_size = (width * MASK_SCALE, height * MASK_SCALE)
    mask = Image.new("L", high_size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        draw.polygon(scale_points(polygon, high_size[0], high_size[1]), fill=255)
    mask = mask.resize(size, Image.Resampling.LANCZOS)
    if feather <= 0:
        return mask
    return mask.filter(ImageFilter.GaussianBlur(max(0.4, width / BASE_W * feather)))


def make_table_mask(size: tuple[int, int]) -> Image.Image:
    return make_polygon_mask(
        size,
        [
            [
                (56, 67),
                (365, 67),
                (365, 36),
                (397, 36),
                (397, 0),
                (883, 0),
                (883, 36),
                (915, 36),
                (915, 67),
                (1223, 67),
                (1223, 133),
                (1185, 210),
                (1140, 304),
                (1070, 384),
                (960, 436),
                (820, 468),
                (640, 477),
                (460, 468),
                (320, 436),
                (210, 384),
                (140, 304),
                (95, 210),
                (56, 133),
            ]
        ],
    )


def make_outer_carpet_mask(size: tuple[int, int]) -> Image.Image:
    return make_polygon_mask(
        size,
        [
            [
                (0, 0),
                (1280, 0),
                (1280, 168),
                (1261, 222),
                (1228, 280),
                (1184, 337),
                (1124, 392),
                (1047, 440),
                (950, 480),
                (822, 520),
                (640, 535),
                (458, 520),
                (330, 480),
                (233, 440),
                (156, 392),
                (96, 337),
                (52, 280),
                (19, 222),
                (0, 168),
            ]
        ],
        feather=0.7,
    )


def keep_border_connected(mask: Image.Image) -> Image.Image:
    data = np.array(mask) > 0
    height, width = data.shape
    visited = np.zeros_like(data, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for x in range(width):
        if data[0, x]:
            visited[0, x] = True
            queue.append((0, x))
        if data[height - 1, x]:
            visited[height - 1, x] = True
            queue.append((height - 1, x))
    for y in range(height):
        if data[y, 0] and not visited[y, 0]:
            visited[y, 0] = True
            queue.append((y, 0))
        if data[y, width - 1] and not visited[y, width - 1]:
            visited[y, width - 1] = True
            queue.append((y, width - 1))

    while queue:
        y, x = queue.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < height and 0 <= nx < width and data[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((ny, nx))

    return Image.fromarray((visited * 255).astype(np.uint8), "L")


def fill_binary_holes(mask: Image.Image) -> Image.Image:
    data = np.array(mask) > 0
    height, width = data.shape
    outside = np.zeros_like(data, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for x in range(width):
        if not data[0, x]:
            outside[0, x] = True
            queue.append((0, x))
        if not data[height - 1, x] and not outside[height - 1, x]:
            outside[height - 1, x] = True
            queue.append((height - 1, x))
    for y in range(height):
        if not data[y, 0] and not outside[y, 0]:
            outside[y, 0] = True
            queue.append((y, 0))
        if not data[y, width - 1] and not outside[y, width - 1]:
            outside[y, width - 1] = True
            queue.append((y, width - 1))

    while queue:
        y, x = queue.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < height and 0 <= nx < width and not data[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                queue.append((ny, nx))

    filled = data | ~outside
    return Image.fromarray((filled * 255).astype(np.uint8), "L")


def keep_largest_components(data: np.ndarray, count: int) -> np.ndarray:
    height, width = data.shape
    visited = np.zeros_like(data, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for start_y in range(height):
        for start_x in range(width):
            if not data[start_y, start_x] or visited[start_y, start_x]:
                continue
            visited[start_y, start_x] = True
            queue: deque[tuple[int, int]] = deque([(start_y, start_x)])
            component: list[tuple[int, int]] = []
            while queue:
                y, x = queue.popleft()
                component.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < height and 0 <= nx < width and data[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            components.append(component)

    kept = np.zeros_like(data, dtype=bool)
    for component in sorted(components, key=len, reverse=True)[:count]:
        for y, x in component:
            kept[y, x] = True
    return kept


def make_color_carpet_mask(base: Image.Image, table_mask: Image.Image) -> Image.Image:
    width, _ = base.size
    hsv = np.array(base.convert("HSV"))
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    red = (((h <= 9) | (h >= 246)) & (s > 62) & (v > 32)) | (
        ((h <= 14) | (h >= 242)) & (s > 105) & (v > 24)
    )
    red_mask = Image.fromarray((red * 255).astype(np.uint8), "L")
    red_mask = red_mask.filter(ImageFilter.MaxFilter(max(3, sx(11, width) | 1)))
    connected = keep_border_connected(red_mask)
    connected = connected.filter(ImageFilter.MaxFilter(max(3, sx(31, width) | 1)))
    connected = fill_binary_holes(connected)
    connected = connected.filter(ImageFilter.MinFilter(max(3, sx(17, width) | 1)))
    carpet = ImageChops.subtract(connected, table_mask)
    return carpet.filter(ImageFilter.GaussianBlur(max(0.5, width / BASE_W * 0.8)))


def make_masks_from_ppt_overlay(mask_path: Path, size: tuple[int, int]) -> dict[str, Image.Image]:
    overlay = Image.open(mask_path).convert("RGB")
    if overlay.size != size:
        overlay = overlay.resize(size, Image.Resampling.LANCZOS)

    rgb = np.array(overlay)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    # PPT mask convention:
    # A = green fill, B = pink/magenta fill, C = blue/cyan fill.
    area_a = (g > 120) & (g > r * 1.18) & (g > b * 1.08)
    area_b = (r > 135) & (b > 85) & (r > g * 1.15)
    area_c = (b > 120) & (b > r * 1.08) & (b > g * 1.04)

    masks = {}
    for key, data in {"a": area_a, "b": area_b, "c": area_c}.items():
        mask = Image.fromarray((data * 255).astype(np.uint8), "L")
        mask = mask.filter(ImageFilter.MaxFilter(3))
        mask = mask.filter(ImageFilter.GaussianBlur(0.7))
        masks[key] = mask
    return masks


def mask_from_condition(data: np.ndarray, feather: float = 0.7, expand: bool = True) -> Image.Image:
    mask = Image.fromarray((data * 255).astype(np.uint8), "L")
    if expand:
        mask = mask.filter(ImageFilter.MaxFilter(3))
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    return mask


def make_masks_from_manual_images(table_path: Path, b_path: Path, size: tuple[int, int]) -> dict[str, Image.Image]:
    table_overlay = Image.open(table_path).convert("RGB")
    b_overlay = Image.open(b_path).convert("RGB")
    if table_overlay.size != size:
        table_overlay = table_overlay.resize(size, Image.Resampling.LANCZOS)
    if b_overlay.size != size:
        b_overlay = b_overlay.resize(size, Image.Resampling.LANCZOS)

    table_hsv = np.array(table_overlay.convert("HSV"))
    table_h = table_hsv[:, :, 0]
    table_s = table_hsv[:, :, 1]
    table_v = table_hsv[:, :, 2]

    orange_table = (table_h >= 8) & (table_h <= 22) & (table_s > 120) & (table_v > 150)
    # The blue C carpet is selected by colour *and* by connectivity to the
    # outer image edge.  This is the same flood-fill idea as the supplied
    # reference script: detached blue graphics inside the table can never
    # leak into the C mask.
    blue_candidate = (table_h >= 132) & (table_h <= 180) & (table_s > 55) & (table_v > 28)
    blue_carpet = np.array(
        keep_border_connected(Image.fromarray((blue_candidate * 255).astype(np.uint8), "L"))
    ) > 0

    b_hsv = np.array(b_overlay.convert("HSV"))
    b_h = b_hsv[:, :, 0]
    b_s = b_hsv[:, :, 1]
    b_v = b_hsv[:, :, 2]
    teal_b = (b_h >= 124) & (b_h <= 150) & (b_s > 95) & (b_v > 82)
    teal_b = keep_largest_components(teal_b, 2)

    # The supplied PPT fills already define clean hard boundaries.  Avoid
    # extra dilation/blur here so A/B/C do not acquire wide soft seams.
    table_mask = mask_from_condition(orange_table, feather=0.12, expand=False)
    # The coloured PPT fills meet at an anti-aliased black contour.  Seal that
    # single-pixel gap on the table side so the original dark contour cannot
    # show through between A and C.
    table_mask = table_mask.filter(ImageFilter.MaxFilter(3))
    area_c = mask_from_condition(blue_carpet, feather=0.12, expand=False)
    raw_area_b = mask_from_condition(teal_b, feather=0.12, expand=False)
    area_b = ImageChops.multiply(raw_area_b, table_mask)
    area_a = ImageChops.subtract(table_mask, area_b)

    return {"a": area_a, "b": area_b, "c": area_c}


def make_area_masks(base: Image.Image) -> dict[str, Image.Image]:
    size = base.size
    width, height = size

    if MANUAL_TABLE_PATH.exists() and MANUAL_B_PATH.exists():
        return make_masks_from_manual_images(MANUAL_TABLE_PATH, MANUAL_B_PATH, size)

    if PPT_MASK_PATH.exists():
        return make_masks_from_ppt_overlay(PPT_MASK_PATH, size)

    table_mask = make_table_mask(size)
    raw_area_b = make_polygon_mask(
        size,
        [
            [(247, 68), (470, 150), (511, 471), (242, 455), (319, 171)],
            [(811, 150), (1033, 68), (961, 171), (1038, 455), (769, 471)],
        ],
        feather=0.6,
    )
    area_b = ImageChops.multiply(raw_area_b, table_mask)
    area_a = ImageChops.subtract(
        table_mask,
        area_b.filter(ImageFilter.GaussianBlur(max(0.4, width / BASE_W * 0.5))),
    )

    area_c = make_color_carpet_mask(base, table_mask)

    return {"a": area_a, "b": area_b, "c": area_c}


def make_c_line_mask(base: Image.Image, area_c: Image.Image) -> Image.Image:
    """Extract the bright geometric linework from the C carpet.

    The carpet's base is deliberately dark while its web, stars, and dashed
    guides are brighter.  Comparing each pixel to a softly blurred local
    background keeps that fine artwork without treating the full carpet as a
    line.
    """
    width, _ = base.size
    rgb = np.asarray(base, dtype=np.float32)
    luminance = rgb[:, :, 0] * 0.2126 + rgb[:, :, 1] * 0.7152 + rgb[:, :, 2] * 0.0722
    luma_image = Image.fromarray(np.clip(luminance, 0, 255).astype(np.uint8), "L")
    local_background = np.asarray(
        luma_image.filter(ImageFilter.GaussianBlur(max(3.0, width / BASE_W * 7.5))),
        dtype=np.float32,
    )
    value = rgb.max(axis=2)
    c_pixels = np.asarray(area_c) > 32

    # Keep only the high-contrast core of every stroke.  This avoids turning
    # the white redraw into a heavy outline on light C colours.
    line_pixels = c_pixels & (luminance > local_background + 28) & (value > 72)
    mask = Image.fromarray((line_pixels * 255).astype(np.uint8), "L")
    return mask.filter(ImageFilter.GaussianBlur(max(0.15, width / BASE_W * 0.2)))


def make_ab_print_mask(base: Image.Image) -> Image.Image:
    """Keep printed betting artwork crisp using its original HSV ink values."""
    width, _ = base.size
    hsv = np.asarray(base.convert("HSV"))
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # The table surface is low-saturation beige.  The typography, outlines,
    # chip labels, and iconography are either saturated ink or dark print.
    # Keeping those two classes separate avoids absorbing beige anti-aliasing
    # into the protection mask.
    # Beige also has moderate saturation in this photo.  Its warm hue range
    # must be excluded before selecting the red, blue, green, and cyan inks.
    beige_surface = (h >= 8) & (h <= 42) & (s >= 20) & (v >= 78)
    saturated_ink = (s > 34) & (v > 42) & ~beige_surface
    dark_ink = (v < 116) & (s > 10)
    ink = saturated_ink | dark_ink

    mask = Image.fromarray((ink * 255).astype(np.uint8), "L")
    # A 3px close reconnects anti-aliased strokes without making them wider:
    # MaxFilter fills hairline breaks, MinFilter returns the outer edge.
    mask = mask.filter(ImageFilter.MaxFilter(max(3, sx(1.1, width) | 1)))
    mask = mask.filter(ImageFilter.MinFilter(max(3, sx(1.1, width) | 1)))
    return mask.filter(ImageFilter.GaussianBlur(max(0.08, width / BASE_W * 0.1)))


def make_protection_masks(base: Image.Image, area_masks: dict[str, Image.Image]) -> dict[str, Image.Image]:
    ab_detail_mask = make_ab_print_mask(base)
    # The original table perimeter is dark too, but it is not betting artwork.
    # Do not paste that contour back over the A/C seam after recolouring.
    c_edge_guard = area_masks["c"].filter(ImageFilter.MaxFilter(7))
    ab_detail_mask = ImageChops.subtract(ab_detail_mask, c_edge_guard)

    return {
        "a": ImageChops.multiply(area_masks["a"], ab_detail_mask),
        "b": ImageChops.multiply(area_masks["b"], ab_detail_mask),
        "c": make_c_line_mask(base, area_masks["c"]),
    }


def clean_hex(value: str, fallback: str) -> str:
    value = (value or "").strip()
    if value.startswith("%23"):
        value = "#" + value[3:]
    if not value.startswith("#"):
        value = "#" + value
    if len(value) == 4:
        value = "#" + "".join(char * 2 for char in value[1:])
    if len(value) != 7:
        return fallback
    try:
        int(value[1:], 16)
    except ValueError:
        return fallback
    return value.lower()


def clean_opacity(value: str, fallback: int) -> float:
    try:
        number = int(value)
    except ValueError:
        number = fallback
    return max(0, min(100, number)) / 100


class ColorRenderer:
    def __init__(self, image_path: Path):
        self.base = Image.open(image_path).convert("RGB")
        self.area_masks = make_area_masks(self.base)
        self.protection_masks = make_protection_masks(self.base, self.area_masks)

    def render(self, params: dict[str, list[str]]) -> Image.Image:
        image = self.base.copy()
        colors = {
            "a": clean_hex(params.get("a", ["#bfe39b"])[0], "#bfe39b"),
            "b": clean_hex(params.get("b", ["#4b9b63"])[0], "#4b9b63"),
            "c": clean_hex(params.get("c", ["#9bdcc5"])[0], "#9bdcc5"),
        }
        opacities = {
            "a": clean_opacity(params.get("oa", ["0"])[0], 0),
            "b": clean_opacity(params.get("ob", ["0"])[0], 0),
            "c": clean_opacity(params.get("oc", ["0"])[0], 0),
        }

        for key in ("c", "a", "b"):
            color_layer = Image.new("RGB", image.size, colors[key])
            blended = Image.blend(image, color_layer, opacities[key])
            image.paste(blended, mask=self.area_masks[key])

        # Keep A/B's printed artwork untouched.  The C carpet artwork gets a
        # consistent soft-white treatment instead: its original red lines
        # look muddy against light C colors, while white remains clean across
        # every palette choice.
        ab_protected = ImageChops.lighter(self.protection_masks["a"], self.protection_masks["b"])
        image.paste(self.base, mask=ab_protected)
        if opacities["c"] > 0:
            c_line_alpha = self.protection_masks["c"].point(lambda value: round(value * 0.58))
            image.paste(Image.new("RGB", image.size, "#f4f7fb"), mask=c_line_alpha)

        if params.get("masks", ["0"])[0] == "1":
            image = self.render_mask_overlay(image)

        return image

    def render_mask_overlay(self, image: Image.Image) -> Image.Image:
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        colors = {"a": (255, 215, 0, 72), "b": (65, 190, 100, 72), "c": (70, 150, 255, 88)}
        for key in ("c", "a", "b"):
            color = colors[key]
            color_img = Image.new("RGBA", image.size, color)
            overlay.paste(color_img, mask=self.area_masks[key])
        return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def html_page() -> bytes:
    palette_json = json.dumps(PALETTE)
    presets_json = json.dumps(PRESETS)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Table Color Simulator</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f4f0e8;
      color: #191815;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: linear-gradient(135deg, #f8f1e5 0%, #edf6f1 48%, #f9edf3 100%);
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 18px;
      min-height: 100vh;
      padding: 18px;
    }}
    .preview {{
      min-width: 0;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    header {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      border-bottom: 1px solid rgb(25 24 21 / 12%);
      padding-bottom: 12px;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(22px, 3vw, 34px);
      line-height: 1.05;
      letter-spacing: 0;
    }}
    .eyebrow {{
      margin: 0 0 5px;
      color: #516f65;
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
    }}
    .status {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
      font-size: 12px;
      font-weight: 700;
    }}
    .pill {{
      border: 1px solid rgb(25 24 21 / 12%);
      border-radius: 999px;
      background: rgb(255 255 255 / 62%);
      padding: 7px 10px;
      white-space: nowrap;
    }}
    .image-wrap {{
      flex: 1;
      display: grid;
      place-items: center;
      min-height: 420px;
      overflow: hidden;
      border: 1px solid rgb(0 0 0 / 14%);
      border-radius: 8px;
      background: #08090b;
      box-shadow: 0 22px 60px rgb(0 0 0 / 14%);
    }}
    #render {{
      display: block;
      width: 100%;
      height: auto;
    }}
    aside {{
      border: 1px solid rgb(25 24 21 / 12%);
      border-radius: 8px;
      background: rgb(255 255 255 / 82%);
      box-shadow: 0 18px 44px rgb(0 0 0 / 8%);
      padding: 16px;
      backdrop-filter: blur(10px);
    }}
    .panel-title {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
    }}
    h2 {{
      margin: 2px 0 0;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .area {{
      margin-top: 14px;
      border: 1px solid rgb(25 24 21 / 12%);
      border-radius: 8px;
      background: #fbfaf6;
      padding: 12px;
    }}
    .area-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }}
    .area-label {{
      display: flex;
      align-items: center;
      gap: 9px;
      min-width: 82px;
      font-weight: 800;
    }}
    .swatch {{
      width: 30px;
      height: 30px;
      border-radius: 6px;
      border: 1px solid rgb(0 0 0 / 18%);
      box-shadow: inset 0 0 0 1px rgb(255 255 255 / 25%);
    }}
    select, input[type="color"] {{
      height: 34px;
      border: 1px solid rgb(25 24 21 / 18%);
      border-radius: 8px;
      background: white;
      color: #191815;
      font: inherit;
    }}
    select {{
      flex: 1;
      min-width: 150px;
      padding: 0 9px;
    }}
    input[type="color"] {{
      width: 46px;
      padding: 3px;
    }}
    .color-code {{
      width: 100%;
      height: 32px;
      margin-top: 9px;
      border: 1px solid rgb(25 24 21 / 18%);
      border-radius: 6px;
      background: white;
      color: #191815;
      font: 700 12px ui-monospace, SFMono-Regular, Menlo, monospace;
      letter-spacing: 0;
      padding: 0 9px;
      text-transform: uppercase;
    }}
    .color-code.invalid {{
      border-color: #bd4b55;
      color: #a83a44;
    }}
    label {{
      display: grid;
      gap: 7px;
      margin-top: 10px;
      color: #5e5a51;
      font-size: 12px;
      font-weight: 700;
    }}
    .range-row {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
    }}
    input[type="range"] {{
      width: 100%;
      accent-color: #2f8068;
    }}
    .opacity-control {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 58px;
      gap: 9px;
      align-items: center;
    }}
    .opacity-number {{
      width: 58px;
      height: 30px;
      border: 1px solid rgb(25 24 21 / 18%);
      border-radius: 6px;
      background: white;
      color: #191815;
      font: 700 12px ui-monospace, SFMono-Regular, Menlo, monospace;
      padding: 0 6px;
    }}
    .buttons {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 14px;
    }}
    button, a.button {{
      display: inline-flex;
      min-height: 34px;
      align-items: center;
      justify-content: center;
      border: 1px solid rgb(25 24 21 / 14%);
      border-radius: 8px;
      background: #191815;
      color: white;
      padding: 8px 10px;
      font: inherit;
      font-size: 13px;
      font-weight: 800;
      text-decoration: none;
      cursor: pointer;
    }}
    button.secondary, a.secondary {{
      background: white;
      color: #191815;
    }}
    .preset-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 8px;
    }}
    .check {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 14px;
    }}
    .check input {{ width: 16px; height: 16px; }}
    .hex-list {{
      display: grid;
      gap: 6px;
      margin-top: 14px;
      border-radius: 8px;
      background: #191b1e;
      color: white;
      padding: 12px;
      font-size: 12px;
    }}
    .hex-line {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }}
    @media (max-width: 980px) {{
      main {{ grid-template-columns: 1fr; }}
      aside {{ order: -1; }}
      header {{ align-items: start; flex-direction: column; }}
      .status {{ justify-content: flex-start; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="preview">
      <header>
        <div>
          <p class="eyebrow">Local PIL color playground</p>
          <h1>Table Color Simulator</h1>
        </div>
        <div class="status">
          <span class="pill">A / B / C only</span>
          <span class="pill">Bet type colors protected</span>
        </div>
      </header>
      <div class="image-wrap">
        <img id="render" src="/render.png" alt="Rendered baccarat layout with editable A, B and C area colors" />
      </div>
    </section>
    <aside>
      <div class="panel-title">
        <div>
          <p class="eyebrow">Controls</p>
          <h2>Color combinations</h2>
        </div>
        <button class="secondary" type="button" id="shuffle">Shuffle</button>
      </div>

      <div id="areas"></div>

      <label class="check">
        <input type="checkbox" id="masks" />
        Show A/B/C mask overlay
      </label>

      <p class="eyebrow" style="margin-top: 16px;">Presets</p>
      <div class="preset-grid" id="presets"></div>

      <div class="buttons">
        <button class="secondary" type="button" id="reset">Reset</button>
        <a class="button" id="download" href="/download.png" download="table-color-simulation.png">Download PNG</a>
      </div>

      <div class="hex-list" id="hexes"></div>
    </aside>
  </main>
  <script>
    const palette = {palette_json};
    const presets = {presets_json};
    const state = {{
      a: '#bba286',
      b: '#bba286',
      c: '#bba286',
      oa: 0,
      ob: 0,
      oc: 0,
      masks: 0
    }};
    const labels = {{ a: 'A areas', b: 'B areas', c: 'C areas' }};
    const areaRoot = document.getElementById('areas');
    const hexRoot = document.getElementById('hexes');
    const img = document.getElementById('render');
    const download = document.getElementById('download');

    function activateColor(key) {{
      if (state['o' + key] > 0) return;
      state['o' + key] = key === 'c' ? 52 : 45;
      const range = document.getElementById(`${{key}}-opacity`);
      const number = document.getElementById(`${{key}}-opacity-number`);
      const label = document.getElementById(`${{key}}-opacity-text`);
      if (range) range.value = state['o' + key];
      if (number) number.value = state['o' + key];
      if (label) label.textContent = `${{state['o' + key]}}%`;
    }}

    function normalizeColorCode(value) {{
      const digits = value.trim().replace(/^#/, '');
      return /^[0-9a-f]{{6}}$/i.test(digits) ? `#${{digits.toLowerCase()}}` : null;
    }}

    function colorOptions(selected) {{
      const known = Object.values(palette).some(([, hex]) => hex.toLowerCase() === selected.toLowerCase());
      const custom = known ? '' : `<option value="${{selected}}" selected>Custom color</option>`;
      return custom + Object.values(palette).map(([name, hex]) =>
        `<option value="${{hex}}" ${{hex.toLowerCase() === selected.toLowerCase() ? 'selected' : ''}}>${{name}}</option>`
      ).join('');
    }}

    function renderControls() {{
      areaRoot.innerHTML = ['a', 'b', 'c'].map((key) => `
        <section class="area">
          <div class="area-head">
            <div class="area-label">
              <span class="swatch" id="${{key}}-swatch" style="background:${{state[key]}}"></span>
              <span>${{labels[key]}}</span>
            </div>
            <select id="${{key}}-select" aria-label="Color for ${{key.toUpperCase()}} area">${{colorOptions(state[key])}}</select>
            <input id="${{key}}-custom" type="color" value="${{state[key]}}" aria-label="Custom color for ${{key.toUpperCase()}} area" />
          </div>
          <input id="${{key}}-code" class="color-code" type="text" value="${{state[key].toUpperCase()}}" spellcheck="false" maxlength="7" aria-label="Color code for ${{key.toUpperCase()}} area" />
          <label>
            <span class="range-row"><span>Overlay strength</span><span id="${{key}}-opacity-text">${{state['o' + key]}}%</span></span>
            <span class="opacity-control">
              <input id="${{key}}-opacity" type="range" min="0" max="70" value="${{state['o' + key]}}" aria-label="Overlay strength for ${{key.toUpperCase()}} area" />
              <input id="${{key}}-opacity-number" class="opacity-number" type="number" min="0" max="70" value="${{state['o' + key]}}" aria-label="Overlay strength percentage for ${{key.toUpperCase()}} area" />
            </span>
          </label>
        </section>
      `).join('');

      for (const key of ['a', 'b', 'c']) {{
        document.getElementById(`${{key}}-select`).addEventListener('change', (event) => {{
          state[key] = event.target.value;
          activateColor(key);
          document.getElementById(`${{key}}-custom`).value = state[key];
          document.getElementById(`${{key}}-code`).value = state[key].toUpperCase();
          update();
        }});
        document.getElementById(`${{key}}-custom`).addEventListener('input', (event) => {{
          state[key] = event.target.value;
          activateColor(key);
          document.getElementById(`${{key}}-code`).value = state[key].toUpperCase();
          update();
        }});
        document.getElementById(`${{key}}-code`).addEventListener('input', (event) => {{
          const code = normalizeColorCode(event.target.value);
          event.target.classList.toggle('invalid', !code && event.target.value.length > 0);
          if (!code) return;
          state[key] = code;
          activateColor(key);
          event.target.value = state[key].toUpperCase();
          document.getElementById(`${{key}}-custom`).value = state[key];
          update();
        }});
        const setOpacity = (value) => {{
          state['o' + key] = Math.max(0, Math.min(70, Number(value) || 0));
          document.getElementById(`${{key}}-opacity`).value = state['o' + key];
          document.getElementById(`${{key}}-opacity-number`).value = state['o' + key];
          document.getElementById(`${{key}}-opacity-text`).textContent = `${{state['o' + key]}}%`;
          update();
        }};
        document.getElementById(`${{key}}-opacity`).addEventListener('input', (event) => {{
          setOpacity(event.target.value);
        }});
        document.getElementById(`${{key}}-opacity-number`).addEventListener('input', (event) => {{
          setOpacity(event.target.value);
        }});
      }}
    }}

    function query() {{
      const params = new URLSearchParams(state);
      params.set('_', String(Date.now()));
      return params.toString();
    }}

    function update() {{
      for (const key of ['a', 'b', 'c']) {{
        const swatch = document.getElementById(`${{key}}-swatch`);
        if (swatch) swatch.style.background = state[key];
      }}
      img.src = `/render.png?${{query()}}`;
      download.href = `/download.png?${{query()}}`;
      hexRoot.innerHTML = ['a', 'b', 'c'].map((key) => `
        <div class="hex-line">
          <strong>${{key.toUpperCase()}}</strong>
          <span>${{state[key].toUpperCase()}} · ${{state['o' + key]}}%</span>
        </div>
      `).join('');
    }}

    document.getElementById('masks').addEventListener('change', (event) => {{
      state.masks = event.target.checked ? 1 : 0;
      update();
    }});
    document.getElementById('shuffle').addEventListener('click', () => {{
      const colors = Object.values(palette).map(([, hex]) => hex);
      state.a = colors[Math.floor(Math.random() * colors.length)];
      state.b = colors[Math.floor(Math.random() * colors.length)];
      state.c = colors[Math.floor(Math.random() * colors.length)];
      state.oa = 45;
      state.ob = 45;
      state.oc = 52;
      renderControls();
      update();
    }});
    document.getElementById('reset').addEventListener('click', () => {{
      Object.assign(state, {{ a: '#bba286', b: '#bba286', c: '#bba286', oa: 0, ob: 0, oc: 0, masks: 0 }});
      document.getElementById('masks').checked = false;
      renderControls();
      update();
    }});

    const presetRoot = document.getElementById('presets');
    presetRoot.innerHTML = Object.entries(presets).map(([key, preset]) =>
      `<button class="secondary" type="button" data-preset="${{key}}">${{preset.name}}</button>`
    ).join('');
    presetRoot.addEventListener('click', (event) => {{
      const button = event.target.closest('[data-preset]');
      if (!button) return;
      Object.assign(state, presets[button.dataset.preset]);
      renderControls();
      update();
    }});

    renderControls();
    update();
  </script>
</body>
</html>""".encode("utf-8")


class AppHandler(BaseHTTPRequestHandler):
    renderer: ColorRenderer

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self.send_head_headers(len(html_page()), "text/html; charset=utf-8")
            return
        if parsed.path in {"/render.png", "/download.png"}:
            self.send_head_headers(0, "image/png")
            return
        if parsed.path == "/original.jpg":
            self.send_head_headers(ORIGINAL_PATH.stat().st_size, "image/jpeg")
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        if parsed.path in {"/", "/index.html"}:
            self.send_bytes(html_page(), "text/html; charset=utf-8")
            return
        if parsed.path in {"/render.png", "/download.png"}:
            image = self.renderer.render(params)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            headers = {}
            if parsed.path == "/download.png":
                headers["Content-Disposition"] = 'attachment; filename="table-color-simulation.png"'
            self.send_bytes(buffer.getvalue(), "image/png", headers=headers)
            return
        if parsed.path == "/original.jpg":
            self.send_file(ORIGINAL_PATH)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args: object) -> None:
        return

    def send_file(self, path: Path) -> None:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_bytes(path.read_bytes(), content_type)

    def send_bytes(self, body: bytes, content_type: str, headers: dict[str, str] | None = None) -> None:
        self.send_head_headers(len(body), content_type, headers)
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_head_headers(
        self, content_length: int, content_type: str, headers: dict[str, str] | None = None
    ) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()


def export_sample(renderer: ColorRenderer, output: Path) -> None:
    params = {
        "a": ["#bfe39b"],
        "b": ["#f7bfd6"],
        "c": ["#74bbe9"],
        "oa": ["72"],
        "ob": ["70"],
        "oc": ["86"],
    }
    image = renderer.render(params)
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local PIL app for recoloring A/B/C baccarat layout areas.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--export-sample", type=Path)
    args = parser.parse_args()

    if not ORIGINAL_PATH.exists():
        raise SystemExit(f"Missing {ORIGINAL_PATH}")

    renderer = ColorRenderer(ORIGINAL_PATH)
    if args.export_sample:
        export_sample(renderer, args.export_sample)
        print(f"Saved {args.export_sample}")
        return

    AppHandler.renderer = renderer
    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"Open http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
