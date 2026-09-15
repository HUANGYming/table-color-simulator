from __future__ import annotations

import argparse
import json
import random
import tempfile
from collections import deque
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter


APP_DIR = Path(__file__).resolve().parent
ORIGINAL_PATH = APP_DIR / "original-hd.jpg"
PPT_MASK_PATH = APP_DIR / "mask.png"
MANUAL_TABLE_PATH = APP_DIR / "manual-table.png"
MANUAL_B_PATH = APP_DIR / "manual-b.png"
SAVED_PRESETS_PATH = APP_DIR / "saved-presets.json"
BASE_W = 1280
BASE_H = 555
MASK_SCALE = 4
AREA_KEYS = ("a", "b", "c")

PALETTE = {
    "original": ("Original area color", "#a98f78"),
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

ORIGINAL_AREA_COLORS = {
    "a": "#a98f78",
    "b": "#bb9d83",
    "c": "#192551",
}

SEGMENTED_AREA_LABELS = {"a": "A areas", "b": "B areas", "c": "C areas"}
BACKGROUND_AREA_LABELS = {"a": "Background"}
GRADIO_CSS = """
#preview img { object-fit: contain; }
.gradio-container { max-width: none !important; }
"""

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


def empty_mask(size: tuple[int, int]) -> Image.Image:
    return Image.new("L", size, 0)


def hue_distance(hue: np.ndarray, target: int) -> np.ndarray:
    diff = np.abs(hue.astype(np.int16) - int(target))
    return np.minimum(diff, 255 - diff)


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


def estimate_dominant_surface(base: Image.Image) -> tuple[str, int, int, int, float]:
    rgb = np.asarray(base.convert("RGB"))
    hsv = np.asarray(base.convert("HSV"))
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    non_white = ~((rgb[:, :, 0] > 245) & (rgb[:, :, 1] > 245) & (rgb[:, :, 2] > 245))
    candidate = (s > 35) & (v > 80) & non_white
    if not np.any(candidate):
        return ORIGINAL_AREA_COLORS["a"], 0, 0, 0, 0.0

    qh = ((h[candidate] // 4) * 4).astype(np.uint8)
    qs = ((s[candidate] // 16) * 16).astype(np.uint8)
    qv = ((v[candidate] // 16) * 16).astype(np.uint8)
    bins: dict[tuple[int, int, int], int] = {}
    for key in zip(qh.tolist(), qs.tolist(), qv.tolist()):
        bins[key] = bins.get(key, 0) + 1

    dominant_bin = max(bins, key=bins.get)
    h0, s0, v0 = (int(value) for value in dominant_bin)
    dominant_pixels = candidate & (hue_distance(h, h0) <= 4) & (np.abs(s.astype(np.int16) - s0) < 16)
    if not np.any(dominant_pixels):
        dominant_pixels = candidate & (hue_distance(h, h0) <= 8)

    median_rgb = tuple(int(value) for value in np.median(rgb[dominant_pixels], axis=0).round())
    share = bins[dominant_bin] / (base.size[0] * base.size[1])
    return rgb_to_hex(median_rgb), h0, s0, v0, share


def is_overall_background_design(base: Image.Image) -> bool:
    _, _, _, _, share = estimate_dominant_surface(base)
    return share > 0.45


def make_overall_background_masks(base: Image.Image) -> tuple[dict[str, Image.Image], str]:
    size = base.size
    dominant_hex, h0, s0, v0, _ = estimate_dominant_surface(base)
    hsv = np.asarray(base.convert("HSV"))
    h = hsv[:, :, 0]
    s = hsv[:, :, 1].astype(np.int16)
    v = hsv[:, :, 2].astype(np.int16)

    background_pixels = (
        (hue_distance(h, h0) <= 9)
        & (np.abs(s - s0) <= 55)
        & (np.abs(v - v0) <= 75)
    )
    background_mask = Image.fromarray((background_pixels * 255).astype(np.uint8), "L")
    background_mask = fill_binary_holes(background_mask)
    background_mask = background_mask.filter(ImageFilter.MaxFilter(5))
    background_mask = background_mask.filter(ImageFilter.GaussianBlur(max(0.18, size[0] / BASE_W * 0.06)))
    return {"a": background_mask, "b": empty_mask(size), "c": empty_mask(size)}, dominant_hex


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


def make_area_masks(base: Image.Image, mode: str) -> tuple[dict[str, Image.Image], dict[str, str]]:
    size = base.size
    width, height = size

    if mode == "background":
        masks, background_hex = make_overall_background_masks(base)
        original_colors = dict(ORIGINAL_AREA_COLORS)
        original_colors["a"] = background_hex
        return masks, original_colors

    if MANUAL_TABLE_PATH.exists() and MANUAL_B_PATH.exists():
        return make_masks_from_manual_images(MANUAL_TABLE_PATH, MANUAL_B_PATH, size), dict(ORIGINAL_AREA_COLORS)

    if PPT_MASK_PATH.exists():
        return make_masks_from_ppt_overlay(PPT_MASK_PATH, size), dict(ORIGINAL_AREA_COLORS)

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

    return {"a": area_a, "b": area_b, "c": area_c}, dict(ORIGINAL_AREA_COLORS)


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


def make_background_detail_mask(base: Image.Image, background_mask: Image.Image, background_hex: str) -> Image.Image:
    rgb = np.asarray(base.convert("RGB"))
    hsv = np.asarray(base.convert("HSV"))
    target = tuple(int(background_hex[index : index + 2], 16) for index in (1, 3, 5))
    target_img = Image.new("RGB", base.size, background_hex)
    target_hsv = np.asarray(target_img.convert("HSV"))
    h0 = int(target_hsv[0, 0, 0])
    s0 = int(target_hsv[0, 0, 1])
    v0 = int(target_hsv[0, 0, 2])

    color_distance = np.sqrt(
        ((rgb[:, :, 0].astype(np.int32) - target[0]) ** 2)
        + ((rgb[:, :, 1].astype(np.int32) - target[1]) ** 2)
        + ((rgb[:, :, 2].astype(np.int32) - target[2]) ** 2)
    )
    h = hsv[:, :, 0]
    s = hsv[:, :, 1].astype(np.int16)
    v = hsv[:, :, 2].astype(np.int16)
    differs_from_background = (
        (color_distance > 28)
        | (hue_distance(h, h0) > 7)
        | (np.abs(s - s0) > 35)
        | (np.abs(v - v0) > 42)
    )
    detail = differs_from_background & (np.asarray(background_mask) > 8)
    detail_mask = Image.fromarray((detail * 255).astype(np.uint8), "L")
    detail_mask = detail_mask.filter(ImageFilter.MaxFilter(3))
    return detail_mask.filter(ImageFilter.GaussianBlur(max(0.12, base.size[0] / BASE_W * 0.08)))


def make_protection_masks(
    base: Image.Image,
    area_masks: dict[str, Image.Image],
    mode: str,
    original_area_colors: dict[str, str],
) -> dict[str, Image.Image]:
    if mode == "background":
        return {
            "a": ImageChops.multiply(
                area_masks["a"], make_background_detail_mask(base, area_masks["a"], original_area_colors["a"])
            ),
            "b": empty_mask(base.size),
            "c": empty_mask(base.size),
        }

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
        self.mode = "background" if is_overall_background_design(self.base) else "segmented"
        self.area_labels = BACKGROUND_AREA_LABELS if self.mode == "background" else SEGMENTED_AREA_LABELS
        self.editable_keys = tuple(self.area_labels.keys())
        self.default_opacities = {"a": 100} if self.mode == "background" else {"a": 45, "b": 45, "c": 52}
        self.area_masks, self.original_area_colors = make_area_masks(self.base, self.mode)
        self.protection_masks = make_protection_masks(
            self.base, self.area_masks, self.mode, self.original_area_colors
        )

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

        if self.mode == "background":
            active_key = max(AREA_KEYS, key=lambda key: opacities[key])
            colors["a"] = colors[active_key]
            opacities["a"] = opacities[active_key]

        for key in ("c", "a", "b"):
            if key not in self.editable_keys:
                continue
            color_layer = Image.new("RGB", image.size, colors[key])
            blended = Image.blend(image, color_layer, opacities[key])
            image.paste(blended, mask=self.area_masks[key])

        # Keep A/B's printed artwork untouched.  The C carpet artwork gets a
        # consistent soft-white treatment instead: its original red lines
        # look muddy against light C colors, while white remains clean across
        # every palette choice.
        if self.mode == "background":
            image.paste(self.base, mask=self.protection_masks["a"])
        else:
            ab_protected = ImageChops.lighter(self.protection_masks["a"], self.protection_masks["b"])
            image.paste(self.base, mask=ab_protected)
        if self.mode == "segmented" and opacities["c"] > 0:
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


def image_to_download(image: Image.Image) -> str:
    file = tempfile.NamedTemporaryFile(prefix="table-color-", suffix=".png", delete=False)
    file.close()
    image.save(file.name, format="PNG")
    return file.name


def params_from_values(
    renderer: ColorRenderer,
    colors: dict[str, str],
    opacities: dict[str, int | float],
    show_mask: bool,
) -> dict[str, list[str]]:
    params = {
        "a": [renderer.original_area_colors["a"]],
        "b": [renderer.original_area_colors["b"]],
        "c": [renderer.original_area_colors["c"]],
        "oa": ["0"],
        "ob": ["0"],
        "oc": ["0"],
        "masks": ["1" if show_mask else "0"],
    }
    for key in renderer.editable_keys:
        params[key] = [clean_hex(str(colors.get(key, "")), renderer.original_area_colors[key])]
        params["o" + key] = [str(round(float(opacities.get(key, 0))))]
    return params


def summarize_combination(renderer: ColorRenderer, colors: dict[str, str], opacities: dict[str, int | float]) -> str:
    lines = ["### Current Colors"]
    for key in renderer.editable_keys:
        label = renderer.area_labels[key]
        color = clean_hex(str(colors.get(key, "")), renderer.original_area_colors[key]).upper()
        opacity = round(float(opacities.get(key, 0)))
        lines.append(f"- **{label}**: `{color}` · `{opacity}%`")
    return "\n".join(lines)


def render_gradio_image(
    renderer: ColorRenderer,
    colors: dict[str, str],
    opacities: dict[str, int | float],
    show_mask: bool,
) -> tuple[Image.Image, str, str]:
    image = renderer.render(params_from_values(renderer, colors, opacities, show_mask))
    return image, summarize_combination(renderer, colors, opacities), image_to_download(image)


def empty_combo(renderer: ColorRenderer) -> dict[str, object]:
    combo: dict[str, object] = {}
    for key in AREA_KEYS:
        combo[key] = renderer.original_area_colors[key]
        combo["o" + key] = 0
    return combo


def values_to_combo(renderer: ColorRenderer, values: tuple[object, ...]) -> dict[str, object]:
    combo = empty_combo(renderer)
    colors = values[: len(renderer.editable_keys)]
    opacities = values[len(renderer.editable_keys) : len(renderer.editable_keys) * 2]
    for key, color, opacity in zip(renderer.editable_keys, colors, opacities):
        combo[key] = clean_hex(str(color), renderer.original_area_colors[key])
        combo["o" + key] = max(0, min(100, round(float(opacity or 0))))
    return combo


def combo_to_values(renderer: ColorRenderer, combo: dict[str, object]) -> list[object]:
    colors = [str(combo.get(key, renderer.original_area_colors[key])) for key in renderer.editable_keys]
    opacities = [int(combo.get("o" + key, 0) or 0) for key in renderer.editable_keys]
    return colors + opacities


def combo_to_render_inputs(renderer: ColorRenderer, combo: dict[str, object]) -> tuple[dict[str, str], dict[str, int]]:
    colors = {
        key: clean_hex(str(combo.get(key, renderer.original_area_colors[key])), renderer.original_area_colors[key])
        for key in renderer.editable_keys
    }
    opacities = {
        key: max(0, min(100, round(float(combo.get("o" + key, 0) or 0))))
        for key in renderer.editable_keys
    }
    return colors, opacities


def preset_label(renderer: ColorRenderer, combo: dict[str, object], index: int) -> str:
    parts = []
    for key in renderer.editable_keys:
        color = clean_hex(str(combo.get(key, renderer.original_area_colors[key])), renderer.original_area_colors[key])
        opacity = int(combo.get("o" + key, 0) or 0)
        parts.append(f"{renderer.area_labels[key]} {color.upper()} {opacity}%")
    return f"Preset {index + 1}: " + " | ".join(parts)


def saved_choices(renderer: ColorRenderer, saved: list[dict[str, object]]) -> list[str]:
    return [preset_label(renderer, combo, index) for index, combo in enumerate(saved)]


def combo_signature(renderer: ColorRenderer, combo: dict[str, object]) -> str:
    parts = []
    for key in renderer.editable_keys:
        parts.append(str(combo.get(key, "")).lower())
        parts.append(str(int(combo.get("o" + key, 0) or 0)))
    return "|".join(parts)


def load_saved_presets(renderer: ColorRenderer) -> list[dict[str, object]]:
    try:
        raw = json.loads(SAVED_PRESETS_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    saved = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        combo = empty_combo(renderer)
        valid = True
        for key in renderer.editable_keys:
            color = clean_hex(str(item.get(key, "")), renderer.original_area_colors[key])
            if not color:
                valid = False
                break
            combo[key] = color
            combo["o" + key] = max(0, min(100, round(float(item.get("o" + key, 0) or 0))))
        if valid:
            saved.append(combo)
    return saved[:12]


def persist_saved_presets(saved: list[dict[str, object]]) -> None:
    SAVED_PRESETS_PATH.write_text(json.dumps(saved[:12], indent=2) + "\n")


def build_gradio_app(renderer: ColorRenderer) -> gr.Blocks:
    title = "Table Color Simulator"
    mode_text = "Background only" if renderer.mode == "background" else "A / B / C only"
    original_combo = empty_combo(renderer)
    saved_initial = load_saved_presets(renderer)

    with gr.Blocks(title=title) as app:
        gr.Markdown(f"# {title}\n{mode_text} · Bet type colors protected")
        saved_state = gr.State(saved_initial)

        with gr.Row():
            with gr.Column(scale=5):
                preview = gr.Image(
                    label="Preview",
                    value=renderer.base,
                    type="pil",
                    height=620,
                    elem_id="preview",
                )
                download = gr.File(label="Download PNG", value=image_to_download(renderer.base))
            with gr.Column(scale=2):
                color_components = []
                opacity_components = []
                for key in renderer.editable_keys:
                    label = renderer.area_labels[key]
                    color = gr.ColorPicker(label=f"{label} color", value=renderer.original_area_colors[key])
                    opacity = gr.Slider(
                        label=f"{label} opacity",
                        minimum=0,
                        maximum=100,
                        step=1,
                        value=0,
                    )
                    color_components.append(color)
                    opacity_components.append(opacity)

                show_mask = gr.Checkbox(label="Show mask overlay", value=False)
                summary = gr.Markdown(summarize_combination(renderer, *combo_to_render_inputs(renderer, original_combo)))

                with gr.Row():
                    original_button = gr.Button("Original")
                    shuffle_button = gr.Button("Shuffle")
                save_button = gr.Button("Save Current")
                save_status = gr.Markdown("")
                saved_dropdown = gr.Dropdown(
                    label="Saved presets",
                    choices=saved_choices(renderer, saved_initial),
                    value=None,
                    interactive=True,
                )
                apply_saved_button = gr.Button("Apply Saved")

        controls = color_components + opacity_components + [show_mask]
        render_outputs = [preview, summary, download]
        control_outputs = color_components + opacity_components + render_outputs

        def render_callback(*values: object) -> tuple[Image.Image, str, str]:
            colors = {
                key: clean_hex(str(value), renderer.original_area_colors[key])
                for key, value in zip(renderer.editable_keys, values[: len(renderer.editable_keys)])
            }
            opacities = {
                key: max(0, min(100, round(float(value or 0))))
                for key, value in zip(
                    renderer.editable_keys,
                    values[len(renderer.editable_keys) : len(renderer.editable_keys) * 2],
                )
            }
            return render_gradio_image(renderer, colors, opacities, bool(values[-1]))

        def original_callback(show_mask_value: bool) -> tuple[object, ...]:
            colors, opacities = combo_to_render_inputs(renderer, original_combo)
            image, text, file_path = render_gradio_image(renderer, colors, opacities, show_mask_value)
            return (*combo_to_values(renderer, original_combo), image, text, file_path)

        def shuffle_callback(show_mask_value: bool) -> tuple[object, ...]:
            palette = [hex_value for name, hex_value in PALETTE.values() if name != "Original area color"]
            combo = empty_combo(renderer)
            for key in renderer.editable_keys:
                combo[key] = random.choice(palette)
                combo["o" + key] = renderer.default_opacities.get(key, 45)
            colors, opacities = combo_to_render_inputs(renderer, combo)
            image, text, file_path = render_gradio_image(renderer, colors, opacities, show_mask_value)
            return (*combo_to_values(renderer, combo), image, text, file_path)

        def save_callback(*values: object) -> tuple[list[dict[str, object]], object, str]:
            saved = list(values[-1] or [])
            combo = values_to_combo(renderer, values[:-1])
            signature = combo_signature(renderer, combo)
            saved = [combo] + [item for item in saved if combo_signature(renderer, item) != signature]
            saved = saved[:12]
            persist_saved_presets(saved)
            choices = saved_choices(renderer, saved)
            return saved, gr.update(choices=choices, value=choices[0]), "Saved."

        def apply_saved_callback(selected: str | None, saved: list[dict[str, object]], show_mask_value: bool) -> tuple[object, ...]:
            choices = saved_choices(renderer, saved or [])
            if not selected or selected not in choices:
                return tuple(gr.update() for _ in control_outputs)
            combo = (saved or [])[choices.index(selected)]
            colors, opacities = combo_to_render_inputs(renderer, combo)
            image, text, file_path = render_gradio_image(renderer, colors, opacities, show_mask_value)
            return (*combo_to_values(renderer, combo), image, text, file_path)

        for component in controls:
            component.change(render_callback, inputs=controls, outputs=render_outputs)

        original_button.click(original_callback, inputs=[show_mask], outputs=control_outputs)
        shuffle_button.click(shuffle_callback, inputs=[show_mask], outputs=control_outputs)
        save_button.click(
            save_callback,
            inputs=color_components + opacity_components + [saved_state],
            outputs=[saved_state, saved_dropdown, save_status],
        )
        apply_saved_button.click(
            apply_saved_callback,
            inputs=[saved_dropdown, saved_state, show_mask],
            outputs=control_outputs,
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Gradio app for recoloring table layout areas.")
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

    app = build_gradio_app(renderer)
    app.launch(server_name=args.host, server_port=args.port, css=GRADIO_CSS, footer_links=[])


if __name__ == "__main__":
    main()
