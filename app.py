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

PRESETS = {
    "original": {
        "name": "Original",
        "a": ORIGINAL_AREA_COLORS["a"],
        "b": ORIGINAL_AREA_COLORS["b"],
        "c": ORIGINAL_AREA_COLORS["c"],
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


def html_page(renderer: ColorRenderer) -> bytes:
    palette_json = json.dumps(PALETTE)
    original_area_colors_json = json.dumps(renderer.original_area_colors)
    area_labels_json = json.dumps(renderer.area_labels)
    presets = {
        "original": {
            "name": "Original",
            "a": renderer.original_area_colors["a"],
            "b": renderer.original_area_colors["b"],
            "c": renderer.original_area_colors["c"],
            "oa": 0,
            "ob": 0,
            "oc": 0,
        },
    }
    presets_json = json.dumps(presets)
    editable_status = "Background only" if renderer.mode == "background" else "A / B / C only"
    mask_status = "Background mask" if renderer.mode == "background" else "A/B/C mask overlay"
    image_alt = (
        "Rendered baccarat layout with editable background color"
        if renderer.mode == "background"
        else "Rendered baccarat layout with editable A, B and C area colors"
    )
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
    .save-actions {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 14px;
    }}
    .saved-list {{
      display: grid;
      gap: 7px;
      margin-top: 8px;
    }}
    .saved-empty {{
      margin-top: 8px;
      border-top: 1px solid rgb(25 24 21 / 10%);
      border-bottom: 1px solid rgb(25 24 21 / 10%);
      color: #777168;
      padding: 10px 2px;
      font-size: 12px;
    }}
    .saved-item {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 7px;
      align-items: stretch;
    }}
    .saved-apply {{
      min-width: 0;
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      gap: 9px;
      align-items: center;
      border-color: rgb(25 24 21 / 12%);
      background: #fbfaf6;
      color: #191815;
      padding: 8px 9px;
      text-align: left;
    }}
    .saved-swatches {{
      display: grid;
      grid-template-columns: repeat(var(--swatch-count, 3), 14px);
      gap: 3px;
    }}
    .saved-swatch {{
      width: 14px;
      height: 28px;
      border: 1px solid rgb(0 0 0 / 14%);
      border-radius: 3px;
    }}
    .saved-copy {{
      min-width: 0;
      display: grid;
      gap: 2px;
    }}
    .saved-name {{ font-size: 12px; font-weight: 800; }}
    .saved-values {{
      overflow: hidden;
      color: #6b665d;
      font: 10px ui-monospace, SFMono-Regular, Menlo, monospace;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .saved-delete {{
      width: 58px;
      min-height: 46px;
      background: white;
      color: #8d3941;
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
          <span class="pill">{editable_status}</span>
          <span class="pill">Bet type colors protected</span>
        </div>
      </header>
      <div class="image-wrap">
        <img id="render" src="/render.png" alt="{image_alt}" />
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
        Show {mask_status}
      </label>

      <p class="eyebrow" style="margin-top: 16px;">Presets</p>
      <div class="preset-grid" id="presets"></div>
      <div class="saved-list" id="saved-combinations"></div>

      <div class="save-actions">
        <button type="button" id="save-combination">Save to Presets</button>
        <button class="secondary" type="button" id="compare-original">Hold to Compare Original</button>
      </div>

      <div class="buttons">
        <a class="button" id="download" href="/download.png" download="table-color-simulation.png">Download PNG</a>
      </div>

      <div class="hex-list" id="hexes"></div>
    </aside>
  </main>
  <script>
    const palette = {palette_json};
    const presets = {presets_json};
    const originalAreaColors = {original_area_colors_json};
    const labels = {area_labels_json};
    const areaKeys = Object.keys(labels);
    const state = {{
      a: originalAreaColors.a,
      b: originalAreaColors.b,
      c: originalAreaColors.c,
      oa: 0,
      ob: 0,
      oc: 0,
      masks: 0
    }};
    const areaRoot = document.getElementById('areas');
    const hexRoot = document.getElementById('hexes');
    const img = document.getElementById('render');
    const download = document.getElementById('download');
    const savedRoot = document.getElementById('saved-combinations');
    const storageKey = 'table-color-simulator-combinations-v1';
    let savedCombinations = loadSavedCombinations();

    function syncOpacityControls(key) {{
      const range = document.getElementById(`${{key}}-opacity`);
      const number = document.getElementById(`${{key}}-opacity-number`);
      const label = document.getElementById(`${{key}}-opacity-text`);
      if (range) range.value = state['o' + key];
      if (number) number.value = state['o' + key];
      if (label) label.textContent = `${{state['o' + key]}}%`;
    }}

    function activateColor(key) {{
      if (state['o' + key] > 0) return;
      state['o' + key] = key === 'c' ? 52 : 45;
      syncOpacityControls(key);
    }}

    function restoreOriginalAreaColor(key) {{
      state[key] = originalAreaColors[key];
      state['o' + key] = 0;
      syncOpacityControls(key);
    }}

    function normalizeColorCode(value) {{
      const digits = value.trim().replace(/^#/, '');
      return /^[0-9a-f]{{6}}$/i.test(digits) ? `#${{digits.toLowerCase()}}` : null;
    }}

    function loadSavedCombinations() {{
      try {{
        const parsed = JSON.parse(localStorage.getItem(storageKey) || '[]');
        if (!Array.isArray(parsed)) return [];
        return parsed.filter((item) =>
          item && areaKeys.every((key) => normalizeColorCode(item[key])) &&
          areaKeys.every((key) => Number.isFinite(Number(item['o' + key])))
        ).slice(0, 12);
      }} catch (_) {{
        return [];
      }}
    }}

    function persistSavedCombinations() {{
      try {{
        localStorage.setItem(storageKey, JSON.stringify(savedCombinations));
      }} catch (_) {{
        // The in-memory list still works when browser storage is unavailable.
      }}
    }}

    function currentCombination() {{
      return {{
        id: Date.now(),
        a: state.a,
        b: state.b,
        c: state.c,
        oa: state.oa,
        ob: state.ob,
        oc: state.oc,
      }};
    }}

    function combinationSignature(item) {{
      return ['a', 'b', 'c', 'oa', 'ob', 'oc'].map((key) => item[key]).join('|');
    }}

    function renderSavedCombinations() {{
      if (!savedCombinations.length) {{
        savedRoot.innerHTML = '<div class="saved-empty">No custom presets yet.</div>';
        return;
      }}
      savedRoot.innerHTML = savedCombinations.map((item, index) => `
        <div class="saved-item">
          <button class="saved-apply" type="button" data-saved-action="apply" data-saved-id="${{item.id}}">
              <span class="saved-swatches" style="--swatch-count: ${{areaKeys.length}}" aria-hidden="true">
              ${{areaKeys.map((key) => `<span class="saved-swatch" style="background:${{item[key]}}"></span>`).join('')}}
            </span>
            <span class="saved-copy">
              <span class="saved-name">Preset ${{index + 1}}</span>
              <span class="saved-values">${{areaKeys.map((key) => `${{labels[key]}} ${{item[key].toUpperCase()}} ${{item['o' + key]}}%`).join(' · ')}}</span>
            </span>
          </button>
          <button class="saved-delete" type="button" data-saved-action="delete" data-saved-id="${{item.id}}">Delete</button>
        </div>
      `).join('');
    }}

    function colorOptions(areaKey, selected) {{
      const known = Object.entries(palette).some(([paletteKey, [, hex]]) => {{
        const optionHex = paletteKey === 'original' ? originalAreaColors[areaKey] : hex;
        return optionHex.toLowerCase() === selected.toLowerCase();
      }});
      const custom = known ? '' : `<option value="${{selected}}" selected>Custom color</option>`;
      return custom + Object.entries(palette).map(([paletteKey, [name, hex]]) => {{
        const optionHex = paletteKey === 'original' ? originalAreaColors[areaKey] : hex;
        return `<option value="${{optionHex}}" data-palette-key="${{paletteKey}}" ${{optionHex.toLowerCase() === selected.toLowerCase() ? 'selected' : ''}}>${{name}}</option>`;
      }}
      ).join('');
    }}

    function renderControls() {{
      areaRoot.innerHTML = areaKeys.map((key) => `
        <section class="area">
          <div class="area-head">
            <div class="area-label">
              <span class="swatch" id="${{key}}-swatch" style="background:${{state[key]}}"></span>
              <span>${{labels[key]}}</span>
            </div>
            <select id="${{key}}-select" aria-label="Color for ${{key.toUpperCase()}} area">${{colorOptions(key, state[key])}}</select>
            <input id="${{key}}-custom" type="color" value="${{state[key]}}" aria-label="Custom color for ${{key.toUpperCase()}} area" />
          </div>
          <input id="${{key}}-code" class="color-code" type="text" value="${{state[key].toUpperCase()}}" spellcheck="false" maxlength="7" aria-label="Color code for ${{key.toUpperCase()}} area" />
          <label>
            <span class="range-row"><span>Opacity</span><span id="${{key}}-opacity-text">${{state['o' + key]}}%</span></span>
            <span class="opacity-control">
              <input id="${{key}}-opacity" type="range" min="0" max="100" value="${{state['o' + key]}}" aria-label="Opacity for ${{key.toUpperCase()}} area" />
              <input id="${{key}}-opacity-number" class="opacity-number" type="number" min="0" max="100" value="${{state['o' + key]}}" aria-label="Opacity percentage for ${{key.toUpperCase()}} area" />
            </span>
          </label>
        </section>
      `).join('');

      for (const key of areaKeys) {{
        document.getElementById(`${{key}}-select`).addEventListener('change', (event) => {{
          const paletteKey = event.target.selectedOptions[0]?.dataset.paletteKey;
          if (paletteKey === 'original') {{
            restoreOriginalAreaColor(key);
          }} else {{
            state[key] = event.target.value;
            activateColor(key);
          }}
          document.getElementById(`${{key}}-custom`).value = state[key];
          document.getElementById(`${{key}}-code`).value = state[key].toUpperCase();
          update();
        }});
        document.getElementById(`${{key}}-custom`).addEventListener('input', (event) => {{
          state[key] = event.target.value;
          if (state[key].toLowerCase() === originalAreaColors[key].toLowerCase()) {{
            restoreOriginalAreaColor(key);
          }} else {{
            activateColor(key);
          }}
          document.getElementById(`${{key}}-code`).value = state[key].toUpperCase();
          update();
        }});
        document.getElementById(`${{key}}-code`).addEventListener('input', (event) => {{
          const code = normalizeColorCode(event.target.value);
          event.target.classList.toggle('invalid', !code && event.target.value.length > 0);
          if (!code) return;
          state[key] = code;
          if (state[key].toLowerCase() === originalAreaColors[key].toLowerCase()) {{
            restoreOriginalAreaColor(key);
          }} else {{
            activateColor(key);
          }}
          event.target.value = state[key].toUpperCase();
          document.getElementById(`${{key}}-custom`).value = state[key];
          update();
        }});
        const setOpacity = (value) => {{
          state['o' + key] = Math.max(0, Math.min(100, Number(value) || 0));
          syncOpacityControls(key);
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
      for (const key of areaKeys) {{
        const swatch = document.getElementById(`${{key}}-swatch`);
        if (swatch) swatch.style.background = state[key];
      }}
      img.src = `/render.png?${{query()}}`;
      download.href = `/download.png?${{query()}}`;
      hexRoot.innerHTML = areaKeys.map((key) => `
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
      const colors = Object.entries(palette)
        .filter(([paletteKey]) => paletteKey !== 'original')
        .map(([, [, hex]]) => hex);
      for (const key of areaKeys) {{
        state[key] = colors[Math.floor(Math.random() * colors.length)];
        state['o' + key] = key === 'c' ? 52 : 45;
      }}
      renderControls();
      update();
    }});
    document.getElementById('save-combination').addEventListener('click', (event) => {{
      const combination = currentCombination();
      const signature = combinationSignature(combination);
      savedCombinations = [
        combination,
        ...savedCombinations.filter((item) => combinationSignature(item) !== signature),
      ].slice(0, 12);
      persistSavedCombinations();
      renderSavedCombinations();
      const button = event.currentTarget;
      button.textContent = 'Saved to Presets';
      setTimeout(() => {{ button.textContent = 'Save to Presets'; }}, 900);
    }});
    savedRoot.addEventListener('click', (event) => {{
      const button = event.target.closest('[data-saved-action]');
      if (!button) return;
      const id = button.dataset.savedId;
      const item = savedCombinations.find((saved) => String(saved.id) === id);
      if (!item) return;
      if (button.dataset.savedAction === 'delete') {{
        savedCombinations = savedCombinations.filter((saved) => String(saved.id) !== id);
        persistSavedCombinations();
        renderSavedCombinations();
        return;
      }}
      Object.assign(state, {{
        a: item.a,
        b: item.b,
        c: item.c,
        oa: Math.max(0, Math.min(100, Number(item.oa ?? 0))),
        ob: Math.max(0, Math.min(100, Number(item.ob ?? 0))),
        oc: Math.max(0, Math.min(100, Number(item.oc ?? 0))),
        masks: 0,
      }});
      document.getElementById('masks').checked = false;
      renderControls();
      update();
    }});

    const compareButton = document.getElementById('compare-original');
    let comparingOriginal = false;
    function showOriginal() {{
      if (comparingOriginal) return;
      comparingOriginal = true;
      img.src = `/original.jpg?_=${{Date.now()}}`;
    }}
    function restoreCurrent() {{
      if (!comparingOriginal) return;
      comparingOriginal = false;
      update();
    }}
    compareButton.addEventListener('pointerdown', (event) => {{
      event.preventDefault();
      showOriginal();
    }});
    compareButton.addEventListener('pointerup', restoreCurrent);
    compareButton.addEventListener('pointerleave', restoreCurrent);
    compareButton.addEventListener('pointercancel', restoreCurrent);
    compareButton.addEventListener('keydown', (event) => {{
      if (event.key === ' ' || event.key === 'Enter') showOriginal();
    }});
    compareButton.addEventListener('keyup', (event) => {{
      if (event.key === ' ' || event.key === 'Enter') restoreCurrent();
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
    renderSavedCombinations();
    update();
  </script>
</body>
</html>""".encode("utf-8")


class AppHandler(BaseHTTPRequestHandler):
    renderer: ColorRenderer

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self.send_head_headers(len(html_page(self.renderer)), "text/html; charset=utf-8")
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
            self.send_bytes(html_page(self.renderer), "text/html; charset=utf-8")
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
