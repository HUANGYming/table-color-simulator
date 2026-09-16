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
LEGACY_ARTWORK_PATH = APP_DIR / "artwork-legacy.jpg"
ROSE_PURPLE_ARTWORK_PATH = APP_DIR / "artwork-rose-purple.jpg"
CYAN_ARTWORK_PATH = APP_DIR / "artwork-cyan.jpg"
REV23_ARTWORK_PATH = APP_DIR / "artwork-rev23.jpg"
PPT_MASK_PATH = APP_DIR / "mask.png"
MANUAL_TABLE_PATH = APP_DIR / "manual-table.png"
MANUAL_B_PATH = APP_DIR / "manual-b.png"
SAVED_PRESETS_PATH = APP_DIR / "saved-presets.json"
MAX_SAVED_PRESETS = 12
BASE_W = 1280
BASE_H = 555
MASK_SCALE = 4

DEFAULT_ARTWORK_KEY = "rev23"
ARTWORKS = {
    "legacy": {
        "name": "Original A/B/C table",
        "path": LEGACY_ARTWORK_PATH,
        "mode": "segmented",
    },
    "rose-purple": {
        "name": "Rose purple background",
        "path": ROSE_PURPLE_ARTWORK_PATH,
        "mode": "background",
    },
    "cyan": {
        "name": "Cyan background",
        "path": CYAN_ARTWORK_PATH,
        "mode": "background",
    },
    "rev23": {
        "name": "Beige Rev23 multi-region",
        "path": REV23_ARTWORK_PATH,
        "mode": "rev23",
    },
}

REV23_REGION_SPECS = [
    {"key": "base", "name": "Base blue", "hex": "#90d0f0", "rgb": (144, 208, 240), "threshold": 40},
    {"key": "left_outer", "name": "Left outer", "hex": "#90d0f0", "rgb": (144, 208, 240), "threshold": 40},
    {"key": "cyan", "name": "Bright cyan", "hex": "#00c0f8", "rgb": (0, 192, 248), "threshold": 50},
    {"key": "periwinkle", "name": "Periwinkle", "hex": "#e0e8f8", "rgb": (224, 232, 248), "threshold": 34},
    {"key": "mint", "name": "Mint", "hex": "#d0f8f8", "rgb": (208, 248, 248), "threshold": 34},
    {"key": "peach", "name": "Peach", "hex": "#f8d0b8", "rgb": (248, 208, 184), "threshold": 38},
    {"key": "mauve", "name": "Mauve", "hex": "#d8c0c8", "rgb": (216, 192, 200), "threshold": 38},
]

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
    return {"a": background_mask}, dominant_hex


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


def looks_like_rev23(base: Image.Image) -> bool:
    """Detect the new Illustrator-rendered Rev23 artwork by its large blue fill."""
    sample = base.resize((320, max(1, round(base.size[1] * 320 / base.size[0]))), Image.Resampling.BILINEAR)
    arr = np.asarray(sample.convert("RGB"), dtype=np.int32)
    target = np.array(REV23_REGION_SPECS[0]["rgb"], dtype=np.int32)
    distance_squared = ((arr - target) ** 2).sum(axis=2)
    blue_share = np.mean(distance_squared <= REV23_REGION_SPECS[0]["threshold"] ** 2)
    return base.size[0] > 2000 and blue_share > 0.20


def make_rev23_region_masks(base: Image.Image) -> dict[str, Image.Image]:
    """Segment the Rev23 artwork into its large editable background fills."""
    arr = np.asarray(base.convert("RGB"), dtype=np.int32)
    refs = np.array([spec["rgb"] for spec in REV23_REGION_SPECS], dtype=np.int32)
    thresholds = np.array([spec["threshold"] ** 2 for spec in REV23_REGION_SPECS], dtype=np.int32)

    distances = ((arr[:, :, None, :] - refs[None, None, :, :]) ** 2).sum(axis=3)
    nearest = distances.argmin(axis=2)
    nearest_distance = distances.min(axis=2)
    max_channel = arr.max(axis=2)
    min_channel = arr.min(axis=2)

    # White page/notch areas and black outside pixels should never recolor.
    valid_background = (nearest_distance <= thresholds[nearest]) & (max_channel > 90)
    valid_background &= ~((max_channel > 245) & (min_channel > 235))

    raw_masks: dict[str, np.ndarray] = {}
    for index, spec in enumerate(REV23_REGION_SPECS):
        raw_masks[spec["key"]] = valid_background & (nearest == index)

    key_to_index = {spec["key"]: index for index, spec in enumerate(REV23_REGION_SPECS)}
    cyan_index = key_to_index["cyan"]
    mauve_index = key_to_index["mauve"]
    height, width = valid_background.shape
    half_width = width // 2

    def connected_component_near(data: np.ndarray, seed_x: int, seed_y: int, radius: int = 120) -> np.ndarray:
        seed_x = max(0, min(width - 1, seed_x))
        seed_y = max(0, min(height - 1, seed_y))
        if not data[seed_y, seed_x]:
            y0 = max(0, seed_y - radius)
            y1 = min(height, seed_y + radius + 1)
            x0 = max(0, seed_x - radius)
            x1 = min(width, seed_x + radius + 1)
            nearby_y, nearby_x = np.where(data[y0:y1, x0:x1])
            if len(nearby_x) == 0:
                return np.zeros_like(data, dtype=bool)
            distances = (nearby_x + x0 - seed_x) ** 2 + (nearby_y + y0 - seed_y) ** 2
            nearest = int(distances.argmin())
            seed_x = int(nearby_x[nearest] + x0)
            seed_y = int(nearby_y[nearest] + y0)

        component = np.zeros_like(data, dtype=bool)
        component[seed_y, seed_x] = True
        queue: deque[tuple[int, int]] = deque([(seed_y, seed_x)])
        while queue:
            y, x = queue.popleft()
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and data[ny, nx] and not component[ny, nx]:
                    component[ny, nx] = True
                    queue.append((ny, nx))
        return component

    def row_edge(data: np.ndarray, side: str) -> np.ndarray:
        edge = np.full(height, -1, dtype=np.int32)
        for y in range(height):
            cols = np.flatnonzero(data[y])
            if len(cols) > 0:
                edge[y] = int(cols[0] if side == "left" else cols[-1])
        return edge

    # The far-left module shares the same fill color as Base blue.  Instead of
    # guessing by color, construct it from four visible boundaries:
    # top = the white arc, right = the neighboring mauve block,
    # bottom = the table edge, left = the mirrored right edge of the far-right block.
    left_outer = np.zeros_like(valid_background, dtype=bool)
    fillable_surface = (max_channel > 90) & ~((max_channel > 245) & (min_channel > 235))
    white_pixels = (arr[:, :, 0] > 240) & (arr[:, :, 1] > 240) & (arr[:, :, 2] > 240)
    right_cyan_pixels = valid_background & (nearest == cyan_index)
    right_cyan_pixels[:, :half_width] = False
    left_mauve_pixels = valid_background & (nearest == mauve_index)
    left_mauve_pixels[:, half_width:] = False

    white_arc = connected_component_near(white_pixels, round(width * 0.17), round(height * 0.08))
    right_cyan_block = keep_largest_components(right_cyan_pixels, 1)
    left_mauve_block = keep_largest_components(left_mauve_pixels, 1)

    white_left_edge = row_edge(white_arc, "left")
    right_cyan_outer_edge = row_edge(right_cyan_block, "right")
    left_mauve_edge = row_edge(left_mauve_block, "left")
    left_outer_edge = np.full(height, -1, dtype=np.int32)
    edge_rows = np.flatnonzero(right_cyan_outer_edge >= 0)
    left_outer_edge[edge_rows] = width - 1 - right_cyan_outer_edge[edge_rows]
    if len(edge_rows) >= 20:
        fit_rows = edge_rows[-min(140, len(edge_rows)) :]
        slope, intercept = np.polyfit(fit_rows, left_outer_edge[fit_rows], 1)
        last_row = int(edge_rows[-1])
        for y in range(last_row + 1, height):
            predicted = int(round(slope * y + intercept))
            if predicted <= 0 or predicted >= half_width:
                break
            left_outer_edge[y] = predicted

    for y in range(height):
        if left_outer_edge[y] < 0:
            continue

        left_boundary = int(left_outer_edge[y])
        right_candidates = []
        if white_left_edge[y] > left_boundary:
            right_candidates.append(int(white_left_edge[y]) - 1)
        if left_mauve_edge[y] > left_boundary:
            right_candidates.append(int(left_mauve_edge[y]) - 1)
        if not right_candidates:
            continue

        right_boundary = min(right_candidates)
        if 0 <= left_boundary < right_boundary < half_width:
            left_outer[y, left_boundary : right_boundary + 1] = fillable_surface[
                y, left_boundary : right_boundary + 1
            ]

    left_outer = keep_largest_components(left_outer, 1)
    raw_masks["left_outer"] = left_outer
    raw_masks["base"] = raw_masks["base"] & ~left_outer

    masks: dict[str, Image.Image] = {}
    for spec in REV23_REGION_SPECS:
        data = raw_masks[spec["key"]]
        mask = Image.fromarray((data * 255).astype(np.uint8), "L")
        masks[spec["key"]] = mask.filter(ImageFilter.GaussianBlur(0.2))
    return masks


def make_rev23_detail_mask(base: Image.Image) -> Image.Image:
    """Protect printed artwork while leaving the large fill surfaces editable."""
    arr = np.asarray(base.convert("RGB"), dtype=np.int32)
    refs = np.array([spec["rgb"] for spec in REV23_REGION_SPECS], dtype=np.int32)
    thresholds = np.array([spec["threshold"] ** 2 for spec in REV23_REGION_SPECS], dtype=np.int32)

    distances = ((arr[:, :, None, :] - refs[None, None, :, :]) ** 2).sum(axis=3)
    nearest = distances.argmin(axis=2)
    nearest_distance = distances.min(axis=2)
    background = nearest_distance <= thresholds[nearest]

    hsv = np.asarray(base.convert("HSV"))
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    dark_print = (v < 150) & (s > 8)
    saturated_print = (s > 55) & ~background
    white_linework = (v > 210) & (s < 35) & ~background
    detail = dark_print | saturated_print | white_linework

    mask = Image.fromarray((detail * 255).astype(np.uint8), "L")
    mask = mask.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    return mask.filter(ImageFilter.GaussianBlur(0.12))


def make_area_masks(base: Image.Image) -> dict[str, Image.Image]:
    size = base.size
    width, height = size

    if looks_like_rev23(base):
        return make_rev23_region_masks(base)

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
            )
        }

    if mode == "rev23":
        detail_mask = make_rev23_detail_mask(base)
        return {
            key: ImageChops.multiply(mask, detail_mask)
            for key, mask in area_masks.items()
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


def clean_opacity_percent(value: object, fallback: int = 0) -> int:
    try:
        number = int(float(str(value)))
    except (TypeError, ValueError):
        number = fallback
    return max(0, min(100, number))


def clean_artwork_key(params: dict[str, list[str]]) -> str:
    value = params.get("artwork", [DEFAULT_ARTWORK_KEY])[0]
    return value if value in ARTWORKS else DEFAULT_ARTWORK_KEY


def artwork_metadata() -> dict[str, str]:
    return {key: spec["name"] for key, spec in ARTWORKS.items()}


def normalize_saved_preset(item: object, area_keys: tuple[str, ...]) -> dict[str, object] | None:
    if not isinstance(item, dict):
        return None

    normalized: dict[str, object] = {}
    for key in area_keys:
        color = clean_hex(str(item.get(key, "")), "")
        if not color:
            return None
        normalized[key] = color
        normalized[f"o{key}"] = clean_opacity_percent(item.get(f"o{key}", 0))

    signature = "|".join(str(normalized[value]) for key in area_keys for value in (key, f"o{key}"))
    preset_id = str(item.get("id") or signature)
    normalized["id"] = preset_id[:80]
    return normalized


def load_saved_store() -> dict[str, list[dict[str, object]]]:
    if not SAVED_PRESETS_PATH.exists():
        return {}
    try:
        payload = json.loads(SAVED_PRESETS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(payload, list):
        return {"legacy": payload}
    return payload if isinstance(payload, dict) else {}


def load_saved_presets(artwork_key: str, renderer: ColorRenderer) -> list[dict[str, object]]:
    store = load_saved_store()
    items = store.get(artwork_key, [])
    if not isinstance(items, list):
        return []

    presets: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in items:
        normalized = normalize_saved_preset(item, renderer.area_keys)
        if not normalized:
            continue
        signature = "|".join(str(normalized[value]) for key in renderer.area_keys for value in (key, f"o{key}"))
        if signature in seen:
            continue
        seen.add(signature)
        presets.append(normalized)
        if len(presets) >= MAX_SAVED_PRESETS:
            break
    return presets


def save_saved_presets(
    artwork_key: str,
    renderer: ColorRenderer,
    items: object,
) -> list[dict[str, object]]:
    if not isinstance(items, list):
        raise ValueError("Expected a list of presets")

    presets: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in items:
        normalized = normalize_saved_preset(item, renderer.area_keys)
        if not normalized:
            continue
        signature = "|".join(str(normalized[value]) for key in renderer.area_keys for value in (key, f"o{key}"))
        if signature in seen:
            continue
        seen.add(signature)
        presets.append(normalized)
        if len(presets) >= MAX_SAVED_PRESETS:
            break

    store = load_saved_store()
    store[artwork_key] = presets
    temp_path = SAVED_PRESETS_PATH.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(store, indent=2), encoding="utf-8")
    temp_path.replace(SAVED_PRESETS_PATH)
    return presets


class ColorRenderer:
    def __init__(self, image_path: Path, mode: str, name: str):
        self.image_path = image_path
        self.mode = mode
        self.name = name
        self.base = Image.open(image_path).convert("RGB")

        if mode == "rev23":
            self.area_masks = make_rev23_region_masks(self.base)
            self.area_labels = {spec["key"]: spec["name"] for spec in REV23_REGION_SPECS}
            self.original_area_colors = {spec["key"]: spec["hex"] for spec in REV23_REGION_SPECS}
        elif mode == "background":
            self.area_masks, background_hex = make_overall_background_masks(self.base)
            self.area_labels = {"a": "Background"}
            self.original_area_colors = {"a": background_hex}
        else:
            self.area_masks = make_area_masks(self.base)
            self.area_labels = {"a": "A areas", "b": "B areas", "c": "C areas"}
            self.original_area_colors = ORIGINAL_AREA_COLORS

        self.area_keys = tuple(self.area_masks.keys())
        self.protection_masks = make_protection_masks(
            self.base, self.area_masks, self.mode, self.original_area_colors
        )
        self.presets = {
            "original": {
                "name": "Original",
                **self.original_area_colors,
                **{f"o{key}": 0 for key in self.area_keys},
            }
        }

    def render(self, params: dict[str, list[str]]) -> Image.Image:
        image = self.base.copy()
        colors = {
            key: clean_hex(params.get(key, [self.original_area_colors[key]])[0], self.original_area_colors[key])
            for key in self.area_keys
        }
        opacities = {
            key: clean_opacity(params.get(f"o{key}", ["0"])[0], 0)
            for key in self.area_keys
        }

        for key in self.area_keys:
            color_layer = Image.new("RGB", image.size, colors[key])
            blended = Image.blend(image, color_layer, opacities[key])
            image.paste(blended, mask=self.area_masks[key])

        if self.mode == "segmented":
            # Keep A/B's printed artwork untouched.  The C carpet artwork gets
            # a consistent soft-white treatment instead.
            ab_protected = ImageChops.lighter(self.protection_masks["a"], self.protection_masks["b"])
            image.paste(self.base, mask=ab_protected)
            c_line_alpha = self.protection_masks["c"].point(lambda value: round(value * 0.58))
            if opacities["c"] > 0:
                image.paste(Image.new("RGB", image.size, "#f4f7fb"), mask=c_line_alpha)
        else:
            for key in self.area_keys:
                image.paste(self.base, mask=self.protection_masks[key])

        if params.get("masks", ["0"])[0] == "1":
            image = self.render_mask_overlay(image)

        return image

    def render_mask_overlay(self, image: Image.Image) -> Image.Image:
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        colors = [
            (35, 130, 190, 92),
            (0, 190, 230, 104),
            (120, 135, 235, 104),
            (70, 210, 180, 104),
            (245, 150, 70, 104),
            (185, 95, 140, 104),
            (255, 215, 0, 92),
        ]
        for index, key in enumerate(self.area_keys):
            color = colors[index % len(colors)]
            color_img = Image.new("RGBA", image.size, color)
            overlay.paste(color_img, mask=self.area_masks[key])
        return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def legacy_html_page() -> bytes:
    palette_json = json.dumps(PALETTE)
    presets_json = json.dumps(PRESETS)
    original_area_colors_json = json.dumps(ORIGINAL_AREA_COLORS)
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
      grid-template-columns: repeat(3, 14px);
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
    const state = {{
      a: originalAreaColors.a,
      b: originalAreaColors.b,
      c: originalAreaColors.c,
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
          item && ['a', 'b', 'c'].every((key) => normalizeColorCode(item[key])) &&
          ['oa', 'ob', 'oc'].every((key) => Number.isFinite(Number(item[key])))
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
            <span class="saved-swatches" aria-hidden="true">
              <span class="saved-swatch" style="background:${{item.a}}"></span>
              <span class="saved-swatch" style="background:${{item.b}}"></span>
              <span class="saved-swatch" style="background:${{item.c}}"></span>
            </span>
            <span class="saved-copy">
              <span class="saved-name">Preset ${{index + 1}}</span>
              <span class="saved-values">A ${{item.a.toUpperCase()}} ${{item.oa}}% · B ${{item.b.toUpperCase()}} ${{item.ob}}% · C ${{item.c.toUpperCase()}} ${{item.oc}}%</span>
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
      areaRoot.innerHTML = ['a', 'b', 'c'].map((key) => `
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

      for (const key of ['a', 'b', 'c']) {{
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
      const colors = Object.entries(palette)
        .filter(([paletteKey]) => paletteKey !== 'original')
        .map(([, [, hex]]) => hex);
      state.a = colors[Math.floor(Math.random() * colors.length)];
      state.b = colors[Math.floor(Math.random() * colors.length)];
      state.c = colors[Math.floor(Math.random() * colors.length)];
      state.oa = 45;
      state.ob = 45;
      state.oc = 52;
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
        oa: Math.max(0, Math.min(100, Number(item.oa))),
        ob: Math.max(0, Math.min(100, Number(item.ob))),
        oc: Math.max(0, Math.min(100, Number(item.oc))),
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


def html_page(renderer: ColorRenderer, artwork_key: str) -> bytes:
    area_keys_json = json.dumps(renderer.area_keys)
    area_labels_json = json.dumps(renderer.area_labels)
    original_area_colors_json = json.dumps(renderer.original_area_colors)
    presets_json = json.dumps(renderer.presets)
    artworks_json = json.dumps(artwork_metadata())
    artwork_key_json = json.dumps(artwork_key)
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Table Color Simulator</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f7f7;
      color: #17191a;
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; background: #f5f7f7; }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 380px;
      gap: 16px;
      min-height: 100vh;
      padding: 16px;
    }
    .preview { min-width: 0; display: flex; flex-direction: column; gap: 12px; }
    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      border-bottom: 1px solid rgb(23 25 26 / 12%);
      padding-bottom: 12px;
    }
    h1 { margin: 0; font-size: clamp(22px, 3vw, 34px); line-height: 1.05; letter-spacing: 0; }
    .eyebrow {
      margin: 0 0 5px;
      color: #4d6b70;
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
    }
    .status { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; font-size: 12px; font-weight: 700; }
    .pill {
      border: 1px solid rgb(23 25 26 / 12%);
      border-radius: 999px;
      background: white;
      padding: 7px 10px;
      white-space: nowrap;
    }
    .image-wrap {
      flex: 1;
      display: grid;
      place-items: center;
      min-height: 420px;
      overflow: hidden;
      border: 1px solid rgb(0 0 0 / 14%);
      border-radius: 8px;
      background: white;
      box-shadow: 0 18px 46px rgb(0 0 0 / 10%);
    }
    #render { display: block; width: 100%; height: auto; }
    aside {
      border: 1px solid rgb(23 25 26 / 12%);
      border-radius: 8px;
      background: white;
      box-shadow: 0 18px 44px rgb(0 0 0 / 8%);
      padding: 16px;
      max-height: calc(100vh - 32px);
      overflow: auto;
    }
    .panel-title { display: flex; justify-content: space-between; gap: 12px; align-items: start; }
    h2 { margin: 2px 0 0; font-size: 18px; letter-spacing: 0; }
    .artwork-select {
      width: 100%;
      height: 36px;
      margin-top: 14px;
      border: 1px solid rgb(23 25 26 / 18%);
      border-radius: 8px;
      background: white;
      color: #17191a;
      font: inherit;
      font-weight: 700;
      padding: 0 10px;
    }
    .area {
      margin-top: 12px;
      border: 1px solid rgb(23 25 26 / 12%);
      border-radius: 8px;
      background: #fbfcfc;
      padding: 12px;
    }
    .area-head {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 48px;
      gap: 10px;
      align-items: center;
    }
    .area-label { display: flex; align-items: center; gap: 9px; min-width: 0; font-weight: 800; }
    .area-label span:last-child { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .swatch {
      width: 30px;
      height: 30px;
      border-radius: 6px;
      border: 1px solid rgb(0 0 0 / 18%);
      box-shadow: inset 0 0 0 1px rgb(255 255 255 / 35%);
      flex: 0 0 auto;
    }
    input[type="color"] {
      width: 48px;
      height: 34px;
      border: 1px solid rgb(23 25 26 / 18%);
      border-radius: 8px;
      background: white;
      padding: 3px;
    }
    .color-code, .opacity-number {
      border: 1px solid rgb(23 25 26 / 18%);
      border-radius: 6px;
      background: white;
      color: #17191a;
      font: 700 12px ui-monospace, SFMono-Regular, Menlo, monospace;
      letter-spacing: 0;
    }
    .color-code {
      width: 100%;
      height: 32px;
      margin-top: 9px;
      padding: 0 9px;
      text-transform: uppercase;
    }
    .color-code.invalid { border-color: #bd4b55; color: #a83a44; }
    label { display: grid; gap: 7px; margin-top: 10px; color: #5d6364; font-size: 12px; font-weight: 700; }
    .range-row { display: flex; justify-content: space-between; gap: 10px; }
    input[type="range"] { width: 100%; accent-color: #297a86; }
    .opacity-control { display: grid; grid-template-columns: minmax(0, 1fr) 58px; gap: 9px; align-items: center; }
    .opacity-number { width: 58px; height: 30px; padding: 0 6px; }
    button, a.button {
      display: inline-flex;
      min-height: 34px;
      align-items: center;
      justify-content: center;
      border: 1px solid rgb(23 25 26 / 14%);
      border-radius: 8px;
      background: #17191a;
      color: white;
      padding: 8px 10px;
      font: inherit;
      font-size: 13px;
      font-weight: 800;
      text-decoration: none;
      cursor: pointer;
    }
    button.secondary, a.secondary { background: white; color: #17191a; }
    .actions, .preset-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 14px; }
    .saved-list { display: grid; gap: 7px; margin-top: 8px; }
    .saved-empty {
      margin-top: 8px;
      border-top: 1px solid rgb(23 25 26 / 10%);
      border-bottom: 1px solid rgb(23 25 26 / 10%);
      color: #73797a;
      padding: 10px 2px;
      font-size: 12px;
    }
    .saved-item { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 7px; align-items: stretch; }
    .saved-apply {
      min-width: 0;
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      gap: 9px;
      align-items: center;
      border-color: rgb(23 25 26 / 12%);
      background: #fbfcfc;
      color: #17191a;
      padding: 8px 9px;
      text-align: left;
    }
    .saved-swatches { display: grid; grid-auto-flow: column; grid-auto-columns: 10px; gap: 2px; }
    .saved-swatch { width: 10px; height: 28px; border: 1px solid rgb(0 0 0 / 14%); border-radius: 3px; }
    .saved-copy { min-width: 0; display: grid; gap: 2px; }
    .saved-name { font-size: 12px; font-weight: 800; }
    .saved-values {
      overflow: hidden;
      color: #686f70;
      font: 10px ui-monospace, SFMono-Regular, Menlo, monospace;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .saved-delete { width: 58px; min-height: 46px; background: white; color: #8d3941; }
    .check { display: flex; align-items: center; gap: 8px; margin-top: 14px; color: #424748; }
    .check input { width: 16px; height: 16px; }
    .hex-list { display: grid; gap: 6px; margin-top: 14px; border-radius: 8px; background: #17191a; color: white; padding: 12px; font-size: 12px; }
    .hex-line { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      aside { order: -1; max-height: none; }
      header { align-items: start; flex-direction: column; }
      .status { justify-content: flex-start; }
    }
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
          <span class="pill" id="region-count"></span>
          <span class="pill">Printed artwork protected</span>
        </div>
      </header>
      <div class="image-wrap">
        <img id="render" src="/render.png" alt="Rendered table layout with editable color regions" />
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

      <label>
        <span>Artwork</span>
        <select class="artwork-select" id="artwork-select"></select>
      </label>

      <div id="areas"></div>

      <label class="check">
        <input type="checkbox" id="masks" />
        Show mask overlay
      </label>

      <p class="eyebrow" style="margin-top: 16px;">Presets</p>
      <div class="preset-grid" id="presets"></div>
      <div class="saved-list" id="saved-combinations"></div>

      <div class="actions">
        <button type="button" id="save-combination">Save Preset</button>
        <button class="secondary" type="button" id="compare-original">Hold Original</button>
      </div>

      <div class="actions" id="legacy-import-actions">
        <button class="secondary" type="button" id="import-legacy-presets">Import 8765 Presets</button>
      </div>

      <div class="actions">
        <a class="button" id="download" href="/download.png" download="table-color-simulation.png">Download PNG</a>
      </div>

      <div class="hex-list" id="hexes"></div>
    </aside>
  </main>
  <script>
    const selectedArtwork = __ARTWORK_KEY__;
    const artworks = __ARTWORKS__;
    const areaKeys = __AREA_KEYS__;
    const labels = __AREA_LABELS__;
    const originalAreaColors = __ORIGINAL_AREA_COLORS__;
    const presets = __PRESETS__;
    const pastelColors = ['#a9d8cf', '#b8d7f4', '#d6d2f2', '#f3d2df', '#f2d7ba', '#c8e6bd', '#f1bfc9', '#c2efe8'];
    const state = { masks: 0 };
    for (const key of areaKeys) {
      state[key] = originalAreaColors[key];
      state['o' + key] = 0;
    }

    const areaRoot = document.getElementById('areas');
    const hexRoot = document.getElementById('hexes');
    const img = document.getElementById('render');
    const download = document.getElementById('download');
    const savedRoot = document.getElementById('saved-combinations');
    document.getElementById('region-count').textContent = `${areaKeys.length} editable regions`;
    const storageKey = `table-color-simulator-combinations-${selectedArtwork}`;
    const fallbackStorageKeys = selectedArtwork === 'legacy' ? ['table-color-simulator-combinations-v1'] : [];
    let savedCombinations = loadSavedCombinations();

    const artworkSelect = document.getElementById('artwork-select');
    artworkSelect.innerHTML = Object.entries(artworks).map(([key, name]) =>
      `<option value="${key}" ${key === selectedArtwork ? 'selected' : ''}>${name}</option>`
    ).join('');
    artworkSelect.addEventListener('change', (event) => {
      window.location.href = `/?artwork=${encodeURIComponent(event.target.value)}`;
    });

    function normalizeColorCode(value) {
      const digits = String(value || '').trim().replace(/^#/, '');
      return /^[0-9a-f]{6}$/i.test(digits) ? `#${digits.toLowerCase()}` : null;
    }

    function clampOpacity(value) {
      return Math.max(0, Math.min(100, Number(value) || 0));
    }

    function syncOpacityControls(key) {
      const range = document.getElementById(`${key}-opacity`);
      const number = document.getElementById(`${key}-opacity-number`);
      const label = document.getElementById(`${key}-opacity-text`);
      if (range) range.value = state['o' + key];
      if (number) number.value = state['o' + key];
      if (label) label.textContent = `${state['o' + key]}%`;
    }

    function activateColor(key) {
      if (state['o' + key] > 0) return;
      state['o' + key] = 100;
      syncOpacityControls(key);
    }

    function normalizedCombination(item) {
      if (!item) return null;
      const normalized = {};
      for (const key of areaKeys) {
        const color = normalizeColorCode(item[key]);
        if (!color) return null;
        normalized[key] = color;
        normalized['o' + key] = clampOpacity(item['o' + key]);
      }
      normalized.id = String(item.id || combinationSignature(normalized));
      return normalized;
    }

    function sanitizeCombinations(items) {
      if (!Array.isArray(items)) return [];
      const result = [];
      const seen = new Set();
      for (const item of items) {
        const normalized = normalizedCombination(item);
        if (!normalized) continue;
        const signature = combinationSignature(normalized);
        if (seen.has(signature)) continue;
        seen.add(signature);
        result.push(normalized);
        if (result.length >= 12) break;
      }
      return result;
    }

    function mergeCombinations(...groups) {
      return sanitizeCombinations(groups.flat());
    }

    function loadSavedCombinations() {
      try {
        const groups = [sanitizeCombinations(JSON.parse(localStorage.getItem(storageKey) || '[]'))];
        for (const key of fallbackStorageKeys) {
          groups.push(sanitizeCombinations(JSON.parse(localStorage.getItem(key) || '[]')));
        }
        return mergeCombinations(...groups);
      } catch (_) {
        return [];
      }
    }

    function persistLocalSavedCombinations() {
      try {
        localStorage.setItem(storageKey, JSON.stringify(savedCombinations));
      } catch (_) {}
    }

    async function loadServerSavedCombinations() {
      try {
        const response = await fetch(`/saved-presets.json?artwork=${encodeURIComponent(selectedArtwork)}`, { cache: 'no-store' });
        if (!response.ok) return;
        const serverCombinations = sanitizeCombinations(await response.json());
        const merged = mergeCombinations(savedCombinations, serverCombinations);
        const serverSignature = serverCombinations.map(combinationSignature).join('\\n');
        const mergedSignature = merged.map(combinationSignature).join('\\n');
        savedCombinations = merged;
        persistLocalSavedCombinations();
        renderSavedCombinations();
        if (mergedSignature !== serverSignature) await persistSavedCombinations();
      } catch (_) {}
    }

    async function persistSavedCombinations() {
      persistLocalSavedCombinations();
      try {
        const response = await fetch(`/saved-presets.json?artwork=${encodeURIComponent(selectedArtwork)}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(savedCombinations)
        });
        if (!response.ok) return;
        savedCombinations = sanitizeCombinations(await response.json());
        persistLocalSavedCombinations();
        renderSavedCombinations();
      } catch (_) {}
    }

    async function importLegacyPortPresets() {
      if (selectedArtwork !== 'legacy') return;
      const origins = ['http://127.0.0.1:8765', 'http://localhost:8765'];
      let importedCount = 0;
      for (const origin of origins) {
        importedCount += await new Promise((resolve) => {
        const iframe = document.createElement('iframe');
        iframe.style.display = 'none';
        const timeout = setTimeout(done, 1600);
        function done(count = 0) {
          clearTimeout(timeout);
          window.removeEventListener('message', handleMessage);
          iframe.remove();
          resolve(count);
        }
        async function handleMessage(event) {
          if (event.origin !== origin) return;
          if (event.data?.source !== 'table-color-simulator-preset-export') return;
          const imported = sanitizeCombinations(event.data.presets || []);
          if (imported.length) {
            const before = savedCombinations.map(combinationSignature).join('\\n');
            savedCombinations = mergeCombinations(savedCombinations, imported);
            const after = savedCombinations.map(combinationSignature).join('\\n');
            if (after !== before) {
              renderSavedCombinations();
              await persistSavedCombinations();
            }
          }
          done(imported.length);
        }
        window.addEventListener('message', handleMessage);
        iframe.src = `${origin}/preset-export.html`;
        document.body.appendChild(iframe);
      });
      }
      return importedCount;
    }

    function currentCombination() {
      const item = { id: Date.now() };
      for (const key of areaKeys) {
        item[key] = state[key];
        item['o' + key] = state['o' + key];
      }
      return item;
    }

    function combinationSignature(item) {
      return areaKeys.flatMap((key) => [item[key], item['o' + key]]).join('|');
    }

    function renderSavedCombinations() {
      if (!savedCombinations.length) {
        savedRoot.innerHTML = '<div class="saved-empty">No custom presets yet.</div>';
        return;
      }
      savedRoot.innerHTML = savedCombinations.map((item, index) => {
        const swatches = areaKeys.map((key) => `<span class="saved-swatch" style="background:${item[key]}"></span>`).join('');
        const values = areaKeys.map((key) => `${labels[key]} ${item[key].toUpperCase()} ${item['o' + key]}%`).join(' · ');
        return `
          <div class="saved-item">
            <button class="saved-apply" type="button" data-saved-action="apply" data-saved-id="${item.id}">
              <span class="saved-swatches" aria-hidden="true">${swatches}</span>
              <span class="saved-copy">
                <span class="saved-name">Preset ${index + 1}</span>
                <span class="saved-values">${values}</span>
              </span>
            </button>
            <button class="saved-delete" type="button" data-saved-action="delete" data-saved-id="${item.id}">Delete</button>
          </div>
        `;
      }).join('');
    }

    function renderControls() {
      areaRoot.innerHTML = areaKeys.map((key) => `
        <section class="area">
          <div class="area-head">
            <div class="area-label">
              <span class="swatch" id="${key}-swatch" style="background:${state[key]}"></span>
              <span>${labels[key]}</span>
            </div>
            <input id="${key}-custom" type="color" value="${state[key]}" aria-label="Custom color for ${labels[key]}" />
          </div>
          <input id="${key}-code" class="color-code" type="text" value="${state[key].toUpperCase()}" spellcheck="false" maxlength="7" aria-label="Color code for ${labels[key]}" />
          <label>
            <span class="range-row"><span>Opacity</span><span id="${key}-opacity-text">${state['o' + key]}%</span></span>
            <span class="opacity-control">
              <input id="${key}-opacity" type="range" min="0" max="100" value="${state['o' + key]}" aria-label="Opacity for ${labels[key]}" />
              <input id="${key}-opacity-number" class="opacity-number" type="number" min="0" max="100" value="${state['o' + key]}" aria-label="Opacity percentage for ${labels[key]}" />
            </span>
          </label>
        </section>
      `).join('');

      for (const key of areaKeys) {
        document.getElementById(`${key}-custom`).addEventListener('input', (event) => {
          state[key] = event.target.value;
          activateColor(key);
          document.getElementById(`${key}-code`).value = state[key].toUpperCase();
          update();
        });
        document.getElementById(`${key}-code`).addEventListener('input', (event) => {
          const code = normalizeColorCode(event.target.value);
          event.target.classList.toggle('invalid', !code && event.target.value.length > 0);
          if (!code) return;
          state[key] = code;
          activateColor(key);
          event.target.value = state[key].toUpperCase();
          document.getElementById(`${key}-custom`).value = state[key];
          update();
        });
        const setOpacity = (value) => {
          state['o' + key] = clampOpacity(value);
          syncOpacityControls(key);
          update();
        };
        document.getElementById(`${key}-opacity`).addEventListener('input', (event) => setOpacity(event.target.value));
        document.getElementById(`${key}-opacity-number`).addEventListener('input', (event) => setOpacity(event.target.value));
      }
    }

    function query() {
      const params = new URLSearchParams();
      params.set('artwork', selectedArtwork);
      for (const key of areaKeys) {
        params.set(key, state[key]);
        params.set('o' + key, state['o' + key]);
      }
      params.set('masks', state.masks);
      params.set('_', String(Date.now()));
      return params.toString();
    }

    function update() {
      for (const key of areaKeys) {
        const swatch = document.getElementById(`${key}-swatch`);
        if (swatch) swatch.style.background = state[key];
      }
      img.src = `/render.png?${query()}`;
      download.href = `/download.png?${query()}`;
      hexRoot.innerHTML = areaKeys.map((key) => `
        <div class="hex-line">
          <strong>${labels[key]}</strong>
          <span>${state[key].toUpperCase()} · ${state['o' + key]}%</span>
        </div>
      `).join('');
    }

    document.getElementById('masks').addEventListener('change', (event) => {
      state.masks = event.target.checked ? 1 : 0;
      update();
    });
    document.getElementById('shuffle').addEventListener('click', () => {
      for (const key of areaKeys) {
        state[key] = pastelColors[Math.floor(Math.random() * pastelColors.length)];
        state['o' + key] = 100;
      }
      renderControls();
      update();
    });
    const legacyImportActions = document.getElementById('legacy-import-actions');
    const legacyImportButton = document.getElementById('import-legacy-presets');
    if (selectedArtwork !== 'legacy') {
      legacyImportActions.style.display = 'none';
    } else {
      legacyImportButton.addEventListener('click', async () => {
        legacyImportButton.textContent = 'Importing...';
        const count = await importLegacyPortPresets();
        legacyImportButton.textContent = count ? 'Imported' : 'No More Presets Found';
        setTimeout(() => { legacyImportButton.textContent = 'Import 8765 Presets'; }, 1400);
      });
    }
    document.getElementById('save-combination').addEventListener('click', async (event) => {
      const combination = currentCombination();
      const signature = combinationSignature(combination);
      savedCombinations = [
        combination,
        ...savedCombinations.filter((item) => combinationSignature(item) !== signature),
      ].slice(0, 12);
      renderSavedCombinations();
      await persistSavedCombinations();
      const button = event.currentTarget;
      button.textContent = 'Saved';
      setTimeout(() => { button.textContent = 'Save Preset'; }, 900);
    });
    savedRoot.addEventListener('click', async (event) => {
      const button = event.target.closest('[data-saved-action]');
      if (!button) return;
      const id = button.dataset.savedId;
      const item = savedCombinations.find((saved) => String(saved.id) === id);
      if (!item) return;
      if (button.dataset.savedAction === 'delete') {
        savedCombinations = savedCombinations.filter((saved) => String(saved.id) !== id);
        renderSavedCombinations();
        await persistSavedCombinations();
        return;
      }
      for (const key of areaKeys) {
        state[key] = item[key];
        state['o' + key] = clampOpacity(item['o' + key]);
      }
      state.masks = 0;
      document.getElementById('masks').checked = false;
      renderControls();
      update();
    });

    const compareButton = document.getElementById('compare-original');
    let comparingOriginal = false;
    function showOriginal() {
      if (comparingOriginal) return;
      comparingOriginal = true;
      img.src = `/original.jpg?artwork=${encodeURIComponent(selectedArtwork)}&_=${Date.now()}`;
    }
    function restoreCurrent() {
      if (!comparingOriginal) return;
      comparingOriginal = false;
      update();
    }
    compareButton.addEventListener('pointerdown', (event) => {
      event.preventDefault();
      showOriginal();
    });
    compareButton.addEventListener('pointerup', restoreCurrent);
    compareButton.addEventListener('pointerleave', restoreCurrent);
    compareButton.addEventListener('pointercancel', restoreCurrent);
    compareButton.addEventListener('keydown', (event) => {
      if (event.key === ' ' || event.key === 'Enter') showOriginal();
    });
    compareButton.addEventListener('keyup', (event) => {
      if (event.key === ' ' || event.key === 'Enter') restoreCurrent();
    });

    const presetRoot = document.getElementById('presets');
    presetRoot.innerHTML = Object.entries(presets).map(([key, preset]) =>
      `<button class="secondary" type="button" data-preset="${key}">${preset.name}</button>`
    ).join('');
    presetRoot.addEventListener('click', (event) => {
      const button = event.target.closest('[data-preset]');
      if (!button) return;
      const preset = presets[button.dataset.preset];
      for (const key of areaKeys) {
        state[key] = preset[key];
        state['o' + key] = clampOpacity(preset['o' + key]);
      }
      renderControls();
      update();
    });

    renderControls();
    renderSavedCombinations();
    loadServerSavedCombinations().then(importLegacyPortPresets);
    update();
  </script>
</body>
</html>"""
    return (
        template
        .replace("__ARTWORK_KEY__", artwork_key_json)
        .replace("__ARTWORKS__", artworks_json)
        .replace("__AREA_KEYS__", area_keys_json)
        .replace("__AREA_LABELS__", area_labels_json)
        .replace("__ORIGINAL_AREA_COLORS__", original_area_colors_json)
        .replace("__PRESETS__", presets_json)
        .encode("utf-8")
    )


class AppHandler(BaseHTTPRequestHandler):
    renderers: dict[str, ColorRenderer]

    def renderer_for_params(self, params: dict[str, list[str]]) -> tuple[str, ColorRenderer]:
        artwork_key = clean_artwork_key(params)
        return artwork_key, self.renderers[artwork_key]

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        artwork_key, renderer = self.renderer_for_params(params)
        if parsed.path in {"/", "/index.html"}:
            self.send_head_headers(len(html_page(renderer, artwork_key)), "text/html; charset=utf-8")
            return
        if parsed.path in {"/render.png", "/download.png"}:
            self.send_head_headers(0, "image/png")
            return
        if parsed.path == "/original.jpg":
            self.send_head_headers(renderer.image_path.stat().st_size, "image/jpeg")
            return
        if parsed.path == "/saved-presets.json":
            self.send_json(load_saved_presets(artwork_key, renderer), head_only=True)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        artwork_key, renderer = self.renderer_for_params(params)
        if parsed.path in {"/", "/index.html"}:
            self.send_bytes(html_page(renderer, artwork_key), "text/html; charset=utf-8")
            return
        if parsed.path in {"/render.png", "/download.png"}:
            image = renderer.render(params)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            headers = {}
            if parsed.path == "/download.png":
                headers["Content-Disposition"] = f'attachment; filename="table-color-simulation-{artwork_key}.png"'
            self.send_bytes(buffer.getvalue(), "image/png", headers=headers)
            return
        if parsed.path == "/original.jpg":
            self.send_file(renderer.image_path)
            return
        if parsed.path == "/saved-presets.json":
            self.send_json(load_saved_presets(artwork_key, renderer))
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        artwork_key, renderer = self.renderer_for_params(params)
        if parsed.path != "/saved-presets.json":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            presets = save_saved_presets(artwork_key, renderer, payload)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, OSError):
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid preset data")
            return
        self.send_json(presets)

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

    def send_json(self, payload: object, head_only: bool = False) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_head_headers(len(body), "application/json")
        if head_only:
            return
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
    sample_colors = ["#a9d8cf", "#b8d7f4", "#d6d2f2", "#f3d2df", "#f2d7ba", "#c8e6bd"]
    params = {}
    for index, key in enumerate(renderer.area_keys):
        params[key] = [sample_colors[index % len(sample_colors)]]
        params[f"o{key}"] = ["100"]
    image = renderer.render(params)
    image.save(output)


def build_renderers() -> dict[str, ColorRenderer]:
    renderers: dict[str, ColorRenderer] = {}
    for key, spec in ARTWORKS.items():
        image_path = spec["path"]
        if not image_path.exists():
            raise SystemExit(f"Missing {image_path}")
        renderers[key] = ColorRenderer(image_path, spec["mode"], spec["name"])
    return renderers


def main() -> None:
    parser = argparse.ArgumentParser(description="Local PIL app for recoloring table layout areas.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--export-sample", type=Path)
    args = parser.parse_args()

    renderers = build_renderers()
    if args.export_sample:
        export_sample(renderers[DEFAULT_ARTWORK_KEY], args.export_sample)
        print(f"Saved {args.export_sample}")
        return

    AppHandler.renderers = renderers
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
