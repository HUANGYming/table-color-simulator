# Table Color Simulator

A local Python and Pillow web app for testing color combinations across multiple table layouts in one service. The interface can switch between segmented A/B/C artwork, full-background artwork, and Rev23 multi-region artwork while keeping printed betting artwork protected.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.11 or newer (installed automatically by uv when needed)

## Start Locally

```bash
uv sync --locked
uv run python app.py
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765).

To make the service available to other devices on the same network:

```bash
uv run python app.py --host 0.0.0.0 --port 8765
```

## Project Files

- `app.py`: web interface, mask extraction, and Pillow rendering logic.
- `artwork-legacy.jpg`: original segmented A/B/C source layout.
- `artwork-rose-purple.jpg`: rose/purple full-background source layout.
- `artwork-cyan.jpg`: cyan full-background source layout.
- `artwork-rev23.jpg`: Rev23 multi-region source layout.
- `original-hd.jpg`: default source layout used by older single-image runs.
- `manual-table.png`: manually marked table and C-area reference.
- `manual-b.png`: manually marked B-area reference.
- `pyproject.toml` and `uv.lock`: reproducible Python dependencies.

## Controls

- Choose an artwork from the Artwork dropdown.
- Pick editable region colors with the color picker or enter a six-digit hex code.
- Adjust color opacity from 0% to 100%.
- Save up to 12 custom presets per artwork and reapply them later.
- Hold Compare Original to temporarily view the untouched source image.
- Use Original to return editable regions to their untouched source colors.
- Download the current result as a full-resolution PNG.
