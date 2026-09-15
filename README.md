# Table Color Simulator

A local Python and Pillow web app for testing color combinations on a table layout. It supports the original segmented A/B/C layout and full-background layouts such as rose/purple source art. Printed betting artwork keeps its source colors, and the C-area line pattern is redrawn in soft white whenever C is recolored on segmented artwork.

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
- `original-hd.jpg`: high-resolution source layout.
- `manual-table.png`: manually marked table and C-area reference.
- `manual-b.png`: manually marked B-area reference.
- `pyproject.toml` and `uv.lock`: reproducible Python dependencies.

## Controls

- Pick area colors with the color picker or enter a six-digit hex code.
- Adjust color opacity from 0% to 100%.
- Save up to 12 custom presets in the browser and reapply them later.
- Hold Compare Original to temporarily view the untouched source image.
- Use Original to return editable areas to their untouched source colors.
- Download the current result as a full-resolution PNG.
