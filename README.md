# Table Color Simulator

A local Python and Pillow web app for testing color combinations on the A, B, and C background areas of a table layout. Printed betting artwork keeps its source colors, and the C-area line pattern is redrawn in soft white whenever C is recolored.

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

- Pick A, B, and C colors with the color picker or enter a six-digit hex code.
- Adjust overlay strength from 0% to 70%.
- Use Original to return to the untouched source image.
- Download the current result as a full-resolution PNG.
