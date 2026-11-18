"""Paths and constants of the Task 3 assistant.

Every path is derived from the location of this file, so the project runs on any machine
as long as the three task folders keep their relative layout:

    CW/
      datasets/   checkerboard/ (Task 1A photos), fruits/raw + fruits/splits (Task 2 data)
      task1/      Task 1 engine  (src/, artifacts/calibration.npz)
      task2/      Task 2 model   (src/, artifacts/model.pth)
      task3/      this application

A different layout can be configured with environment variables (no code change):
FCVG_TASK1_DIR, FCVG_TASK2_DIR, FCVG_DATASETS_DIR, FCVG_CALIBRATION_FILE, FCVG_MODEL_FILE.
"""
from __future__ import annotations

import os
from pathlib import Path

TASK3_DIR = Path(__file__).resolve().parents[1]
COURSEWORK_DIR = TASK3_DIR.parent


def _path(env_name: str, default: Path) -> Path:
    value = os.environ.get(env_name)
    return Path(value).expanduser().resolve() if value else default


TASK1_DIR = _path("FCVG_TASK1_DIR", COURSEWORK_DIR / "task1")
TASK2_DIR = _path("FCVG_TASK2_DIR", COURSEWORK_DIR / "task2")
DATASETS_DIR = _path("FCVG_DATASETS_DIR", COURSEWORK_DIR / "datasets")
FRUIT_SPLITS = DATASETS_DIR / "fruits" / "splits"
CALIBRATION_FILE = _path("FCVG_CALIBRATION_FILE", TASK1_DIR / "artifacts" / "calibration.npz")
MODEL_FILE = _path("FCVG_MODEL_FILE", TASK2_DIR / "artifacts" / "model.pth")

# Sample images offered in the UI (only folders that exist are listed).
SAMPLE_DIRS = {
    "Fruit test set (Task 2)": FRUIT_SPLITS / "test",
    "Checkerboard photos (Task 1A)": DATASETS_DIR / "checkerboard" / "calibration",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MAX_UPLOAD_MB = 25          # also set in .streamlit/config.toml
WORK_MAX_SIDE = 800         # analysis resolution for filtering / edges / features / segmentation
DISPLAY_MAX_SIDE = 900      # images shown in the UI are downscaled to this size
LOW_CONFIDENCE = 0.60       # below this softmax score the prediction is flagged as uncertain
