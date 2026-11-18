"""Load the two engines the assistant integrates.

* Task 1 lives in ``task1/src`` and Task 2 in ``task2/src``: both packages are
  called ``src``. Task 2's own code imports ``src.model`` absolutely, so Task 2 keeps the
  name ``src``; Task 1 is imported under the alias ``fcvg_task1`` to avoid the clash.
* Nothing is copied: the app runs exactly the code that produced the Task 1 / Task 2 results.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import settings
from .errors import AssistantError

TASK1_ALIAS = "fcvg_task1"


# ----------------------------------------------------------------------------
# Task 1 - classical image analysis engine
# ----------------------------------------------------------------------------
def load_task1(task1_dir: Path | None = None) -> SimpleNamespace:
    """Import ``task1/src`` as package ``fcvg_task1``; returns its calibration / filtering / features modules."""
    src = Path(task1_dir or settings.TASK1_DIR) / "src"
    required = [src / f"{name}.py" for name in ("calibration", "filtering", "features")]
    missing = [p.name for p in required if not p.is_file()]
    if missing:
        raise AssistantError(
            f"Task 1 source code not found (missing {', '.join(missing)} in {src}).",
            "Keep the task1/ folder next to task3/, or set the FCVG_TASK1_DIR environment variable.",
        )
    if TASK1_ALIAS not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            TASK1_ALIAS, src / "__init__.py", submodule_search_locations=[str(src)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[TASK1_ALIAS] = package
        spec.loader.exec_module(package)
    return SimpleNamespace(
        calibration=importlib.import_module(f"{TASK1_ALIAS}.calibration"),
        filtering=importlib.import_module(f"{TASK1_ALIAS}.filtering"),
        features=importlib.import_module(f"{TASK1_ALIAS}.features"),
    )


@dataclass
class Calibration:
    """Intrinsics from Task 1A, valid for one camera at one resolution."""

    K: np.ndarray
    dist: np.ndarray
    size: tuple[int, int]          # (width, height) of the calibration photos
    rms: float | None
    path: Path

    def for_image(self, width: int, height: int, tolerance: float = 0.01) -> np.ndarray:
        """K rescaled to an image of the same camera that was resized (aspect ratio must match).

        fx, cx scale with the width ratio; fy, cy with the height ratio.
        """
        sx, sy = width / self.size[0], height / self.size[1]
        if abs(sx - sy) > tolerance * max(sx, sy):
            raise AssistantError(
                f"Image is {width}x{height} but the camera was calibrated at {self.size[0]}x{self.size[1]} "
                f"(aspect ratio {width / height:.3f} vs {self.size[0] / self.size[1]:.3f}).",
                "K and the distortion coefficients only apply to photos from the calibrated phone in the same "
                "orientation (landscape 4:3). Rotate/crop-free originals work; cropped or portrait photos do not.",
            )
        K = self.K.astype(float).copy()
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
        return K


def load_calibration(path: Path | None = None) -> Calibration:
    """Read ``calibration.npz`` written by Task 1A (via Task 1's own ``load_calibration``)."""
    path = Path(path or settings.CALIBRATION_FILE)
    if not path.is_file():
        raise AssistantError(
            f"Calibration file not found: {path}",
            "Run notebook task1/1A_calibration.ipynb (it writes artifacts/calibration.npz), "
            "or set FCVG_CALIBRATION_FILE to an existing file.",
        )
    data = load_task1().calibration.load_calibration(path)
    if "K" not in data or "dist" not in data:
        raise AssistantError(f"{path.name} has no K / dist arrays.", "Re-run notebook 1A to rewrite the file.")
    return Calibration(
        K=np.asarray(data["K"], float),
        dist=np.asarray(data["dist"], float).ravel(),
        size=(int(data["width"]), int(data["height"])),
        rms=float(data["rms"]) if "rms" in data else None,
        path=path,
    )


# ----------------------------------------------------------------------------
# Task 2 - deep-learning classifier
# ----------------------------------------------------------------------------
@dataclass
class Classifier:
    """Trained FruitCNN plus the exact preprocessing used in training."""

    model: object
    idx_to_class: dict
    transform: object
    predict_pil: object            # Task 2's own inference function
    meta: dict = field(default_factory=dict)
    path: Path | None = None

    @property
    def classes(self) -> list[str]:
        return [self.idx_to_class[i] for i in sorted(self.idx_to_class)]


def _import_task2(task2_dir: Path):
    """Import Task 2's ``src`` package, making sure another package called ``src`` is not picked up."""
    src = task2_dir / "src"
    if not (src / "inference.py").is_file() or not (src / "model.py").is_file():
        raise AssistantError(
            f"Task 2 source code not found in {src}.",
            "Keep the task2 folder next to task3/, or set the FCVG_TASK2_DIR environment variable.",
        )
    cached = sys.modules.get("src")
    if cached is not None and Path(getattr(cached, "__file__", "") or "").resolve().parent != src.resolve():
        for name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
            del sys.modules[name]
    if str(task2_dir) not in sys.path:
        sys.path.insert(0, str(task2_dir))
    return importlib.import_module("src.inference")


def load_classifier(model_path: Path | None = None, task2_dir: Path | None = None) -> Classifier:
    """Load ``model.pth`` with Task 2's ``load_model`` (CPU; the CNN is tiny)."""
    task2_dir = Path(task2_dir or settings.TASK2_DIR)
    model_path = Path(model_path or settings.MODEL_FILE)
    if not model_path.is_file():
        raise AssistantError(
            f"Trained model artifact not found: {model_path}",
            "Train it with notebook task2/Task2_pipeline.ipynb (writes artifacts/model.pth), or download the "
            "shared model.pth into that folder, or set FCVG_MODEL_FILE.",
        )
    try:
        inference = _import_task2(task2_dir)
    except ModuleNotFoundError as exc:          # torch / torchvision not installed
        raise AssistantError(f"Cannot import the Task 2 model code: {exc}",
                             "Install the dependencies: python -m pip install -r CW/requirements.txt") from exc
    try:
        model, idx_to_class, transform = inference.load_model(str(model_path))
        checkpoint = inference.read_checkpoint(str(model_path))
    except Exception as exc:                    # corrupt file / architecture mismatch
        raise AssistantError(f"Could not load {model_path.name}: {exc}",
                             "The checkpoint must come from task2/Task2_pipeline.ipynb (same FruitCNN definition).") from exc
    meta = {k: v for k, v in checkpoint.items() if k != "model_state"}
    return Classifier(model, idx_to_class, transform, inference.predict_pil, meta, model_path)
