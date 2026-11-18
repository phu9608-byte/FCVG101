"""Generate the Task 3 figures and routing table used in the report.

Every figure is produced by the assistant's own code path:
    detect_intent(request) -> tools.run(mode, images, params) -> ToolResult
so the panels are exactly what the Streamlit app shows for that request.

Run with the Task 3 environment from the CourseWork folder:
    C:\\venvs\\fcvg101\\Scripts\\python.exe report/build/make_task3_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COURSEWORK = Path(__file__).resolve().parents[2]   # the CW folder
TASK3 = COURSEWORK / "task3"
sys.path.insert(0, str(TASK3))

from assistant import engines, settings, tools  # noqa: E402
from assistant.intent import MODES, detect_intent  # noqa: E402

OUT = COURSEWORK / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
TEST = settings.FRUIT_SPLITS / "test"
BOARDS = settings.SAMPLE_DIRS["Checkerboard photos (Task 1A)"]


def img(path: Path):
    return tools.read_image_file(path)


def fruit(cls: str, i: int = 0) -> Path:
    return sorted((TEST / cls).glob("*.jpg"))[i]


def rgb(a):
    return a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def figure(name: str, request: str, images, clf, calib, max_panels: int = 4, extra_params=None, pick=None):
    intent = detect_intent(request)
    params = {**(extra_params or {}), **intent.params}
    res = tools.run(intent.mode, images, params, clf=clf, calib=calib)
    panels = [res.images[i] for i in pick] if pick else res.images[:max_panels]
    aspect = sum(im.shape[0] / im.shape[1] for _, im in panels) / len(panels)
    width = min(14.0, 3.6 * len(panels) / max(aspect, 0.6) * 0.75)
    fig, axes = plt.subplots(1, len(panels), figsize=(width, width / len(panels) * aspect + 1.0),
                             layout="constrained")
    axes = [axes] if len(panels) == 1 else axes
    for ax, (caption, im) in zip(axes, panels):
        ax.imshow(rgb(im), cmap="gray" if im.ndim == 2 else None)
        ax.set_title(caption, fontsize=8, wrap=True)
        ax.axis("off")
    head = f'Request: "{request}"  ->  mode: {intent.mode}  ({intent.reason})'
    if intent.params:
        head += "  | params: " + ", ".join(f"{k}={v}" for k, v in intent.params.items())
    summary = res.summary.replace("**", "")
    fig.suptitle(head + "\n" + summary, fontsize=8.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(OUT / f"{name}.png", dpi=130, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return {"figure": name, "request": request, "mode": intent.mode, "reason": intent.reason,
            "params": {k: str(v) for k, v in intent.params.items()}, "summary": summary,
            "metrics": {k: str(v) for k, v in res.metrics.items()},
            "table": res.table, "warnings": res.warnings, "elapsed_ms": round(res.elapsed_ms, 1)}


def main():
    clf = engines.load_classifier()
    calib = engines.load_calibration()
    log = []
    log.append(figure("t3_classify", "Phân loại quả này giúp tôi", [img(fruit("apple", 9))], clf, calib, 2))
    log.append(figure("t3_inspect", "Inspect this fruit: classify and measure it", [img(fruit("orange", 2))],
                      clf, calib, 2))
    log.append(figure("t3_filter", "Lọc nhiễu muối tiêu bằng median 5x5", [img(fruit("orange", 12))], clf, calib, 3))
    log.append(figure("t3_fft", "Low-pass filter in the frequency domain, cutoff 25", [img(fruit("banana", 4))],
                      clf, calib, 4))
    log.append(figure("t3_edges", "Find the edges with Canny and Sobel", [img(fruit("orange", 0))], clf, calib, 4))
    boards = sorted(BOARDS.glob("*.jpg"))
    log.append(figure("t3_features", "So khớp SIFT giữa hai ảnh", [img(boards[0]), img(boards[1])], clf, calib,
                      pick=[-1]))
    log.append(figure("t3_calibrate", "Khử méo ảnh này", [img(boards[2])], clf, calib, 3))

    requests = [
        "phân loại quả này", "What fruit is this?", "tìm cạnh bằng Canny", "find the edges, thresholds 50 150",
        "lọc nhiễu muối tiêu bằng median 7x7", "apply a low pass FFT filter", "so khớp ORB giữa hai ảnh, ratio 0.8",
        "detect Harris corners", "khử méo ảnh này", "show the camera intrinsic matrix K",
        "đo kích thước quả ở 30 cm", "how big is this fruit?", "phân loại và đo kích thước", "hello there",
    ]
    routing = []
    for r in requests:
        it = detect_intent(r)
        routing.append({"request": r, "mode": it.mode or "(ask user)", "reason": it.reason,
                        "params": ", ".join(f"{k}={v}" for k, v in it.params.items())})
    (OUT / "t3_log.json").write_text(json.dumps({"runs": log, "routing": routing}, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print(json.dumps({"runs": log, "routing": routing}, ensure_ascii=False, indent=1)[:6000])


if __name__ == "__main__":
    main()
