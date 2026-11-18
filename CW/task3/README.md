# FCVG101 – Task 3: FruitSort Vision Assistant (application integration / tool use)

A Streamlit app that links the **Task 1 classical CV engine** (`../task1/src`) and the
**Task 2 CNN** (`../task2`). The user picks
an image and asks in English or Vietnamese ("phân loại quả này", "find the edges with Canny").
The assistant recognises the task, runs the matching module and shows figures and numbers.
Nothing from Task 1/2 is copied: the app imports and calls their code directly.

## 1. Install (one environment for Tasks 1–3)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate        macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, if `pip install torch` fails with a *Long Path* error, create the venv in a short
path (e.g. `python -m venv C:\venvs\fcvg101`). A CPU build of PyTorch is enough, because the CNN is tiny
(98 KB). Classification plus Grad-CAM takes under 0.1 s per image in the app.

Tested with Python 3.12.10, Streamlit 1.67, PyTorch 2.14.1, OpenCV 5.1, Windows 11.

## 2. Run

```bash
cd task3
streamlit run app.py
```

The app opens at <http://localhost:8501>. Required artifacts (produced by Tasks 1 and 2):

| Artifact | Default location | Produced by |
|---|---|---|
| Calibration `K`, `dist` | `../task1/artifacts/calibration.npz` | `task1/1A_calibration.ipynb` |
| Trained model | `../task2/artifacts/model.pth` + `classes.json` | `task2/Task2_pipeline.ipynb` |
| Sample images | `../datasets/fruits/splits/test/`, `../datasets/checkerboard/calibration/` | Task 2 split, Task 1A photos |

If the folders are laid out differently, set `FCVG_TASK1_DIR`, `FCVG_TASK2_DIR`, `FCVG_DATASETS_DIR`,
`FCVG_CALIBRATION_FILE` or `FCVG_MODEL_FILE`. The code contains no machine-specific paths.
If an artifact is missing, the app still starts. The sidebar shows what is missing and how to fix it,
the tools that need that artifact answer with the same hint, and every other tool keeps working.

## 3. Using the assistant

1. Choose **Upload** (JPG/PNG/BMP/TIFF/WEBP, ≤ 25 MB) or **Sample images** (fruit test split,
   checkerboard photos).
2. Type a request in the chat box, or press a quick-action button.
3. Optional: force a mode or change parameters in the sidebar. Values written in the request,
   such as `median 7x7`, `ORB`, `ratio 0.8`, `threshold 50 150`, `at 30 cm`, override the sidebar.

| Mode | Example request | Module called | Output |
|---|---|---|---|
| Calibrate / undistort | "khử méo ảnh này" | `task1/src/calibration.py`: `undistort_image`, `detect_corners`, `line_straightness` | undistorted image, shift map, scaled `K`, line straightness before/after; with ≥ 3 checkerboards a new `calibrateCamera` run |
| Filter | "lọc nhiễu muối tiêu bằng median 5x5" | `filtering.apply_filter`, `add_*noise`, `quality`, FFT masks | filtered image, residual, MSE/PSNR/SSIM table for all filters; spectrum + mask for FFT |
| Edges | "tìm cạnh bằng Canny" | `filtering.sobel`, `canny_auto`, `fruit_outline` | Sobel magnitude/orientation, Canny map, fruit outline, circularity |
| Match | "so khớp SIFT giữa hai ảnh" | `features.evaluate_pair`, `synthetic_view`, `draw_matches` | keypoints, ratio-test matches, RANSAC inliers; with one image a synthetic view with known H gives match precision |
| Classify | "đây là quả gì?" | `task2/src/inference.predict_pil` + Grad-CAM | class, score per class, Grad-CAM heat map |
| Measure | "đo kích thước quả ở 30 cm" | calibration `K` + `fruit_outline` | size in px; in mm via `size_mm = size_px · Z / f` when Z is given and the photo comes from the calibrated camera |
| Inspect | "phân loại và đo kích thước" | Classify + Measure | both results |

## 4. How it works

```
app.py                   Streamlit UI: image input, chat, sidebar status/parameters, result rendering
assistant/settings.py    relative paths to Task 1/2 code + artifacts (overridable by env vars), limits
assistant/engines.py     imports task1/src as package `fcvg_task1` (both tasks name their package `src`),
                         imports Task 2's `src.inference`, loads calibration.npz and model.pth once
assistant/intent.py      rule-based intent router: strip Vietnamese diacritics, weighted keyword
                         patterns per mode -> best mode (classify+measure -> inspect; tie/no match -> ask),
                         plus parameter extraction (filter, kernel, SIFT/ORB, ratio, thresholds, distance)
assistant/tools.py       one function per mode -> ToolResult(summary, images, metrics, table, warnings);
                         input validation (non-image, empty, too large, wrong aspect ratio, missing artifact)
assistant/gradcam.py     Grad-CAM on the last conv layer of FruitCNN
Task3_assistant.ipynb    engines, intent router, one demo per tool, integration tests T1–T7 + routing + headless UI test
```

* **Application logic (3.1).** Every request goes through `detect_intent`. The chat shows the chosen
  mode and the keywords that triggered it. Unclear requests are answered with the closest
  candidates, and nothing runs. The sidebar can force a mode.
* **Tool integration (3.2).** `tools.run(mode, …)` dispatches to the Task 1 / Task 2 code. Models
  and calibration are cached with `st.cache_resource`, so they load once per server process.
  Errors become a message with a fix hint and never crash the app.

## 5. Notebook and tests

Open `Task3_assistant.ipynb` with the working directory set to `task3/` and run all cells. It loads the
engines, shows how requests are routed, runs every tool on sample images and ends with the integration tests
(last run: **12/12 passed**).

| Case | Check | Result |
|---|---|---|
| T1 Filtering | median 5×5 on 5 % salt & pepper | PSNR 17.93 → 27.22 dB, best of 4 filters |
| T2 Edges | auto and manual Canny thresholds | edge map + thresholds shown |
| T3 Features | real checkerboard pair, SIFT / ORB / Harris+SIFT | SIFT 172 matches / 81 inliers; synthetic view precision 0.97 |
| T4 Classify | whole Task 2 test split through the app | 37/38 = 97.4 % (same as the Task 2 evaluation) |
| T5 Bad input | text file renamed `.jpg`, empty file, PDF, no image | clear message + hint |
| T6 Missing artifacts | missing `model.pth` / `calibration.npz` | message + how to fix; measurement falls back to pixels |
| T7 Clean machine | no absolute paths in code, env-var overrides | pass |
| Extra | undistortion straightens board lines | 0.95 → 0.26 px RMS |
| Extra | new calibration from 6 board photos | RMS 0.137 px, fx within 1.2 % of Task 1A |
| Extra | 19 English/Vietnamese requests routed | all correct |
| UI | 5 chat requests + quick action via `streamlit.testing` AppTest | no exceptions |

## 6. Limitations

* The intent router is keyword-based. It is transparent and needs no internet or API key, but
  it does not understand arbitrary paraphrases. Unknown requests are answered with suggestions.
* `K`/`dist` belong to one phone at 4000×3000 in landscape orientation. Other cameras, cropped
  or portrait photos are refused for undistortion, and measurement then stays in pixels.
* Metric size assumes a roughly fronto-parallel fruit at a known distance. The classical
  segmentation expects one fruit on a plain light background. Clutter is flagged
  (solidity / region count) instead of being measured silently.
* The CNN only knows apple, banana and orange, and its softmax score is not a calibrated probability.
