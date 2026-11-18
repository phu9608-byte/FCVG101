# FCVG101 – Task 1: Core Image Analysis Engine (FruitSort Vision Assistant)

Classical computer-vision engine of the FruitSort project: camera calibration on checkerboard
photos (1A), filtering / edges / Fourier analysis on fruit images (1B) and interest-point
detection and matching (1C). The algorithms live in `src/` so that the Task 3 app can import
them; each notebook runs the experiments, saves figures and tables to `results/`, and prints
an auto-generated summary from the actual numbers.

## Structure

```
task1/
├── 1A_calibration.ipynb          # pinhole model, Zhang from scratch vs OpenCV, model selection, undistortion
├── 1B_filtering_edges_fft.ipynb  # convolution from scratch, denoising, Sobel/Canny, FFT filters, notch filter
├── 1C_features_matching.ipynb    # Harris from scratch, SIFT/ORB/Harris+SIFT, ratio test, RANSAC
├── src/
│   ├── calibration.py   # corner detection, Zhang's method (DLT + closed form + LM), CV model selection, undistort
│   ├── filtering.py     # correlate/convolve/median from scratch, noise, PSNR/SSIM, Sobel, Canny, FFT masks
│   └── features.py      # Harris from scratch, detectors/descriptors, ratio test, RANSAC, ground-truth evaluation
├── artifacts/calibration.npz     # K, dist, image size (written by 1A, used by Task 3)
├── results/1A, 1B, 1C/           # figures (.png), tables (.csv), metrics.json
└── requirements.txt
```

## Data

| Notebook | Input | Location |
|---|---|---|
| 1A | 15 checkerboard photos (9×6 inner corners), one phone, 4000×3000 | `../datasets/checkerboard/calibration/` |
| 1A | independent test photo + floor-tile photo, same phone | `../datasets/checkerboard/mau/`, `../datasets/checkerboard/check_test.jpg` |
| 1B, 1C | fruit images (Kaggle dataset used in Task 2) | `../datasets/fruits/raw/` |

## Run

```powershell
cd CW\task1
python -m pip install -r requirements.txt     # inside the project environment (see ../README.md)
```

Open the notebooks with the working directory set to `task1/` (VS Code / Jupyter)
and run all cells in order 1A → 1B → 1C. Total run time is about 1 minute.

## Reusing the engine (Task 3)

```python
from src import calibration, filtering, features

calib = calibration.load_calibration("artifacts/calibration.npz")
undistorted, _, _ = calibration.undistort_image(img, calib["K"], calib["dist"], alpha=0)
edges, thresholds = filtering.canny_auto(filtering.to_gray(img))
contour, mask, boundary = filtering.fruit_outline(img)
result = features.evaluate_pair(gray1, gray2, "SIFT")      # keypoints, ratio-test matches, RANSAC inliers
```

`K`/`dist` only apply to photos from the same phone at 4000×3000; scale `fx, fy, cx, cy` if the
image is resized.
