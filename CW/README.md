# FCVG101 Coursework – FruitSort Vision Assistant

A visual inspection assistant for fruit sorting: camera calibration and classical image analysis (Task 1),
a data pipeline and a CNN fruit classifier (Task 2), and a tool-using assistant app that runs both (Task 3).

Repository: <https://github.com/phu9608-byte/FCVG101>

## Structure

```
CW/
├── datasets/
│   ├── checkerboard/      # 15 calibration photos (calibration/), independent test photo (mau/), floor tiles (check_test.jpg)
│   └── fruits/            # Kaggle "Fruit Images for Object Detection" (CC0): raw/ (255 cleaned images), splits/ (train/val/test)
├── task1/                 # 1A_calibration.ipynb, 1B_filtering_edges_fft.ipynb, 1C_features_matching.ipynb + src/ engine
├── task2/                 # Task2_pipeline.ipynb + src/ (model, preprocessing, inference) + artifacts/model.pth
├── task3/                 # Task3_assistant.ipynb + Streamlit app (app.py, assistant/)
└── report/                # FCVG101_Coursework_Report.pdf / .docx, figures, build scripts
```

| Task | Notebook | Main result |
|---|---|---|
| 1A | `task1/1A_calibration.ipynb` | K, dist from Zhang's method (scratch = OpenCV); RMS 0.458 px |
| 1B | `task1/1B_filtering_edges_fft.ipynb` | filters, Sobel/Canny, FFT and notch filtering with PSNR/SSIM/F1 |
| 1C | `task1/1C_features_matching.ipynb` | Harris (scratch), SIFT/ORB, ratio test, RANSAC; SIFT precision 97.4 % |
| 2 | `task2/Task2_pipeline.ipynb` | FruitCNN, test accuracy 97.4 %, macro-F1 0.973 |
| 3 | `task3/Task3_assistant.ipynb` | intent router + 7 tools; 12/12 integration tests |

Trained model artifact: [`task2/artifacts/model.pth`](task2/artifacts/model.pth) (≈ 98 KB, with `classes.json`).

## Setup (Windows, Python 3.12)

```powershell
python -m venv C:\venvs\fcvg101            # a short path avoids the Windows "Long Path" error of PyTorch
C:\venvs\fcvg101\Scripts\Activate.ps1
cd CW
python -m pip install -r task1\requirements.txt -r task2\requirements.txt -r task3\requirements.txt
python -m pip install ipykernel
python -m ipykernel install --user --name fcvg101 --display-name "Python (fcvg101)"
```

## Run order

1. **Task 1** – open `task1/` and run `1A` → `1B` → `1C` (1A writes `task1/artifacts/calibration.npz`).
2. **Task 2** – open `task2/Task2_pipeline.ipynb` and run all cells (set `RUN_TRAINING = True` to retrain).
3. **Task 3** – open `task3/Task3_assistant.ipynb` and run all cells, or start the GUI:

   ```powershell
   cd task3
   streamlit run app.py
   ```

Each notebook expects its own task folder as the working directory. Every task folder has its own `README.md`.
