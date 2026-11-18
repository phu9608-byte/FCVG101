"""Copy the result figures of Tasks 1-3 into report/figures as compressed JPEGs (PDF limit: 5 MB).

    C:\\venvs\\fcvg101\\Scripts\\python.exe report/build/prepare_figures.py
"""
from pathlib import Path
import glob
from PIL import Image, ImageDraw

CW = Path(__file__).resolve().parents[2]
OUT = CW / "report" / "figures" / "jpg"
OUT.mkdir(parents=True, exist_ok=True)
T1 = CW / "task1" / "results"
T2 = CW / "task2"
T3F = CW / "report" / "figures"

FIGS = {
    # Task 1A
    "1a_corners": T1 / "1A/corners_overlay.png", "1a_coverage": T1 / "1A/corner_coverage.png",
    "1a_reproj": T1 / "1A/reprojection_error.png", "1a_modelcv": T1 / "1A/distortion_model_cv.png",
    "1a_convergence": T1 / "1A/convergence.png", "1a_field": T1 / "1A/distortion_field.png",
    "1a_undistort": T1 / "1A/undistort_sample.png", "1a_straight": T1 / "1A/straightness.png",
    "1a_tiles": T1 / "1A/undistort_floor_tiles.png",
    # Task 1B
    "1b_kernels": T1 / "1B/kernels.png", "1b_denoise_grid": T1 / "1B/denoising_grid.png",
    "1b_psnr_k": T1 / "1B/denoising_psnr_vs_k.png", "1b_sobel_canny": T1 / "1B/sobel_canny.png",
    "1b_canny_thr": T1 / "1B/canny_thresholds.png", "1b_edges_noise": T1 / "1B/edges_under_noise.png",
    "1b_outline": T1 / "1B/fruit_outline.png", "1b_fft": T1 / "1B/fft_filters.png",
    "1b_ringing": T1 / "1B/ringing_profile.png", "1b_notch": T1 / "1B/periodic_noise_notch.png",
    # Task 1C
    "1c_harris": T1 / "1C/harris_scratch.png", "1c_matches": T1 / "1C/matches_methods.png",
    "1c_ratio": T1 / "1C/ratio_sweep.png", "1c_robust": T1 / "1C/robustness.png",
    "1c_ransac": T1 / "1C/ransac_alignment.png", "1c_unrelated": T1 / "1C/related_vs_unrelated.png",
    # Task 2
    "2_curves": T2 / "results/learning_curves.png", "2_cm": T2 / "results/confusion_matrix.png",
    # Task 3 (made by make_task3_figures.py)
    **{p.stem: p for p in sorted(T3F.glob("t3_*.png"))},
}


def save(im: Image.Image, name: str, max_w: int = 1400, q: int = 72):
    im = im.convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    path = OUT / f"{name}.jpg"
    im.save(path, quality=q, optimize=True, progressive=True)
    return path


def main():
    total = 0
    for name, src in FIGS.items():
        im = Image.open(src)
        if name == "t3_ui_home":
            im = im.crop((0, 0, 1040, im.height))          # drop a Streamlit promo toast at the top right
        p = save(im, name)
        total += p.stat().st_size
        print(f"{name:16s} {Image.open(p).size} {p.stat().st_size // 1024} KB")
    # near-duplicate review sheet: keep the first rows only (cross-split pairs come first)
    nd = Image.open(T2 / "results/near_duplicates.png")
    p = save(nd.crop((0, 0, nd.width, min(nd.height, 970))), "2_near_dup")
    total += p.stat().st_size
    # dataset sample grid: 6 training images per class
    cls = ["apple", "banana", "orange"]
    cell = 150
    grid = Image.new("RGB", (6 * cell, 3 * (cell + 18)), "white")
    d = ImageDraw.Draw(grid)
    for r, c in enumerate(cls):
        files = sorted(glob.glob(str(CW / f"datasets/fruits/splits/train/{c}/*.jpg")))[::9][:6]
        for i, f in enumerate(files):
            t = Image.open(f).convert("RGB")
            t.thumbnail((cell - 6, cell - 6))
            grid.paste(t, (i * cell + (cell - t.width) // 2, r * (cell + 18) + 18 + (cell - t.height) // 2))
        d.text((4, r * (cell + 18) + 2), f"class: {c}", fill="black")
    p = save(grid, "2_samples")
    total += p.stat().st_size
    print(f"TOTAL {total / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
