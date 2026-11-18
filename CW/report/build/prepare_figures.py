"""Copy the result figures of Tasks 1-3 into report/figures as compressed JPEGs (CMS limit: 2 MB per PDF).

    C:\\venvs\\fcvg101\\Scripts\\python.exe report/build/prepare_figures.py
"""
from pathlib import Path

from PIL import Image

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


def save(im: Image.Image, name: str, max_w: int = 1000, q: int = 50):   # CMS limit: 2 MB per PDF
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
    # Dataset images themselves are not put in the report; they are in CW/datasets/ in the repository.
    print(f"TOTAL {total / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
