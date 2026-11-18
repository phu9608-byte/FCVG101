"""The assistant's tools: one function per mode (rubric 3.2 - tool integration).

Each tool receives decoded images (OpenCV BGR uint8) plus parameters, calls the Task 1 engine
(``task1/src``) or the Task 2 model (``task2/src``) and returns a :class:`ToolResult`
holding figures, numbers and warnings. The tools know nothing about Streamlit, so they can be
tested headless (see ``tests/run_integration_tests.py``) and reused by any other interface.
"""
from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import gradcam, settings
from .engines import Calibration, Classifier, load_task1
from .errors import AssistantError


@dataclass
class ToolResult:
    tool: str
    summary: str
    images: list[tuple[str, np.ndarray]] = field(default_factory=list)   # (caption, BGR or grey uint8)
    metrics: dict = field(default_factory=dict)
    table: list[dict] | None = None
    warnings: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0


# ----------------------------------------------------------------------------
# Input handling
# ----------------------------------------------------------------------------
def decode_image(data: bytes, name: str = "upload") -> np.ndarray:
    """Bytes of an uploaded file -> BGR uint8 image; friendly error for anything that is not an image."""
    if not data:
        raise AssistantError(f"'{name}' is empty.", "Upload a JPG or PNG photo.")
    if len(data) > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise AssistantError(f"'{name}' is larger than {settings.MAX_UPLOAD_MB} MB.", "Use a smaller photo.")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise AssistantError(f"'{name}' is not a readable image.",
                             "Only image files (JPG, PNG, BMP, TIFF, WEBP) are accepted. The file may be corrupt "
                             "or have a wrong extension.")
    if img.dtype != np.uint8:                                   # 16-bit PNG/TIFF -> 8-bit
        img = cv2.convertScaleAbs(img, alpha=255.0 / max(float(img.max()), 1.0))
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:                                     # drop alpha channel
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if min(img.shape[:2]) < 32:
        raise AssistantError(f"'{name}' is only {img.shape[1]}x{img.shape[0]} px.", "Use an image of at least 32x32 px.")
    return img


def read_image_file(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise AssistantError(f"File not found: {path}")
    return decode_image(path.read_bytes(), path.name)


def resize_max_side(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s == 1.0:
        return img, 1.0
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA), s


def _need(images: list, n: int, mode: str) -> None:
    if len(images) < n:
        raise AssistantError(f"Mode '{mode}' needs {n} image(s), got {len(images)}.",
                             "Upload an image (or choose a sample image) first.")


def _colour(gray: np.ndarray, cmap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    return cv2.applyColorMap(gray, cmap)


# ----------------------------------------------------------------------------
# Calibrate / undistort  (Task 1A)
# ----------------------------------------------------------------------------
def calibrate(images: list[np.ndarray], calib: Calibration | None, alpha: float = 0.0,
              pattern: tuple[int, int] = (9, 6), square_mm: float = 18.0) -> ToolResult:
    """One photo  -> undistort it with the stored K/dist and verify on a checkerboard if one is visible.
    >= 3 photos -> run a fresh checkerboard calibration (cv2.calibrateCamera via Task 1) and compare with K."""
    _need(images, 1, "calibrate")
    t1 = load_task1()
    if len(images) >= 3:
        return _calibrate_new(images, calib, pattern, square_mm)
    if calib is None:
        raise AssistantError("No calibration loaded.", "Run notebook task1/1A_calibration.ipynb or upload >= 3 "
                                                       "checkerboard photos to calibrate now.")
    img = images[0]
    h, w = img.shape[:2]
    K = calib.for_image(w, h)
    undistorted, newK, _roi = t1.calibration.undistort_image(img, K, calib.dist, alpha=alpha)

    # how far undistortion moves each pixel (visualises the lens model)
    ys, xs = np.mgrid[0:h:max(1, h // 60), 0:w:max(1, w // 80)]
    grid = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    moved = t1.calibration.undistort_points_px(grid, K, calib.dist)
    shift = np.linalg.norm(moved - grid, axis=1).reshape(xs.shape)
    shift_map = cv2.resize(np.uint8(255 * shift / max(shift.max(), 1e-9)), (w, h), interpolation=cv2.INTER_LINEAR)

    metrics = {
        "image_size": f"{w} x {h}",
        "calibrated_at": f"{calib.size[0]} x {calib.size[1]}",
        "fx, fy (px, this size)": f"{K[0, 0]:.1f}, {K[1, 1]:.1f}",
        "cx, cy (px, this size)": f"{K[0, 2]:.1f}, {K[1, 2]:.1f}",
        "dist (k1,k2,p1,p2,k3)": ", ".join(f"{d:+.4f}" for d in calib.dist[:5]),
        "calibration RMS (px)": round(calib.rms, 3) if calib.rms is not None else "n/a",
        "max undistortion shift (px)": round(float(shift.max()), 1),
        "horizontal FOV (deg)": round(float(np.degrees(2 * np.arctan(w / (2 * K[0, 0])))), 1),
    }
    first = img.copy()
    warnings = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, corners, detector = t1.calibration.detect_corners(gray, pattern)
    if ok:
        pts = corners.reshape(-1, 2).astype(np.float64)
        before = t1.calibration.line_straightness(pts, pattern)
        after = t1.calibration.line_straightness(t1.calibration.undistort_points_px(pts, K, calib.dist), pattern)
        metrics["board corners"] = f"{len(pts)} found ({detector})"
        metrics["line straightness before (px RMS)"] = round(float(before.mean()), 3)
        metrics["line straightness after (px RMS)"] = round(float(after.mean()), 3)
        cv2.drawChessboardCorners(first, pattern, corners, True)
        summary = (f"Undistorted with K from Task 1A. Checkerboard rows/columns became straighter: "
                   f"{before.mean():.2f} px -> {after.mean():.2f} px RMS deviation from a straight line.")
    else:
        warnings.append(f"No {pattern[0]}x{pattern[1]} checkerboard found, so straightness cannot be measured; "
                        "the undistortion is still applied.")
        summary = f"Undistorted with K from Task 1A (largest pixel shift {shift.max():.1f} px, at the image corners)."
    return ToolResult("calibrate", summary,
                      [("Input" + (" + detected corners" if ok else ""), first),
                       (f"Undistorted (alpha={alpha:g})", undistorted),
                       ("Undistortion shift magnitude (bright = larger)", _colour(shift_map))],
                      metrics, warnings=warnings)


def _calibrate_new(images, calib, pattern, square_mm) -> ToolResult:
    t1 = load_task1()
    sizes = {im.shape[:2] for im in images}
    if len(sizes) > 1:
        raise AssistantError("Calibration photos have different resolutions.",
                             "Use photos from one camera at one resolution (do not mix resized/rotated images).")
    with tempfile.TemporaryDirectory() as tmp:            # Task 1 reads calibration photos from files
        paths = []
        for i, im in enumerate(images):
            p = Path(tmp) / f"view_{i:02d}.png"
            cv2.imwrite(str(p), im)
            paths.append(p)
        data = t1.calibration.collect_correspondences(paths, pattern, square_mm)
    if data.n_views < 3:
        raise AssistantError(f"Checkerboard ({pattern[0]}x{pattern[1]} inner corners) found in only "
                             f"{data.n_views}/{len(images)} photos; at least 3 are needed.",
                             "Photograph the whole board, sharp, from different angles; check the pattern size.")
    res = t1.calibration.calibrate_opencv(data.object_points, data.image_points, data.image_size)
    K = res["K"]
    rows = [{"view": i + 1, "status": r["status"], "detector": r["detector"]} for i, r in enumerate(data.log)]
    for row, err in zip([r for r in rows if r["status"] == "ACCEPTED"], res["per_view_rms"]):
        row["reprojection RMS (px)"] = round(float(err), 3)
    metrics = {"views used": f"{data.n_views}/{len(images)}", "RMS reprojection error (px)": round(res["rms"], 3),
               "fx, fy": f"{K[0, 0]:.1f}, {K[1, 1]:.1f}", "cx, cy": f"{K[0, 2]:.1f}, {K[1, 2]:.1f}",
               "dist (k1,k2,p1,p2,k3)": ", ".join(f"{d:+.4f}" for d in res["dist"][:5])}
    warnings = []
    if data.n_views < 10:
        warnings.append("Fewer than 10 views: K is poorly constrained (Zhang's method needs varied board poses).")
    if calib is not None:
        try:
            Ks = calib.for_image(*data.image_size)
            metrics["fx vs stored K (%)"] = round(100 * (K[0, 0] / Ks[0, 0] - 1), 2)
        except AssistantError:
            warnings.append("Photos have a different size/aspect than the stored calibration; no comparison made.")
    vis = images[0].copy()
    cv2.drawChessboardCorners(vis, pattern, data.image_points[0], True)
    und, _, _ = t1.calibration.undistort_image(images[0], K, res["dist"], alpha=0.0)
    return ToolResult("calibrate", f"New calibration from {data.n_views} checkerboard views: "
                                   f"RMS reprojection error {res['rms']:.3f} px.",
                      [("View 1 + detected corners", vis), ("View 1 undistorted with the new K", und)],
                      metrics, table=rows, warnings=warnings)


# ----------------------------------------------------------------------------
# Filtering  (Task 1B)
# ----------------------------------------------------------------------------
SPATIAL_FILTERS = ("box", "gaussian", "median", "bilateral")
FREQ_FILTERS = ("fft_lowpass", "fft_highpass")


def filter_image(images, filter: str = "gaussian", ksize: int = 5, noise: str = "none",
                 cutoff: int = 30, profile: str = "gaussian") -> ToolResult:
    """Spatial filters (box / Gaussian / median / bilateral) or Fourier low/high-pass on the grey image.
    With ``noise`` != 'none' the clean image is corrupted first and every spatial filter is scored
    (MSE / PSNR / SSIM against the clean image) - the denoising experiment of notebook 1B."""
    _need(images, 1, "filter")
    f = load_task1().filtering
    if filter not in SPATIAL_FILTERS + FREQ_FILTERS:
        raise AssistantError(f"Unknown filter '{filter}'.", f"Choose one of {', '.join(SPATIAL_FILTERS + FREQ_FILTERS)}.")
    ksize = int(min(31, max(3, int(ksize) | 1)))
    work, _ = resize_max_side(images[0], settings.WORK_MAX_SIDE)
    clean = f.to_gray(work)
    if noise == "gaussian":
        src = f.add_gaussian_noise(clean, sigma=20)
    elif noise == "salt_pepper":
        src = f.add_salt_pepper(clean, amount=0.05)
    else:
        src = clean
    noisy_label = {"gaussian": "Gaussian noise sigma=20", "salt_pepper": "salt & pepper 5%"}.get(noise, "")
    metrics, table, panels = {"filter": filter, "working size": f"{clean.shape[1]} x {clean.shape[0]}"}, None, []
    panels.append(("Input (grey" + (f", {noisy_label})" if noisy_label else ")"), src))

    if filter in SPATIAL_FILTERS:
        out = f.apply_filter(src, filter, ksize)
        metrics["kernel"] = f"{ksize} x {ksize}"
        panels.append((f"{filter.capitalize()} {ksize}x{ksize}", out))
        panels.append(("|input - output| (what was removed)", _colour(f.to_uint8(np.abs(src.astype(float) - out)))))
        if noise != "none":
            table = [{"image": "noisy input", **{k: round(v, 3) for k, v in f.quality(clean, src).items()}}]
            for name in SPATIAL_FILTERS:
                q = f.quality(clean, f.apply_filter(src, name, ksize))
                table.append({"image": f"{name} {ksize}x{ksize}", **{k: round(v, 3) for k, v in q.items()}})
            best = max(table[1:], key=lambda r: r["PSNR_dB"])
            metrics["PSNR noisy -> filtered (dB)"] = f"{table[0]['PSNR_dB']} -> {f.psnr(clean, out):.2f}"
            summary = (f"{filter.capitalize()} filter on {noisy_label}: PSNR {table[0]['PSNR_dB']:.2f} -> "
                       f"{f.psnr(clean, out):.2f} dB. Best filter for this noise: {best['image']} "
                       f"({best['PSNR_dB']:.2f} dB).")
        else:
            summary = (f"{filter.capitalize()} {ksize}x{ksize} smoothing: mean absolute change "
                       f"{np.abs(src.astype(float) - out).mean():.2f} grey levels.")
    else:
        kind = "lowpass" if filter == "fft_lowpass" else "highpass"
        F = f.fft_shifted(src)
        H = (f.lowpass_mask if kind == "lowpass" else f.highpass_mask)(src.shape, cutoff, profile)
        out_f = f.apply_frequency_filter(src, H)
        out = np.clip(out_f, 0, 255).astype(np.uint8) if kind == "lowpass" else f.to_uint8(out_f)
        power = np.abs(F) ** 2
        kept = float((power * H**2).sum() / power.sum())
        metrics.update({"transfer function": f"{profile} {kind}", "cut-off D0 (frequency px)": cutoff,
                        "spectral energy kept (%)": round(100 * kept, 2)})
        panels += [("log |F(u,v)| (centred spectrum)", _colour(f.to_uint8(f.log_magnitude(F)), cv2.COLORMAP_VIRIDIS)),
                   (f"H(u,v): {profile} {kind}, D0={cutoff}", f.to_uint8(H)),
                   (f"Inverse FFT of F x H", out)]
        if noise != "none":
            metrics["PSNR noisy -> filtered (dB)"] = f"{f.psnr(clean, src):.2f} -> {f.psnr(clean, out):.2f}"
        summary = (f"Fourier {kind} ({profile}, D0={cutoff}) keeps {100 * kept:.1f}% of the spectral energy"
                   + (": fine detail and noise are removed." if kind == "lowpass" else ": only edges/texture remain."))
        if profile == "ideal" and kind == "lowpass":
            summary += " The ideal mask causes ringing (Gibbs) around edges."
    return ToolResult("filter", summary, panels, metrics, table=table)


# ----------------------------------------------------------------------------
# Edges and fruit outline  (Task 1B)
# ----------------------------------------------------------------------------
def detect_edges(images, canny: tuple[int, int] | None = None, blur: int = 5) -> ToolResult:
    """Sobel gradient magnitude/orientation, Canny (automatic or manual thresholds) and the
    classical fruit outline (Otsu on HSV saturation + morphology)."""
    _need(images, 1, "edges")
    f = load_task1().filtering
    work, _ = resize_max_side(images[0], settings.WORK_MAX_SIDE)
    gray = f.to_gray(work)
    _gx, _gy, mag, ang = f.sobel(gray)
    if canny is None:
        edges, (lo, hi) = f.canny_auto(gray, blur=blur)
        mode = "auto (median-based)"
    else:
        lo, hi = sorted(int(v) for v in canny)
        smooth = cv2.GaussianBlur(gray, (blur, blur), 0) if blur else gray
        edges = cv2.Canny(smooth, lo, hi)
        mode = "manual"
    hsv_angle = np.zeros((*gray.shape, 3), np.uint8)               # orientation as hue, magnitude as value
    hsv_angle[..., 0] = np.uint8((np.degrees(ang) % 180))
    hsv_angle[..., 1] = 255
    hsv_angle[..., 2] = f.to_uint8(mag)
    contour, mask, _boundary = f.fruit_outline(work)
    outline = work.copy()
    metrics = {"working size": f"{gray.shape[1]} x {gray.shape[0]}", "Canny thresholds": f"{lo} / {hi} ({mode})",
               "edge pixels (%)": round(100 * float((edges > 0).mean()), 2),
               "mean gradient magnitude": round(float(mag.mean()), 2)}
    warnings = []
    if contour is not None:
        cv2.drawContours(outline, [contour], -1, (0, 255, 0), 2)
        area, perim = cv2.contourArea(contour), cv2.arcLength(contour, True)
        metrics["fruit area (% of image)"] = round(100 * area / gray.size, 1)
        metrics["fruit circularity 4*pi*A/P^2"] = round(4 * np.pi * area / max(perim**2, 1e-9), 3)
    else:
        warnings.append("No fruit region found (segmentation expects a saturated fruit on a plain light background).")
    summary = (f"Canny found edges on {metrics['edge pixels (%)']}% of pixels with thresholds {lo}/{hi}."
               + (f" Fruit outline circularity {metrics['fruit circularity 4*pi*A/P^2']}." if contour is not None else ""))
    return ToolResult("edges", summary,
                      [("Sobel gradient magnitude", f.to_uint8(mag)),
                       ("Gradient orientation (hue) x magnitude", cv2.cvtColor(hsv_angle, cv2.COLOR_HSV2BGR)),
                       (f"Canny {lo}/{hi}", edges),
                       ("Fruit outline (Otsu on saturation)", outline)],
                      metrics, warnings=warnings)


# ----------------------------------------------------------------------------
# Keypoints and matching  (Task 1C)
# ----------------------------------------------------------------------------
def match_features(images, method: str = "SIFT", ratio: float = 0.75) -> ToolResult:
    """Detect + describe keypoints, Lowe ratio test, RANSAC homography.
    One image -> a synthetic second view with a KNOWN homography is generated, so match
    precision can be measured against ground truth (as in notebook 1C)."""
    _need(images, 1, "features")
    feats, f = load_task1().features, load_task1().filtering
    if method not in feats.METHODS:
        raise AssistantError(f"Unknown method '{method}'.", f"Choose one of {', '.join(feats.METHODS)}.")
    img1, _ = resize_max_side(images[0], settings.WORK_MAX_SIDE)
    H_true, warnings = None, []
    if len(images) >= 2:
        img2, _ = resize_max_side(images[1], settings.WORK_MAX_SIDE)
        pair = "two uploaded views"
    else:
        img2, H_true = feats.synthetic_view(img1, angle=25, scale=0.85, perspective=0.2, brightness=-15,
                                            noise_sigma=4)
        pair = "synthetic view (rotation 25 deg, scale 0.85, tilt, darker, noise) with known H"
    r = feats.evaluate_pair(f.to_gray(img1), f.to_gray(img2), method, H_true=H_true, ratio=ratio)
    kp1, kp2, good, inliers = r["_kp1"], r["_kp2"], r["_good"], r["_inliers"]
    metrics = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if not k.startswith("_")}
    metrics.update({"ratio": ratio, "pair": pair})
    if len(kp1) == 0 or len(kp2) == 0:
        warnings.append("No keypoints detected in one of the images (too smooth / too small).")
    if len(good) < 4:
        warnings.append("Fewer than 4 good matches: no homography can be estimated. Try SIFT or a higher ratio.")
    elif r["ransac_inliers"] / max(len(good), 1) < 0.25:
        warnings.append("Less than 25% of the matches agree with one homography: the views may not show the "
                        "same planar scene.")
    kp_vis = cv2.drawKeypoints(img1, kp1, None, (0, 255, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    matches_vis = feats.draw_matches(img1, kp1, img2, kp2, good, inliers, max_draw=60)
    summary = (f"{method}: {len(kp1)} / {len(kp2)} keypoints, {len(good)} matches pass the ratio test {ratio}, "
               f"{r['ransac_inliers']} are RANSAC inliers.")
    if H_true is not None:
        summary += f" Ground-truth precision {100 * r.get('precision', 0):.1f}%."
    return ToolResult("features", summary,
                      [(f"{method} keypoints (size + orientation)", kp_vis),
                       ("Matches: green = RANSAC inlier, red = outlier (best 60 shown)", matches_vis)],
                      metrics, warnings=warnings)


# ----------------------------------------------------------------------------
# Classification  (Task 2)
# ----------------------------------------------------------------------------
def classify(images, clf: Classifier) -> ToolResult:
    """Run the Task 2 FruitCNN through Task 2's ``predict_pil`` and explain it with Grad-CAM."""
    _need(images, 1, "classify")
    if clf is None:
        raise AssistantError("The Task 2 model is not loaded.", "See the model status in the sidebar.")
    img = images[0]
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    label, score = clf.predict_pil(clf.model, clf.idx_to_class, clf.transform, pil)     # Task 2 code path
    x = clf.transform(pil).unsqueeze(0)                                                  # same preprocessing
    scores, idx, cam = gradcam.scores_and_cam(clf.model, x)
    if clf.idx_to_class[idx] != label:                                                   # sanity check
        raise AssistantError("Grad-CAM and Task 2 inference disagree.", "Reload the app (model state changed).")
    work, _ = resize_max_side(img, settings.WORK_MAX_SIDE)
    table = [{"class": clf.idx_to_class[i], "score": round(float(s), 4)} for i, s in enumerate(scores)]
    table.sort(key=lambda r: -r["score"])
    metrics = {"predicted class": label, "score (softmax)": round(score, 4),
               "margin to 2nd class": round(float(table[0]["score"] - table[1]["score"]), 4) if len(table) > 1 else 1.0,
               "model": clf.meta.get("architecture", "FruitCNN"),
               "input": f"{clf.meta.get('image_size', '?')} px, mean/std {clf.meta.get('mean')}/{clf.meta.get('std')}"}
    warnings = []
    if score < settings.LOW_CONFIDENCE:
        warnings.append(f"Low confidence ({score:.2f} < {settings.LOW_CONFIDENCE}): the image may not show one of "
                        f"{', '.join(clf.classes)}, or shows several fruits. Check it manually.")
    warnings.append("Softmax scores are not calibrated probabilities; the model only knows "
                    f"{', '.join(clf.classes)} and will always answer one of them.")
    return ToolResult("classify", f"Predicted **{label}** with score {score:.3f}.",
                      [("Input", work), (f"Grad-CAM for '{label}' (red = most influential)", gradcam.overlay(work, cam))],
                      metrics, table=table, warnings=warnings)


# ----------------------------------------------------------------------------
# Measurement  (Task 1A intrinsics + Task 1B segmentation)
# ----------------------------------------------------------------------------
def measure(images, calib: Calibration | None, distance_mm: float | None = None) -> ToolResult:
    """Fruit size from the outline. Pixels -> millimetres with the pinhole model
    size_mm = size_px * Z / f_px, valid for a roughly fronto-parallel object at distance Z."""
    _need(images, 1, "measure")
    f = load_task1().filtering
    img, warnings = images[0], []
    K = None
    if calib is not None:
        try:
            K = calib.for_image(img.shape[1], img.shape[0])
            # undistort first; the undistorted image has its own camera matrix newK (alpha=1 keeps all pixels)
            img, K, _ = load_task1().calibration.undistort_image(img, K, calib.dist, alpha=1.0)
        except AssistantError as exc:
            warnings.append(f"{exc.message} Size is reported in pixels only.")
            K = None
    else:
        warnings.append("No calibration loaded: size is reported in pixels only.")
    work, s = resize_max_side(img, settings.WORK_MAX_SIDE)
    contour, mask, _ = f.fruit_outline(work)
    if contour is None or cv2.contourArea(contour) < 0.002 * mask.size:
        raise AssistantError("No fruit region found to measure.",
                             "Photograph one fruit on a plain, light background, fully inside the frame.")
    area_px = cv2.contourArea(contour) / s**2                       # back to full-resolution pixels
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
    length_px, width_px = max(rw, rh) / s, min(rw, rh) / s
    eq_diam_px = 2 * np.sqrt(area_px / np.pi)
    vis = work.copy()
    cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)
    cv2.drawContours(vis, [np.int32(cv2.boxPoints(((cx, cy), (rw, rh), angle)))], -1, (255, 0, 0), 2)
    metrics = {"area (px)": int(area_px), "equivalent diameter (px)": round(eq_diam_px, 1),
               "length x width (px, min-area box)": f"{length_px:.1f} x {width_px:.1f}",
               "touches image border": bool(contour[:, 0, 0].min() <= 1 or contour[:, 0, 1].min() <= 1
                                            or contour[:, 0, 0].max() >= work.shape[1] - 2
                                            or contour[:, 0, 1].max() >= work.shape[0] - 2)}
    if metrics["touches image border"]:
        warnings.append("The fruit touches the image border: it may be cut off, so the size is a lower bound.")
    # single-fruit sanity checks: a lone fruit is a compact, almost convex blob
    solidity = cv2.contourArea(contour) / max(cv2.contourArea(cv2.convexHull(contour)), 1e-9)
    regions, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n_regions = sum(cv2.contourArea(c) > 0.01 * mask.size for c in regions)
    metrics["solidity (area / convex hull)"] = round(float(solidity), 3)
    metrics["foreground regions > 1% of image"] = int(n_regions)
    if solidity < 0.85 or n_regions > 1:
        warnings.append(f"The outline does not look like one isolated fruit (solidity {solidity:.2f}, "
                        f"{n_regions} regions): several fruits or a coloured background were merged. "
                        "The size is unreliable; use one fruit on a plain light background.")
    summary = f"Fruit outline: equivalent diameter {eq_diam_px:.0f} px, {length_px:.0f} x {width_px:.0f} px."
    if K is not None:
        f_px = float(np.sqrt(K[0, 0] * K[1, 1]))
        metrics["focal length f (px, this size)"] = round(f_px, 1)
        if distance_mm:
            mm = distance_mm / f_px
            metrics["camera-object distance Z (mm)"] = distance_mm
            metrics["equivalent diameter (mm)"] = round(eq_diam_px * mm, 1)
            metrics["length x width (mm)"] = f"{length_px * mm:.1f} x {width_px * mm:.1f}"
            summary = (f"Fruit about **{length_px * mm:.0f} x {width_px * mm:.0f} mm** (equivalent diameter "
                       f"{eq_diam_px * mm:.0f} mm) at Z = {distance_mm:.0f} mm, using f = {f_px:.0f} px from Task 1A.")
            warnings.append("Metric size assumes the fruit is roughly parallel to the image plane at the given "
                            "distance; a 10% distance error gives a 10% size error.")
        else:
            warnings.append("Give the camera-object distance (e.g. 'measure at 30 cm') to convert pixels to mm.")
    return ToolResult("measure", summary,
                      [("Outline (green) and minimum-area box (blue)", vis), ("Segmentation mask", mask)],
                      metrics, warnings=warnings)


def inspect(images, clf: Classifier | None, calib: Calibration | None, distance_mm: float | None = None) -> ToolResult:
    """Chain both engines: classify (Task 2) then measure (Task 1)."""
    _need(images, 1, "inspect")
    c = classify(images, clf)
    try:
        m = measure(images, calib, distance_mm)
    except AssistantError as exc:
        m = ToolResult("measure", "", warnings=[f"Measurement skipped: {exc.message} {exc.hint}"])
    metrics = {**c.metrics, **m.metrics}
    summary = c.summary + (" " + m.summary if m.summary else "")
    return ToolResult("inspect", summary, c.images[1:] + m.images[:1], metrics, table=c.table,
                      warnings=c.warnings + m.warnings)


# ----------------------------------------------------------------------------
# Dispatcher: mode name -> tool
# ----------------------------------------------------------------------------
def run(mode: str, images: list[np.ndarray], params: dict, clf: Classifier | None = None,
        calib: Calibration | None = None) -> ToolResult:
    """Call the tool for ``mode`` with the parameters it accepts; time it."""
    t0 = time.perf_counter()
    p = params
    if mode == "calibrate":
        res = calibrate(images, calib, alpha=p.get("alpha", 0.0))
    elif mode == "filter":
        res = filter_image(images, p.get("filter", "gaussian"), p.get("ksize", 5), p.get("noise", "none"),
                           p.get("cutoff", 30), p.get("profile", "gaussian"))
    elif mode == "edges":
        res = detect_edges(images, p.get("canny"))
    elif mode == "features":
        res = match_features(images, p.get("method", "SIFT"), p.get("ratio", 0.75))
    elif mode == "classify":
        res = classify(images, clf)
    elif mode == "measure":
        res = measure(images, calib, p.get("distance_mm"))
    elif mode == "inspect":
        res = inspect(images, clf, calib, p.get("distance_mm"))
    else:
        raise AssistantError(f"Unknown mode '{mode}'.")
    res.elapsed_ms = (time.perf_counter() - t0) * 1000
    return res
