"""Task 1B - Spatial filtering, denoising, image gradients and frequency-domain filtering.

From-scratch parts (NumPy only) are verified against OpenCV in the notebook:
``convolve2d`` / ``correlate2d``, ``median_filter``, ``gaussian_kernel``, ``ssim`` and
the frequency-domain masks.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------
def load_image(path: str | Path, max_side: int | None = 512) -> np.ndarray:
    """Read a BGR image and optionally shrink it so its longest side is ``max_side``."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if max_side and max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Min-max scale any array to 0..255 uint8 for display."""
    img = np.asarray(img, float)
    lo, hi = img.min(), img.max()
    return np.zeros(img.shape, np.uint8) if hi == lo else ((img - lo) / (hi - lo) * 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# Linear filtering from scratch
# ----------------------------------------------------------------------------
def correlate2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2-D cross-correlation with reflect-101 borders (same as cv2.filter2D default)."""
    kernel = np.asarray(kernel, float)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(np.asarray(img, float), ((ph, kh - 1 - ph), (pw, kw - 1 - pw)), mode="reflect")
    windows = sliding_window_view(padded, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, kernel)


def convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2-D convolution = correlation with the kernel flipped in both axes."""
    return correlate2d(img, np.asarray(kernel)[::-1, ::-1])


def box_kernel(k: int) -> np.ndarray:
    return np.full((k, k), 1.0 / (k * k))


def gaussian_kernel(k: int, sigma: float) -> np.ndarray:
    """Normalised k x k Gaussian kernel (separable outer product)."""
    ax = np.arange(k) - (k - 1) / 2
    g = np.exp(-(ax**2) / (2 * sigma**2))
    g /= g.sum()
    return np.outer(g, g)


def median_filter(img: np.ndarray, k: int) -> np.ndarray:
    """k x k median filter from scratch (non-linear, so it is not a convolution)."""
    p = k // 2
    padded = np.pad(img, p, mode="edge")  # replicate border, as cv2.medianBlur
    return np.median(sliding_window_view(padded, (k, k)), axis=(-2, -1)).astype(img.dtype)


def apply_filter(gray: np.ndarray, name: str, k: int = 5, sigma: float | None = None) -> np.ndarray:
    """OpenCV implementation of the filters compared in the notebook."""
    if name == "box":
        return cv2.blur(gray, (k, k))
    if name == "gaussian":
        return cv2.GaussianBlur(gray, (k, k), sigma or 0.3 * ((k - 1) * 0.5 - 1) + 0.8)
    if name == "median":
        return cv2.medianBlur(gray, k)
    if name == "bilateral":
        return cv2.bilateralFilter(gray, k, 50, 50)
    raise ValueError(f"Unknown filter: {name}")


# ----------------------------------------------------------------------------
# Noise models and quality metrics
# ----------------------------------------------------------------------------
def add_gaussian_noise(gray: np.ndarray, sigma: float = 20, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(gray.astype(float) + rng.normal(0, sigma, gray.shape), 0, 255).astype(np.uint8)


def add_salt_pepper(gray: np.ndarray, amount: float = 0.05, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = gray.copy()
    u = rng.random(gray.shape)
    out[u < amount / 2] = 0
    out[u > 1 - amount / 2] = 255
    return out


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(float) - b.astype(float)) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, peak: float = 255.0) -> float:
    m = mse(a, b)
    return float("inf") if m == 0 else float(10 * np.log10(peak**2 / m))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural similarity (Wang et al. 2004), Gaussian window sigma = 1.5."""
    a, b = a.astype(float), b.astype(float)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a**2
    var_b = blur(b * b) - mu_b**2
    cov = blur(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))
    return float(s.mean())


def quality(reference: np.ndarray, test: np.ndarray) -> dict:
    return {"MSE": mse(reference, test), "PSNR_dB": psnr(reference, test), "SSIM": ssim(reference, test)}


# ----------------------------------------------------------------------------
# Gradients and edges
# ----------------------------------------------------------------------------
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
SOBEL_Y = SOBEL_X.T


def sobel(gray: np.ndarray):
    """Sobel gradients: returns gx, gy, magnitude and orientation (radians)."""
    g = gray.astype(float)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    return gx, gy, np.hypot(gx, gy), np.arctan2(gy, gx)


def canny_auto(gray: np.ndarray, sigma: float = 0.33, blur: int = 5):
    """Canny with thresholds derived from the median intensity of a blurred image."""
    smooth = cv2.GaussianBlur(gray, (blur, blur), 0) if blur else gray
    m = float(np.median(smooth))
    lo, hi = int(max(0, (1 - sigma) * m)), int(min(255, (1 + sigma) * m))
    return cv2.Canny(smooth, lo, hi), (lo, hi)


def edge_f1(reference: np.ndarray, test: np.ndarray, tolerance: int = 2) -> dict:
    """Precision / recall / F1 of an edge map against a reference edge map.

    An edge pixel counts as correct when a reference edge lies within ``tolerance``
    pixels (matching through a dilated reference), and vice versa for recall.
    """
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance + 1,) * 2)
    ref, tst = reference > 0, test > 0
    ref_d = cv2.dilate(ref.astype(np.uint8), ker) > 0
    tst_d = cv2.dilate(tst.astype(np.uint8), ker) > 0
    precision = (tst & ref_d).sum() / max(tst.sum(), 1)
    recall = (ref & tst_d).sum() / max(ref.sum(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": float(precision), "recall": float(recall), "F1": float(f1), "edge_pixels": int(tst.sum())}


def fruit_mask(bgr: np.ndarray, blur: int = 7):
    """Classical fruit/background segmentation.

    Fruit skins (red, yellow, orange, green) are strongly saturated while studio
    backgrounds are white/grey, so Otsu's threshold on the smoothed HSV saturation
    channel separates them; morphology removes specks and fills holes.
    Returns (binary mask 0/255, Otsu threshold).
    """
    sat = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1], (blur, blur), 0)
    t, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ell = lambda s: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, ell(7)), cv2.MORPH_CLOSE, ell(15))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)  # fill holes (e.g. specular highlights)
    return filled, float(t)


def fruit_outline(bgr: np.ndarray, blur: int = 7):
    """Largest fruit region from :func:`fruit_mask` and its boundary edges.

    Returns (largest contour or None, mask, boundary edges from Canny on the mask).
    """
    mask, _ = fruit_mask(bgr, blur)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary = cv2.Canny(mask, 50, 150)
    if not contours:
        return None, mask, boundary
    return max(contours, key=cv2.contourArea), mask, boundary


# ----------------------------------------------------------------------------
# Frequency domain
# ----------------------------------------------------------------------------
def fft_shifted(gray: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(gray.astype(float)))


def log_magnitude(F: np.ndarray) -> np.ndarray:
    return np.log1p(np.abs(F))


def _radius_grid(shape) -> np.ndarray:
    h, w = shape
    y, x = np.ogrid[:h, :w]
    return np.hypot(y - h / 2, x - w / 2)


def lowpass_mask(shape, cutoff: float, kind: str = "gaussian", order: int = 2) -> np.ndarray:
    """Centred low-pass transfer function H(u, v); cutoff = radius in frequency pixels."""
    D = _radius_grid(shape)
    if kind == "ideal":
        return (D <= cutoff).astype(float)
    if kind == "gaussian":
        return np.exp(-(D**2) / (2 * cutoff**2))
    if kind == "butterworth":
        return 1 / (1 + (D / cutoff) ** (2 * order))
    raise ValueError(kind)


def highpass_mask(shape, cutoff: float, kind: str = "gaussian", order: int = 2) -> np.ndarray:
    return 1.0 - lowpass_mask(shape, cutoff, kind, order)


def notch_mask(shape, centres, radius: float = 4) -> np.ndarray:
    """Gaussian notch-reject mask removing the given (row, col) peaks and their mirrors."""
    h, w = shape
    y, x = np.ogrid[:h, :w]
    H = np.ones(shape)
    for r, c in centres:
        for rr, cc in [(r, c), (h - r, w - c)]:
            H *= 1 - np.exp(-((y - rr) ** 2 + (x - cc) ** 2) / (2 * radius**2))
    return H


def apply_frequency_filter(gray: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Multiply the centred spectrum by H and return the real inverse transform."""
    F = fft_shifted(gray)
    return np.real(np.fft.ifft2(np.fft.ifftshift(F * H)))


def add_periodic_noise(gray: np.ndarray, amplitude: float = 30, freq=(0.08, 0.05)) -> np.ndarray:
    """Add a 2-D sinusoid (e.g. scanner / electrical interference)."""
    h, w = gray.shape
    y, x = np.mgrid[:h, :w]
    noise = amplitude * np.sin(2 * np.pi * (freq[0] * x + freq[1] * y))
    return np.clip(gray.astype(float) + noise, 0, 255).astype(np.uint8)


def find_spectral_peaks(F: np.ndarray, n: int = 1, exclude_radius: int = 10):
    """Strongest off-centre peaks in the upper half of a centred spectrum."""
    mag = np.abs(F).copy()
    h, w = mag.shape
    mag[_radius_grid(mag.shape) < exclude_radius] = 0
    mag[h // 2:, :] = 0  # each peak has a mirror; keep only the upper half
    peaks = []
    for _ in range(n):
        r, c = np.unravel_index(np.argmax(mag), mag.shape)
        peaks.append((int(r), int(c)))
        mag[max(0, r - 5):r + 6, max(0, c - 5):c + 6] = 0
    return peaks
