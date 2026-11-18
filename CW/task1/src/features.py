"""Task 1C - Interest points, descriptors and matching across two views.

* ``harris_response`` / ``harris_corners`` - Harris detector implemented from scratch
  (structure tensor + non-maximum suppression), checked against cv2.cornerHarris.
* ``detect_and_describe`` - SIFT, ORB and "Harris + SIFT descriptor".
* ``ratio_test_matches`` - Lowe's ratio test on 2-nearest-neighbour matches.
* ``ransac_homography`` - geometric verification.
* ``synthetic_view`` - a second view with a KNOWN homography, so match correctness can
  be measured objectively (ground truth) instead of judged by eye.
"""
from __future__ import annotations

import time

import cv2
import numpy as np


# ----------------------------------------------------------------------------
# Harris corner detector from scratch
# ----------------------------------------------------------------------------
def harris_response(gray: np.ndarray, k: float = 0.04, sigma: float = 1.5, window: str = "gaussian",
                    block_size: int = 3) -> np.ndarray:
    """Harris corner response R = det(M) - k * trace(M)^2.

    M is the structure tensor: window-weighted sums of Ix^2, Iy^2 and Ix*Iy.
    ``window="gaussian"`` (classic Harris, isotropic) or ``"box"`` with ``block_size``
    (the window cv2.cornerHarris uses - handy for verifying this implementation).
    """
    g = gray.astype(np.float64) / 255.0
    Ix = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    if window == "box":
        smooth = lambda a: cv2.boxFilter(a, cv2.CV_64F, (block_size, block_size), normalize=False)
    else:
        ksz = int(2 * round(3 * sigma) + 1)
        smooth = lambda a: cv2.GaussianBlur(a, (ksz, ksz), sigma)
    Sxx, Syy, Sxy = smooth(Ix * Ix), smooth(Iy * Iy), smooth(Ix * Iy)
    det = Sxx * Syy - Sxy**2
    trace = Sxx + Syy
    return det - k * trace**2


def harris_corners(gray: np.ndarray, k: float = 0.04, sigma: float = 1.5, rel_threshold: float = 0.01,
                   nms_size: int = 7, max_corners: int = 1000) -> np.ndarray:
    """Corner (x, y) coordinates: threshold relative to max(R) + non-maximum suppression."""
    R = harris_response(gray, k, sigma)
    local_max = cv2.dilate(R, np.ones((nms_size, nms_size), np.uint8))
    mask = (R == local_max) & (R > rel_threshold * R.max())
    ys, xs = np.nonzero(mask)
    order = np.argsort(R[ys, xs])[::-1][:max_corners]
    return np.column_stack([xs[order], ys[order]]).astype(np.float32)


# ----------------------------------------------------------------------------
# Detectors / descriptors
# ----------------------------------------------------------------------------
METHODS = ("SIFT", "ORB", "Harris+SIFT")


def norm_for(method: str) -> int:
    """Distance used to compare descriptors: L2 for float, Hamming for binary."""
    return cv2.NORM_HAMMING if method == "ORB" else cv2.NORM_L2


def detect_and_describe(gray: np.ndarray, method: str = "SIFT", n_features: int = 2000):
    """Return (keypoints, descriptors) for one of ``METHODS``."""
    if method == "SIFT":
        return cv2.SIFT_create(nfeatures=n_features).detectAndCompute(gray, None)
    if method == "ORB":
        return cv2.ORB_create(nfeatures=n_features).detectAndCompute(gray, None)
    if method == "Harris+SIFT":
        # Harris gives WHERE (fixed scale, no orientation); SIFT describes a fixed 16 px patch
        pts = harris_corners(gray, max_corners=n_features)
        kps = [cv2.KeyPoint(float(x), float(y), 16) for x, y in pts]
        return cv2.SIFT_create().compute(gray, kps)
    raise ValueError(f"Unknown method {method}; choose from {METHODS}")


def ratio_test_matches(desc1, desc2, norm: int, ratio: float = 0.75) -> list[cv2.DMatch]:
    """Lowe's ratio test: keep m1 only if d(m1) < ratio * d(m2) (2 nearest neighbours)."""
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []
    knn = cv2.BFMatcher(norm).knnMatch(desc1, desc2, k=2)
    return [p[0] for p in knn if len(p) == 2 and p[0].distance < ratio * p[1].distance]


def ransac_homography(kp1, kp2, matches, threshold: float = 4.0):
    """Fit a homography with RANSAC; returns (H or None, boolean inlier mask)."""
    if len(matches) < 4:
        return None, np.zeros(len(matches), bool)
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, threshold)
    return H, (mask.ravel().astype(bool) if mask is not None else np.zeros(len(matches), bool))


# ----------------------------------------------------------------------------
# Ground-truth evaluation with a synthetic second view
# ----------------------------------------------------------------------------
def synthetic_view(img: np.ndarray, angle: float = 30, scale: float = 0.8, perspective: float = 0.0,
                   brightness: float = 0, noise_sigma: float = 0, seed: int = 0):
    """Warp ``img`` with a known homography H (rotation, scale, perspective tilt),
    then change brightness and add noise. Returns (view, H) with view = H(img)."""
    h, w = img.shape[:2]
    A = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    H = np.vstack([A, [0, 0, 1]])
    if perspective:
        P = np.eye(3)
        P[2, 0] = perspective / w
        T = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]])
        H = np.linalg.inv(T) @ P @ T @ H
        H /= H[2, 2]
    view = cv2.warpPerspective(img, H, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
                               if img.ndim == 3 else 255)
    view = view.astype(float) + brightness
    if noise_sigma:
        view += np.random.default_rng(seed).normal(0, noise_sigma, view.shape)
    return np.clip(view, 0, 255).astype(np.uint8), H


def transfer(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Map (x, y) points through homography H."""
    return cv2.perspectiveTransform(np.asarray(pts, np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)


def correct_matches(kp1, kp2, matches, H_true: np.ndarray, tolerance: float = 3.0) -> np.ndarray:
    """Boolean array: is each match consistent with the true homography (within px)?"""
    if not matches:
        return np.zeros(0, bool)
    p1 = np.float64([kp1[m.queryIdx].pt for m in matches])
    p2 = np.float64([kp2[m.trainIdx].pt for m in matches])
    return np.linalg.norm(transfer(p1, H_true) - p2, axis=1) < tolerance


def repeatability(kp1, kp2, H_true: np.ndarray, shape, tolerance: float = 3.0) -> float:
    """Share of view-1 keypoints (visible in view 2) re-detected within ``tolerance`` px."""
    if not kp1 or not kp2:
        return 0.0
    p1 = transfer(np.float64([k.pt for k in kp1]), H_true)
    h, w = shape[:2]
    visible = (p1[:, 0] >= 0) & (p1[:, 0] < w) & (p1[:, 1] >= 0) & (p1[:, 1] < h)
    if not visible.any():
        return 0.0
    p2 = np.float64([k.pt for k in kp2])
    d = np.linalg.norm(p1[visible, None, :] - p2[None, :, :], axis=2).min(axis=1)
    return float(np.mean(d < tolerance))


def evaluate_pair(gray1, gray2, method: str, H_true=None, ratio: float = 0.75, tolerance: float = 3.0) -> dict:
    """Detect, describe, ratio-test match and RANSAC-verify one image pair."""
    t0 = time.perf_counter()
    kp1, d1 = detect_and_describe(gray1, method)
    kp2, d2 = detect_and_describe(gray2, method)
    good = ratio_test_matches(d1, d2, norm_for(method), ratio)
    elapsed = (time.perf_counter() - t0) * 1000
    H, inliers = ransac_homography(kp1, kp2, good)
    row = {"method": method, "kp1": len(kp1), "kp2": len(kp2), "good_matches": len(good),
           "ransac_inliers": int(inliers.sum()), "time_ms": elapsed}
    if H_true is not None:
        ok = correct_matches(kp1, kp2, good, H_true, tolerance)
        row["correct_matches"] = int(ok.sum())
        row["precision"] = float(ok.mean()) if len(ok) else 0.0
        row["repeatability"] = repeatability(kp1, kp2, H_true, gray2.shape, tolerance)
        if H is not None:
            corners = np.float64([[0, 0], [gray1.shape[1], 0], [gray1.shape[1], gray1.shape[0]], [0, gray1.shape[0]]])
            row["H_corner_error_px"] = float(np.mean(np.linalg.norm(transfer(corners, H) - transfer(corners, H_true), axis=1)))
        else:
            row["H_corner_error_px"] = float("nan")
    return {**row, "_kp1": kp1, "_kp2": kp2, "_good": good, "_inliers": inliers, "_H": H}


def draw_matches(img1, kp1, img2, kp2, matches, inliers=None, max_draw: int = 50) -> np.ndarray:
    """Side-by-side match drawing: green = RANSAC inlier, red = outlier."""
    matches = list(matches)
    inliers = np.ones(len(matches), bool) if inliers is None else np.asarray(inliers, bool)
    order = np.argsort([m.distance for m in matches])[:max_draw]
    canvas = cv2.drawMatches(img1, kp1, img2, kp2, [matches[i] for i in order if not inliers[i]], None,
                             matchColor=(0, 0, 255), singlePointColor=None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return cv2.drawMatches(img1, kp1, img2, kp2, [matches[i] for i in order if inliers[i]], canvas,
                           matchColor=(0, 200, 0), singlePointColor=None,
                           flags=cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG | cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
