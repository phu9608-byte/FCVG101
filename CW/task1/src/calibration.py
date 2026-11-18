"""Task 1A - Camera model, checkerboard calibration and undistortion.

Two calibration back-ends are provided:

* ``calibrate_zhang``  - Zhang's method implemented from scratch with NumPy/SciPy
  (normalised DLT homographies -> closed-form K -> extrinsics -> Levenberg-Marquardt
  refinement of all parameters including lens distortion).
* ``calibrate_opencv`` - reference result from ``cv2.calibrateCamera``.

Both use the same pinhole + Brown-Conrady distortion model as OpenCV:

    x = X/Z, y = Y/Z                         (normalised pinhole projection)
    r^2 = x^2 + y^2
    x_d = x(1 + k1 r^2 + k2 r^4 + k3 r^6) + 2 p1 x y + p2 (r^2 + 2 x^2)
    y_d = y(1 + k1 r^2 + k2 r^4 + k3 r^6) + p1 (r^2 + 2 y^2) + 2 p2 x y
    u = fx x_d + cx,  v = fy y_d + cy        (intrinsic matrix K, zero skew)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ----------------------------------------------------------------------------
# Data collection
# ----------------------------------------------------------------------------
def list_images(folder: str | Path) -> list[Path]:
    """Return the sorted image files inside ``folder``."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder not found: {folder.resolve()}")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def board_points(pattern: tuple[int, int], square_size: float) -> np.ndarray:
    """3-D coordinates of the inner corners on the board plane (Z = 0).

    ``pattern`` = (columns, rows) of INNER corners; ``square_size`` in any unit
    (the unit of the returned translation vectors).
    """
    cols, rows = pattern
    pts = np.zeros((cols * rows, 3), np.float32)
    pts[:, :2] = np.mgrid[:cols, :rows].T.reshape(-1, 2) * square_size
    return pts


def detect_corners(gray: np.ndarray, pattern: tuple[int, int], work_width: int = 1200):
    """Find the inner chessboard corners of a (possibly very large) grey image.

    Detection runs on a down-scaled copy (fast, and less sensitive to the moire of a
    board shown on a screen); the corners are then scaled back and refined with
    ``cv2.cornerSubPix`` on the full-resolution image.

    Returns (ok, corners[N,1,2] float32, detector_name).
    """
    scale = min(1.0, work_width / gray.shape[1])
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    ok, corners = cv2.findChessboardCornersSB(small, pattern, cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE)
    name = "SB"
    if not ok:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, corners = cv2.findChessboardCorners(small, pattern, flags)
        name = "classic"
    if not ok:
        return False, None, "-"
    corners = (corners.astype(np.float32) / scale).reshape(-1, 1, 2)
    # sub-pixel window ~ 1/4 of a square, measured in full-resolution pixels
    pts = corners.reshape(-1, 2)
    square_px = np.median(np.linalg.norm(np.diff(pts.reshape(pattern[1], pattern[0], 2), axis=1), axis=2))
    win = int(np.clip(square_px / 4, 5, 25))
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
    corners = cv2.cornerSubPix(gray, corners, (win, win), (-1, -1), criteria)
    return True, corners, f"{name} + subpixel"


@dataclass
class CalibrationData:
    """Corner correspondences collected from a folder of checkerboard photos."""

    pattern: tuple[int, int]
    square_size: float
    image_size: tuple[int, int] | None = None  # (width, height)
    object_points: list[np.ndarray] = field(default_factory=list)
    image_points: list[np.ndarray] = field(default_factory=list)
    accepted: list[Path] = field(default_factory=list)
    log: list[dict] = field(default_factory=list)

    @property
    def n_views(self) -> int:
        return len(self.accepted)


def collect_correspondences(paths, pattern, square_size, work_width: int = 1200) -> CalibrationData:
    """Detect corners in every image and build the 3-D / 2-D point lists."""
    data = CalibrationData(tuple(pattern), float(square_size))
    world = board_points(pattern, square_size)
    for path in paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            data.log.append(dict(file=Path(path).name, status="UNREADABLE", detector="-"))
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        size = (gray.shape[1], gray.shape[0])
        if data.image_size is None:
            data.image_size = size
        if size != data.image_size:
            raise ValueError(f"{Path(path).name} is {size}, expected {data.image_size}: "
                             "all calibration photos must share one resolution")
        ok, corners, det = detect_corners(gray, pattern, work_width)
        data.log.append(dict(file=Path(path).name, status="ACCEPTED" if ok else "REJECTED", detector=det))
        if ok:
            data.object_points.append(world.copy())
            data.image_points.append(corners)
            data.accepted.append(Path(path))
    return data


# ----------------------------------------------------------------------------
# Projection model (shared by the from-scratch implementation)
# ----------------------------------------------------------------------------
def distort_normalised(xy: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Apply Brown-Conrady distortion (k1, k2, p1, p2, k3) to normalised points."""
    d = np.zeros(5)
    d[: len(dist)] = dist
    k1, k2, p1, p2, k3 = d
    x, y = xy[:, 0], xy[:, 1]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.column_stack([xd, yd])


def project_points(obj: np.ndarray, rvec, tvec, K: np.ndarray, dist) -> np.ndarray:
    """Pinhole projection with distortion: world points -> pixel coordinates."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))
    cam = obj @ R.T + np.asarray(tvec, float).reshape(1, 3)
    xy = cam[:, :2] / cam[:, 2:3]
    xy = distort_normalised(xy, np.asarray(dist, float).ravel())
    return np.column_stack([K[0, 0] * xy[:, 0] + K[0, 2], K[1, 1] * xy[:, 1] + K[1, 2]])


# ----------------------------------------------------------------------------
# Zhang's method from scratch
# ----------------------------------------------------------------------------
def _normalisation_matrix(pts: np.ndarray) -> np.ndarray:
    """Hartley normalisation: move centroid to origin, mean distance sqrt(2)."""
    mean = pts.mean(axis=0)
    scale = np.sqrt(2) / np.mean(np.linalg.norm(pts - mean, axis=1))
    return np.array([[scale, 0, -scale * mean[0]], [0, scale, -scale * mean[1]], [0, 0, 1]])


def homography_dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Normalised Direct Linear Transform: H such that dst ~ H @ src (N >= 4)."""
    Ts, Td = _normalisation_matrix(src), _normalisation_matrix(dst)
    s = (Ts @ np.column_stack([src, np.ones(len(src))]).T).T
    d = (Td @ np.column_stack([dst, np.ones(len(dst))]).T).T
    A = []
    for (x, y, _), (u, v, _) in zip(s, d):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def _v(H: np.ndarray, i: int, j: int) -> np.ndarray:
    """Zhang's constraint vector v_ij built from columns i, j of H."""
    h = H.T  # rows of h are the columns of H
    return np.array([
        h[i, 0] * h[j, 0],
        h[i, 0] * h[j, 1] + h[i, 1] * h[j, 0],
        h[i, 1] * h[j, 1],
        h[i, 2] * h[j, 0] + h[i, 0] * h[j, 2],
        h[i, 2] * h[j, 1] + h[i, 1] * h[j, 2],
        h[i, 2] * h[j, 2],
    ])


def intrinsics_from_homographies(Hs: list[np.ndarray], zero_skew: bool = True) -> np.ndarray:
    """Closed-form K from >= 3 plane homographies (Zhang 2000, Sec. 3.1 / App. B)."""
    V = []
    for H in Hs:
        V.append(_v(H, 0, 1))
        V.append(_v(H, 0, 0) - _v(H, 1, 1))
    V = np.asarray(V)
    if zero_skew:  # extra constraint B12 = 0  (gamma = 0)
        V = np.vstack([V, [0, 1, 0, 0, 0, 0]])
    _, _, Vt = np.linalg.svd(V)
    b11, b12, b22, b13, b23, b33 = Vt[-1]
    v0 = (b12 * b13 - b11 * b23) / (b11 * b22 - b12**2)
    lam = b33 - (b13**2 + v0 * (b12 * b13 - b11 * b23)) / b11
    alpha = np.sqrt(lam / b11)
    beta = np.sqrt(lam * b11 / (b11 * b22 - b12**2))
    gamma = -b12 * alpha**2 * beta / lam
    u0 = gamma * v0 / beta - b13 * alpha**2 / lam
    return np.array([[alpha, 0.0 if zero_skew else gamma, u0], [0, beta, v0], [0, 0, 1]])


def extrinsics_from_homography(K: np.ndarray, H: np.ndarray):
    """Rotation vector and translation of one board view from its homography."""
    Kinv = np.linalg.inv(K)
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
    lam = 1.0 / np.linalg.norm(Kinv @ h1)
    r1, r2 = lam * Kinv @ h1, lam * Kinv @ h2
    t = lam * Kinv @ h3
    if t[2] < 0:  # board must be in front of the camera
        r1, r2, t = -r1, -r2, -t
    R = np.column_stack([r1, r2, np.cross(r1, r2)])
    U, _, Vt = np.linalg.svd(R)  # closest true rotation matrix
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    rvec, _ = cv2.Rodrigues(R)
    return rvec.ravel(), t


def _pack(K, dist, rvecs, tvecs):
    return np.concatenate([[K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dist, np.ravel(rvecs), np.ravel(tvecs)])


def _unpack(p, n_dist, n_views):
    K = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]])
    dist = p[4:4 + n_dist]
    rest = p[4 + n_dist:]
    rvecs = rest[: 3 * n_views].reshape(n_views, 3)
    tvecs = rest[3 * n_views:].reshape(n_views, 3)
    return K, dist, rvecs, tvecs


def _residuals(p, objs, imgs, n_dist):
    K, dist, rvecs, tvecs = _unpack(p, n_dist, len(objs))
    res = [project_points(o, r, t, K, dist) - i.reshape(-1, 2) for o, i, r, t in zip(objs, imgs, rvecs, tvecs)]
    return np.concatenate(res).ravel()


def calibrate_zhang(object_points, image_points, n_dist: int = 5) -> dict:
    """Zhang's calibration from scratch.

    1. one normalised-DLT homography per view
    2. closed-form intrinsics K (zero skew)
    3. extrinsics (R, t) of every view
    4. non-linear least squares (Levenberg-Marquardt) over K, ``n_dist`` distortion
       coefficients (k1, k2, p1, p2, k3 order) and all extrinsics.
    """
    objs = [np.asarray(o, float)[:, :3] for o in object_points]
    imgs = [np.asarray(i, float).reshape(-1, 2) for i in image_points]
    Hs = [homography_dlt(o[:, :2], i) for o, i in zip(objs, imgs)]
    K0 = intrinsics_from_homographies(Hs)
    ext = [extrinsics_from_homography(K0, H) for H in Hs]
    rvecs0 = np.array([e[0] for e in ext])
    tvecs0 = np.array([e[1] for e in ext])
    dist0 = np.zeros(n_dist)
    p0 = _pack(K0, dist0, rvecs0, tvecs0)
    rms0 = np.sqrt(np.mean(_residuals(p0, objs, imgs, n_dist) ** 2) * 2)
    sol = least_squares(_residuals, p0, args=(objs, imgs, n_dist), method="lm", x_scale="jac", max_nfev=20000)
    K, dist, rvecs, tvecs = _unpack(sol.x, n_dist, len(objs))
    r = sol.fun.reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum(r**2, axis=1))))
    return dict(K=K, dist=np.pad(dist, (0, 5 - n_dist)), rvecs=rvecs, tvecs=tvecs, rms=rms,
                K_closed_form=K0, rms_closed_form=float(rms0), homographies=Hs,
                n_iterations=int(sol.nfev), success=bool(sol.success))


# ----------------------------------------------------------------------------
# OpenCV reference and evaluation
# ----------------------------------------------------------------------------
DISTORTION_MODELS = {
    # name: (calibrateCamera flags, number of free distortion coefficients)
    "0 - pinhole only": (cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST, 0),
    "1 - k1": (cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST, 1),
    "2 - k1,k2": (cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST, 2),
    "4 - k1,k2,p1,p2": (cv2.CALIB_FIX_K3, 4),
    "5 - k1,k2,p1,p2,k3": (0, 5),
    "8 - rational": (cv2.CALIB_RATIONAL_MODEL, 8),
}


def calibrate_opencv(object_points, image_points, image_size, flags: int = 0) -> dict:
    """Reference calibration with cv2.calibrateCameraExtended (also gives std-devs)."""
    out = cv2.calibrateCameraExtended(object_points, image_points, image_size, None, None, flags=flags)
    rms, K, dist, rvecs, tvecs, std_intr, std_extr, per_view = out
    return dict(K=K, dist=dist.ravel(), rvecs=np.array(rvecs).reshape(-1, 3), tvecs=np.array(tvecs).reshape(-1, 3),
                rms=float(rms), std_intrinsics=std_intr.ravel(), per_view_rms=per_view.ravel())


def reprojection_errors(object_points, image_points, K, dist, rvecs, tvecs) -> list[np.ndarray]:
    """Euclidean pixel distance between detected and re-projected corners, per view."""
    errs = []
    for o, i, r, t in zip(object_points, image_points, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, np.asarray(r, float), np.asarray(t, float), K, np.asarray(dist, float))
        errs.append(np.linalg.norm(i.reshape(-1, 2) - proj.reshape(-1, 2), axis=1))
    return errs


def cross_validate_models(object_points, image_points, image_size, models=None, k_folds: int | None = None,
                          seed: int = 0) -> list[dict]:
    """Hold-out reprojection error of each distortion model.

    Views are split into folds; each model is calibrated on the training views, then
    the pose of every held-out view is estimated with solvePnP (K, dist fixed) and its
    reprojection error is measured. A model that over-fits shows a low training RMS
    but a higher hold-out error.
    """
    models = models or DISTORTION_MODELS
    n = len(object_points)
    k_folds = k_folds or n  # default: leave-one-view-out
    folds = np.array_split(np.random.default_rng(seed).permutation(n), k_folds)
    rows = []
    for name, (flags, n_free) in models.items():
        train_rms, test_err = [], []
        for fold in folds:
            train = [j for j in range(n) if j not in set(fold)]
            cal = calibrate_opencv([object_points[j] for j in train], [image_points[j] for j in train], image_size, flags)
            train_rms.append(cal["rms"])
            for j in fold:
                ok, rv, tv = cv2.solvePnP(object_points[j], image_points[j], cal["K"], cal["dist"],
                                          flags=cv2.SOLVEPNP_ITERATIVE)
                proj, _ = cv2.projectPoints(object_points[j], rv, tv, cal["K"], cal["dist"])
                test_err.append(np.linalg.norm(image_points[j].reshape(-1, 2) - proj.reshape(-1, 2), axis=1))
        test_err = np.concatenate(test_err)
        rows.append({"model": name, "free_dist_coeffs": n_free,
                     "train_rms_px": float(np.mean(train_rms)),
                     "holdout_rms_px": float(np.sqrt(np.mean(test_err**2))),
                     "holdout_mean_px": float(np.mean(test_err))})
    return rows


def line_straightness(points: np.ndarray, pattern: tuple[int, int]) -> np.ndarray:
    """RMS distance (px) of each row/column of board corners from its best-fit line.

    Straight lines in the world stay straight under a pinhole camera; lens distortion
    bends them. After a correct undistortion this value should shrink towards 0.
    """
    cols, rows = pattern
    grid = points.reshape(rows, cols, 2)
    lines = [grid[r] for r in range(rows)] + [grid[:, c] for c in range(cols)]
    out = []
    for pts in lines:
        centred = pts - pts.mean(axis=0)
        _, s, _ = np.linalg.svd(centred, full_matrices=False)
        out.append(s[-1] / np.sqrt(len(pts)))  # RMS orthogonal residual
    return np.asarray(out)


def undistort_points_px(points: np.ndarray, K: np.ndarray, dist) -> np.ndarray:
    """Undistort pixel coordinates, returned in the same pixel frame (P = K)."""
    return cv2.undistortPoints(points.reshape(-1, 1, 2).astype(np.float64), K, np.asarray(dist, float), P=K).reshape(-1, 2)


def undistort_image(img: np.ndarray, K: np.ndarray, dist, alpha: float = 0.0):
    """Undistort with an optimal new camera matrix.

    alpha = 0 -> crop so that only valid pixels remain (no black border)
    alpha = 1 -> keep every source pixel (black border appears)
    """
    h, w = img.shape[:2]
    newK, roi = cv2.getOptimalNewCameraMatrix(K, np.asarray(dist, float), (w, h), alpha, (w, h))
    return cv2.undistort(img, K, np.asarray(dist, float), None, newK), newK, roi


def save_calibration(path: str | Path, K, dist, image_size, **extra) -> None:
    """Store K / dist / image size so other modules (and the Task 3 app) can reuse them."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, K=K, dist=np.asarray(dist), width=image_size[0], height=image_size[1], **extra)


def load_calibration(path: str | Path) -> dict:
    """Load a calibration file written by :func:`save_calibration`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}. Run notebook 1A first.")
    d = np.load(path)
    return {k: d[k] for k in d.files}
