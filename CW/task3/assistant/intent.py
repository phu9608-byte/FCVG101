"""Recognise which module the user is asking for (rubric 3.1 - application logic).

The request is free text in English or Vietnamese ("tìm cạnh bằng Canny", "what fruit is this?").
A transparent rule-based router is enough for seven tools and needs no external service:

1. normalise: lower case, strip Vietnamese diacritics (đ -> d), collapse spaces;
2. score every mode by the weighted keyword patterns it matches;
3. pick the best mode - "classify" + "measure" together become "inspect";
   no match or a tie -> ``mode = None`` and the UI asks the user to choose;
4. extract optional parameters (filter type, kernel size, SIFT/ORB, ratio, distance...).

The result keeps the matched keywords, so the UI can explain WHY a mode was chosen.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

MODES = {
    "calibrate": "Calibrate / undistort (Task 1A)",
    "filter": "Filtering & frequency domain (Task 1B)",
    "edges": "Edges & fruit outline (Task 1B)",
    "features": "Keypoints & matching (Task 1C)",
    "classify": "Classify fruit - CNN (Task 2)",
    "measure": "Measure fruit size (Task 1A + 1B)",
    "inspect": "Full inspection: classify + measure (Task 1 + 2)",
}

# (regex on normalised text, weight). Specific technique names weigh more than generic words.
KEYWORDS: dict[str, list[tuple[str, int]]] = {
    "calibrate": [
        (r"calibrat\w*", 3), (r"undistort\w*", 3), (r"distort\w*", 2), (r"hieu chinh", 3), (r"hieu chuan", 3),
        (r"khu meo", 3), (r"meo", 1), (r"bien dang", 2), (r"intrinsic\w*", 2), (r"ma tran k", 3),
        (r"camera matrix", 3), (r"thong so (camera|may anh)", 3), (r"checker ?board", 2), (r"chess ?board", 2),
        (r"ban co", 2), (r"ong kinh", 2), (r"lens", 2),
    ],
    "filter": [
        (r"filter\w*", 2), (r"loc", 2), (r"blur\w*", 2), (r"lam mo", 2), (r"smooth\w*", 2), (r"denois\w*", 3),
        (r"khu nhieu", 3), (r"giam nhieu", 3), (r"noise", 1), (r"nhieu", 1), (r"gauss\w*", 1), (r"median", 2),
        (r"trung vi", 2), (r"box", 1), (r"bilateral", 2), (r"fft", 3), (r"fourier", 3), (r"tan so", 3),
        (r"frequenc\w*", 3), (r"spectrum", 3), (r"pho", 1), (r"low ?pass", 3), (r"high ?pass", 3),
        (r"thong thap", 3), (r"thong cao", 3),
    ],
    "edges": [
        (r"edges?", 3), (r"canh", 3), (r"bien canh", 3), (r"duong bien", 3), (r"duong vien", 3), (r"vien", 1),
        (r"sobel", 3), (r"canny", 3), (r"gradients?", 2), (r"dao ham", 2), (r"contours?", 2), (r"outline", 2),
        (r"tach nen", 2), (r"segment\w*", 2),
    ],
    "features": [
        (r"features?", 2), (r"key ?points?", 3), (r"diem dac trung", 3), (r"dac trung", 2), (r"match\w*", 3),
        (r"so khop", 3), (r"khop", 2), (r"ghep", 1), (r"sift", 3), (r"orb", 3), (r"harris", 3), (r"corners?", 2),
        (r"diem goc", 2), (r"ransac", 3), (r"homograph\w*", 2), (r"tuong ung", 2), (r"correspond\w*", 2),
    ],
    "classify": [
        (r"classif\w*", 3), (r"phan loai", 3), (r"nhan dien", 3), (r"nhan dang", 3), (r"qua gi", 3),
        (r"loai qua", 3), (r"(what|which) (fruit|kind|type)", 3), (r"what is (this|it)", 2), (r"la gi", 2),
        (r"predict\w*", 2), (r"du doan", 2), (r"recogni[sz]\w*", 2), (r"identify", 2), (r"cnn", 2),
        (r"apple|banana|orange", 1), (r"grad ?cam", 2),
    ],
    "measure": [
        (r"measur\w*", 3), (r"sizes?", 2), (r"kich thuoc", 3), (r"duong kinh", 3), (r"diameter", 3),
        (r"area", 2), (r"dien tich", 3), (r"how (big|large)", 3), (r"bao nhieu (cm|mm)", 3),
        (r"(to|lon) bao nhieu", 3), (r"do (kich thuoc|duong kinh|dien tich|chieu|size)", 3),
        (r"chieu (dai|rong|cao)", 2),
    ],
    "inspect": [
        (r"inspect\w*", 3), (r"kiem tra", 2), (r"kiem dinh", 3), (r"toan bo", 2), (r"full", 2),
        (r"everything", 2), (r"tat ca", 2),
    ],
}

FILTER_WORDS = [  # (pattern, filter name) - first match wins
    (r"low ?pass|thong thap", "fft_lowpass"), (r"high ?pass|thong cao", "fft_highpass"),
    (r"median|trung vi", "median"), (r"bilateral|song phuong", "bilateral"),
    (r"gauss\w* (blur|filter|loc)|loc gauss|gaussian", "gaussian"), (r"box|mean|trung binh", "box"),
    (r"fft|fourier|tan so|frequenc", "fft_lowpass"),
]


def normalise(text: str) -> str:
    """Lower case, remove diacritics (Vietnamese included), keep letters/digits/basic symbols."""
    text = (text or "").replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn").lower()
    text = re.sub(r"[-_]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _find(pattern: str, text: str):
    return re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", text)


@dataclass
class Intent:
    text: str
    mode: str | None
    scores: dict = field(default_factory=dict)
    hits: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    reason: str = ""

    @property
    def candidates(self) -> list[str]:
        """Modes with a positive score, best first."""
        return [m for m, s in sorted(self.scores.items(), key=lambda kv: -kv[1]) if s > 0]


def extract_params(text: str) -> dict:
    """Optional parameters written in the request; anything not mentioned keeps the UI value."""
    t = normalise(text)
    p: dict = {}
    for pattern, name in FILTER_WORDS:
        if _find(pattern, t):
            p["filter"] = name
            break
    if m := re.search(r"(?<!\d)(\d{1,2})\s*[x×*]\s*(\d{1,2})(?!\d)", t):
        p["ksize"] = int(m.group(1))
    elif m := re.search(r"(?:kernel|ksize|k)\s*=?\s*(\d{1,2})(?!\d)", t):
        p["ksize"] = int(m.group(1))
    if "ksize" in p:
        p["ksize"] = int(min(31, max(3, p["ksize"] | 1)))          # odd, 3..31
    if _find(r"salt|muoi tieu|s&p|impulse", t):
        p["noise"] = "salt_pepper"
    elif _find(r"(add|them|cho|co) (gaussian )?(noise|nhieu)|nhieu gauss\w*|gaussian noise", t):
        p["noise"] = "gaussian"
    for method, pattern in (("Harris+SIFT", r"harris"), ("ORB", r"orb"), ("SIFT", r"sift")):
        if _find(pattern, t):
            p["method"] = method
            break
    if m := re.search(r"ratio\s*=?\s*(0?\.\d+|1(?:\.0+)?)", t):
        p["ratio"] = float(m.group(1))
    if m := re.search(r"(?:cut ?off|d0|ban kinh)\s*=?\s*(\d{1,3})", t):
        p["cutoff"] = int(m.group(1))
    if m := re.search(r"(?:threshold\w*|nguong)\s*=?\s*(\d{1,3})\s*(?:,|/|va|and|den|to|\s)\s*(\d{1,3})", t):
        lo, hi = sorted((int(m.group(1)), int(m.group(2))))
        p["canny"] = (lo, hi)
    if m := re.search(r"(\d+(?:[.,]\d+)?)\s*(mm|cm|m)(?![a-z])", t):
        value = float(m.group(1).replace(",", "."))
        p["distance_mm"] = value * {"mm": 1, "cm": 10, "m": 1000}[m.group(2)]
    return p


def detect_intent(text: str) -> Intent:
    """Map a free-text request to one of :data:`MODES` (or ``None`` when unclear)."""
    t = normalise(text)
    scores, hits = {}, {}
    for mode, patterns in KEYWORDS.items():
        found = [(m.group(0), w) for pat, w in patterns if (m := _find(pat, t))]
        hits[mode] = [word for word, _ in found]
        scores[mode] = sum(w for _, w in found)
    params = extract_params(text)
    intent = Intent(text=text, mode=None, scores=scores, hits=hits, params=params)

    if not t:
        intent.reason = "Empty request."
        return intent
    if scores["classify"] and scores["measure"]:
        scores["inspect"] += scores["classify"] + scores["measure"]
    ranked = intent.candidates
    if not ranked:
        intent.reason = "No known task keyword found."
        return intent
    best = ranked[0]
    if len(ranked) > 1 and scores[ranked[1]] == scores[best]:
        intent.reason = f"Ambiguous: '{ranked[0]}' and '{ranked[1]}' match equally."
        return intent
    intent.mode = best
    words = hits[best] or [w for m in ("classify", "measure") for w in hits[m]]
    intent.reason = "Matched: " + ", ".join(f"'{w}'" for w in words)
    return intent
