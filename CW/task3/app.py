"""FCVG101 Task 3 - FruitSort Vision Assistant (Streamlit GUI).

Run from the task3 folder:   streamlit run app.py

Flow of one request
    user text / quick button --> intent.detect_intent (which tool? which parameters?)
        --> tools.run(mode, images, params)  --> Task 1 engine (task1/src) or Task 2 model (task2/src)
        --> ToolResult (figures + numbers + warnings) rendered in the chat
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from assistant import engines, settings, tools
from assistant.errors import AssistantError
from assistant.intent import MODES, detect_intent

st.set_page_config(page_title="FruitSort Vision Assistant", page_icon="🍎", layout="wide")

AUTO = "Auto - understand my request"
EXAMPLES = {
    "calibrate": "Undistort this photo with the camera calibration",
    "filter": "Remove salt and pepper noise with a median 5x5 filter",
    "edges": "Find the edges with Canny and Sobel",
    "features": "Match SIFT keypoints between the two views",
    "classify": "What fruit is this?",
    "measure": "Measure the fruit size at 30 cm",
    "inspect": "Inspect this fruit: classify and measure it",
}
SHORT = {"calibrate": "📷 Undistort", "filter": "🌫️ Filter", "edges": "✏️ Edges", "features": "🔗 Match",
         "classify": "🏷️ Classify", "measure": "📏 Measure", "inspect": "🔍 Inspect"}
HISTORY_LIMIT = 6


# ----------------------------------------------------------------------------
# Engines: loaded once per server process (st.cache_resource); failures are not cached,
# so fixing a missing file and pressing "Reload" is enough.
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the Task 1 engine...")
def get_task1():
    return engines.load_task1()


@st.cache_resource(show_spinner="Loading the camera calibration...")
def get_calibration():
    return engines.load_calibration()


@st.cache_resource(show_spinner="Loading the Task 2 CNN...")
def get_classifier():
    return engines.load_classifier()


def attempt(loader):
    try:
        return loader(), None
    except AssistantError as exc:
        return None, exc


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def for_display(img: np.ndarray) -> np.ndarray:
    """BGR/grey uint8 -> RGB/grey, downscaled for the browser."""
    img, _ = tools.resize_max_side(img, settings.DISPLAY_MAX_SIDE)
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@st.cache_data(show_spinner=False)
def list_samples() -> dict[str, str]:
    """{label: path} of the sample images that exist on this machine."""
    out = {}
    for group, folder in settings.SAMPLE_DIRS.items():
        if folder.is_dir():
            for p in sorted(folder.rglob("*")):
                if p.suffix.lower() in settings.IMAGE_EXTENSIONS:
                    out[f"{group} / {p.relative_to(folder).as_posix()}"] = str(p)
    return out


def show_result(res: dict) -> None:
    """Render one assistant turn stored in the history."""
    if res.get("intent_line"):
        st.caption(res["intent_line"])
    if res.get("error"):
        st.error(res["error"])
        if res.get("hint"):
            st.info("How to fix: " + res["hint"])
        return
    st.markdown(res["summary"])
    images = res["images"]
    for row in range(0, len(images), 2):
        cols = st.columns(2)
        for col, (caption, img) in zip(cols, images[row:row + 2]):
            col.image(img, caption=caption, width="stretch")
    for w in res["warnings"]:
        st.warning(w)
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Results**")
        st.dataframe(pd.DataFrame({"value": [str(v) for v in res["metrics"].values()]},
                                  index=list(res["metrics"].keys())), width="stretch")
    if res.get("table"):
        with right:
            st.markdown("**Details**")
            st.dataframe(pd.DataFrame(res["table"]), hide_index=True, width="stretch")
    st.caption(f"Tool `{res['tool']}` ran in {res['elapsed_ms']:.0f} ms.")


# ----------------------------------------------------------------------------
# Sidebar: system status, mode override, parameters
# ----------------------------------------------------------------------------
task1, task1_err = attempt(get_task1)
calib, calib_err = attempt(get_calibration) if task1 else (None, task1_err)
clf, clf_err = attempt(get_classifier)

with st.sidebar:
    st.header("System status")
    if task1:
        st.success("Task 1 engine: task1/src loaded")
    else:
        st.error(f"Task 1 engine: {task1_err.message}")
        st.caption(task1_err.hint)
    if calib:
        st.success(f"Calibration: {calib.size[0]}x{calib.size[1]}, fx = {calib.K[0, 0]:.0f} px"
                   + (f", RMS {calib.rms:.2f} px" if calib.rms is not None else ""))
    else:
        st.warning(f"Calibration: {calib_err.message}")
        st.caption(calib_err.hint)
    if clf:
        acc = clf.meta.get("best_val_accuracy")
        st.success(f"Task 2 model: {clf.meta.get('architecture', 'FruitCNN')} - classes {', '.join(clf.classes)}"
                   + (f" (val acc {acc:.1%})" if acc is not None else ""))
    else:
        st.warning(f"Task 2 model: {clf_err.message}")
        st.caption(clf_err.hint)
    if st.button("Reload engines", width="stretch"):
        st.cache_resource.clear()
        st.rerun()

    st.header("Mode")
    mode_choice = st.selectbox("Which tool should run?", [AUTO] + list(MODES.values()), key="mode_choice",
                               help="Auto reads your request and picks the tool; choose a tool to force it.")
    forced_mode = None if mode_choice == AUTO else next(k for k, v in MODES.items() if v == mode_choice)

    st.header("Parameters")
    st.caption("Values written in the request (e.g. 'median 7x7', 'ORB', 'at 40 cm') override these.")
    params: dict = {}
    with st.expander("Filtering"):
        params["filter"] = st.selectbox("Filter", list(tools.SPATIAL_FILTERS + tools.FREQ_FILTERS), index=1,
                                        key="p_filter")
        params["ksize"] = st.slider("Kernel size", 3, 15, 5, step=2, key="p_ksize")
        params["noise"] = st.radio("Add noise first (denoising test)", ["none", "gaussian", "salt_pepper"],
                                   key="p_noise", horizontal=True)
        params["cutoff"] = st.slider("FFT cut-off D0", 5, 120, 30, key="p_cutoff")
        params["profile"] = st.selectbox("FFT mask", ["gaussian", "butterworth", "ideal"], key="p_profile")
    with st.expander("Edges"):
        if st.checkbox("Automatic Canny thresholds", True, key="p_auto_canny"):
            params["canny"] = None
        else:
            params["canny"] = st.slider("Canny low / high", 0, 255, (50, 150), key="p_canny")
    with st.expander("Features"):
        params["method"] = st.selectbox("Detector / descriptor", ["SIFT", "ORB", "Harris+SIFT"], key="p_method")
        params["ratio"] = st.slider("Lowe ratio", 0.5, 1.0, 0.75, step=0.05, key="p_ratio")
    with st.expander("Calibration & measurement"):
        params["alpha"] = st.slider("Undistort alpha (0 = crop, 1 = keep all)", 0.0, 1.0, 0.0, key="p_alpha")
        dist = st.number_input("Camera-object distance Z (mm, 0 = unknown)", 0, 5000, 0, step=10, key="p_dist")
        params["distance_mm"] = float(dist) or None

# ----------------------------------------------------------------------------
# Main area: images + chat
# ----------------------------------------------------------------------------
st.title("🍎 FruitSort Vision Assistant")
st.caption("Ask in English or Vietnamese. The assistant picks the right module - Task 1 classical CV "
           "(calibration, filtering, edges, features, measurement) or the Task 2 CNN (classification) - "
           "runs it on your image and shows the result.")

source = st.radio("Image source", ["Upload", "Sample images"], horizontal=True, key="source")
inputs: list[tuple[str, bytes]] = []
if source == "Upload":
    files = st.file_uploader("Images (1 for most tools, 2 for matching, 3+ checkerboards to re-calibrate)",
                             type=sorted(e.lstrip(".") for e in settings.IMAGE_EXTENSIONS),
                             accept_multiple_files=True, key="uploads")
    inputs = [(f.name, f.getvalue()) for f in files or []]
else:
    samples = list_samples()
    if not samples:
        st.info("No sample folders found on this machine; use Upload.")
    else:
        picked = st.multiselect("Sample images", list(samples), default=list(samples)[:1], key="samples",
                                max_selections=20)
        inputs = [(Path(samples[k]).name, Path(samples[k]).read_bytes()) for k in picked]

decoded, decoded_names, input_errors = [], [], []
for name, data in inputs:
    try:
        decoded.append(tools.decode_image(data, name))
        decoded_names.append(name)
    except AssistantError as exc:
        input_errors.append(exc)
for exc in input_errors:
    st.error(exc.message)
    st.caption("How to fix: " + exc.hint)
if decoded:
    thumbs = st.columns(max(4, min(len(decoded), 6)))       # fixed small thumbnails even for one image
    for col, img, name in zip(thumbs, decoded, decoded_names):
        col.image(for_display(img), caption=f"{name} ({img.shape[1]}x{img.shape[0]})", width="stretch")

st.markdown("**Quick actions**")
quick = None
cols = st.columns(4) + st.columns(4)
for col, mode in zip(cols, MODES):
    if col.button(SHORT[mode], key=f"quick_{mode}", width="stretch", help=f"{MODES[mode]}: '{EXAMPLES[mode]}'"):
        quick = (EXAMPLES[mode], mode)

if "history" not in st.session_state:
    st.session_state.history = []

request = st.chat_input("e.g. 'phân loại quả này', 'tìm cạnh bằng Canny', 'so khớp ORB', 'đo kích thước ở 30 cm'")
pending = quick or ((request, None) if request else None)

if pending:
    text, button_mode = pending
    intent = detect_intent(text)
    mode = button_mode or forced_mode or intent.mode
    turn = {"request": text}
    if button_mode:
        turn["intent_line"] = f"Quick action -> {MODES[mode]}"
    elif forced_mode:
        turn["intent_line"] = f"Mode forced in the sidebar -> {MODES[mode]}"
    elif mode:
        turn["intent_line"] = f"Understood: {MODES[mode]}  ({intent.reason})"
    run_params = {**params, **intent.params}
    if intent.params:
        turn["intent_line"] = turn.get("intent_line", "") + "  |  parameters from text: " + ", ".join(
            f"{k}={v}" for k, v in intent.params.items())

    if mode is None:
        options = ", ".join(MODES[m].split(" (")[0] for m in intent.candidates) or "none"
        turn.update(error=f"I could not tell which tool you want. {intent.reason}",
                    hint=f"Mention a task - calibrate, filter/denoise, edges, keypoints/matching, classify, measure "
                         f"or inspect - or pick a mode in the sidebar. Closest matches: {options}.")
    elif input_errors and not decoded:
        turn.update(error="None of the selected files could be read as an image.", hint=input_errors[0].hint)
    elif not decoded:
        turn.update(error="No image yet.", hint="Upload a photo or switch to 'Sample images' above, then ask again.")
    else:
        try:
            res = tools.run(mode, decoded, run_params, clf=clf, calib=calib)
            turn.update(tool=res.tool, summary=res.summary, metrics=res.metrics, table=res.table,
                        warnings=res.warnings, elapsed_ms=res.elapsed_ms,
                        images=[(c, for_display(i)) for c, i in res.images])
        except AssistantError as exc:
            turn.update(error=exc.message, hint=exc.hint)
        except Exception as exc:  # never crash the UI; show the technical message instead
            turn.update(error=f"Unexpected error in '{mode}': {type(exc).__name__}: {exc}",
                        hint="Try another image or other parameters. If it persists, check the terminal log.")
    st.session_state.history = (st.session_state.history + [turn])[-HISTORY_LIMIT:]

if not st.session_state.history:
    with st.chat_message("assistant"):
        st.markdown("Hi! Choose an image, then tell me what to do, for example:\n\n"
                    + "\n".join(f"- *{EXAMPLES[m]}*" for m in MODES))
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["request"])
    with st.chat_message("assistant"):
        show_result(turn)
