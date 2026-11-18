// Builds report/FCVG101_Coursework_Report.docx from the project results.
// Run from the CourseWork folder:  node report/build/build_report.js
// Tables are read from the CSV/JSON files written by Tasks 1-3, figures from report/figures/jpg,
// and Appendix A lists the source code files verbatim.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, BorderStyle,
  ShadingType, AlignmentType, HeadingLevel, PageBreak, Header, Footer, PageNumber, TableOfContents,
  LevelFormat, VerticalAlign, TabStopType, ExternalHyperlink,
} = require("docx");

const CW = path.resolve(__dirname, "..", "..");
const FIG = path.join(CW, "report", "figures", "jpg");
const T1 = path.join(CW, "task1");
const T2 = path.join(CW, "task2");
const REPO = "https://github.com/phu9608-byte/FCVG101";
const MODEL_URL = "https://github.com/phu9608-byte/FCVG101/blob/main/CW/task2/artifacts/model.pth";
const T3 = path.join(CW, "task3");
const OUT = path.join(CW, "report", "FCVG101_Coursework_Report.docx");

const PAGE_W = 11906, MARGIN = 1134, CONTENT = PAGE_W - 2 * MARGIN; // A4, 2 cm margins
const FONT = "Calibri", MONO = "Consolas", MATH = "Cambria Math";
const ACCENT = "1F3864";

// ---------------------------------------------------------------- helpers: data
const readJSON = (p) => JSON.parse(fs.readFileSync(p, "utf8").replace(/^\uFEFF/, "").replace(/\bNaN\b/g, "null"));
function readCSV(p) {
  const text = fs.readFileSync(p, "utf8").replace(/^\uFEFF/, "");
  const rows = [];
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue;
    const cells = []; let cur = "", q = false;
    for (const ch of line) {
      if (ch === '"') q = !q; else if (ch === "," && !q) { cells.push(cur); cur = ""; } else cur += ch;
    }
    cells.push(cur); rows.push(cells);
  }
  return rows;
}
const f = (x, d = 2) => (x === null || x === undefined || x === "" || isNaN(Number(x)) ? "–" : Number(x).toFixed(d));
const pct = (x, d = 1) => (Number(x) * 100).toFixed(d) + " %";

function jpegSize(file) {
  const b = fs.readFileSync(file);
  let i = 2;
  while (i < b.length) {
    if (b[i] !== 0xff) { i++; continue; }
    const m = b[i + 1], len = b.readUInt16BE(i + 2);
    if (m >= 0xc0 && m <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(m)) return { h: b.readUInt16BE(i + 5), w: b.readUInt16BE(i + 7) };
    i += 2 + len;
  }
  throw new Error("no SOF in " + file);
}

// ---------------------------------------------------------------- helpers: numbering of figures/tables
let figNo = {}, tabNo = {}, figCount = 0, tabCount = 0;
function resolve(text) {
  return text.replace(/\{(fig|tab):([\w-]+)\}/g, (_, kind, key) => {
    const n = (kind === "fig" ? figNo : tabNo)[key];
    return (kind === "fig" ? "Figure " : "Table ") + (n || "?");
  });
}

// inline markup: **bold**, *italic*, `code`
function runs(text, base = {}) {
  text = resolve(text);
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`|\*[^*\s][^*]*\*)/g).filter((s) => s !== "");
  return parts.map((s) => {
    if (s.startsWith("**")) return new TextRun({ ...base, text: s.slice(2, -2), bold: true });
    if (s.startsWith("`")) return new TextRun({ ...base, text: s.slice(1, -1), font: MONO, size: (base.size || 21) - 2 });
    if (s.startsWith("*") && s.length > 2) return new TextRun({ ...base, text: s.slice(1, -1), italics: true });
    return new TextRun({ ...base, text: s });
  });
}

const P = (text, opts = {}) => new Paragraph({ children: runs(text, opts.run || {}), alignment: opts.align || AlignmentType.JUSTIFIED, spacing: { after: 120, line: 276 }, ...(opts.para || {}) });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)], pageBreakBefore: true });
const H1n = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const bullet = (t, level = 0) => new Paragraph({ numbering: { reference: "bullets", level }, children: runs(t), spacing: { after: 60, line: 264 }, alignment: AlignmentType.LEFT });
const num = (t, ref = "num1") => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(t), spacing: { after: 60, line: 264 } });
const eq = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 120 }, children: [new TextRun({ text: t, font: MATH, size: 22 })] });
const spacer = () => new Paragraph({ children: [], spacing: { after: 0 } });

function figure(key, caption, widthFrac = 1.0) {
  figNo[key] = ++figCount;
  const file = path.join(FIG, key + ".jpg");
  const { w, h } = jpegSize(file);
  const maxPx = Math.round((CONTENT / 1440) * 96 * widthFrac);
  const width = maxPx;
  const height = Math.round((width * h) / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "jpg", data: fs.readFileSync(file), transformation: { width, height },
        altText: { title: key, description: caption.replace(/[*`]/g, ""), name: key } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: `Figure ${figCount}. `, bold: true, size: 18, color: ACCENT }), ...runs(caption, { size: 18, italics: true })] }),
  ];
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "A6A6A6" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(key, caption, headers, rows, fracs, opts = {}) {
  tabNo[key] = ++tabCount;
  const total = fracs.reduce((a, b) => a + b, 0);
  const widths = fracs.map((x) => Math.floor((x / total) * CONTENT));
  widths[widths.length - 1] += CONTENT - widths.reduce((a, b) => a + b, 0);
  const size = opts.size || 18;
  const cell = (text, i, head, shade) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
    shading: head ? { fill: "D9E2F3", type: ShadingType.CLEAR, color: "auto" } : shade ? { fill: "F2F2F2", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({ alignment: i === 0 || opts.left ? AlignmentType.LEFT : AlignmentType.CENTER, spacing: { after: 0 },
      children: runs(String(text), { size, bold: head }) })],
  });
  return [
    new Paragraph({ keepNext: true, spacing: { before: 160, after: 80 }, alignment: AlignmentType.LEFT,
      children: [new TextRun({ text: `Table ${tabCount}. `, bold: true, size: 18, color: ACCENT }), ...runs(caption, { size: 18, italics: true })] }),
    new Table({ width: { size: CONTENT, type: WidthType.DXA }, columnWidths: widths,
      rows: [new TableRow({ tableHeader: true, children: headers.map((t, i) => cell(t, i, true)) }),
        ...rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, i, false, opts.highlight && opts.highlight(ri))) }))] }),
    new Paragraph({ children: [], spacing: { after: 120 } }),
  ];
}

// code block: one paragraph per line, light grey shading
function codeBlock(lines, size = 16) {
  const shade = { fill: "F5F5F5", type: ShadingType.CLEAR, color: "auto" };
  return lines.map((ln, i) => new Paragraph({
    shading: shade, keepLines: true, keepNext: i < lines.length - 1 && lines.length < 40,
    spacing: { before: 0, after: 0, line: Math.round(size * 11.5), lineRule: "exact" },
    children: [new TextRun({ text: ln.replace(/\t/g, "    ") || " ", font: MONO, size })],
  }));
}
function snippet(file, startPat, endPat, label) {
  const lines = fs.readFileSync(file, "utf8").replace(/^\uFEFF/, "").split(/\r?\n/);
  const s = lines.findIndex((l) => l.includes(startPat));
  let e = endPat ? lines.findIndex((l, i) => i > s && l.includes(endPat)) : lines.length;
  if (s < 0) throw new Error("snippet start not found: " + startPat);
  if (e < 0) e = lines.length;
  const rel = path.relative(CW, file).replace(/\\/g, "/");
  return [new Paragraph({ keepNext: true, spacing: { before: 120, after: 40 }, children: [new TextRun({ text: `${label || "Listing"} – ${rel}, lines ${s + 1}–${e}`, size: 17, italics: true, color: "595959" })] }),
    ...codeBlock(lines.slice(s, e).map((l) => l.replace(/\s+$/, ""))), spacer()];
}

// Emoji only occur inside Python string literals (app.py button labels); they are printed as the
// equivalent escape sequences so that the PDF does not embed the 4 MB Segoe UI Emoji font.
function escapeSymbols(line) {
  return Array.from(line).map((ch) => {
    const cp = ch.codePointAt(0);
    if (cp > 0xffff) return "\\U" + cp.toString(16).toUpperCase().padStart(8, "0");
    if ((cp >= 0x2600 && cp <= 0x27bf) || (cp >= 0xfe00 && cp <= 0xfe0f)) return "\\u" + cp.toString(16).toUpperCase().padStart(4, "0");
    return ch;
  }).join("");
}

// ---------------------------------------------------------------- data from the project
const m1a = readJSON(path.join(T1, "results/1A/metrics.json"));
const m1b = readJSON(path.join(T1, "results/1B/metrics.json"));
const m1c = readJSON(path.join(T1, "results/1C/metrics.json"));
const m2 = readJSON(path.join(T2, "results/test_metrics.json"));
const m2base = readJSON(path.join(T2, "runs/30ep_baseline/test_metrics.json"));
const t3log = readJSON(path.join(CW, "report/figures/t3_log.json"));
const run3 = Object.fromEntries(t3log.runs.map((r) => [r.figure, r]));

// ================================================================ CONTENT
function content() {
  figNo = figNo; figCount = 0; tabCount = 0;
  const C = [];
  const add = (...xs) => xs.forEach((x) => (Array.isArray(x) ? C.push(...x) : C.push(x)));

  // ------------------------------------------------------------ title page
  add(new Paragraph({ spacing: { before: 1800, after: 120 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: "FCVG101 – Introduction to Computer Vision", size: 28, color: "595959" })] }));
  add(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: "Coursework Report", size: 32, color: "595959" })] }));
  add(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 400, after: 200 }, children: [new TextRun({ text: "FruitSort Vision Assistant", bold: true, size: 56, color: ACCENT })] }));
  add(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 800 }, children: [new TextRun({ text: "A visual inspection assistant for fruit sorting: camera calibration, classical image analysis, a CNN classifier and a tool-using assistant app", size: 24, italics: true })] }));
  const hl = (t) => new TextRun({ text: t, highlight: "yellow", size: 22 });
  const lab = (t) => new TextRun({ text: t, bold: true, size: 22 });
  for (const [k, v] of [["Student name: ", "[your full name]"], ["Student ID: ", "[your student ID]"], ["Class / group: ", "[class]"], ["Module leader: ", "[module leader]"]]) {
    add(new Paragraph({ alignment: AlignmentType.LEFT, indent: { left: 1700 }, spacing: { after: 100 }, children: [lab(k), hl(v)] }));
  }
  for (const [k, v] of [["Source code repository: ", REPO], ["Trained model artifact (model.pth): ", MODEL_URL]]) {
    add(new Paragraph({ alignment: AlignmentType.LEFT, indent: { left: 1700, hanging: 0 }, spacing: { after: 100 }, children: [lab(k), new ExternalHyperlink({ link: v, children: [new TextRun({ text: v, style: "Hyperlink", size: 20 })] })] }));
  }
  add(new Paragraph({ alignment: AlignmentType.LEFT, indent: { left: 1700 }, spacing: { after: 100 }, children: [lab("Submission: "), new TextRun({ text: "single PDF, contribution 60 % of the module grade", size: 22 })] }));
  add(new Paragraph({ children: [new PageBreak()] }));

  // ------------------------------------------------------------ contents
  add(new Paragraph({ children: [new TextRun({ text: "Contents", bold: true, size: 32, color: ACCENT })], spacing: { after: 200 } }));
  add(new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }));

  // ============================================================ 1. INTRODUCTION
  add(H1("1. Introduction"));
  add(H2("1.1 Scenario and goals"));
  add(P("I worked as a computer vision engineer on an outsourced MVP: **FruitSort Vision Assistant**, a visual inspection assistant for an organic fruit farm or a fruit warehouse. A manager takes a photo of a fruit (or of a crate) and asks the assistant, in English or Vietnamese, what to do with it: identify the fruit, measure it, check image quality, find edges and outlines, or compare two views. The assistant decides which module the request needs, runs it and returns the result as figures and numbers."));
  add(P("The MVP covers the three minimum requirements of the brief:"));
  add(bullet("**(a) Image formation and camera geometry.** A phone camera is calibrated with a checkerboard. The pinhole model with Brown–Conrady lens distortion gives the intrinsic matrix *K* and the distortion coefficients, which are used to undistort photos and to convert fruit size from pixels to millimetres (Task 1A)."));
  add(bullet("**(b) Classical image processing with quantitative results.** Convolution, denoising, Sobel/Canny edges, Fourier filtering, Harris/SIFT/ORB features and ratio-test matching, each measured with an objective metric (PSNR, SSIM, F1, match precision) (Task 1B, 1C)."));
  add(bullet("**(c) A trained deep-learning model.** A small CNN classifies apple / banana / orange photos from a public Kaggle dataset (Task 2). Its saved checkpoint is called by the assistant, which also explains each prediction with Grad-CAM (Task 3)."));
  add(H2("1.2 System architecture"));
  add(P("Everything is in the folder `CW/` of the repository: `datasets/` (checkerboard photos and fruit images), one folder per task, and `report/`. Every task is run from Jupyter notebooks; only the modules imported by other tasks and the Streamlit app are `.py` files. Task 1 and Task 2 are libraries with notebooks or scripts that produce results and artifacts. Task 3 imports their code directly; nothing is copied, so the app runs exactly the code that produced the numbers in this report."));
  add(table("arch", "Components of the system and how they are connected.", ["Component", "Folder", "Role", "Produces / consumes"], [
    ["Task 1 engine", "task1/src/", "calibration.py, filtering.py, features.py: classical CV functions, several implemented from scratch and checked against OpenCV", "Writes artifacts/calibration.npz (K, dist, image size)"],
    ["Task 1 experiments", "task1/*.ipynb", "Notebooks 1A, 1B, 1C run the experiments and save figures/tables", "results/1A, 1B, 1C (PNG, CSV, metrics.json)"],
    ["Task 2 pipeline", "task2/Task2_pipeline.ipynb, task2/src/", "Notebook: ingestion, cleaning, near-duplicate check, grouped split, training, evaluation; src/: model, preprocessing, inference API", "artifacts/model.pth, classes.json, data_manifest.csv; results/"],
    ["Task 3 assistant", "task3/Task3_assistant.ipynb, app.py, assistant/", "Streamlit GUI, intent router, tool dispatcher, Grad-CAM, input validation; notebook with demos and integration tests", "Reads calibration.npz and model.pth; calls Task 1/2 code"],
  ], [1.3, 1.6, 3.4, 2.6], { left: true }));
  add(P("A request flows as follows: *user text* → `intent.detect_intent` (which tool, which parameters) → `tools.run(mode, images, params)` → Task 1 engine or Task 2 model → `ToolResult` (figures, metrics, table, warnings) → rendered in the chat."));
  add(H2("1.3 Where each marking criterion is addressed"));
  add(table("rubric", "Mapping between the marking scheme and the sections of this report.", ["Criterion", "Section(s)", "Main evidence"], [
    ["1.1 Image formation & calibration", "2.1", "Zhang's method from scratch = OpenCV to 2.3e-8; K, dist with 95 % CI; RMS 0.458 px; undistortion straightens lines 0.743 → 0.227 px"],
    ["1.2 Filtering & edges", "2.2", "Convolution from scratch = OpenCV to 1.4e-13; PSNR/SSIM of 4 filters × 3 sizes × 2 noises; Canny F1 0.26 → 0.92; notch filter +6.7 dB"],
    ["1.3 Features & matching", "2.3", "Harris from scratch (corr. 1.0); SIFT/ORB/Harris+SIFT on ground truth; ratio sweep; rotation/scale robustness; RANSAC decision"],
    ["1.4 Core code quality", "2.4, App. A", "Reusable modules with docstrings, verified against OpenCV, used unchanged by Task 3"],
    ["2.1 Data pipeline", "3.2", "Ingestion log, manual + automatic cleaning, SHA-256 and pHash duplicate checks, grouped stratified split, train-only augmentation"],
    ["2.2 Model implementation", "3.3, 3.4", "FruitCNN (23,779 parameters), justified for 255 images; early stopping on validation loss"],
    ["2.3 Evaluation", "3.5", "Test accuracy 97.4 %, macro F1 0.973, per-class report, confusion matrix, confidence interval, error analysis"],
    ["3.1 Application logic", "4.3", "Bilingual weighted-keyword intent router, parameter extraction, ambiguity handling; 19/19 routing tests"],
    ["3.2 Tool integration", "4.4–4.6", "7 modes calling Task 1/2 code; Grad-CAM; error handling; 12/12 integration tests"],
    ["4.1 / 4.2 Report, README", "All, 5, App. A", "This report; README.md in each task folder; run instructions in Section 5"],
  ], [2.2, 1.1, 5.6], { left: true }));
  add(H2("1.4 Software environment"));
  add(P("Python 3.12.10 on Windows 11. NumPy 2.5.4, OpenCV 5.x (the notebooks ran with 5.0.0, the app was tested with 5.1.0), SciPy, pandas, Matplotlib, PyTorch 2.14.1 (+cu130) with torchvision 0.29.1, scikit-learn, Pillow 13 and Streamlit 1.67. The CNN was trained on an NVIDIA RTX 4060 Laptop GPU; inference in the app runs on the CPU."));

  // ============================================================ 2. TASK 1
  add(H1("2. Task 1 – Core Image Analysis Engine"));
  add(P("Task 1 is the classical computer-vision engine. The algorithms live in `task1/src/` so that the Task 3 app can import them, and three notebooks run the experiments: **1A** camera model and calibration, **1B** filtering, edges and the Fourier domain, **1C** interest points and matching. Where the brief recommends it for a higher mark, the key techniques are implemented **from scratch** (Zhang's calibration with DLT and Levenberg–Marquardt, 2-D convolution and median filter, Harris corners) and each implementation is checked numerically against the OpenCV function. Every notebook ends with a summary printed from the actual numbers, so the text in this report can be traced to the saved CSV and JSON files."));

  // ---- 2.1
  add(H2("2.1 Image formation and camera calibration (1A)"));
  add(H3("2.1.1 Pinhole camera model and lens distortion"));
  add(P("A 3-D point (X, Y, Z) in the camera frame projects to normalised image coordinates x = X/Z, y = Y/Z. The intrinsic matrix *K* converts them to pixels:"));
  add(eq("[u, v, 1]ᵀ = K · [x, y, 1]ᵀ ,     K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]"));
  add(P("For every calibration photo, a board point in world coordinates (Zw = 0 on the board plane) is first moved into the camera frame by the extrinsic parameters [R | t] of that view. Real lenses bend straight lines, so before applying *K* the normalised point is distorted with the Brown–Conrady model used by OpenCV, where r² = x² + y²:"));
  add(eq("xd = x(1 + k1r² + k2r⁴ + k3r⁶) + 2p1xy + p2(r² + 2x²)"));
  add(eq("yd = y(1 + k1r² + k2r⁴ + k3r⁶) + p1(r² + 2y²) + 2p2xy"));
  add(P("Calibration estimates fx, fy, cx, cy, the distortion coefficients (k1, k2, p1, p2, k3) and one pose per view by minimising the **reprojection error**: the pixel distance between each detected board corner and the corner projected with the current parameters."));

  add(H3("2.1.2 Data and corner detection"));
  add(P(`The data are **${m1a.n_images_total} photos of a checkerboard with ${m1a.pattern_inner_corners[0]} × ${m1a.pattern_inner_corners[1]} inner corners** (square side ${m1a.square_mm} mm, shown full-screen on a flat laptop display), all taken with the main camera of one phone at ${m1a.image_size_wh[0]} × ${m1a.image_size_wh[1]} px, plus an independent test photo of the board and a photo of floor tiles that were not used for calibration. The views vary in distance, tilt and position in the frame.`));
  add(P("Corners are found with `cv2.findChessboardCornersSB` on a copy shrunk to 1200 px width (fast, and robust to the moiré pattern of a photographed screen) and then refined to sub-pixel accuracy with `cv2.cornerSubPix` on the full 12 MP image (`calibration.detect_corners`). All 15 photos were accepted with 54 corners each, giving 810 3-D/2-D correspondences ({fig:1a_corners}). {fig:1a_coverage} shows how much of the frame the corners cover: the farthest corner reaches a normalised radius of 0.84 (1.0 = image corner), so the distortion model is extrapolated in the four extreme corners."));
  add(figure("1a_corners", "Six of the fifteen calibration photos with the 54 detected inner corners (SB detector + cornerSubPix)."));
  add(figure("1a_coverage", "Corner coverage of the 4000 × 3000 frame. Left: all corners, one colour per photo. Right: number of corners per cell of an 8 × 6 grid (every cell contains corners; the centre is sampled most).", 0.9));

  add(H3("2.1.3 Zhang's method implemented from scratch"));
  add(P("`src/calibration.py` implements Zhang's method (Zhang, 2000) with NumPy SVD and SciPy `least_squares` only:"));
  add(num("**Homography per view** (`homography_dlt`). The board is planar, so its image is a homography H = K[r1 r2 t]. *Ah* = 0 is solved by SVD with Hartley normalisation for numerical stability. The scratch H reprojects the corners within 0.015 px of `cv2.findHomography`.", "num1"));
  add(num("**Closed-form K** (`intrinsics_from_homographies`). r1 ⊥ r2 and ‖r1‖ = ‖r2‖ give two linear equations per view in B = K⁻ᵀK⁻¹; B is the null vector of the stacked system (SVD) and K follows by Cholesky-like decomposition.", "num1"));
  add(num("**Extrinsics** (`extrinsics_from_homography`). r1 = λK⁻¹h1, r2 = λK⁻¹h2, r3 = r1 × r2, t = λK⁻¹h3, then the nearest true rotation by SVD.", "num1"));
  add(num("**Non-linear refinement** (`calibrate_zhang`). All intrinsics, the 5 distortion coefficients and the 6 pose parameters of every view are refined together with Levenberg–Marquardt on the reprojection residuals.", "num1"));
  add(...snippet(path.join(T1, "src/calibration.py"), "def intrinsics_from_homographies", "def extrinsics_from_homography", "Listing 1"));
  const zh = readCSV(path.join(T1, "results/1A/zhang_vs_opencv.csv"));
  const zname = ["Scratch Zhang, closed form (step 2, no distortion)", "Scratch Zhang, after LM (step 4)", "OpenCV cv2.calibrateCamera"];
  add(table("zhang", "Scratch implementation versus OpenCV on all 15 views (pixels; distortion coefficients are unitless).", ["Method", "fx", "fy", "cx", "cy", "k1", "k2", "k3", "RMS (px)"],
    zh.slice(1).map((r, i) => [zname[i], f(r[1]), f(r[2]), f(r[3]), f(r[4]), f(r[5], 4), f(r[6], 4), f(r[9], 4), f(r[10], 4)]), [3.2, 1, 1, 1, 1, 0.9, 0.9, 0.9, 0.9]));
  add(P(`The closed-form solution ignores distortion, so its RMS error (3.50 px) is large; it is only the starting point. After LM refinement the scratch result is **identical to OpenCV**: the largest relative difference of fx, fy, cx, cy is **${m1a.zhang_scratch_vs_opencv_max_rel_diff.toExponential(1)}**, i.e. rounding error. OpenCV follows the same chain (homography initialisation, then LM), which confirms that the implementation is correct.`));

  add(H3("2.1.4 Reprojection error and outlier rejection"));
  add(P("Each corner is re-projected with the estimated K, dist, R and t (`cv2.projectPoints`) and compared with the detected corner. With all 15 views the RMS error is 0.508 px (mean 0.422 px, median 0.360 px, maximum 2.129 px on a 4000 × 3000 image). A view is treated as abnormal if its mean error exceeds **2 × the median view error = 0.812 px**. Only `20261021_153036.jpg` (mean 0.828 px, maximum 2.13 px) exceeds it; its error vectors point in the same direction ({fig:1a_reproj}, right), which suggests a slightly moved board or motion blur rather than random noise. Removing it reduces the RMS error from 0.508 px to **0.458 px**, and the remaining 14 views are used in all later steps."));
  add(figure("1a_reproj", "Reprojection error. Left: mean error per view (dashed line = mean). Middle: distribution over all corners. Right: error vectors of the worst view, magnified 200×."));

  add(H3("2.1.5 Choosing the distortion model with cross-validation"));
  add(P("RMS error on the calibration images always decreases when parameters are added, even when the model starts to fit noise. The notebook therefore uses **leave-one-view-out cross-validation**: one view is held out, the camera is calibrated on the others, and for the held-out view only the pose is estimated (`cv2.solvePnP` with K and dist fixed) before its reprojection error is measured. Six models are compared. **Selection rule:** the simplest model whose hold-out error is within 5 % of the best model."));
  const cv = readCSV(path.join(T1, "results/1A/distortion_model_cv.csv"));
  add(table("cv", "Leave-one-view-out comparison of distortion models (14 views). The chosen model is shaded.", ["Model", "Free coefficients", "Train RMS (px)", "Hold-out RMS (px)", "Hold-out mean (px)"],
    cv.slice(1).map((r) => [r[0] + (r[5] ? "  ◄ chosen" : ""), r[1], f(r[2], 3), f(r[3], 3), f(r[4], 3)]), [3, 1.4, 1.4, 1.5, 1.5], { highlight: (i) => !!cv[i + 1][5] }));
  add(P("Two large drops appear: adding k2 (2.32 → 1.11 px) and adding k3 (1.11 → 0.52 px). The 8-coefficient rational model is only 3.8 % better than the 5-coefficient model (0.500 vs 0.519 px), so by the rule the **5-coefficient model (k1, k2, p1, p2, k3)** is chosen. The strong high-order term is typical of a wide-angle phone lens whose image processor already compensates part of the distortion."));
  add(figure("1a_modelcv", "Train versus hold-out RMS error of each distortion model. The gap between the two bars of a model shows over-fitting.", 0.7));

  add(H3("2.1.6 Final intrinsic parameters"));
  add(P("The final calibration (`cv2.calibrateCameraExtended`, 14 views, 5 coefficients) gives an RMS reprojection error of **0.458 px** and the following camera matrix and distortion vector:"));
  const K = m1a.K;
  add(eq(`K = [[${f(K[0][0], 1)}, 0, ${f(K[0][2], 1)}], [0, ${f(K[1][1], 1)}, ${f(K[1][2], 1)}], [0, 0, 1]]  (px)`));
  add(eq(`dist = (k1, k2, p1, p2, k3) = (${m1a.dist.map((d) => f(d, 4)).join(", ")})`));
  const fp = readCSV(path.join(T1, "results/1A/final_parameters.csv"));
  add(table("final", "Final parameters with standard deviation and 95 % confidence interval (from calibrateCameraExtended).", ["Parameter", "Value", "Std. dev.", "95 % CI low", "95 % CI high"],
    fp.slice(1).map((r) => { const d = ["k1", "k2", "p1", "p2", "k3"].includes(r[0]) ? 5 : 2; return [r[0], f(r[1], d), f(r[2], d === 5 ? 6 : 2), f(r[3], d), f(r[4], d)]; }), [1.4, 1.6, 1.6, 1.6, 1.6]));
  add(P("Interpretation: fx ≈ fy (square pixels, aspect difference 0.005 %), so the sensor has no measurable skew. The horizontal and vertical fields of view are 2·atan(W/2fx) = **72.0°** and **57.1°**, plausible for a phone main camera. The principal point is 4.5 px left of and 21.2 px above the image centre. The 95 % interval of fx is ±11.4 px (±0.4 %)."));
  add(P("**Is the number of views sufficient?** The notebook draws n views at random (20 times for each n) and recalibrates ({fig:1a_convergence}). The standard deviation of fx falls from 344 px with 4 views to 41 px with 8 and 2.1 px with 13 views, and cx, cy settle within a few pixels, so 14 views are enough for a stable K. The curve is not perfectly monotonic (101.7 px at 9 views) because some random subsets contain only similar poses."));
  add(figure("1a_convergence", "Stability of fx, cx and cy versus the number of calibration views (mean ± 1 std over 20 random subsets)."));

  add(H3("2.1.7 Undistortion and quantitative check"));
  add(P("{fig:1a_field} shows how far the lens model moves each pixel (`cv2.undistortPoints` on a grid): 0 px at the centre, 16.6 px on average and **58.2 px at the image corners**. Inside the region covered by corners (r ≤ 0.84) the maximum is 23.0 px; values in the four extreme corners are extrapolated and less reliable."));
  add(figure("1a_field", "Distortion displacement field: colour = shift in pixels, arrows = direction of correction (barrel distortion pulled outwards at the corners).", 0.62));
  add(P("Undistortion uses `cv2.getOptimalNewCameraMatrix`: **alpha = 0** crops to valid pixels only, **alpha = 1** keeps every source pixel and shows black borders ({fig:1a_undistort})."));
  add(figure("1a_undistort", "Independent test photo (not used for calibration): original, undistorted with alpha = 0 (cropped) and alpha = 1 (all pixels kept)."));
  add(P("**Straightness test.** A straight line in the world stays straight through an ideal pinhole camera. Each row and column of board corners is a real straight line, so a line is fitted to it (SVD) and the RMS perpendicular distance of its corners is measured before and after `undistortPoints`. Over the calibration images the deviation falls from **0.718 px to 0.207 px** on average; on the **independent** photo it falls from 1.092 px to 0.506 px (−53.6 %). Every image improves ({fig:1a_straight}). The floor-tile photo gives a qualitative check: the tile edges at the image corner become straight after undistortion ({fig:1a_tiles})."));
  add(figure("1a_straight", "RMS deviation of board rows/columns from a straight line, before (blue) and after (orange) undistortion. The first bar pair is the independent test photo.", 0.85));
  add(figure("1a_tiles", "Floor tiles (real straight edges). Top: original and zoom on the top-left corner. Bottom: undistorted (alpha = 1) with the same zoom.", 0.72));

  add(H3("2.1.8 Limitations"));
  add(bullet("K and dist are valid only for this phone, its main lens, 4000 × 3000 resolution and no zoom. For a resized photo fx, cx are scaled with the width ratio and fy, cy with the height ratio (done in Task 3). Internet images such as the Kaggle fruit photos cannot be undistorted with this K."));
  add(bullet("The board was displayed on a laptop screen: it is flat, but causes moiré, and the physical square size depends on the screen zoom. The square size only scales the translation t, not K or dist; metric measurement would need the square measured with a ruler, or a printed board / ChArUco board."));
  add(bullet("Autofocus changes the focal length slightly with distance; this is not modelled. More views with corners near the frame edges would reduce the extrapolation at the corners."));

  // ---- 2.2
  add(H2("2.2 Filtering and edge processing (1B)"));
  add(P(`The inputs are photos from the Task 2 fruit dataset, i.e. the same kind of image a user uploads to the app. The main image is \`${m1b.image}\` (${m1b.size_wh[0]} × ${m1b.size_wh[1]} px); images are shrunk to a longest side of 512 px so that the comparisons are fair.`));
  add(H3("2.2.1 Correlation and convolution from scratch"));
  add(P("Correlation computes g(x, y) = Σ k(i, j) f(x + i, y + j); convolution is the same with the kernel flipped in both axes. For symmetric kernels (box, Gaussian) the two are equal; for an antisymmetric kernel such as Sobel the sign changes. `cv2.filter2D` actually computes correlation. The scratch version (`filtering.correlate2d`) builds all k × k windows with `numpy.lib.stride_tricks.sliding_window_view` and contracts them with `einsum`, with reflect-101 borders like OpenCV. The median filter is implemented the same way with `numpy.median` over the window axis."));
  add(...snippet(path.join(T1, "src/filtering.py"), "def correlate2d", "def box_kernel", "Listing 2"));
  const conv = readCSV(path.join(T1, "results/1B/convolution_verification.csv"));
  const convName = ["correlate2d vs cv2.filter2D (Gaussian 5×5)", "correlate2d (box 5×5) vs cv2.blur", "convolve2d (Sobel X) vs filter2D (flipped kernel)", "convolve2d(Sobel X) + correlate2d(Sobel X) (must be 0: sign flips)", "gaussian_kernel vs cv2.getGaussianKernel", "median_filter (scratch) vs cv2.medianBlur (5×5)"];
  add(table("conv", "Verification of the scratch filters against OpenCV (maximum absolute difference, grey levels 0–255).", ["Check", "Max |difference|"], conv.slice(1).map((r, i) => [convName[i], Number(r[1]) === 0 ? "0" : Number(r[1]).toExponential(2)]), [6, 2]));
  add(P("All differences are at floating-point rounding level. The scratch Gaussian 5 × 5 takes 6.7 ms against 2.3 ms for OpenCV, which uses SIMD and multi-threading, so the app uses the OpenCV versions and the scratch versions serve as verified references."));
  add(figure("1b_kernels", "The kernels used: box 5 × 5, Gaussian 5 × 5 (σ = 1) and Sobel X.", 0.7));

  add(H3("2.2.2 Denoising"));
  add(P("Two noise types are added to the clean image with a fixed seed: **Gaussian noise σ = 20** (sensor noise in low light) and **5 % salt-and-pepper noise** (transmission errors). Box, Gaussian, median and bilateral filters are applied with kernel sizes 3, 5 and 7 and compared with the clean image by **MSE** (lower is better), **PSNR** in dB (higher is better) and **SSIM** (Wang et al., 2004; 0–1, sensitive to structure and edges)."));
  const dn = readCSV(path.join(T1, "results/1B/denoising_metrics.csv")).slice(1);
  const get = (noise, fil, k) => dn.find((r) => r[0].startsWith(noise) && r[1] === fil && r[2] === String(k));
  const dRows = [["no filter", "0"]].concat(["box", "gaussian", "median", "bilateral"].flatMap((fl) => [3, 5, 7].map((k) => [fl, String(k)])));
  add(table("denoise", "Denoising results (PSNR in dB / SSIM). Best per noise type in bold.", ["Filter", "k", "Gaussian σ=20: PSNR", "SSIM", "Salt & pepper 5 %: PSNR", "SSIM"],
    dRows.map(([fl, k]) => {
      const a = get("Gaussian", fl === "no filter" ? "(không lọc)" : fl, k), b = get("Mu", fl === "no filter" ? "(không lọc)" : fl, k);
      const bestA = fl === m1b.best_filter_gaussian_noise.filter && Number(k) === m1b.best_filter_gaussian_noise.k;
      const bestB = fl === m1b.best_filter_salt_pepper.filter && Number(k) === m1b.best_filter_salt_pepper.k;
      const w = (s, on) => (on ? `**${s}**` : s);
      return [fl, k === "0" ? "–" : `${k}×${k}`, w(f(a[4]), bestA), w(f(a[5], 3), bestA), w(f(b[4]), bestB), w(f(b[5], 3), bestB)];
    }), [1.6, 0.8, 1.6, 1.1, 1.8, 1.1]));
  add(bullet("**Gaussian noise** has zero mean, so averaging cancels it. The edge-preserving **bilateral 7 × 7** is best (PSNR 23.30 → 28.77 dB, SSIM 0.39 → 0.82). Box filtering removes noise but blurs edges, and its PSNR falls as k grows."));
  add(bullet("**Salt-and-pepper noise** consists of outliers. Linear filters only spread them over the neighbourhood, and the bilateral filter treats them as edges and keeps them (PSNR stays at 17.3–17.5 dB). The **median 3 × 3** removes them because the median is not pulled by a few 0 or 255 values (17.10 → 30.52 dB, SSIM 0.33 → 0.91)."));
  add(bullet("Larger kernels remove more noise but also more detail ({fig:1b_psnr_k}); there is no single best filter, the choice depends on the noise type."));
  add(figure("1b_denoise_grid", "Denoising with 5 × 5 kernels. Rows 1–2: Gaussian noise, rows 3–4: salt-and-pepper noise, each with a zoomed crop. Columns: clean, noisy, box, Gaussian, median, bilateral."));
  add(figure("1b_psnr_k", "PSNR versus kernel size for each filter. Dotted line = noisy input.", 0.85));

  add(H3("2.2.3 Image gradients and Canny edges"));
  add(P("Sobel kernels approximate the derivatives Gx and Gy; the gradient magnitude is √(Gx² + Gy²) and the orientation atan2(Gy, Gx), shown in colour (hue = orientation, brightness = magnitude). **Canny** (Canny, 1986) = Gaussian smoothing → Sobel → non-maximum suppression (1-pixel-wide edges) → double threshold with hysteresis. The thresholds are chosen automatically from the median intensity m of the blurred image as [(1 − 0.33)m, (1 + 0.33)m] (`filtering.canny_auto`); for the orange they are **156 / 255** because the white background makes the median bright."));
  add(figure("1b_sobel_canny", "Sobel Gx, Gy, gradient magnitude, gradient orientation (colour) and automatic Canny."));
  add(figure("1b_canny_thr", "Effect of the Canny thresholds: low thresholds keep texture and noise (22,653 edge pixels), high thresholds keep only the strong outline."));
  add(P("**How much does noise hurt edges and how much does pre-filtering help?** The Canny map of the clean image is the reference. On the image with Gaussian noise, Canny is run with the same thresholds after each pre-filter, and precision / recall / F1 are measured with a 2-pixel tolerance."));
  const ef = readCSV(path.join(T1, "results/1B/edge_f1.csv"));
  add(table("edgef1", "Canny on the noisy image after different pre-filters (reference = Canny of the clean image, 2 px tolerance).", ["Pre-filter", "Precision", "Recall", "F1", "Edge pixels"],
    ef.slice(1).map((r) => [r[0] === "không lọc trước" ? "none" : r[0], f(r[1], 3), f(r[2], 3), f(r[3], 3), r[4]]), [2, 1, 1, 1, 1.2]));
  add(P("Without pre-filtering, noise creates many false edges (precision 0.15, F1 0.26). A **Gaussian 5 × 5** raises F1 to **0.918**; median and bilateral are close. Too much smoothing (Gaussian 9 × 9) removes real edges: precision 1.0 but recall 0.26. This shows the trade-off that the low-pass step of Canny controls."));
  add(figure("1b_edges_noise", "Reference edges of the clean image and Canny on the noisy image after each pre-filter, with F1."));

  add(H3("2.2.4 Application: fruit segmentation and outline"));
  add(P("Fruit skin (red, yellow, orange) has high colour **saturation**, while product-photo backgrounds are white or grey. `filtering.fruit_mask` therefore smooths the S channel of HSV, applies **Otsu's threshold** (Otsu, 1979; the threshold that maximises the between-class variance), cleans the mask with morphological opening/closing, fills holes and keeps the **largest contour**; the outline is the boundary of that mask. This function is reused by the app's *edges*, *measure* and *inspect* modes."));
  add(table("outline", "Otsu segmentation on four photos.", ["Image", "Otsu threshold (S)", "Fruit area (px)", "Area ratio", "Bounding box (px)"], [
    ["kaggle_train_apple_16.jpg", "84", "≈ 45,300", "0.453", "246 × 248"], ["kaggle_train_banana_29.jpg", "83", "≈ 30,600", "0.156", "361 × 227"],
    ["kaggle_train_orange_16.jpg", "111", "≈ 89,000", "0.442", "404 × 334"], ["kaggle_train_apple_24.jpg (cluttered)", "117", "≈ 95,000", "0.543", "273 × 474"]], [2.8, 1.4, 1.3, 1.1, 1.4]));
  add(P("The fourth photo, with a bowl and a cloth, was kept on purpose: the coloured background is merged with the fruit. A method based on one simple feature fails on cluttered scenes, which is one reason why Task 2 uses a CNN, and why the app flags low solidity or several regions instead of measuring silently."));
  add(figure("1b_outline", "Saturation channel (top), Otsu mask (middle) and largest contour with bounding box (bottom) for four photos. The last column shows the failure case on a cluttered background.", 0.9));

  add(H3("2.2.5 Frequency-domain filtering"));
  add(P("After `fftshift` the low frequencies (large smooth regions) are at the centre of the spectrum F(u, v) and the high frequencies (edges, detail, noise) towards the outside. Filtering in the frequency domain multiplies F by a transfer function H(u, v) and takes the real part of the inverse transform (`apply_frequency_filter`). Three low-pass shapes with the same cut-off D0 = 30 are compared:"));
  add(table("fft", "Low-pass filters with D0 = 30. Ringing = standard deviation of the difference to the Gaussian result in flat regions.", ["Transfer function", "PSNR vs original (dB)", "Ringing (std, grey levels)"], [["Ideal (step)", "25.86", "3.335"], ["Butterworth, order 2", "26.38", "1.063"], ["Gaussian", "26.76", "0"]], [3, 2, 2]));
  add(P("The ideal filter is a step in frequency, i.e. a convolution with a sinc in space, which produces **ringing** next to edges ({fig:1b_ringing}); the Gaussian has no side lobes and no ringing; Butterworth lies in between. A Gaussian high-pass keeps only edges and texture."));
  add(figure("1b_fft", "Top: original and the masks (ideal, Gaussian, Butterworth low-pass D0 = 30, Gaussian high-pass). Bottom: log spectrum and the filtered results."));
  add(figure("1b_ringing", "Intensity profile along row 224: the ideal low-pass oscillates around the edges, the Gaussian does not.", 0.85));
  add(P(`**Convolution theorem check.** A spatial Gaussian of standard deviation σ has the transfer function H(fx, fy) = exp(−2π²σ²(fx² + fy²)) (f in cycles per pixel). With σ = 3, the scratch spatial correlation with *periodic* borders (the DFT's assumption) and the frequency-domain product differ by ${f(m1b.conv_theorem_mean_abs_diff, 4)} grey levels on average (maximum 0.279), the small rest coming from truncating the spatial kernel at ±3σ.`));
  add(P("**Application: periodic noise.** Periodic stripes (electrical interference, scanners, flickering light) cover the whole image in space, so spatial filters handle them badly, but in the spectrum they are a single pair of bright peaks. The notebook adds a 2-D sinusoid, finds the peak automatically (`find_spectral_peaks`) and removes it and its mirror with a Gaussian notch-reject mask."));
  const pn = readCSV(path.join(T1, "results/1B/periodic_noise.csv"));
  const pnName = ["Image with periodic noise", "Notch filter (frequency)", "Gaussian low-pass D0 = 30 (frequency)", "Median 5×5 (spatial)", "Gaussian 7×7 (spatial)"];
  add(table("notch", "Removing periodic noise (reference = clean image).", ["Method", "MSE", "PSNR (dB)", "SSIM"], pn.slice(1).map((r, i) => [i === 1 ? `**${pnName[i]}**` : pnName[i], f(r[1], 1), f(r[2]), f(r[3], 3)]), [3.5, 1, 1, 1]));
  add(P("The notch filter raises PSNR from 22.79 dB to **29.53 dB** and SSIM from 0.50 to 0.86, while the spatial median does nothing and the low-pass blurs the fruit. Faint stripes remain on the white background because the noisy image was clipped at 255 there; clipping is non-linear and creates harmonics outside the removed peaks."));
  add(figure("1b_notch", "Periodic noise: noisy image, spectrum with the detected peaks, notch mask, notch result and median result."));

  // ---- 2.3
  add(H2("2.3 Feature detection and matching (1C)"));
  add(P(`The test image is \`${m1c.image}\` (an orange with a wooden board and slices, 512 × 384 px). Smooth, round, shiny fruit skin has few stable corners, so a photo with some texture was chosen; this limitation is discussed below.`));
  add(H3("2.3.1 Harris corner detector from scratch"));
  add(P("For gradients Ix, Iy (Sobel), the structure tensor at each pixel is M = Σw [[Ix², IxIy], [IxIy, Iy²]], a Gaussian-weighted sum over a window w. The corner response is R = det M − k (trace M)² with k ≈ 0.04–0.06: R ≫ 0 at a corner (both eigenvalues large), R ≪ 0 on an edge, |R| small on a flat region. Points with R > τ · max R are kept after 7 × 7 non-maximum suppression (`features.harris_response`, `harris_corners`)."));
  add(...snippet(path.join(T1, "src/features.py"), "def harris_response", "def harris_corners", "Listing 3"));
  add(P(`**Verification:** the correlation between the scratch R and \`cv2.cornerHarris\` is ${m1c.harris_scratch_vs_opencv_corr.toFixed(10)}, and the 200 strongest corners coincide 100 % (≤ 1 px). The number of corners is controlled mainly by τ (k = 0.04: 414, 112 and 19 corners for τ = 0.001, 0.01, 0.05); a larger k lowers it slightly (τ = 0.01: 112, 99, 91 for k = 0.04, 0.06, 0.1).`));
  add(figure("1c_harris", "Harris response from scratch, OpenCV response, and detected corners (τ = 0.01, k = 0.04)."));

  add(H3("2.3.2 Comparing SIFT, ORB and Harris + SIFT on a pair with ground truth"));
  add(P("Judging matches by eye is subjective, so the second view is generated from the first with a **known homography** H_true: rotation 25°, scale 0.85, perspective tilt, brightness +20 and Gaussian noise σ = 5. A match (p1, p2) is **correct** if ‖H_true p1 − p2‖ < 3 px. Metrics: *precision* = correct / matches after the ratio test; *repeatability* = share of view-1 points re-detected in view 2; *H corner error* = mean distance at the four image corners between the RANSAC homography and H_true. Each detector uses brute-force matching (L2 for SIFT, Hamming for ORB), Lowe's ratio test ρ = 0.75 and RANSAC with a 4 px threshold."));
  const mc = m1c.methods;
  add(table("methods", "Detector/descriptor comparison on the ground-truth pair (ratio 0.75).", ["Method", "Keypoints 1 / 2", "Ratio-test matches", "RANSAC inliers", "Correct", "Precision", "Repeatability", "H corner error (px)", "Time (ms)"],
    Object.entries(mc).map(([k, v]) => [k, `${v.kp1} / ${v.kp2}`, v.good_matches, v.ransac_inliers, v.correct_matches, f(v.precision, 3), f(v.repeatability, 3), v.H_corner_error_px == null ? "fails" : f(v.H_corner_error_px, 3), f(v.time_ms, 0)]),
    [1.3, 1.3, 1.1, 1, 0.9, 0.9, 1.1, 1.1, 0.8]));
  add(bullet("**SIFT** (Lowe, 2004) is the most accurate: 97.4 % precision and the RANSAC homography is only **0.20 px** from H_true."));
  add(bullet("**ORB** (Rublee et al., 2011) finds more matches (521 correct) and has higher repeatability, but its homography error is larger (1.29 px). Here ORB is slower than SIFT because it detected about twice as many points (`nfeatures = 2000`) and brute-force matching grows with the product of the two point counts; its 32-byte binary descriptor is still much more compact than SIFT's 128 floats."));
  add(bullet("**Harris + SIFT descriptor** fails (precision 33 %) despite good repeatability (0.77). Harris gives a stable *position* but no orientation or scale, so the descriptor is computed on a fixed, unrotated 16-px patch, which looks different after a 25° rotation and 0.85 scaling. SIFT and ORB assign each keypoint a dominant orientation and a scale (scale space / image pyramid)."));
  add(figure("1c_matches", "Ratio-test matches of SIFT, ORB and Harris + SIFT on the ground-truth pair (green = correct, red = wrong).", 0.8));

  add(H3("2.3.3 Choosing the ratio-test threshold with data"));
  add(P("For each descriptor in view 1 the two nearest neighbours in view 2 are found, and the match is kept only if d1 < ρ · d2, i.e. the best match is clearly better than the second (not ambiguous). Small ρ gives few but precise matches; large ρ gives many matches with more errors."));
  const rs = readCSV(path.join(T1, "results/1C/ratio_sweep.csv")).slice(1);
  const ratios = [...new Set(rs.map((r) => r[1]))];
  add(table("ratio", "Ratio sweep on the ground-truth pair: matches / correct matches / precision.", ["ρ", "SIFT matches", "SIFT correct", "SIFT precision", "ORB matches", "ORB correct", "ORB precision"],
    ratios.map((q) => { const s = rs.find((r) => r[0] === "SIFT" && r[1] === q), o = rs.find((r) => r[0] === "ORB" && r[1] === q); return [q === "0.75" ? "**0.75**" : q, s[2], s[3], f(s[4], 3), o[2], o[3], f(o[4], 3)]; }),
    [0.8, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2], { highlight: (i) => ratios[i] === "0.75" }));
  add(P("With SIFT, ρ = 0.75 keeps 97.4 % precision with 331 correct matches; raising ρ to 0.90 adds only 18 correct matches but 100 wrong ones (precision 77.7 %). Lowe's recommended range 0.7–0.8 is therefore confirmed on this data, and 0.75 is the default in the app."));
  add(figure("1c_ratio", "Precision and number of correct matches as functions of the ratio threshold ρ.", 0.85));

  add(H3("2.3.4 Robustness to rotation and scale"));
  const rb = readCSV(path.join(T1, "results/1C/robustness.csv")).slice(1);
  const pick = (test, vals) => vals.map((v) => { const g = (m) => rb.find((r) => r[0] === test && Number(r[1]) === v && r[2] === m); return [test === "xoay" ? `rotation ${v}°` : `scale ×${v}`, g("SIFT")[3], g("ORB")[3], g("Harris+SIFT")[3]]; });
  add(table("robust", "Number of correct matches when the second view is only rotated or only scaled (ratio 0.75).", ["Transformation", "SIFT", "ORB", "Harris+SIFT"],
    [...pick("xoay", [0, 15, 30, 45, 90, 135, 180]), ...pick("tỉ lệ", [0.4, 0.6, 0.8, 1.2, 1.6])], [2.5, 1.5, 1.5, 1.5]));
  add(P("SIFT keeps at least 340 correct matches at every rotation and ORB at least 573, while Harris + SIFT drops to 1 correct match at 30° and 0 beyond. All methods lose matches when the scale changes strongly, but SIFT and ORB still have 135–185 correct matches at ×0.4 and ×1.6. The identity case (0°, ×1) is shown only as a reference."));
  add(figure("1c_robust", "Correct matches versus rotation angle (left) and scale factor (right).", 0.85));

  add(H3("2.3.5 RANSAC and deciding whether two photos show the same scene"));
  add(P("The ratio test still lets some wrong matches through. **RANSAC** (Fischler and Bolles, 1981) repeatedly picks 4 matches at random, fits a homography and counts the matches consistent with it (inliers); the homography with the most inliers wins. This gives a simple **decision rule**: two photos show the same scene if there are **at least 15 inliers and an inlier ratio of at least 50 %**."));
  const un = m1c.unrelated_pair; const uk = Object.keys(un);
  add(table("scene", "RANSAC on a related pair and on an unrelated pair (orange vs banana).", ["Pair", "Ratio-test matches", "RANSAC inliers", "Inlier ratio", "Decision"],
    [["Same scene (transformed view)", un[uk[0]].good_matches, un[uk[0]].ransac_inliers, pct(un[uk[0]].inlier_ratio), "SAME SCENE"], ["Unrelated (banana photo)", un[uk[1]].good_matches, un[uk[1]].ransac_inliers, pct(un[uk[1]].inlier_ratio), "DIFFERENT"]], [2.8, 1.4, 1.3, 1.2, 1.3]));
  add(P("Even for unrelated photos, 13 matches pass the ratio test by coincidence, but only 5 of them agree on one homography. A large number of ratio-test matches alone is therefore not evidence that two photos match: geometric verification is required. For the related pair, warping view 1 with the estimated H aligns it with view 2 ({fig:1c_ransac})."));
  add(figure("1c_ransac", "View 1 warped by the RANSAC homography and overlaid on view 2 (corner error 0.20 px)."));
  add(figure("1c_unrelated", "Inlier matches for the related pair (left) and for the unrelated orange/banana pair (right)."));
  add(P("**Limitation.** The second view is synthetic. This makes precision objective, but there is no real 3-D change (parallax, occlusion, specular highlights) as when one fruit is photographed from two positions. With two real photos the same code still reports RANSAC inliers; only the ground-truth metrics are unavailable."));

  // ---- 2.4
  add(H2("2.4 Code structure and quality"));
  add(table("t1code", "Task 1 modules (full listings in Appendix A).", ["Module", "Lines", "Main functions"], [
    ["src/calibration.py", "381", "detect_corners, collect_correspondences, homography_dlt, intrinsics_from_homographies, extrinsics_from_homography, calibrate_zhang, calibrate_opencv, reprojection_errors, cross_validate_models, line_straightness, undistort_points_px, undistort_image, save/load_calibration"],
    ["src/filtering.py", "273", "correlate2d, convolve2d, gaussian_kernel, median_filter, apply_filter, add_gaussian_noise, add_salt_pepper, mse, psnr, ssim, quality, sobel, canny_auto, edge_f1, fruit_mask, fruit_outline, lowpass/highpass/notch masks, apply_frequency_filter, find_spectral_peaks"],
    ["src/features.py", "183", "harris_response, harris_corners, detect_and_describe, ratio_test_matches, ransac_homography, synthetic_view, evaluate_pair, draw_matches"],
  ], [1.6, 0.6, 6.5], { left: true }));
  add(bullet("**Separation of library and experiments.** Algorithms are plain functions in `src/` with type hints and docstrings; the notebooks only call them, plot and save. Task 3 imports the same functions, so the app and the report use one implementation."));
  add(bullet("**Verified implementations.** Each from-scratch method is compared with its OpenCV counterpart (Tables {tab:zhang}, {tab:conv} and the Harris check), so correctness is shown with numbers, not assumed."));
  add(bullet("**Reproducibility.** Fixed random seeds for noise, sampling and synthetic views; every figure and table is written to `results/` together with a `metrics.json`; `artifacts/calibration.npz` stores K, dist and the image size for other modules."));
  add(bullet("**Quantitative evaluation everywhere.** Each technique is judged by a metric with a reference (reprojection error, hold-out error, line straightness, PSNR/SSIM, edge F1, match precision against a known homography)."));

  // ============================================================ 3. TASK 2
  add(H1("3. Task 2 – Data Pipeline and Deep-Learning Model"));
  add(H2("3.1 Dataset and justification"));
  add(P("**Dataset:** *Fruit Images for Object Detection* by Muhammed Buyukkinaci on Kaggle (licence CC0: public domain). It contains 300 photos of apples, bananas, oranges and mixed fruit, with bounding-box XML files for detection. For this MVP the task is **3-class image classification (apple, banana, orange)**, so the XML files are not used and mixed photos are excluded."));
  add(P("**Why this dataset?** (1) It matches the scenario: fruit photographed in varied, realistic conditions (single fruit, bunches, cut fruit, white and natural backgrounds), the kind of image a farm worker would upload. (2) The licence allows use and redistribution. (3) The same photos feed Task 1B and 1C, so the classical engine and the CNN work on one domain. (4) It is small enough to clean by hand, which shows a careful pipeline; the small size is also a limitation, discussed in Section 3.7."));
  add(figure("2_samples", "Training examples of the three classes (every ninth image of each class folder)."));
  add(H2("3.2 Data pipeline"));
  add(P("The pipeline is the notebook `task2/Task2_pipeline.ipynb`, one section per step, each step writing a log or manifest so that every image can be traced from the original archive to the split it ended in."));
  add(table("pipeline", "Steps of the data pipeline.", ["Step", "Script", "What it does", "Output"], [
    ["1. Ingestion", "Task2_pipeline.ipynb § 1", "Copies apple/banana/orange photos from the extracted archive (train + test folders merged) into datasets/fruits/raw/<class>/; skips mixed_* and .xml; never modifies the source", "datasets/fruits/raw/, artifacts/import_log.csv"],
    ["2. Manual cleaning", "§ 1 (EXCLUDE list)", "Removes 20 images found by visual inspection: drawings/clipart/vector art, toys, jam, large overlaid text, and photos that contain a fruit of another class", "logged as 'excluded' with reason"],
    ["3. Automatic cleaning", "§ 2", "Opens and verifies every file (PIL verify + full decode with EXIF orientation), rejects images < 32 px, removes exact duplicates by SHA-256 and stops if one file appears in two classes", "artifacts/rejected_images.csv"],
    ["4. Grouped stratified split", "§ 2", "70/15/15 split per class by group id (seed 42) so that photos of the same object never cross splits; re-encodes to RGB JPEG", "datasets/fruits/splits/, data_manifest.csv"],
    ["5. Near-duplicate audit", "§ 3", "64-bit perceptual hash (DCT of a 32×32 grey image), also against the mirrored image; lists pairs with Hamming distance ≤ 14 for visual review, cross-split pairs first", "artifacts/near_duplicates.csv, results/near_duplicates.png"],
    ["6. Transformation / augmentation", "src/data_utils.py (§ 4)", "Resize 128×128, ToTensor, normalise mean = std = 0.5; train only: random rotation ±12°, horizontal flip p = 0.5, colour jitter ±15 % brightness/contrast", "used by § 6–8 and inference.py"],
  ], [1.4, 1.7, 4.2, 2.1], { left: true, size: 17 }));
  add(H3("3.2.1 Ingestion and cleaning"));
  add(P("Of the 300 images in the archive, **25 mixed** photos have no single label and are skipped, and **20 are excluded manually** (5 apple, 11 banana, 4 orange) with the reason written to the import log, for example *'drawing, not a photo'*, *'toy shaped like a banana'*, *'contains an apple (another class)'*. **255 images** are imported (89 apple, 80 banana, 86 orange). The automatic checks of § 2 then found no corrupt, tiny or byte-identical files (`rejected_images.csv` is empty), so all 255 images enter the split."));
  add(P("Handling of missing values and outliers in an image dataset means: unreadable or truncated files (verify + decode), images too small to carry information (< 32 px), label noise (images of the wrong object or containing two classes, removed manually) and duplicates (SHA-256 for exact copies, pHash for near copies). Byte-identical files across classes would be a labelling error, so the script stops instead of choosing a label."));
  add(H3("3.2.2 Splitting without leakage"));
  add(P("Random splitting per image can leak information when several photos show the same object. The naming convention `<group_id>__<view>.jpg` puts all photos of one object in one group, and the split is done **per class over groups**, so a group is entirely in train, validation or test. The Kaggle file names carry no object information, so each photo is its own group here, and leakage is checked separately by the near-duplicate audit."));
  add(table("split", "Final split (images per class).", ["Split", "Apple", "Banana", "Orange", "Total", "Share"], [["Train", "63", "56", "60", "179", "70.2 %"], ["Validation", "13", "12", "13", "38", "14.9 %"], ["Test", "13", "12", "13", "38", "14.9 %"], ["**Total**", "**89**", "**80**", "**86**", "**255**", "100 %"]], [1.6, 1, 1, 1, 1, 1]));
  add(P("The classes are nearly balanced (largest/smallest = 1.11), so plain cross-entropy without class weights is appropriate, and macro-F1 is reported next to accuracy."));
  add(H3("3.2.3 Near-duplicate audit"));
  add(P("SHA-256 only finds byte-identical files. A resized, recompressed, cropped or mirrored copy has a different hash but almost the same content, and if it falls into train and test the test score is inflated. § 3 of the notebook computes a perceptual hash: grey 32 × 32 → 2-D DCT → 8 × 8 low-frequency block → one bit per coefficient (above/below the median). The Hamming distance between hashes (0–64) is compared both to the image and to its mirror. On this dataset two crops of one original give d = 14, while the median distance between different photos is 28, so the threshold is 14."));
  add(P("The audit flagged **73 pairs** (minimum d = 8), **29 across splits** and **33 across classes**. Cross-class pairs cannot be true duplicates (e.g. the closest pair is a mirrored apple/orange pair at d = 8), which shows that at this threshold pHash mostly finds *similar layouts*: one round fruit on a white background. The list is therefore a review aid, not an automatic filter: the contact sheet ({fig:2_near_dup}) is inspected by eye, cross-split pairs first, and a confirmed duplicate would be given a shared group id so that it moves into one split."));
  add(figure("2_near_dup", "First rows of the near-duplicate contact sheet (red label = pair across splits; d = Hamming distance; 'lat' = mirrored match).", 0.85));
  add(H3("3.2.4 Transformation and augmentation"));
  add(P("One function, `make_transform`, defines the preprocessing for training, validation, test and inference, so the app sees exactly the input distribution the model was trained on. Images are resized to 128 × 128 (fruit photos are recognisable at this size and the small input keeps the model and training fast) and normalised to [−1, 1]. **Augmentation is applied on the fly and only in the training loader** (no offline copies, so augmented versions of one photo cannot leak into validation/test): rotation ±12°, horizontal flip and brightness/contrast jitter ±15 %, which mimic how a phone photo of a fruit varies. Vertical flips and strong hue changes are not used because they would create unrealistic images (a banana's colour is part of its identity)."));
  add(...snippet(path.join(T2, "src/data_utils.py"), "IMAGE_SIZE = 128", null, "Listing 4"));

  add(H2("3.3 Model: FruitCNN"));
  add(P("The model is a compact VGG-style CNN of three Conv → ReLU (→ MaxPool) blocks, followed by **global average pooling** and one linear layer that outputs logits (no softmax during training, because `CrossEntropyLoss` applies log-softmax itself)."));
  add(...snippet(path.join(T2, "src/model.py"), "class FruitCNN", null, "Listing 5"));
  add(table("arch2", "FruitCNN layers for a 3 × 128 × 128 input.", ["Layer", "Output shape", "Parameters"], [
    ["Conv2d 3→16, 3×3, pad 1 + ReLU", "16 × 128 × 128", "448"], ["MaxPool 2×2", "16 × 64 × 64", "0"],
    ["Conv2d 16→32, 3×3, pad 1 + ReLU", "32 × 64 × 64", "4,640"], ["MaxPool 2×2", "32 × 32 × 32", "0"],
    ["Conv2d 32→64, 3×3, pad 1 + ReLU", "64 × 32 × 32", "18,496"], ["AdaptiveAvgPool → flatten", "64", "0"],
    ["Linear 64→3", "3 (logits)", "195"], ["**Total**", "", "**23,779**"]], [4, 2.2, 1.6]));
  add(P("**Why this architecture?** With 179 training images a large network (e.g. VGG-16 with 138 M parameters) would memorise the training set. FruitCNN has 23,779 parameters, about 130 per training image, and global average pooling replaces a large fully-connected layer, which reduces over-fitting and makes the last convolutional maps directly usable for **Grad-CAM** in Task 3. Three 3 × 3 convolutions with two poolings give each output unit a receptive field of 18 × 18 px on the 128-px input; colour and local texture (smooth orange peel, banana ridges, apple shine) are the discriminative cues, and global pooling aggregates them over the image. The checkpoint is only about 98 KB, so the app loads it in milliseconds and runs on a CPU. Transfer learning from an ImageNet backbone (e.g. ResNet-18) is the natural next step if the dataset grows, but for three visually distinct classes the small CNN already reaches 97 % test accuracy."));

  add(H2("3.4 Training and hyperparameters"));
  add(table("hparams", "Training configuration (Task2_pipeline.ipynb § 6).", ["Setting", "Value", "Reason"], [
    ["Loss", "Cross-entropy", "Standard for single-label multi-class classification"],
    ["Optimiser", "Adam, learning rate 1e-3", "Robust default for small CNNs; no tuning of the schedule needed"],
    ["Batch size", "32", "Fits easily in memory; about 6 updates per epoch"],
    ["Epochs", "max 200, early stopping patience 20", "Stops when validation loss has not improved for 20 epochs"],
    ["Checkpoint", "Best validation loss", "Loss is smoother than accuracy on 38 validation images"],
    ["Seed", "42 (Python, NumPy, PyTorch, CUDA)", "Reproducible split and initialisation"],
    ["Device", "CUDA (RTX 4060 Laptop), CPU fallback", "CUDA is used automatically when available"],
  ], [1.5, 2.6, 4.6], { left: true }));
  add(P("Training ran for 74 epochs: the validation loss reached its minimum **0.302 at epoch 54** (validation accuracy 94.7 %, training accuracy 98.9 %) and did not improve in the next 20 epochs, so training stopped and the epoch-54 weights were kept ({fig:2_curves}). The training loss keeps falling to about 0.02 while the validation loss levels off around 0.30–0.38, the expected sign of mild over-fitting on a small dataset; the checkpoint rule protects against it."));
  add(figure("2_curves", "Learning curves (loss and accuracy). The dashed line marks the best epoch (54) chosen by validation loss."));
  add(P("The checkpoint stores more than the weights, so that the model can be used without the training script: `model_state`, `class_to_idx`, `image_size`, normalisation `mean`/`std`, `architecture` name, `best_epoch`, `best_val_loss`, `best_val_accuracy` and `seed`."));

  add(H2("3.5 Evaluation"));
  add(P("§ 7 of the notebook loads the frozen checkpoint and evaluates it **once** on the 38 test images, which were not used for training or for choosing the checkpoint. It checks that the class-to-index mapping of the test folder equals the checkpoint's, then reports accuracy, macro-F1, the per-class report and the confusion matrix (scikit-learn), and saves every prediction to `results/test_predictions.csv`."));
  const pc = m2.per_class;
  add(table("test", `Test results (n = ${m2.test_samples}).`, ["Class", "Precision", "Recall", "F1-score", "Support"],
    [...["apple", "banana", "orange"].map((c) => [c, f(pc[c].precision, 3), f(pc[c].recall, 3), f(pc[c]["f1-score"], 3), pc[c].support]),
      ["**Macro average**", f(pc["macro avg"].precision, 3), f(pc["macro avg"].recall, 3), `**${f(pc["macro avg"]["f1-score"], 3)}**`, 38],
      ["Weighted average", f(pc["weighted avg"].precision, 3), f(pc["weighted avg"].recall, 3), f(pc["weighted avg"]["f1-score"], 3), 38],
      ["**Accuracy**", "", "", `**${f(m2.accuracy, 3)}**`, 38]], [2.2, 1.2, 1.2, 1.2, 1]));
  add(P(`The model classifies **37 of 38 test images correctly: accuracy ${pct(m2.accuracy)} and macro-F1 ${f(m2.macro_f1, 3)}**. Apple is perfect; the single error is a banana photo (\`kaggle_train_banana_14.jpg\`) predicted as orange, which lowers banana recall to 0.917 and orange precision to 0.929. With only 38 test images the estimate is uncertain: the 95 % Wilson confidence interval of the accuracy is **86.5 %–99.5 %**, so the result should be read as "about 90 % or better", not as an exact 97.4 %.`));
  add(figure("2_cm", "Confusion matrix on the test set.", 0.5));
  add(P(`**Comparison with a baseline run.** A first run with a fixed 30 epochs and no early stopping (kept in \`runs/30ep_baseline/\`) reached ${pct(m2base.accuracy)} accuracy and macro-F1 ${f(m2base.macro_f1, 3)} (3 errors: one apple and two bananas). Training longer with early stopping and checkpoint selection by validation loss reduced the test errors from 3 to 1. Banana is the hardest class in both runs: several banana photos show peeled or cut fruit, bunches on dark backgrounds or very small fruit, where the yellow-orange colour cue is ambiguous.`));
  add(P("**Is the model ready for deployment?** For the MVP, yes: it is trained, evaluated on a held-out test set, saved with its preprocessing metadata, and called by Task 3 through a stable API. Section 4.6 shows that the app reproduces the 37/38 test result exactly, which confirms that training, evaluation and the app use the same preprocessing."));

  add(H2("3.6 Model artifact and inference API"));
  add(P("The trained model is `task2/artifacts/model.pth` (≈ 98 KB) with `classes.json`, stored in the repository (link on the title page and in Section 7). `src/inference.py` provides the interface used by § 8 of the notebook and by Task 3:"));
  add(...snippet(path.join(T2, "src/inference.py"), "def load_model", null, "Listing 6"));
  add(P("`load_model` rebuilds `FruitCNN` with the number of classes in the checkpoint, loads the weights with `weights_only=True` (safe loading), and recreates the evaluation transform from the stored image size and normalisation. `predict_pil` applies EXIF orientation, converts to RGB and returns the class and its softmax score. The softmax score is a *model score*, not a calibrated probability; the app says so in its output."));
  add(H2("3.7 Limitations and next steps"));
  add(bullet("**Small dataset** (255 images, 38 in test): wide confidence interval; more data and k-fold cross-validation would give a more reliable estimate."));
  add(bullet("**Closed set:** the model always answers apple, banana or orange, even for a pear or a non-fruit image. The app warns when the score is below 0.60; an 'unknown' class or out-of-distribution detection would be better."));
  add(bullet("**Near-duplicate audit is manual:** pHash finds candidates, a human decides. Kaggle file names do not identify objects, so group-aware splitting cannot use real object ids here."));
  add(bullet("**Next steps:** transfer learning (ResNet-18 / MobileNetV3) with the same pipeline, quality classes (ripe / unripe / defective) that a sorting line needs, and temperature scaling to calibrate the scores."));

  // ============================================================ 4. TASK 3
  add(H1("4. Task 3 – Assistant Feature (Application Integration / Tool Use)"));
  add(H2("4.1 Overview"));
  add(P("**FruitSort Vision Assistant** is a Streamlit web app (`task3/app.py`). The user uploads one or more images (or picks sample images: fruit test split, checkerboard photos), then types a request in English or Vietnamese, for example *'phân loại quả này'* (classify this fruit), *'find the edges with Canny'* or *'đo kích thước quả ở 30 cm'* (measure the fruit at 30 cm). The assistant identifies the task, runs the matching module of Task 1 or Task 2 and answers in the chat with figures, a metrics table, and warnings. Quick-action buttons and a sidebar mode selector are alternatives to typing; the sidebar also shows whether the Task 1 engine, the calibration file and the model are loaded."));
  add(figure("t3_ui_home", "The app at start-up: system status (Task 1 engine, calibration K, Task 2 model), mode selector and parameters in the sidebar; image source, quick actions, example requests and chat input in the main area.", 0.78));
  add(H2("4.2 Request flow"));
  add(num("**Input.** Uploaded bytes are decoded with `tools.decode_image`: empty files, files over 25 MB, non-images (e.g. a text file renamed to .jpg, a PDF), images under 32 px are rejected with a clear message; 16-bit, grey and RGBA images are converted to 8-bit BGR.", "num2"));
  add(num("**Intent.** `intent.detect_intent(text)` returns the mode, the matched keywords (shown to the user as the reason) and parameters extracted from the text.", "num2"));
  add(num("**Mode choice.** Priority: quick-action button > mode forced in the sidebar > detected intent. If no mode is found or two modes tie, nothing runs and the assistant lists the closest candidates.", "num2"));
  add(num("**Parameters.** Sidebar values are the defaults; values written in the request (e.g. 'median 7x7', 'ORB', 'ratio 0.8', 'threshold 50 150', 'at 30 cm') override them.", "num2"));
  add(num("**Tool call.** `tools.run(mode, images, params, clf, calib)` dispatches to one function per mode, which calls Task 1 / Task 2 code and returns a `ToolResult(summary, images, metrics, table, warnings, elapsed_ms)`.", "num2"));
  add(num("**Rendering.** The chat shows the request, the routing explanation, the summary, the figures, the metrics and a table; the last six turns are kept.", "num2"));

  add(H2("4.3 Application logic: recognising the user's intent"));
  add(P("A transparent **rule-based router** was chosen instead of a large language model: seven tools with distinctive vocabulary do not need one, it works offline without an API key, it answers in under a millisecond, and every decision can be explained by the keywords that triggered it. The algorithm (`assistant/intent.py`):"));
  add(num("**Normalise** the text: lower case, remove Vietnamese diacritics (Unicode NFD, drop combining marks, đ → d), replace '-' and '_' by spaces, collapse whitespace. 'Phân loại' and 'phan loai' therefore match the same pattern.", "num3"));
  add(num("**Score** every mode by the weighted regular-expression patterns it matches, with word boundaries. Technique names weigh more than generic words: 'canny', 'sobel', 'sift', 'ransac', 'khử méo' (undistort) weigh 3; 'filter', 'gradient', 'size' weigh 2; ambiguous words such as 'noise', 'nhiễu', 'méo' weigh 1.", "num3"));
  add(num("**Combine:** if both *classify* and *measure* score, their scores are added to *inspect* ('phân loại và đo kích thước' → full inspection).", "num3"));
  add(num("**Decide:** the highest score wins. No match or a tie at the top → `mode = None` with a reason ('No known task keyword found.' / 'Ambiguous: …') and the UI asks the user to choose.", "num3"));
  add(num("**Extract parameters** with regular expressions: filter type, kernel size (made odd and clamped to 3–31), noise type to add, detector (SIFT/ORB/Harris), ratio, FFT cut-off, Canny thresholds, and camera–object distance converted to mm.", "num3"));
  add(...snippet(path.join(T3, "assistant/intent.py"), "def detect_intent", null, "Listing 7"));
  add(table("routing", "Requests routed by the assistant (output of detect_intent; Vietnamese requests translated in brackets).", ["Request", "Mode", "Reason shown to the user", "Parameters extracted"],
    t3log.routing.map((r) => {
      const tr = { "phân loại quả này": " [classify this fruit]", "tìm cạnh bằng Canny": " [find edges with Canny]", "lọc nhiễu muối tiêu bằng median 7x7": " [remove salt-and-pepper noise with median 7x7]", "so khớp ORB giữa hai ảnh, ratio 0.8": " [match ORB between two images, ratio 0.8]", "khử méo ảnh này": " [undistort this image]", "đo kích thước quả ở 30 cm": " [measure the fruit at 30 cm]", "phân loại và đo kích thước": " [classify and measure]" }[r.request] || "";
      return [r.request + tr, r.mode, r.reason, r.params || "–"];
    }), [3.4, 1.1, 2.6, 2.4], { left: true, size: 16 }));
  add(P("The integration tests route 19 English and Vietnamese requests, all correctly (Section 4.6). Typical robust cases: 'detect Harris corners' goes to *features* with method Harris+SIFT; 'show the camera intrinsic matrix K' goes to *calibrate*; 'hello there' runs nothing and asks the user to name a task."));

  add(H2("4.4 Tool integration"));
  add(table("tools", "Modes, the Task 1 / Task 2 code each one calls, and what is returned.", ["Mode", "Code called", "Result shown"], [
    ["calibrate", "Task 1A: calibration.undistort_image, detect_corners, line_straightness, undistort_points_px; with ≥ 3 board photos a new calibration (cv2.calibrateCamera)", "Input with corners, undistorted image, shift map, scaled K, FOV, straightness before/after; new K compared with Task 1A"],
    ["filter", "Task 1B: apply_filter, add_gaussian_noise / add_salt_pepper, quality (MSE/PSNR/SSIM), lowpass/highpass masks, apply_frequency_filter", "Filtered image, residual, comparison table of all filters; spectrum, mask and result for FFT filters"],
    ["edges", "Task 1B: sobel, canny_auto (or given thresholds), fruit_outline", "Gradient magnitude and orientation, Canny map, outline, circularity"],
    ["features", "Task 1C: evaluate_pair (detect, describe, ratio test, RANSAC), synthetic_view, draw_matches", "Keypoints, matches coloured by RANSAC inlier/outlier; precision against ground truth when only one image is given"],
    ["classify", "Task 2: src.inference.predict_pil with the loaded model.pth; assistant/gradcam.py", "Class, score per class, margin, Grad-CAM heat map, low-confidence warning"],
    ["measure", "Task 1A K + Task 1B fruit_outline; size_mm = size_px · Z / f", "Area, equivalent diameter, min-area box in px (mm when Z is given and the photo is from the calibrated camera); solidity check"],
    ["inspect", "classify + measure", "Both results in one answer"],
  ], [1.1, 4.2, 3.6], { left: true, size: 17 }));
  add(H3("4.4.1 Loading the two engines"));
  add(P("`assistant/engines.py` imports the code of both tasks instead of copying it. Both tasks name their package `src`, and Task 2's own modules import `src.model` absolutely, so Task 2 keeps the name `src` while Task 1 is loaded under the alias `fcvg_task1` with `importlib` (and a stale `src` package pointing elsewhere is removed from `sys.modules`). Paths are derived from the location of the app, so the project runs on any machine with the same folder layout; the environment variables `FCVG_TASK1_DIR`, `FCVG_TASK2_DIR`, `FCVG_CALIBRATION_FILE`, `FCVG_MODEL_FILE` override them. The engines are cached with `st.cache_resource`, so `model.pth` and `calibration.npz` are loaded once per server process; failures are not cached, so fixing a file and pressing *Reload engines* is enough."));
  add(P("**Camera matrix for resized photos.** `Calibration.for_image(w, h)` scales fx, cx by w/4000 and fy, cy by h/3000, and refuses photos whose aspect ratio differs by more than 1 % (cropped or portrait photos), because K would then be wrong. In that case undistortion is refused and measurement falls back to pixels with an explanation."));
  add(H3("4.4.2 Explaining the classifier with Grad-CAM"));
  add(P("For the predicted class c with logit yc and the feature maps Aᵏ of the last convolutional block (64 maps of 32 × 32 for a 128-px input), Grad-CAM (Selvaraju et al., 2017) weights each map by its average gradient and keeps positive evidence:"));
  add(eq("αk = mean(i,j) ∂yc / ∂Aᵏij ,      Lc = ReLU( Σk αk Aᵏ )"));
  add(P("Lc is up-sampled to the photo and blended as a heat map (red = most influential). A forward hook on the ReLU after the last convolution stores Aᵏ and a tensor hook stores its gradient. The app checks that Grad-CAM's class equals Task 2's `predict_pil` answer, so the explanation always belongs to the prediction shown."));
  add(H3("4.4.3 Error handling"));
  add(P("Problems the user can fix raise `AssistantError(message, hint)`: no image, unreadable file, wrong number of images for a mode, missing `model.pth` or `calibration.npz`, a photo that does not match the calibrated camera, no fruit region found. The chat shows the message and a 'How to fix' hint instead of a traceback. Any unexpected exception is also caught and shown, so the app never crashes. If an artifact is missing, the app still starts, the sidebar shows what is missing, and every tool that does not need it keeps working."));

  add(H2("4.5 Results returned by each mode"));
  add(P("The following figures were produced by passing the request text through `detect_intent` and `tools.run`, i.e. the same code path as the chat (Section 3 of `Task3_assistant.ipynb`; the copies in this report are made by `report/build/make_task3_figures.py`). Each title shows the request, the detected mode with its reason, and the assistant's summary."));
  const r3 = (k) => run3[k];
  add(figure("t3_classify", `Classification request in Vietnamese ('classify this fruit for me'). ${r3("t3_classify").summary} Grad-CAM shows that the decision is based on the red skin of the fruit, not on the white background. Scores: ${r3("t3_classify").table.map((t) => `${t.class} ${t.score}`).join(", ")}.`, 0.75));
  add(figure("t3_inspect", `Inspect = classify + measure. ${r3("t3_inspect").summary} Solidity ${r3("t3_inspect").metrics["solidity (area / convex hull)"]} and one foreground region confirm a single fruit. This Kaggle photo is not from the calibrated phone, so the size stays in pixels and the app explains why.`, 0.62));
  add(figure("t3_filter", `Filtering with parameters taken from the text (median, 5×5, add salt-and-pepper noise). ${r3("t3_filter").summary} The app also reports a table of all filters (box ${r3("t3_filter").table[1].PSNR_dB} dB, Gaussian ${r3("t3_filter").table[2].PSNR_dB} dB, bilateral ${r3("t3_filter").table[4].PSNR_dB} dB).`));
  add(figure("t3_fft", `Frequency-domain request with a cut-off read from the text. ${r3("t3_fft").summary}`));
  add(figure("t3_edges", `Edges request. ${r3("t3_edges").summary} The automatic thresholds are high because of the white background, so only the strongest edges remain; thresholds can be given in the request or the sidebar.`));
  add(figure("t3_features", `Matching two real checkerboard photos with SIFT. ${r3("t3_features").summary} Many matches are rejected by RANSAC because the board is a repetitive pattern: many squares have almost the same descriptor, which is exactly the case where geometric verification is needed.`, 0.75));
  add(figure("t3_calibrate", `Undistortion of a checkerboard photo with K from Task 1A. ${r3("t3_calibrate").summary} The shift map shows the radial correction (largest at the corners, ${r3("t3_calibrate").metrics["max undistortion shift (px)"]} px).`));

  add(H2("4.6 Testing"));
  add(P("Section 4 of `task3/Task3_assistant.ipynb` covers the integration test cases T1–T7 of the coursework guide plus routing and a headless UI test with `streamlit.testing.v1.AppTest`. The output of the last run (**12/12 passed**):"));
  add(table("tests", "Integration tests of the assistant (actual output).", ["Case", "Check", "Result"], [
    ["T1 Filtering", "Median 5×5 on 5 % salt & pepper; FFT low-pass", "PSNR 17.93 → 27.22 dB, best of 4 filters; Gaussian low-pass D0 = 20 keeps 95.4 % of spectral energy"],
    ["T2 Edges", "Edge map present, thresholds adjustable", "Canny thresholds 64/129, 7.08 % edge pixels"],
    ["T3 Features", "Two real views; SIFT / ORB / Harris+SIFT", "SIFT 172 matches / 81 inliers; ORB 227 / 29; Harris+SIFT 9 / 7; synthetic-view precision 0.97"],
    ["T4 Classify", "Whole Task 2 test split through the app", "37/38 = 97.4 %, identical to the Task 2 evaluation"],
    ["T5 Bad input", "Text file renamed .jpg, empty file, PDF, no image", "Clear message + hint, no crash"],
    ["T6 Missing artifacts", "No model.pth / no calibration.npz", "Message + how to fix; other tools keep working"],
    ["T7 Clean machine", "No personal absolute paths; env-var overrides", "Pass"],
    ["Calibrate", "Undistort a board photo", "Line straightness 0.95 → 0.26 px RMS"],
    ["Re-calibrate", "≥ 3 board photos → new calibration", "6 views: RMS 0.137 px, fx within −1.19 % of Task 1A"],
    ["Measure + Inspect", "Size in px; mm when distance is given", "Equivalent diameter 328 px, 392 × 349 px"],
    ["Intent router", "19 English/Vietnamese requests", "All routed correctly"],
    ["UI (AppTest)", "5 chat requests + 1 quick action", "Rendered without exceptions"],
  ], [1.6, 3.2, 4.2], { left: true, size: 17 }));
  add(P("Test T4 is the key integration check: running every test image through the app gives exactly the accuracy of the Task 2 evaluation (§ 7), so preprocessing and model loading in the app are identical to the evaluation pipeline. Re-calibrating from six board photos inside the app gives fx within 1.2 % of the Task 1A value, which also cross-checks Task 1A."));
  add(H2("4.7 Limitations"));
  add(bullet("The keyword router does not understand arbitrary paraphrases; unknown requests are answered with suggestions rather than guessed. An LLM with function calling could be added behind the same `tools.run` interface."));
  add(bullet("Metric size assumes a roughly fronto-parallel fruit at a known distance (a 10 % distance error gives a 10 % size error) and a photo from the calibrated phone; segmentation assumes one fruit on a plain, light background, and clutter is flagged rather than measured."));
  add(bullet("The CNN knows three classes only and its score is not a calibrated probability (Section 3.7)."));

  // ============================================================ 5. RUNNING
  add(H1("5. How to Run the Whole System"));
  add(P("Tested on Windows 11 with Python 3.12.10. One virtual environment can serve all three tasks. On Windows, create it in a short path to avoid the *Long Path* error when installing PyTorch."));
  add(P("Clone the repository (" + REPO + "); all paths below are relative to its folder `CW/`.", { align: AlignmentType.LEFT }));
  add(H3("Step 1 – Environment"));
  add(...codeBlock(["git clone git@github.com:phu9608-byte/FCVG101.git", "python -m venv C:\\venvs\\fcvg101", "C:\\venvs\\fcvg101\\Scripts\\Activate.ps1", "cd FCVG101\\CW", "python -m pip install -r task1\\requirements.txt -r task2\\requirements.txt -r task3\\requirements.txt", "python -m pip install ipykernel", "python -m ipykernel install --user --name fcvg101 --display-name \"Python (fcvg101)\"", "# optional, NVIDIA GPU for training:", "# python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130"], 17), spacer());
  add(H3("Step 2 – Task 1 (classical engine)"));
  add(P("Open `task1/` in VS Code or Jupyter with the kernel above and run all cells of `1A_calibration.ipynb`, `1B_filtering_edges_fft.ipynb` and `1C_features_matching.ipynb` in this order (about 1 minute). 1A writes `task1/artifacts/calibration.npz`; all figures and tables go to `task1/results/1A|1B|1C/`. The checkerboard photos are in `datasets/checkerboard/`, the fruit images in `datasets/fruits/raw/`.", { align: AlignmentType.LEFT }));
  add(H3("Step 3 – Task 2 (data pipeline and CNN)"));
  add(P("Open `task2/Task2_pipeline.ipynb` and run all cells (§ 1 ingestion → § 8 inference). Two switches in the first code cell: `KAGGLE_ARCHIVE` (path of the extracted Kaggle archive, only needed to rebuild `datasets/fruits/raw/` from scratch) and `RUN_TRAINING` (`False` evaluates the submitted `artifacts/model.pth` and reproduces Section 3.5; `True` retrains and overwrites it). The split in § 2 is deterministic, so re-running it recreates the same files.", { align: AlignmentType.LEFT }));
  add(H3("Step 4 – Task 3 (assistant)"));
  add(P("Open `task3/Task3_assistant.ipynb` and run all cells (engines, intent router, one request per tool, 12 integration tests). To use the GUI:", { align: AlignmentType.LEFT }));
  add(...codeBlock(["cd task3", "streamlit run app.py              # opens http://localhost:8501"], 17), spacer());
  add(P("If the folders are arranged differently, set `FCVG_TASK1_DIR`, `FCVG_TASK2_DIR`, `FCVG_DATASETS_DIR`, `FCVG_CALIBRATION_FILE` or `FCVG_MODEL_FILE`. `CW/README.md` and the `README.md` of each task repeat these instructions."));

  // ============================================================ 6. CONCLUSION
  add(H1("6. Conclusion"));
  add(P("The project delivers a working MVP of a visual inspection assistant. **Task 1** calibrates a real phone camera with a from-scratch implementation of Zhang's method that matches OpenCV to 2.3 × 10⁻⁸, selects the distortion model by cross-validation (hold-out RMS 0.52 px) and shows that undistortion makes lines straighter on an independent photo; it compares filters, edge detectors, frequency-domain filters and feature detectors with objective metrics (e.g. Canny F1 0.26 → 0.92 with pre-filtering, notch filter +6.7 dB, SIFT precision 97.4 % against a known homography). **Task 2** builds a documented pipeline from a public dataset (ingestion log, manual and automatic cleaning, duplicate checks, grouped split, train-only augmentation) and a compact CNN that reaches 97.4 % test accuracy and 0.973 macro-F1, saved with its preprocessing metadata. **Task 3** links both through a bilingual assistant that recognises the requested task, calls the original Task 1/2 code, explains predictions with Grad-CAM and handles bad input gracefully; 12/12 integration tests pass and the app reproduces the Task 2 test accuracy exactly."));
  add(P("The main limitations are the small fruit dataset and test set, a calibration valid for one phone only, classical segmentation that needs a plain background, and a keyword-based intent router. The next steps are more data with quality classes (ripe / defective), transfer learning, a printed or ChArUco calibration target, and an LLM-based router behind the same tool interface."));

  // ============================================================ 7. LINKS
  add(H1n("7. Project Repository and Model Artifact"));
  const linkPara = (label, url) => new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: label, bold: true }), new ExternalHyperlink({ link: url, children: [new TextRun({ text: url, style: "Hyperlink" })] })] });
  add(linkPara("Source code, datasets and report (GitHub): ", REPO));
  add(linkPara("Trained model artifact (model.pth): ", MODEL_URL));
  add(P("The repository contains the folder `CW/` with `datasets/`, `task1/`, `task2/`, `task3/` and `report/`, as described in this report."));

  // ============================================================ REFERENCES
  add(H1n("References"));
  const refs = [
    "Bradski, G. (2000) 'The OpenCV Library', *Dr. Dobb's Journal of Software Tools*. Documentation: https://docs.opencv.org/.",
    "Brown, D. C. (1966) 'Decentering distortion of lenses', *Photogrammetric Engineering*, 32(3), pp. 444–462.",
    "Buyukkinaci, M. (2018) *Fruit Images for Object Detection* [dataset]. Kaggle, licence CC0. https://www.kaggle.com/datasets/mbkinaci/fruit-images-for-object-detection.",
    "Canny, J. (1986) 'A computational approach to edge detection', *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 8(6), pp. 679–698.",
    "Fischler, M. A. and Bolles, R. C. (1981) 'Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography', *Communications of the ACM*, 24(6), pp. 381–395.",
    "Harris, C. and Stephens, M. (1988) 'A combined corner and edge detector', *Proceedings of the Alvey Vision Conference*, pp. 147–151.",
    "Hartley, R. and Zisserman, A. (2003) *Multiple View Geometry in Computer Vision*. 2nd edn. Cambridge: Cambridge University Press.",
    "Kingma, D. P. and Ba, J. (2015) 'Adam: a method for stochastic optimization', *International Conference on Learning Representations (ICLR)*.",
    "Lowe, D. G. (2004) 'Distinctive image features from scale-invariant keypoints', *International Journal of Computer Vision*, 60(2), pp. 91–110.",
    "Otsu, N. (1979) 'A threshold selection method from gray-level histograms', *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1), pp. 62–66.",
    "Paszke, A. et al. (2019) 'PyTorch: an imperative style, high-performance deep learning library', *Advances in Neural Information Processing Systems*, 32.",
    "Rublee, E., Rabaud, V., Konolige, K. and Bradski, G. (2011) 'ORB: an efficient alternative to SIFT or SURF', *IEEE International Conference on Computer Vision*, pp. 2564–2571.",
    "Selvaraju, R. R. et al. (2017) 'Grad-CAM: visual explanations from deep networks via gradient-based localization', *IEEE International Conference on Computer Vision*, pp. 618–626.",
    "Szeliski, R. (2022) *Computer Vision: Algorithms and Applications*. 2nd edn. Cham: Springer.",
    "Tomasi, C. and Manduchi, R. (1998) 'Bilateral filtering for gray and color images', *IEEE International Conference on Computer Vision*, pp. 839–846.",
    "Wang, Z., Bovik, A. C., Sheikh, H. R. and Simoncelli, E. P. (2004) 'Image quality assessment: from error visibility to structural similarity', *IEEE Transactions on Image Processing*, 13(4), pp. 600–612.",
    "Zhang, Z. (2000) 'A flexible new technique for camera calibration', *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 22(11), pp. 1330–1334.",
    "Streamlit (2025) *Streamlit documentation*. https://docs.streamlit.io/.",
  ];
  refs.forEach((r) => add(new Paragraph({ children: runs(r, { size: 20 }), indent: { left: 400, hanging: 400 }, spacing: { after: 80 }, alignment: AlignmentType.LEFT })));

  // ============================================================ APPENDIX
  add(H1("Appendix A – Source Code"));
  add(P("All source code of the project is listed below as selectable text: the code cells of every notebook and the Python modules they import, in the order Task 1, Task 2, Task 3. Comments in some files are written in Vietnamese. The emoji in the button labels of `task3/app.py` are printed as the equivalent Python escape sequences (e.g. `\\U0001F34E` for the apple icon), which give the same string at run time. The figure-generation helpers used only for this report are in `report/build/`."));
  const pyFiles = [
    ["A.1 Task 1 – engine (task1/src)", ["task1/src/calibration.py", "task1/src/filtering.py", "task1/src/features.py"]],
    ["A.4 Task 2 – modules imported by the notebook and by Task 3 (task2/src)", ["task2/src/model.py", "task2/src/data_utils.py", "task2/src/inference.py", "task2/requirements.txt"]],
    ["A.6 Task 3 – Streamlit app and assistant package (task3)", ["task3/app.py", "task3/assistant/__init__.py", "task3/assistant/settings.py", "task3/assistant/errors.py", "task3/assistant/engines.py", "task3/assistant/intent.py", "task3/assistant/tools.py", "task3/assistant/gradcam.py", "task3/.streamlit/config.toml", "task3/requirements.txt"]],
  ];
  const listFile = (rel) => {
    const lines = fs.readFileSync(path.join(CW, rel), "utf8").replace(/^\uFEFF/, "").split(/\r?\n/);
    while (lines.length && !lines[lines.length - 1].trim()) lines.pop();
    const w = String(lines.length).length;
    add(H3(rel + `  (${lines.length} lines)`));
    // no line numbers: code copied from the PDF must run unchanged
    add(...lines.map((l) => new Paragraph({ spacing: { before: 0, after: 0, line: 176, lineRule: "exact" },
      children: [new TextRun({ text: escapeSymbols(l.replace(/\t/g, "    ").replace(/\s+$/, "")), font: MONO, size: 14 })] })));
  };
  const listNotebook = (rel) => {
    const d = JSON.parse(fs.readFileSync(path.join(CW, rel), "utf8"));
    add(H3(rel));
    d.cells.forEach((c, i) => {
      if (c.cell_type !== "code") return;
      const text = Array.isArray(c.source) ? c.source.join("") : c.source;
      const src = text.replace(/\s+$/, "").split(/\r?\n/);
      add(new Paragraph({ spacing: { before: 60, after: 0 }, keepNext: true, children: [new TextRun({ text: `# ---- cell ${i + 1}`, font: MONO, size: 14, color: "2E74B5", bold: true })] }));
      add(...src.map((l) => new Paragraph({ spacing: { before: 0, after: 0, line: 176, lineRule: "exact" }, children: [new TextRun({ text: escapeSymbols(l.replace(/\t/g, "    ")) || " ", font: MONO, size: 14 })] })));
    });
  };
  add(H2(pyFiles[0][0])); pyFiles[0][1].forEach(listFile);
  add(H2("A.2 Task 1 – notebooks (code cells)"));
  ["task1/1A_calibration.ipynb", "task1/1B_filtering_edges_fft.ipynb", "task1/1C_features_matching.ipynb"].forEach(listNotebook);
  add(H2("A.3 Task 2 – pipeline notebook (code cells)"));
  listNotebook("task2/Task2_pipeline.ipynb");
  add(H2(pyFiles[1][0])); pyFiles[1][1].forEach(listFile);
  add(H2("A.5 Task 3 – assistant notebook (code cells)"));
  listNotebook("task3/Task3_assistant.ipynb");
  add(H2(pyFiles[2][0])); pyFiles[2][1].forEach(listFile);
  add(new Paragraph({ spacing: { before: 360 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Repository: ", bold: true }), new ExternalHyperlink({ link: REPO, children: [new TextRun({ text: REPO, style: "Hyperlink" })] })] }));
  return C;
}

// two passes: the first fixes figure/table numbers so that {fig:..}/{tab:..} references resolve
content();
const body = content();

const doc = new Document({
  creator: "FCVG101 student", title: "FCVG101 Coursework Report – FruitSort Vision Assistant",
  description: "Report for Tasks 1–3 of the FCVG101 coursework",
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 34, bold: true, color: ACCENT }, paragraph: { spacing: { before: 240, after: 200 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 27, bold: true, color: "2E74B5" }, paragraph: { spacing: { before: 280, after: 120 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 23, bold: true, color: "404040" }, paragraph: { spacing: { before: 200, after: 80 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }, { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1000, hanging: 270 } } } }] },
    ...["num1", "num2", "num3"].map((r) => ({ reference: r, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] })),
  ] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN, header: 560, footer: 560 } }, titlePage: true },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF", space: 2 } }, children: [new TextRun({ text: "FCVG101 Coursework – FruitSort Vision Assistant", size: 16, color: "808080" })] })] }), first: new Header({ children: [] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES], size: 16, color: "808080" })] })] }), first: new Footer({ children: [] }) },
    children: body,
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT, (buf.length / 1048576).toFixed(2), "MB; figures", figCount, "tables", tabCount); });
