"""Shrink the exported report below the CMS limit (2 MB per PDF) without touching its text.

export_pdf.ps1 already asks Word for its minimum-size PDF. This step rewrites the file with compressed
object streams and drops unused objects; only if that is not enough are the embedded images re-compressed.
All text, including the source-code appendix, stays selectable.

    python report/build/compress_pdf.py            (run from the CW folder, after export_pdf.ps1)
Requires: pip install pymupdf
"""
from pathlib import Path
import shutil
import sys

import pymupdf

LIMIT_BYTES = 2_000_000          # "2 MB" read strictly (decimal), so the file also fits a 2 MiB limit
PDF = Path(__file__).resolve().parents[1] / "FCVG101_Coursework_Report.pdf"


def rewrite(src: Path, dst: Path, image_setting=None) -> int:
    doc = pymupdf.open(src)
    if image_setting:
        dpi, quality = image_setting
        doc.rewrite_images(dpi_threshold=dpi + 10, dpi_target=dpi, quality=quality, lossless=False, bitonal=False)
    doc.save(dst, garbage=3, deflate=True, use_objstms=1)
    doc.close()
    return dst.stat().st_size


def main():
    before = PDF.stat().st_size
    tmp = PDF.with_suffix(".tmp.pdf")
    for setting in [None, (130, 55), (100, 45)]:          # gentlest option that fits wins
        size = rewrite(PDF, tmp, setting)
        print(f"{'object streams only' if setting is None else f'images {setting[0]} dpi, quality {setting[1]}'}: {size:,} bytes")
        if size < LIMIT_BYTES * 0.97:
            break
    else:
        tmp.unlink()
        sys.exit("Still above the limit; reduce the figures in prepare_figures.py.")
    shutil.move(tmp, PDF)
    print(f"{PDF.name}: {before:,} -> {PDF.stat().st_size:,} bytes ({PDF.stat().st_size / 1048576:.2f} MiB)")


if __name__ == "__main__":
    main()
