"""Convert one PDF to text, using OCR for image-based pages."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pymupdf

PDF_PATH = "data/test/test_book.pdf"


def pdf_to_txt(pdf, output=None, *, language="uzb", tesseract="tesseract", tessdata=None, force_ocr=False):
    """Convert one PDF to a text file and return its path."""
    pdf = Path(pdf).expanduser()
    output = Path(output).expanduser() if output else pdf.with_suffix(".txt")
    if pdf.resolve() == output.resolve():
        raise ValueError("Output must differ from the input PDF")
    env = os.environ.copy()
    local_ocr = Path(__file__).resolve().parents[1] / ".local" / "ocr" / "usr"
    local_tesseract = local_ocr / "bin" / "tesseract"
    if tesseract == "tesseract" and not shutil.which(tesseract) and local_tesseract.is_file():
        tesseract = local_tesseract
        library_dirs = [str(p) for p in (local_ocr / "lib").glob("*-linux-gnu") if p.is_dir()]
        env["LD_LIBRARY_PATH"] = os.pathsep.join(library_dirs + [env.get("LD_LIBRARY_PATH", "")])
        if not tessdata:
            tessdata = local_ocr / "share" / "tesseract-ocr" / "5" / "tessdata"
    if tessdata:
        env["TESSDATA_PREFIX"] = str(Path(tessdata).expanduser().resolve())
    pages = []
    with pymupdf.open(pdf) as doc, tempfile.TemporaryDirectory() as temp_dir:
        image = Path(temp_dir) / "page.png"
        for number, page in enumerate(doc, 1):
            text = page.get_text().strip()
            if force_ocr or len(text) < 20:
                print(f"OCR page {number}/{len(doc)}", file=sys.stderr)
                page.get_pixmap(dpi=300, alpha=False).save(image)
                result = subprocess.run(
                    [str(tesseract), str(image), "stdout", "-l", language],
                    check=True, stdout=subprocess.PIPE, env=env,
                )
                text = result.stdout.decode("utf-8").strip()
            pages.append(text)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(pages) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", nargs="?", default=PDF_PATH, help="PDF path (defaults to PDF_PATH above)")
    parser.add_argument("-o", "--output", help="output path (default: same name as PDF, with .txt)")
    parser.add_argument("--language", default="uzb", help="Tesseract language code (default: uzb)")
    parser.add_argument("--tesseract", default="tesseract", help="path to Tesseract executable")
    parser.add_argument("--tessdata", help="directory containing Tesseract language data")
    parser.add_argument("--force-ocr", action="store_true", help="use OCR on every page")
    args = parser.parse_args()
    try:
        output = pdf_to_txt(args.pdf, args.output, language=args.language,
                            tesseract=args.tesseract, tessdata=args.tessdata,
                            force_ocr=args.force_ocr)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Conversion failed: {error}", file=sys.stderr)
        return 1
    print(f"Saved text to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
