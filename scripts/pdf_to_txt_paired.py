"""Create book.txt in paired-book folders, extracting text or OCRing scanned pages."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata

import pymupdf


ROOT = Path(__file__).resolve().parents[1]
PAIRED_DIR = ROOT / "data" / "paired"
CACHE_VERSION = 2
LETTER_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
LATIN = re.compile(r"[A-Za-z]")
CYRILLIC = re.compile(r"[\u0400-\u04ff]")


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     delete=False) as temporary:
        name = Path(temporary.name)
        try:
            temporary.write(text)
        except BaseException:
            name.unlink(missing_ok=True)
            raise
    try:
        name.replace(path)
    finally:
        name.unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def poor_text(text):
    """Detect missing text, broken encodings and conspicuously garbled OCR layers."""
    text = re.sub(r"(?im)^\s*Kitob\.uz - Respublika bolalar kutubxonasi\s*$", "", text)
    letters = [char for char in text if char.isalpha()]
    if len(letters) < 30:
        return True
    if text.count("\ufffd") > max(2, len(letters) * 0.01):
        return True
    words = LETTER_WORD.findall(text)
    mixed = sum(bool(LATIN.search(word) and CYRILLIC.search(word)) for word in words)
    if len(words) >= 30 and mixed / len(words) > 0.12:
        return True
    # Some Latin scans have an old Cyrillic OCR layer: Ь stands for b everywhere.
    cyrillic = CYRILLIC.findall(text)
    return (len(cyrillic) > 100
            and sum(char in "ьЬъЪ" for char in cyrillic) / len(cyrillic) > 0.055)


def analyze_pdf(path):
    with pymupdf.open(path) as document:
        if document.needs_pass:
            raise ValueError(f"Encrypted PDF requires a password: {path}")
        pages = []
        for page in document:
            text = page.get_text(sort=True)
            pages.append({"text": text, "needs_ocr": poor_text(text)})
        if not pages:
            raise ValueError(f"PDF contains no pages: {path}")
    garbled = sum(page["needs_ocr"] and sum(c.isalpha() for c in page["text"]) >= 30
                  for page in pages)
    if garbled > len(pages) / 2:
        # A consistently broken OCR layer is unreliable even on short pages
        # that happen not to trip the per-page script/encoding heuristics.
        for page in pages:
            page["needs_ocr"] = True
    front = " ".join(page["text"] for page in pages[:3]).casefold()
    if len(pages) <= 5 and "taqvimiy" in front and "dars" in front:
        role = "lesson_plan"
    elif "metodik" in front and ("qo‘llanma" in front or "qo'llanma" in front):
        role = "teacher_guide"
    else:
        role = "book"
    useful = [page for page in pages if not page["needs_ocr"]]
    summary = {
        "filename": path.name,
        "sha256": file_hash(path),
        "pages": len(pages),
        "usable_text_pages": len(useful),
        "pages_needing_ocr": sum(page["needs_ocr"] for page in pages),
        "native_letters": sum(sum(c.isalpha() for c in page["text"]) for page in pages),
        "role": role,
        "preview": " ".join(front.split())[:500],
    }
    return summary, pages


def source_rank(summary):
    # Prefer the actual book over schedules/guides, then a readable edition.
    useful = summary["usable_text_pages"]
    return (summary["role"] == "book", bool(useful),
            useful or summary["pages"], summary["native_letters"])


def ocr_settings(tesseract, tessdata, language):
    env = os.environ.copy()
    local = ROOT / ".local" / "ocr" / "usr"
    if tesseract == "tesseract" and not shutil.which(tesseract):
        if (local / "bin" / "tesseract").is_file():
            tesseract = str(local / "bin" / "tesseract")
            libraries = [str(p) for p in (local / "lib").glob("*-linux-gnu")]
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                libraries + [env.get("LD_LIBRARY_PATH", "")])
            if tessdata is None:
                tessdata = local / "share" / "tesseract-ocr" / "5" / "tessdata"
        else:
            raise RuntimeError("OCR needs Tesseract and the uzb/uzb_cyrl language packs; "
                               "install them or use --tesseract and --tessdata")
    command = [str(tesseract)]
    if tessdata:
        command.extend(["--tessdata-dir", str(Path(tessdata).expanduser().resolve())])
    env["OMP_THREAD_LIMIT"] = "1"  # Each page already runs in its own worker.
    result = subprocess.run(command + ["--list-langs"], env=env, capture_output=True,
                            text=True, check=True)
    missing = set(language.split("+")) - set(result.stdout.splitlines())
    if missing:
        raise RuntimeError("Missing Tesseract language packs: " + ", ".join(sorted(missing)))
    return command, env


def ocr_page(path, index, cache_path, command, env, language, dpi):
    """Workers open their own documents; PyMuPDF objects are never shared."""
    with pymupdf.open(path) as document, tempfile.TemporaryDirectory() as temporary:
        page = document[index]
        image = Path(temporary) / "page.png"
        page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False).save(image)
        def recognize(languages):
            return subprocess.run(command + [str(image), "stdout", "-l", languages],
                                  env=env, capture_output=True, text=True, timeout=240)

        result = recognize(language)
        if result.returncode:
            raise RuntimeError(f"OCR failed for {path.name} page {index + 1}: {result.stderr.strip()}")
        # On a single-script page, a second pass avoids Latin lookalikes in
        # Cyrillic words (and vice versa). Bilingual pages retain both models.
        used_language = language
        if set(language.split("+")) == {"uzb", "uzb_cyrl"}:
            latin = len(LATIN.findall(result.stdout))
            cyrillic = len(CYRILLIC.findall(result.stdout))
            total = latin + cyrillic
            if total >= 100 and max(latin, cyrillic) / total > 0.85:
                used_language = "uzb" if latin > cyrillic else "uzb_cyrl"
                refined = recognize(used_language)
                if refined.returncode == 0 and refined.stdout.strip():
                    result = refined
        value = {"page": index + 1, "method": "ocr", "language": used_language,
                 "text": result.stdout.strip()}
        write_json(cache_path, value)
        return value


def clean_pages(pages):
    """Keep original scripts and verse lines; remove repeated margins and wrap hyphens."""
    ligatures = str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
                               "\u00ad": "", "\u00a0": " ", "\u200b": "", "\ufeff": ""})
    lines_by_page = []
    margins = Counter()
    for page in pages:
        text = unicodedata.normalize("NFC", page["text"])
        text = re.sub(r"([^\W\d_])\u00ad\s*\n\s*(?=[^\W\d_])", r"\1", text)
        text = text.translate(ligatures)
        lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.splitlines()]
        lines = [line for line in lines if line]
        lines_by_page.append(lines)
        margins.update(set(lines[:1] + lines[-1:]))
    repeated = {line for line, count in margins.items()
                if count >= max(4, math.ceil(len(pages) * 0.5)) and len(line) <= 120}
    output = []
    for number, lines in enumerate(lines_by_page, 1):
        kept = []
        for i, line in enumerate(lines):
            margin = i == 0 or i == len(lines) - 1
            page_number = line.strip("—–- .[]()")
            if margin and (line in repeated or (page_number.isdecimal()
                           and abs(int(page_number) - number) <= 5)):
                continue
            if re.fullmatch(r"(?:www\.)?(?:ziyouz\.(?:com|uz)|kitob\.uz|eduportal\.uz)\S*", line, re.I):
                continue
            if line.casefold().startswith("kitob.uz - respublika bolalar kutubxonasi"):
                continue
            kept.append(line)
        text = "\n".join(kept)
        text = re.sub(r"([^\W\d_])[-‐]\n(?=[^\W\d_])", r"\1", text)
        if text.strip():
            output.append(text.strip())
    text = "\n\n".join(output)
    return re.sub(r"([^\W\d_])[-‐]\n\n(?=[^\W\d_])", r"\1", text) + "\n"


def source_warnings(book, selected):
    warnings = []
    if selected["role"] != "book":
        warnings.append(f"Only a {selected['role'].replace('_', ' ')} was supplied; this is not a full textbook.")
    metadata = book / "meta.json"
    if metadata.is_file():
        meta = json.loads(metadata.read_text(encoding="utf-8"))
        title = meta.get("name", "")
        sources = [item for item in meta.get("text", [])
                   if Path(item.get("path", "")).name == selected["filename"]]
        source_title = sources[0].get("book", "") if sources else ""
        def tokens(value):
            words = LETTER_WORD.findall(value.casefold())
            # Metadata sometimes has a Cyrillic А in an otherwise Latin title.
            lookalikes = str.maketrans("аесорхуікмт", "aecopxyikmt")
            return {word.translate(lookalikes) if LATIN.search(word) else word for word in words}
        wanted, actual = tokens(title), tokens(source_title)
        if wanted and actual and len(wanted & actual) / len(wanted | actual) < 0.25:
            warnings.append(f"Folder title {title!r} differs from supplied PDF title {source_title!r}.")
    for pattern, label in [(r"(\d+)\s*-?\s*sinf", "grade"), (r"(\d+)\s*-?\s*qism", "part")]:
        wanted = re.search(pattern, book.name)
        actual = re.search(pattern, selected["preview"])
        if wanted and actual and wanted[1] != actual[1]:
            warnings.append(f"Folder {label} {wanted[1]} differs from PDF {label} {actual[1]}.")
    return warnings


def positive_integer(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-dir", type=Path, default=PAIRED_DIR)
    parser.add_argument("--book", action="append", help="process only this folder name (repeatable)")
    parser.add_argument("--workers", type=positive_integer, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--dpi", type=positive_integer, default=300)
    parser.add_argument("--language", default="uzb+uzb_cyrl")
    parser.add_argument("--tesseract", default="tesseract")
    parser.add_argument("--tessdata", type=Path)
    parser.add_argument("--force", action="store_true", help="replace book.txt; reuse completed OCR pages")
    parser.add_argument("--force-ocr", action="store_true", help="OCR every page, even if it has text")
    parser.add_argument("--analyze-only", action="store_true", help="write report without extracting/OCRing")
    args = parser.parse_args()
    args.paired_dir = args.paired_dir.expanduser().resolve()
    if not args.paired_dir.is_dir():
        parser.error(f"Paired directory not found: {args.paired_dir}")
    books = sorted(path for path in args.paired_dir.iterdir()
                   if path.is_dir() and not path.name.startswith("."))
    if args.book:
        missing = set(args.book) - {path.name for path in books}
        if missing:
            parser.error("Unknown book folders: " + ", ".join(sorted(missing)))
        books = [path for path in books if path.name in args.book]
    report = {"language": args.language, "dpi": args.dpi, "books": []}
    cache_key = hashlib.sha256(json.dumps([CACHE_VERSION, args.language, args.dpi,
                                          args.tesseract, str(args.tessdata)]).encode()).hexdigest()[:16]
    jobs, documents = {}, {}
    for book in books:
        entry = {"book": book.name}
        report["books"].append(entry)
        if ((book / "book.txt").is_file() and (book / "book.txt").stat().st_size
                and not args.force and not args.analyze_only):
            entry["status"] = "skipped_existing"
            print(f"Skipping {book.name}: book.txt exists (use --force)", flush=True)
            continue
        try:
            paths = sorted(path for directory in (book / "text", book) if directory.is_dir()
                           for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")
            if not paths:
                raise ValueError("No PDF supplied")
            candidates = []
            for path in paths:
                try:
                    summary, pages = analyze_pdf(path)
                    candidates.append((summary, pages, path))
                except (OSError, RuntimeError, ValueError) as error:
                    entry.setdefault("candidate_errors", []).append({"filename": path.name, "error": str(error)})
            if not candidates:
                raise ValueError("No readable PDF found")
            selected, native, path = max(candidates, key=lambda candidate: source_rank(candidate[0]))
            entry.update(status="analyzed", candidates=[candidate[0] for candidate in candidates],
                         source=str(path.relative_to(book)), sha256=selected["sha256"],
                         pages=selected["pages"], warnings=source_warnings(book, selected))
            for warning in entry["warnings"]:
                print(f"WARNING {book.name}: {warning}", flush=True)
            print(f"{book.name}: {path.name}, {len(native)} pages, "
                  f"{selected['pages_needing_ocr']} need OCR", flush=True)
            if args.analyze_only:
                continue
            identity = selected["sha256"]
            if identity not in documents:
                extracted = []
                for index, page in enumerate(native):
                    if page["needs_ocr"] or args.force_ocr:
                        cache = args.paired_dir / ".pdf_text_cache" / identity / cache_key / f"{index + 1:05d}.json"
                        if cache.is_file():
                            try:
                                value = json.loads(cache.read_text(encoding="utf-8"))
                                if (value["page"] != index + 1 or value["method"] != "ocr"
                                        or not isinstance(value["text"], str)):
                                    raise ValueError("Invalid cached page")
                                extracted.append(value)
                                continue
                            except (KeyError, ValueError):
                                pass
                        extracted.append(None)
                        jobs[(identity, index)] = (path, index, cache)
                    else:
                        extracted.append({"page": index + 1, "method": "native", "text": page["text"]})
                documents[identity] = extracted
        except (OSError, RuntimeError, ValueError) as error:
            entry.update(status="failed", error=str(error))
            print(f"Failed {book.name}: {error}", file=sys.stderr, flush=True)

    def save_ready_books():
        for entry in report["books"]:
            if entry["status"] != "analyzed":
                continue
            pages = documents[entry["sha256"]]
            if any(page is None for page in pages):
                continue
            book = args.paired_dir / entry["book"]
            try:
                text = clean_pages(pages)
                if not any(char.isalpha() for char in text):
                    raise ValueError("PDF extraction produced no text")
                entry.update(status="completed", characters=len(text), words=len(text.split()),
                             native_pages=sum(page["method"] == "native" for page in pages),
                             ocr_pages=sum(page["method"] == "ocr" for page in pages),
                             empty_pages=[page["page"] for page in pages if not page["text"].strip()])
                atomic_write(book / "book.txt", text)
                write_json(book / "book_extraction.json", entry)
                print(f"Saved {book.name}/book.txt ({entry['words']:,} words)", flush=True)
            except (OSError, ValueError) as error:
                entry.update(status="failed", error=str(error))
                print(f"Failed {book.name}: {error}", file=sys.stderr, flush=True)

    if not args.analyze_only:
        save_ready_books()
        if jobs:
            print(f"OCR: {len(jobs)} uncached pages, {args.workers} workers", flush=True)
            try:
                command, env = ocr_settings(args.tesseract, args.tessdata, args.language)
                with ProcessPoolExecutor(max_workers=args.workers) as executor:
                    futures = {executor.submit(ocr_page, *job, command, env, args.language, args.dpi): key
                               for key, job in jobs.items()}
                    for completed, future in enumerate(as_completed(futures), 1):
                        identity, index = futures[future]
                        try:
                            documents[identity][index] = future.result()
                        except Exception as error:
                            for entry in report["books"]:
                                if entry.get("sha256") == identity:
                                    entry.update(status="failed", error=f"Page {index + 1}: {error}")
                            print(f"OCR page failed: {error}", file=sys.stderr, flush=True)
                        if completed % 25 == 0 or completed == len(jobs):
                            print(f"OCR progress: {completed}/{len(jobs)} pages", flush=True)
                            save_ready_books()
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                for entry in report["books"]:
                    if entry["status"] == "analyzed":
                        entry.update(status="failed", error=str(error))
                print(f"OCR setup failed: {error}", file=sys.stderr, flush=True)
        save_ready_books()
    counts = Counter(entry["status"] for entry in report["books"])
    report["summary"] = dict(counts)
    report_path = args.paired_dir / "pdf_extraction_report.json"
    write_json(report_path, report)
    print(f"Summary: {dict(counts)}. Report: {report_path}", flush=True)
    return int(bool(counts["failed"]))


if __name__ == "__main__":
    raise SystemExit(main())
