"""Every folder and file the pipeline uses, in one place.

Notebooks and scripts import their paths from here instead of typing
"/home/intern/..." or "../../data/...". If a folder moves, change it here once.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = ROOT / "data" / "test"
TEST_AUDIO = TEST_DIR / "test_audio.mp3"
BOOK_TEXT = TEST_DIR / "book.txt"
RAW_RESPONSES = TEST_DIR / "raw_api_responses"
BOOK_MATCHES = TEST_DIR / "book_matches_draft.json"

SECTIONS_DIR = ROOT / "src" / "notebooks" / "temporary_sections"
SECTIONS_JSON = ROOT / "src" / "notebooks" / "temporary_sections.json"

ALIGNMENT_JSON = TEST_DIR / "alignment.json"

CHUNKS_DIR = TEST_DIR / "chunks"
RECORDING_WAV = CHUNKS_DIR / "recording_16k.wav"
CLIPS_DIR = CHUNKS_DIR / "wavs"
CHUNKS_METADATA = CHUNKS_DIR / "metadata.json"
MANIFEST_ALL = TEST_DIR / "manifest_all.jsonl"
MANIFEST_CLEAN = TEST_DIR / "manifest_clean.jsonl"

PAIRED_DIR = ROOT / "data" / "paired"


def book_paths(book_dir):
    """Every file the pipeline reads or writes for one book folder in data/paired.

    Everything the pipeline makes goes into <book>/pipeline/, so the book's
    own files (audio/, book.txt, metadata.json) are never touched.
    """
    book_dir = Path(book_dir)
    output = book_dir / "pipeline"
    return {
        "book_text": book_dir / "book.txt",
        "output": output,
        "recording_wav": output / "chunks" / "recording_16k.wav",
        "sections_dir": output / "sections",
        "sections_json": output / "sections.json",
        "raw_responses": output / "raw_api_responses",
        "alignment_json": output / "alignment.json",
        "clips_dir": output / "chunks" / "wavs",
        "chunks_metadata": output / "chunks" / "metadata.json",
        "manifest_all": output / "manifest_all.jsonl",
        "manifest_clean": output / "manifest_clean.jsonl",
    }