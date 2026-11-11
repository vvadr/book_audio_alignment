"""Every folder and file the pipeline uses, in one place.

Notebooks and scripts import their paths from here instead of typing
"/home/intern/..." or "../../data/...". If a folder moves, change it here once.
"""

from pathlib import Path

# The project folder: this file is in scripts/, and scripts/ is in the project folder
ROOT = Path(__file__).resolve().parent.parent

# The test book: inputs we already have
TEST_DIR = ROOT / "data" / "test"
TEST_AUDIO = TEST_DIR / "test_audio.mp3"             # the whole recording
BOOK_TEXT = TEST_DIR / "book.txt"                    # the book after OCR
RAW_RESPONSES = TEST_DIR / "raw_api_responses"       # one .response file per section
BOOK_MATCHES = TEST_DIR / "book_matches_draft.json"  # where each section is in the book

# The ~60 s sections that vad.ipynb cut (they still live next to the notebooks)
SECTIONS_DIR = ROOT / "src" / "notebooks" / "temporary_sections"
SECTIONS_JSON = ROOT / "src" / "notebooks" / "temporary_sections.json"

# Outputs of the next stages (they don't exist yet)
ALIGNMENT_JSON = TEST_DIR / "alignment.json"   # stage 8: every word with its time
CHUNKS_DIR = TEST_DIR / "chunks"               # stages 11-12: the training clips