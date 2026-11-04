"""Combine each book's MP3 recordings into one file using ffmpeg.

Run from any directory: python scripts/combine_recordings.py
The original recordings are kept in each book's audio directory.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path


PAIRED_DIR = Path(__file__).resolve().parents[1] / "data" / "paired"


def natural_key(path: Path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


def combine(book: Path) -> None:
    recordings = sorted((book / "audio").glob("*.mp3"), key=natural_key)
    if not recordings:
        return

    output = book / "combined.mp3"
    if output.exists():
        print(f"Skipping {book.name}: {output.name} already exists")
        return

    # Use a temporary directory on the same filesystem for an atomic final move.
    with tempfile.TemporaryDirectory(dir=book) as temp_dir:
        temp = Path(temp_dir)
        manifest = temp / "recordings.txt"
        manifest.write_text(
            "".join(
                "file '" + str(path.resolve()).replace("'", "'\\''") + "'\n"
                for path in recordings
            ),
            encoding="utf-8",
        )
        combined = temp / "combined.mp3"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-f", "concat", "-safe", "0",
             "-i", str(manifest), "-vn", "-c:a", "libmp3lame", str(combined)],
            check=True,
        )
        combined.replace(output)
    print(f"Combined {len(recordings)} recordings: {output}")


def main() -> int:
    if not PAIRED_DIR.is_dir():
        print(f"Paired directory not found: {PAIRED_DIR}", file=sys.stderr)
        return 1

    failed = False
    for book in sorted(path for path in PAIRED_DIR.iterdir() if path.is_dir()):
        try:
            combine(book)
        except (OSError, subprocess.CalledProcessError) as error:
            print(f"Failed to combine {book.name}: {error}", file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
