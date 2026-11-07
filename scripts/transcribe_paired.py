"""Transcribe paired-book recordings, one book at a time."""

import argparse
import http.client
import json
import sys
import tempfile
from pathlib import Path

from transcribe import transcribe


PAIRED_DIR = Path(__file__).resolve().parents[1] / "data" / "paired"
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"}


def parse_response(response):
    """Return the JSON response and transcript in the API's segment order."""
    payload = json.loads(response.decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object from the transcription API")

    segments = payload.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("API response has no transcription segments")
    if any(not isinstance(s, dict) or not isinstance(s.get("text"), str)
           for s in segments):
        raise ValueError("Each transcription segment must have a text field")
    text = " ".join(s["text"].strip() for s in segments if s["text"].strip())
    if not text:
        raise ValueError("API response contains no transcription text")
    return payload, text


def process_book(book, *, url, timeout, force):
    # Some books have an older combined file at the root as well.
    audio = next((path for path in (book / "audio" / "combined.mp3",
                                    book / "combined.mp3") if path.is_file()), None)
    if audio is None:
        recordings = [path for directory in (book / "audio", book)
                      if directory.is_dir() for path in directory.iterdir()
                      if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS]
        if len(recordings) != 1:
            print(f"Skipping {book.name}: found {len(recordings)} audio files and no combined.mp3")
            return
        audio = recordings[0]

    transcript_path = book / "transcription.txt"
    metadata_path = book / "metadata.json"
    if not force and (transcript_path.exists() or metadata_path.exists()):
        print(f"Skipping {book.name}: output already exists (use --force to replace)")
        return

    print(f"Transcribing {audio}...", flush=True)
    response = transcribe(audio, url=url, timeout=timeout)
    metadata, text = parse_response(response)

    # Stage both outputs before replacing either file.
    with tempfile.TemporaryDirectory(dir=book) as temporary:
        temp = Path(temporary)
        (temp / "transcription.txt").write_text(text + "\n", encoding="utf-8")
        (temp / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (temp / "metadata.json").replace(metadata_path)
        (temp / "transcription.txt").replace(transcript_path)
    print(f"Saved {transcript_path} and {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-dir", type=Path, default=PAIRED_DIR)
    parser.add_argument("--url", default=None, help="override TRANSCRIBE_URL / API_URL")
    parser.add_argument("--timeout", type=float, default=600, help="socket timeout in seconds")
    parser.add_argument("--force", action="store_true", help="replace existing output files")
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not args.paired_dir.is_dir():
        parser.error(f"paired directory not found: {args.paired_dir}")

    failed = False
    for book in sorted(path for path in args.paired_dir.iterdir() if path.is_dir()):
        try:
            process_book(book, url=args.url, timeout=args.timeout, force=args.force)
        except (OSError, ValueError, RuntimeError, http.client.HTTPException) as error:
            print(f"Failed to transcribe {book.name}: {error}", file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
