"""Transcribe temporary audiobook sections, preserving original audio times."""

import argparse
import json
from pathlib import Path

from transcribe import extract_text, transcribe

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METADATA = ROOT / "src/notebooks/temporary_sections.json"
DEFAULT_OUTPUT = ROOT / "data/test/section_transcriptions.json"


def save_results(path, results):
    """Save progress after each successful section (and avoid partial JSON files)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--audio-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--url", default=None, help="override TRANSCRIBE_URL / API_URL")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--limit", type=int, default=None, help="test first N sections")
    parser.add_argument("--force", action="store_true", help="re-transcribe saved sections")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")

    metadata_path = args.metadata.resolve()
    audio_dir = (
        args.audio_dir.resolve() if args.audio_dir
        else metadata_path.parent / "temporary_sections"
    )
    output_path = args.output.resolve()
    with metadata_path.open(encoding="utf-8") as f:
        sections = json.load(f)
    if not isinstance(sections, list):
        raise ValueError("Metadata must contain a JSON list of sections")

    if output_path.exists():
        with output_path.open(encoding="utf-8") as f:
            existing = json.load(f)
        if not isinstance(existing, list):
            raise ValueError("Existing output must contain a JSON list")
    else:
        existing = []

    by_filename = {item["filename"]: item for item in existing}
    to_process = sections[:args.limit] if args.limit else sections

    for index, section in enumerate(to_process, start=1):
        filename = section["filename"]
        start = section["start"]
        end = section["end"]
        audio_path = audio_dir / filename
        if not audio_path.is_file():
            raise FileNotFoundError(f"Missing audio section: {audio_path}")

        old = by_filename.get(filename)
        if (
            old is not None and not args.force
            and old.get("start") == start
            and old.get("end") == end
            and old.get("transcription")
        ):
            print(f"[{index}/{len(to_process)}] Skipping {filename} (already saved)")
            continue

        print(f"[{index}/{len(to_process)}] Transcribing {filename}...")
        response = transcribe(audio_path, url=args.url, timeout=args.timeout)
        raw_dir = output_path.parent / "raw_api_responses"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / f"{Path(filename).stem}.response").write_bytes(response)

        words = extract_text(response)
        by_filename[filename] = {
            "filename": filename,
            "start": start,
            "end": end,
            "transcription": words,
        }
        results = [by_filename[s["filename"]] for s in sections if s["filename"] in by_filename]
        save_results(output_path, results)

    results = [by_filename[s["filename"]] for s in sections if s["filename"] in by_filename]
    print(f"Done: {len(results)}/{len(sections)} transcriptions in {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())