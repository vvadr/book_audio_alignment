"""Transcribe long paired-book recordings in smaller, resumable chunks."""

import argparse
import copy
import hashlib
import http.client
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from transcribe import API_URL, transcribe
from transcribe_paired import (
    DEFAULT_MAX_DURATION, PAIRED_DIR, audio_duration, find_audio, outputs_exist,
    positive_seconds, save_outputs,
)

DEFAULT_CHUNK_DURATION = 300.0


def cache_directory(book, audio, chunk_duration, url):
    """Keep checkpoints separate when the source, chunk size or endpoint changes."""
    stat = audio.stat()
    identity = [str(audio.resolve()), stat.st_size, stat.st_mtime_ns, chunk_duration,
                url or os.environ.get("TRANSCRIBE_URL") or API_URL]
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]
    return book / ".transcribe_long_chunks" / key


def parse_chunk_response(response):
    """Allow silent chunks while still validating the API's response shape."""
    payload = json.loads(response.decode("utf-8-sig"))
    if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
        raise ValueError("API response must have a segments list")
    segments = payload["segments"]
    if any(not isinstance(segment, dict) or not isinstance(segment.get("text"), str)
           for segment in segments):
        raise ValueError("Each transcription segment must have a text field")
    return payload, " ".join(segment["text"].strip() for segment in segments
                             if segment["text"].strip())


def offset_segment(segment, offset):
    """Translate chunk-relative segment and word timestamps to recording time."""
    segment = copy.deepcopy(segment)
    items = [segment]
    if isinstance(segment.get("words"), list):
        items.extend(word for word in segment["words"] if isinstance(word, dict))
    for item in items:
        for key in ("start", "end"):
            if key in item:
                value = item[key]
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value)):
                    raise ValueError(f"Invalid {key} timestamp in API response")
                item[key] = round(value + offset, 6)
    return segment


def process_book(book, *, url, timeout, force, max_duration, chunk_duration):
    audio = find_audio(book)
    if audio is None or outputs_exist(book, force):
        return
    duration = audio_duration(audio)
    if duration <= max_duration:
        print(f"Skipping {book.name}: {duration:.1f}s is at or below {max_duration:g}s")
        return

    cache = cache_directory(book, audio, chunk_duration, url)
    cache.mkdir(parents=True, exist_ok=True)
    count = math.ceil(duration / chunk_duration)
    segments, chunks, texts = [], [], []
    print(f"Transcribing {audio} in {count} chunks...", flush=True)
    with tempfile.TemporaryDirectory() as temporary:
        chunk_audio = Path(temporary) / "chunk.wav"
        for index in range(count):
            start = index * chunk_duration
            length = min(chunk_duration, duration - start)
            checkpoint = cache / f"{index:06d}.json"
            if checkpoint.is_file() and not force:
                print(f"  Chunk {index + 1}/{count}: using saved response", flush=True)
                response = checkpoint.read_bytes()
                payload, text = parse_chunk_response(response)
            else:
                print(f"  Chunk {index + 1}/{count}: transcribing from {start:g}s", flush=True)
                subprocess.run(
                    ["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(start),
                     "-i", str(audio), "-t", str(length), "-vn", "-ac", "1",
                     "-ar", "16000", "-c:a", "pcm_s16le", str(chunk_audio)],
                    check=True,
                )
                response = transcribe(chunk_audio, url=url, timeout=timeout)
                payload, text = parse_chunk_response(response)
                for segment in payload["segments"]:
                    offset_segment(segment, start)
                with tempfile.NamedTemporaryFile(dir=cache, delete=False) as temp:
                    temp_path = Path(temp.name)
                    try:
                        temp.write(response)
                    except BaseException:
                        temp_path.unlink(missing_ok=True)
                        raise
                temp_path.replace(checkpoint)
            segments.extend(offset_segment(segment, start) for segment in payload["segments"])
            texts.append(text)
            chunks.append({"start": start, "end": start + length, "response": payload})

    text = " ".join(text for text in texts if text)
    if not text:
        raise ValueError("API responses contain no transcription text")
    metadata = {"duration": duration, "segments": segments, "chunks": chunks}
    save_outputs(book, metadata, text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-dir", type=Path, default=PAIRED_DIR)
    parser.add_argument("--url", default=None, help="override TRANSCRIBE_URL / API_URL")
    parser.add_argument("--timeout", type=positive_seconds, default=600, help="socket timeout in seconds")
    parser.add_argument("--max-duration", type=positive_seconds, default=DEFAULT_MAX_DURATION,
                        help="process recordings above this many seconds (default: 600)")
    parser.add_argument("--chunk-duration", type=positive_seconds, default=DEFAULT_CHUNK_DURATION,
                        help="maximum duration of each API request in seconds (default: 300)")
    parser.add_argument("--force", action="store_true", help="replace outputs and re-transcribe all chunks")
    args = parser.parse_args()
    if not args.paired_dir.is_dir():
        parser.error(f"paired directory not found: {args.paired_dir}")

    failed = False
    for book in sorted(path for path in args.paired_dir.iterdir() if path.is_dir()):
        try:
            process_book(book, url=args.url, timeout=args.timeout, force=args.force,
                         max_duration=args.max_duration, chunk_duration=args.chunk_duration)
        except (OSError, ValueError, RuntimeError, http.client.HTTPException, subprocess.SubprocessError) as error:
            print(f"Failed to transcribe {book.name}: {error}", file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
