"""Run the whole pipeline for one or more books in data/paired.

usage:
    python scripts/process_book.py data/paired/187_baxt-daraxti [more book folders ...]

Stages (the same as vad.ipynb, align_sections.ipynb and make_chunks.ipynb):
    6   decode the recording to 16 kHz mono WAV
    7   cut it into ~60 s sections at pauses (VAD)
    8   transcribe every section with the API
    9   align every section with NeMo and stitch the words together
    10  compare every word with the book
    11  cut the words into 3-30 s chunks
    12  grade the chunks, write the clips, manifests and metadata

Everything goes into <book>/pipeline/. Stages 6-8 are slow, so they keep what
is already saved: after an interruption, run the same command again and it
continues. Stages 9-12 are quick and always run again, so they use the newest code.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from book_check import check_against_book
from chunks import make_chunks
from clips import decode_to_wav, write_clips, write_manifest, write_metadata
from grading import grade_chunks
from paths import book_paths
from sections import load_sections, stitch, to_recording_time
from segments import join_segments
from transcribe import transcribe
from transcribe_paired import find_audio
from uzbek_text import fix_apostrophes, to_model_text
from vad_sections import write_sections


def say(stage, message):
    print(f"[{time.strftime('%H:%M:%S')}] stage {stage}: {message}", flush=True)


def transcribe_sections(paths, sections):
    """Stage 8: send every section that has no saved API response yet."""
    paths["raw_responses"].mkdir(parents=True, exist_ok=True)
    missing = [
        section for section in sections
        if not (paths["raw_responses"] / f"{Path(section['filename']).stem}.response").exists()
    ]
    say(8, f"{len(sections) - len(missing)} of {len(sections)} sections already transcribed")

    for number, section in enumerate(missing, start=1):
        response = transcribe(paths["sections_dir"] / section["filename"])
        json.loads(response)
        response_path = paths["raw_responses"] / f"{Path(section['filename']).stem}.response"
        response_path.write_bytes(response)
        if number % 25 == 0 or number == len(missing):
            say(8, f"transcribed {number} of {len(missing)}")


def align_book(paths, model):
    """Stages 8-9 in align_sections.ipynb: section texts, alignment, stitching."""
    from aligner import align

    sections = load_sections(paths["sections_json"])
    book_words = paths["book_text"].read_text(encoding="utf-8").split()

    words_per_section = []
    empty = 0
    for section in sections:
        response_path = paths["raw_responses"] / f"{Path(section['filename']).stem}.response"
        response = json.loads(response_path.read_text(encoding="utf-8"))
        text = to_model_text(fix_apostrophes(join_segments(response["segments"], book_words)))

        if not text:
            empty += 1
            words_per_section.append([])
            continue

        words = align(model, paths["sections_dir"] / section["filename"], text)
        words_per_section.append(to_recording_time(words, section))

    all_words = stitch(sections, words_per_section)
    out_of_order = sum(after["start"] < before["start"] for before, after in zip(all_words, all_words[1:]))
    say(9, f"{len(all_words)} words from {len(sections)} sections ({empty} sections had no speech), "
           f"{out_of_order} out of time order")

    paths["alignment_json"].write_text(json.dumps(all_words, ensure_ascii=False, indent=2), encoding="utf-8")
    return all_words


def make_book_chunks(book, paths, words):
    """Stages 10-12 in make_chunks.ipynb: book check, chunks, grades, clips, manifests, metadata."""
    words = check_against_book(words, paths["book_text"].read_text(encoding="utf-8"))
    confirmed = sum(word["confirmed"] for word in words)
    say(10, f"{confirmed} of {len(words)} words confirmed by the book ({100 * confirmed / len(words):.1f}%)")

    chunks = make_chunks(words, paths["recording_wav"])
    durations = [chunk["duration"] for chunk in chunks]
    say(11, f"{len(chunks)} chunks, {min(durations)}-{max(durations)} s, "
            f"average {sum(durations) / len(durations):.1f} s")

    chunks = grade_chunks(chunks, words)
    chunks = write_clips(chunks, paths["recording_wav"], paths["clips_dir"], book.name)
    write_manifest(chunks, paths["manifest_all"], {"clean", "review", "reject"})
    write_manifest(chunks, paths["manifest_clean"], {"clean"})
    write_metadata(chunks, paths["chunks_metadata"])

    for tier in ["clean", "review", "reject"]:
        group = [chunk for chunk in chunks if chunk["tier"] == tier]
        say(12, f"{tier:7} {len(group):4} clips {sum(c['duration'] for c in group) / 60:7.2f} min")
    return chunks


def process(book, model):
    paths = book_paths(book)
    print(f"\n===== {book.name}", flush=True)

    audio = find_audio(book)
    if audio is None:
        return False
    decode_to_wav(audio, paths["recording_wav"])
    say(6, f"recording {paths['recording_wav']}")

    if paths["sections_json"].exists():
        sections = json.loads(paths["sections_json"].read_text(encoding="utf-8"))
        say(7, f"{len(sections)} sections already saved")
    else:
        sections = write_sections(paths["recording_wav"], paths["sections_dir"], paths["sections_json"])
        say(7, f"cut {len(sections)} sections")

    transcribe_sections(paths, sections)
    words = align_book(paths, model)
    make_book_chunks(book, paths, words)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("books", nargs="+", type=Path, help="book folders in data/paired")
    args = parser.parse_args()

    from aligner import load_model
    model = load_model()

    failed = []
    for book in args.books:
        if not process(book.resolve(), model):
            failed.append(book.name)

    if failed:
        print("Skipped:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
