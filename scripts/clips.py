"""Stage 12: cut the chunks out of the recording, write the manifests and the metadata.

A NeMo manifest is a text file with one JSON object per line, one line per clip:
{"audio_filepath": "/.../test_0001.wav", "duration": 12.4, "text": "..."}
Training reads the clips and their texts from it.

metadata.json lives next to the clips: for every clip its path, duration,
text and word count.
"""

import json
import os
import subprocess
from pathlib import Path

import soundfile

SAMPLE_RATE = 16000


def decode_to_wav(audio_path, wav_path):
    """Convert the recording to 16 kHz mono WAV once, so clips can be cut from it quickly."""
    if wav_path.exists():
        return

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(audio_path),
         "-ac", "1", "-ar", str(SAMPLE_RATE), str(wav_path)],
        check=True,
    )


def write_clips(chunks, wav_path, clips_dir, name):
    """Write one WAV file per chunk, and add its path and exact length to the chunk."""
    clips_dir.mkdir(parents=True, exist_ok=True)
    for old_clip in clips_dir.glob("*.wav"):
        old_clip.unlink()

    for number, chunk in enumerate(chunks):
        first_sample = round(chunk["start"] * SAMPLE_RATE)
        last_sample = round(chunk["end"] * SAMPLE_RATE)
        audio, rate = soundfile.read(str(wav_path), start=first_sample, stop=last_sample)

        clip_path = clips_dir / f"{name}_{number:04d}.wav"
        soundfile.write(str(clip_path), audio, rate)

        chunk["audio_filepath"] = str(clip_path)
        chunk["duration"] = round(len(audio) / rate, 3)

    return chunks


def write_manifest(chunks, manifest_path, tiers):
    """Write the chunks whose tier is in `tiers` to a NeMo manifest."""
    lines = []
    for chunk in chunks:
        if chunk["tier"] in tiers:
            entry = {
                "audio_filepath": chunk["audio_filepath"],
                "duration": chunk["duration"],
                "text": chunk["text"],
                "tier": chunk["tier"],
                "reasons": chunk["reasons"],
            }
            lines.append(json.dumps(entry, ensure_ascii=False) + "\n")

    manifest_path.write_text("".join(lines), encoding="utf-8")
    return len(lines)


def write_metadata(chunks, metadata_path):
    """Write metadata.json: a list with one simple entry per clip.

    {"path": "wavs/test_0001.wav", "duration": 12.4, "text": "...", "word_count": 18}

    "path" is relative to the folder of metadata.json, so the chunks folder
    still works after copying it between the Mac and the remote.
    """
    folder = metadata_path.parent
    entries = []
    for chunk in chunks:
        entries.append({
            "path": Path(os.path.relpath(chunk["audio_filepath"], folder)).as_posix(),
            "duration": chunk["duration"],
            "text": chunk["text"],
            "word_count": len(chunk["text"].split()),
        })

    metadata_path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return entries
