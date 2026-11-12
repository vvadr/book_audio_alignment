"""Stage 7: cut the whole recording into sections of about one minute.

The same method as vad.ipynb, as a function that works for any book:
1. Silero VAD finds where someone speaks. The gaps between are pauses.
2. Every MAX_SECONDS we cut at the pause closest to that time (looking up to
   SEARCH_SECONDS either way), so a cut rarely falls inside a word.
3. Each section is saved with OVERLAP seconds of extra audio on both sides,
   so neighbouring sections share 1 s (sections.py explains why).
"""

import json

import soundfile
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

SAMPLE_RATE = 16000
MAX_SECONDS = 60
SEARCH_SECONDS = 10
OVERLAP = 0.5


def find_pauses(audio):
    """The middle of every pause between two speech regions, in seconds."""
    regions = get_speech_timestamps(
        torch.from_numpy(audio),
        load_silero_vad(),
        sampling_rate=SAMPLE_RATE,
        return_seconds=True,
        min_silence_duration_ms=300,
        speech_pad_ms=200,
    )

    pauses = []
    for current, following in zip(regions, regions[1:]):
        if following["start"] > current["end"]:
            pauses.append((current["end"] + following["start"]) / 2)
    return pauses


def plan_sections(pauses, duration):
    """Where to cut: a list of (start, end) times in seconds, before adding the overlap."""
    sections = []
    start = 0.0

    while start < duration:
        target = min(start + MAX_SECONDS, duration)
        nearby = [
            pause for pause in pauses
            if target - SEARCH_SECONDS <= pause <= target + SEARCH_SECONDS and pause > start
        ]

        if nearby and target < duration:
            end = min(nearby, key=lambda pause: abs(pause - target))
        else:
            end = target

        sections.append((start, end))
        start = end

    return sections


def write_sections(wav_path, sections_dir, sections_json):
    """Cut the 16 kHz recording into section WAVs and save their times in sections_json.

    Returns the list that sections_json holds:
    [{"filename": "section_0000.wav", "start": 0.0, "end": 60.4}, ...]
    "start" and "end" include the overlap, like temporary_sections.json.
    """
    audio, rate = soundfile.read(str(wav_path), dtype="float32")
    if rate != SAMPLE_RATE:
        raise ValueError(f"{wav_path} is {rate} Hz, expected {SAMPLE_RATE}")
    duration = len(audio) / rate
    sections_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for number, (start, end) in enumerate(plan_sections(find_pauses(audio), duration)):
        start = max(0.0, start - OVERLAP)
        end = min(duration, end + OVERLAP)
        filename = f"section_{number:04d}.wav"

        soundfile.write(str(sections_dir / filename), audio[int(start * rate):int(end * rate)], rate)
        saved.append({"filename": filename, "start": start, "end": end})

    sections_json.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    return saved
