"""Section times: which part of the recording each section is responsible for.

vad.ipynb saved every section with 0.5 s of extra audio on both sides, so
neighbouring sections share 1 s of audio. Words spoken in that shared second
appear in both sections. To use every word exactly once, each section "owns"
the time up to the middle of its overlap with each neighbour.
"""

import json
from pathlib import Path


def load_sections(path):
    """Read temporary_sections.json and add own_start / own_end to every section."""
    sections = json.loads(Path(path).read_text(encoding="utf-8"))
    sections.sort(key=lambda section: section["start"])

    for index, section in enumerate(sections):
        is_first = index == 0
        is_last = index == len(sections) - 1

        if is_first:
            section["own_start"] = section["start"]
        else:
            previous = sections[index - 1]
            section["own_start"] = (previous["end"] + section["start"]) / 2

        if is_last:
            section["own_end"] = section["end"]
        else:
            following = sections[index + 1]
            section["own_end"] = (section["end"] + following["start"]) / 2

    return sections


def to_recording_time(words, section):
    """The aligner's times count from the start of the section's WAV file.
    Add the section's start so they count from the start of the whole recording."""
    converted = []
    for word in words:
        new_word = dict(word)  # a copy, so the original list stays unchanged
        new_word["start"] = round(section["start"] + word["start"], 3)
        new_word["end"] = round(section["start"] + word["end"], 3)
        converted.append(new_word)
    return converted


def keep_owned(words, section):
    """Keep only the words whose middle lies in the time this section owns."""
    kept = []
    for word in words:
        middle = (word["start"] + word["end"]) / 2
        if section["own_start"] <= middle < section["own_end"]:
            kept.append(word)
    return kept