"""Section times: which part of the recording each section is responsible for.

vad.ipynb saved every section with 0.5 s of extra audio on both sides, so
neighbouring sections share 1 s of audio. Words spoken in that shared second
appear in both sections. To use every word exactly once, each section "owns"
the time up to the middle of its overlap with each neighbour.
"""

import json
from pathlib import Path

from segments import normalize_word

NEAR_SECONDS = 1.0


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
        new_word = dict(word)
        new_word["start"] = round(section["start"] + word["start"], 3)
        new_word["end"] = round(section["start"] + word["end"], 3)
        converted.append(new_word)
    return converted


def owns(section, word):
    """True if the middle of the word lies in the time this section owns."""
    middle = (word["start"] + word["end"]) / 2
    return section["own_start"] <= middle < section["own_end"]


def same_word_nearby(word, others):
    """True if `others` has the same word within NEAR_SECONDS of this one."""
    for other in others:
        same_text = normalize_word(other["word"]) == normalize_word(word["word"])
        close = abs(other["start"] - word["start"]) <= NEAR_SECONDS
        if same_text and close:
            return True
    return False


def stitch(sections, words_per_section):
    """Join the words of all sections into one list for the whole recording.

    sections:          from load_sections
    words_per_section: one list of words per section, already in recording time

    Step 1 keeps every word whose time the section owns.
    Step 2 rescues a word outside its section's time when the neighbouring
    sections did not keep the same word nearby. That happens when only one
    section heard the word ("Men"), or when two sections disagree about its
    time and each leaves it to the other ("Soʻng").
    """
    kept = [[] for section in sections]
    left_out = []

    for i in range(len(sections)):
        for word in words_per_section[i]:
            if owns(sections[i], word):
                kept[i].append(word)
            else:
                left_out.append((i, word))

    for i, word in left_out:
        neighbours = kept[max(0, i - 1)] + kept[min(len(kept) - 1, i + 1)]
        if not same_word_nearby(word, neighbours):
            kept[i].append(word)

    all_words = []
    for section_words in kept:
        all_words.extend(section_words)

    all_words.sort(key=lambda word: word["start"])
    return all_words