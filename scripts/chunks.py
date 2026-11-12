"""Stage 11: cut the word timeline into 3-30 s chunks for training.

A chunk may only end in a real pause: the aligner must see a gap between two
words, AND the audio in that gap must really be quiet. The aligner's word
times are spikes (see aligner.py), so a word often sounds a little longer
than its "end". That is why we listen to the audio and cut in the quietest
moment of the pause, so no word loses its first or last sound.

Among the allowed places, we prefer the end of a sentence and a length near
TARGET_SECONDS.
"""

import numpy
import soundfile

MIN_SECONDS = 3.0
MAX_SECONDS = 30.0
TARGET_SECONDS = 12.0
MIN_PAUSE = 0.3
LONG_PAUSE = 2.0
SENTENCE_BONUS = 4.0
CLAUSE_BONUS = 2.0

STEP = 0.01
QUIET_WINDOW = 0.1
EDGE_SEARCH = 0.6
SILENCE_BELOW_LOUD = 15.0


def load_recording(wav_path):
    """Read the recording and measure how loud it is, every STEP seconds.

    loudness[k] is the loudness in decibels (dB) of the QUIET_WINDOW seconds
    that start at k * STEP. 0 dB is the loudest possible sound, and every
    -10 dB is ten times less sound power: speech here is about -17 dB,
    pauses about -40 dB.

    silence: anything at least SILENCE_BELOW_LOUD dB quieter than the loud
    parts of this recording (its 90th percentile) counts as silence. Using
    the recording's own level means it also works for quieter recordings.
    """
    audio, rate = soundfile.read(str(wav_path))
    step = int(rate * STEP)
    pieces = audio[:len(audio) // step * step].reshape(-1, step)
    power = (pieces ** 2).mean(axis=1)

    window = int(QUIET_WINDOW / STEP)
    power = numpy.convolve(power, numpy.ones(window) / window, mode="valid")
    loudness = 10 * numpy.log10(power + 1e-12)

    return {
        "loudness": loudness,
        "silence": float(numpy.percentile(loudness, 90)) - SILENCE_BELOW_LOUD,
        "seconds": len(audio) / rate,
    }


def quietest(recording, start, end):
    """The middle time and the loudness of the quietest moment between start and end."""
    first = max(0, round(start / STEP))
    last = min(len(recording["loudness"]), round((end - QUIET_WINDOW) / STEP) + 1)
    if last <= first:
        return (start + end) / 2, 0.0

    k = first + int(numpy.argmin(recording["loudness"][first:last]))
    return k * STEP + QUIET_WINDOW / 2, float(recording["loudness"][k])


def cut_after(words, i, recording):
    """Where to cut in the pause after word i. i = -1 means before the first word.

    Returns (end, start, quiet):
    end    where the clip before the pause should end
    start  where the clip after the pause should start
    quiet  True if the audio there is silent, so cutting is safe

    A short pause gets one cut in its quietest moment (end == start).
    A long pause gets two, each within EDGE_SEARCH of its word, so clips
    do not carry seconds of silence.
    """
    if i >= 0:
        after = words[i]["end"]
    else:
        after = max(0.0, words[0]["start"] - EDGE_SEARCH)

    if i + 1 < len(words):
        before = words[i + 1]["start"]
    else:
        before = min(recording["seconds"], words[-1]["end"] + EDGE_SEARCH)

    if before - after <= 2 * EDGE_SEARCH:
        middle, loudness = quietest(recording, after, before)
        return middle, middle, loudness <= recording["silence"]

    end, end_loudness = quietest(recording, after, after + EDGE_SEARCH)
    start, start_loudness = quietest(recording, before - EDGE_SEARCH, before)
    quiet = max(end_loudness, start_loudness) <= recording["silence"]
    return end, start, quiet


def pause_after(words, i):
    """Seconds between word i and the next word (None after the last word)."""
    if i == len(words) - 1:
        return None
    return words[i + 1]["start"] - words[i]["end"]


def cost(words, first, i):
    """How bad it is to end a chunk that starts at word `first` after word i (lower is better)."""
    duration = words[i]["end"] - words[first]["start"]
    value = abs(duration - TARGET_SECONDS)
    if words[i]["word"][-1] in ".!?…":
        value -= SENTENCE_BONUS
    elif words[i]["word"][-1] in ",;:":
        value -= CLAUSE_BONUS
    return value


def choose_end(words, first, recording):
    """Pick the word after which the chunk starting at `first` should end.

    A long pause (LONG_PAUSE) always ends the chunk. If ending earlier would
    leave a piece shorter than MIN_SECONDS before that pause, the chunk runs
    on to the pause instead, so no piece is too short to use.
    """
    best = None
    widest = first

    for i in range(first, len(words)):
        duration = words[i]["end"] - words[first]["start"]
        if duration > MAX_SECONDS:
            break

        pause = pause_after(words, i)
        if pause is not None and pause > pause_after(words, widest):
            widest = i

        if pause is None:
            allowed = True
        else:
            allowed = pause >= MIN_PAUSE and cut_after(words, i, recording)[2]

        if allowed and duration >= MIN_SECONDS:
            if best is None or cost(words, first, i) < cost(words, first, best):
                best = i

        if pause is None or pause >= LONG_PAUSE:
            if best is None:
                return i
            leftover = words[i]["end"] - words[best + 1]["start"] if best < i else 0.0
            if 0.0 < leftover < MIN_SECONDS:
                return i
            return best

    return best if best is not None else widest


def make_chunks(words, wav_path):
    """Cut the whole word list into chunks. Returns a list of dicts, in time order.

    wav_path: the recording the word times belong to (16 kHz WAV from clips.decode_to_wav).
    Each chunk ends where the next one starts, or earlier in a long pause:
    no audio is used twice. "quiet_cuts" is False if the chunk had to be cut
    where the audio is not silent (grading.py sends it to review).
    """
    recording = load_recording(wav_path)
    chunks = []
    first = 0

    while first < len(words):
        last = choose_end(words, first, recording)
        _, start, quiet_start = cut_after(words, first - 1, recording)
        end, _, quiet_end = cut_after(words, last, recording)
        chunks.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "duration": round(end - start, 2),
            "first_word": first,
            "last_word": last,
            "text": " ".join(word["word"] for word in words[first:last + 1]),
            "quiet_cuts": quiet_start and quiet_end,
        })
        first = last + 1

    return chunks