"""Stage 11: cut the word timeline into 3-30 s chunks for training.

A chunk may only end in a pause between two words, never inside a word.
Among the allowed places, we prefer the end of a sentence and a length near
TARGET_SECONDS.
"""

MIN_SECONDS = 3.0
MAX_SECONDS = 30.0
TARGET_SECONDS = 12.0
MIN_PAUSE = 0.3
LONG_PAUSE = 2.0
PADDING = 0.2
SENTENCE_BONUS = 4.0
CLAUSE_BONUS = 2.0


def pause_after(words, i):
    """Seconds of silence between word i and the next word (None after the last word)."""
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


def choose_end(words, first):
    """Pick the word after which the chunk starting at `first` should end."""
    best = None
    widest = first

    for i in range(first, len(words)):
        duration = words[i]["end"] - words[first]["start"]
        if duration > MAX_SECONDS:
            break

        pause = pause_after(words, i)
        if pause is not None and pause > pause_after(words, widest):
            widest = i

        allowed = pause is None or pause >= MIN_PAUSE
        if allowed and duration >= MIN_SECONDS:
            if best is None or cost(words, first, i) < cost(words, first, best):
                best = i

        if pause is None or pause >= LONG_PAUSE:
            return best if best is not None else i

    return best if best is not None else widest


def chunk_times(words, first, last):
    """Start and end of the clip: the words plus up to PADDING of silence on each side."""
    if first == 0:
        before = PADDING
    else:
        before = min(PADDING, max(0.0, pause_after(words, first - 1)) / 2)

    if last == len(words) - 1:
        after = PADDING
    else:
        after = min(PADDING, max(0.0, pause_after(words, last)) / 2)

    start = max(0.0, words[first]["start"] - before)
    end = words[last]["end"] + after
    return round(start, 2), round(end, 2)


def make_chunks(words):
    """Cut the whole word list into chunks. Returns a list of dicts, in time order."""
    chunks = []
    first = 0

    while first < len(words):
        last = choose_end(words, first)
        start, end = chunk_times(words, first, last)
        chunks.append({
            "start": start,
            "end": end,
            "duration": round(end - start, 2),
            "first_word": first,
            "last_word": last,
            "text": " ".join(word["word"] for word in words[first:last + 1]),
        })
        first = last + 1

    return chunks