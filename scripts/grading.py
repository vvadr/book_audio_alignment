"""Stage 12: decide how much to trust each chunk.

clean   the alignment looks right and the book confirms every word
review  the alignment looks right, but something needs a human ear
reject  the alignment looks wrong: the text and the audio may not belong together
"""

from chunks import MAX_SECONDS, MIN_SECONDS

MIN_WORD_SCORE = -3.0
MAX_WORD_SECONDS = 2.0
MIN_LETTERS_PER_SECOND = 5.0
MAX_LETTERS_PER_SECOND = 20.0
MIN_CONFIRMED_SHARE = 1.0
MAX_TRIMMED = 0.5


def grade(chunk, words):
    """Return (tier, reasons) for one chunk. `words` is the whole word list."""
    chunk_words = words[chunk["first_word"]:chunk["last_word"] + 1]
    problems = []
    doubts = []

    if not MIN_SECONDS <= chunk["duration"] <= MAX_SECONDS:
        problems.append(f"length {chunk['duration']} s")

    for word in chunk_words:
        if word["score"] < MIN_WORD_SCORE:
            problems.append(f"unsure word '{word['word']}' (score {word['score']})")
        if word["end"] - word["start"] > MAX_WORD_SECONDS:
            problems.append(f"stretched word '{word['word']}'")

    speech = chunk_words[-1]["end"] - chunk_words[0]["start"]
    letters = sum(char.isalpha() for char in chunk["text"])
    rate = letters / speech
    if not MIN_LETTERS_PER_SECOND <= rate <= MAX_LETTERS_PER_SECOND:
        problems.append(f"speaking rate {rate:.1f} letters/s")

    confirmed = sum(word["confirmed"] for word in chunk_words) / len(chunk_words)
    if confirmed < MIN_CONFIRMED_SHARE:
        missing = [word["word"] for word in chunk_words if not word["confirmed"]]
        doubts.append("not confirmed by the book: " + " ".join(missing))

    for word in chunk_words:
        if word["trimmed"] > MAX_TRIMMED:
            doubts.append(f"repaired word '{word['word']}' (trimmed {word['trimmed']} s)")

    if problems:
        return "reject", problems
    if doubts:
        return "review", doubts
    return "clean", []


def grade_chunks(chunks, words):
    """Add "tier" and "reasons" to every chunk."""
    for chunk in chunks:
        tier, reasons = grade(chunk, words)
        chunk["tier"] = tier
        chunk["reasons"] = reasons
    return chunks